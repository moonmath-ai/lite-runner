"""genai_runner -- Experiment runner for video diffusion models with W&B tracking."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import questionary
import wandb

# Mapping from Param.type strings to Python types for argparse.
_PARAM_TYPE_MAP: dict[str, type] = {"int": int, "float": float, "str": str, "path": str}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Param:
    """A CLI parameter for the model command.

    Args:
        name: Parameter name, used as argparse dest (underscored) and CLI flag (hyphenated).
        type: One of "str", "int", "float", "path", "bool". Determines argparse type and TUI widget.
        default: Default value.  Can be a callable (called at resolve time).
        choices: Allowed values (shown as select in TUI).
        help: Description shown in --help and TUI prompt.
        flag: Override the CLI flag (default: --<name with hyphens>).
        value: Fixed value with $output interpolation. Never prompted.  Can be a list for multi-value flags.
        types: Per-element types for multi-value flags, e.g. ["path", "float", "float"].
            Implies nargs=len(types).  Each part gets its own type in argparse and TUI.
        labels: Names for each part, used in TUI prompts and --help metavar.
            e.g. labels=["path", "start_frame", "strength"] prompts separately for each.
        hidden: If True, not shown in --help.
        log_as: If set, upload the file to W&B. One of "video", "image", "artifact", "text".
        log_when: "before" (input file) or "after" (output file).  Auto-inferred from $output in value.
    """

    name: str
    type: str = "str"
    default: Any = None
    choices: list[str] | None = None
    help: str = ""
    flag: str | None = None
    value: Any = None
    types: list[str] | None = None
    labels: list[str] | None = None
    hidden: bool = False
    log_as: str | None = None
    log_when: str | None = None

    def __post_init__(self) -> None:
        self._dest = self.name.replace("-", "_")
        if self.flag is None:
            self.flag = f"--{self.name.replace('_', '-')}"
        if self.log_when is None and self.log_as is not None:
            self.log_when = "after" if self._value_contains_output() else "before"

    def _value_contains_output(self) -> bool:
        """Check whether $output appears anywhere in self.value."""
        if self.value is None:
            return False
        if isinstance(self.value, list):
            return any("$output" in str(v) for v in self.value)
        return "$output" in str(self.value)

    @property
    def dest(self) -> str:
        return self._dest

    @property
    def nargs(self) -> int | None:
        return len(self.types) if self.types else None

    @property
    def is_fixed(self) -> bool:
        """Params with a value= are never prompted or parsed from CLI."""
        return self.value is not None

    @property
    def needs_prompt(self) -> bool:
        return not self.is_fixed and self.default is None


@dataclass
class Output:
    """An output file not tied to any Param (uncontrolled location).

    Args:
        path: Absolute path or $output-relative path.
        log_as: One of "video", "image", "artifact", "text".
        copy_to: If set, copy the file here ($output interpolated) before logging.
    """

    path: str
    log_as: str = "artifact"
    copy_to: str | None = None


@dataclass
class Metric:
    """A value to extract from stdout via regex.

    Args:
        name: Metric name in W&B summary.
        pattern: Regex with one capture group.
        type: "float" or "str".  Determines how the captured value is stored.
    """

    name: str
    pattern: str
    type: str = "float"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Runner:
    command: str | list[str]
    params: list[Param] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    wandb_project: str | None = None  # default: git repo name

    def run(self, overrides: dict[str, object] | None = None) -> None:
        """Execute the full run lifecycle."""
        if not hasattr(self, "_cli_args"):
            self._cli_args = self._parse_cli_args()
        resolved = self._resolve_values(self._cli_args, overrides or {})
        interactive = not resolved.pop("_no_interactive")
        dry_run = resolved.pop("_dry_run")
        keep_outputs = resolved.pop("_keep_outputs")

        # Prompt for missing params (interactive mode)
        self._prompt_missing(resolved, interactive)

        # Git info
        git_info = _collect_git_info()
        project = self.wandb_project or git_info.get("repo", "genai-runs")

        # Dry-run: show command without W&B or output dir
        if dry_run:
            param_values = self._interpolate_output(resolved, Path("/tmp/dry-run-output"))
            cmd = self._build_command(param_values)
            print(f"[dry-run] Project: {project}")
            print(f"[dry-run] Command:\n  {shlex.join(cmd)}")
            return

        # W&B init
        run_tags = list(self.tags)
        if git_info.get("dirty"):
            run_tags.append("dirty-git")

        run_name = resolved.pop("_run_name", None)
        wb_run = wandb.init(
            project=project,
            name=run_name,
            tags=run_tags,
            save_code=True,
            config={
                **{f"param/{k}": v for k, v in resolved.items() if not k.startswith("_")},
                **{f"git/{k}": v for k, v in git_info.items()},
                "meta/hostname": os.uname().nodename,
                "meta/datetime": datetime.datetime.now().isoformat(),
                "meta/command": shlex.join(self.command) if isinstance(self.command, list) else self.command,
            },
        )

        # Output dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        dir_name = wb_run.name or wb_run.id or "run"
        output_dir = Path.home() / "genai_runs" / project / f"{timestamp}_{dir_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save code snapshot (git archive + dirty diff)
        _save_code_snapshot(wb_run, output_dir, git_info)

        # Interpolate $output in fixed-value params
        param_values = self._interpolate_output(resolved, output_dir)

        # Log input files (log_when == "before")
        self._log_files(wb_run, param_values, output_dir, when="before")

        # Build command
        cmd = self._build_command(param_values)
        wb_run.config.update({"meta/full_command": shlex.join(cmd)})

        print(f"Output dir: {output_dir}")
        print(f"W&B run: {wb_run.url}")
        print(f"Command:\n  {shlex.join(cmd)}")
        print("=" * 60)

        # Execute
        exit_code, duration, stdout_text = self._execute(cmd, output_dir)

        # Extract metrics from stdout
        self._extract_metrics(wb_run, stdout_text)

        # Log output files (log_when == "after")
        self._log_files(wb_run, param_values, output_dir, when="after")

        # Log uncontrolled outputs
        self._log_extra_outputs(wb_run, output_dir)

        # Log the run log files
        log_files = [output_dir / name for name in ("run.log", "stdout.log", "stderr.log")]
        existing_logs = [f for f in log_files if f.exists()]
        if existing_logs:
            artifact = wandb.Artifact(f"logs-{wb_run.id}", type="log")
            for f in existing_logs:
                artifact.add_file(str(f))
            wb_run.log_artifact(artifact)

        # Count outputs for summary
        num_videos = self._count_logged("video")
        num_images = self._count_logged("image")

        # Finalize
        failed = exit_code != 0
        status = "failed" if failed else "success"
        if failed:
            wb_run.tags = [*run_tags, "failed"]
        wb_run.summary.update({
            "exit_code": exit_code,
            "duration_seconds": duration,
            "status": status,
            "num_videos": num_videos,
            "num_images": num_images,
            "output_dir": str(output_dir),
        })
        wb_run.finish(exit_code=exit_code)

        print("=" * 60)
        print(f"Status: {status} (exit code {exit_code})")
        print(f"Duration: {duration:.1f}s")
        print(f"Output dir: {output_dir}")

        if not keep_outputs:
            print(f"(use --keep-outputs to preserve local files, or delete with: rm -rf '{output_dir}')")

    # -----------------------------------------------------------------------
    # CLI parsing
    # -----------------------------------------------------------------------

    def _parse_cli_args(self) -> dict:
        parser = argparse.ArgumentParser(
            description="genai_runner experiment launcher",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Built-in flags
        parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
        parser.add_argument("-n", "--no-interactive", action="store_true",
                            help="Non-interactive mode; fail if required params are missing")
        parser.add_argument("--keep-outputs", action="store_true",
                            help="Keep local output directory (always kept, flag is a no-op reminder)")
        parser.add_argument("--run-name", default=None, help="Override W&B run name")
        parser.add_argument("--wandb-project", default=None, help="Override W&B project name")

        for p in self.params:
            if p.is_fixed:
                continue
            kwargs: dict = {
                "default": None,
                "help": argparse.SUPPRESS if p.hidden else (p.help or None),
            }
            if p.type == "bool":
                kwargs.update(action="store_true", default=False)
            else:
                kwargs["type"] = _PARAM_TYPE_MAP.get(p.type, str)
                if p.nargs is not None:
                    # Multi-value: argparse gets nargs=N, all as str (typed later)
                    kwargs["nargs"] = p.nargs
                    kwargs["type"] = str
                    if p.labels and not p.hidden:
                        kwargs["metavar"] = tuple(p.labels)
                        kwargs["help"] = f"{p.help or ''} ({' '.join(p.labels)})".strip()
            if p.choices:
                kwargs["choices"] = p.choices
            parser.add_argument(p.flag, dest=p.dest, **kwargs)

        ns = parser.parse_args()
        result = {}
        for p in self.params:
            if not p.is_fixed:
                result[p.dest] = getattr(ns, p.dest, None)
        result["_dry_run"] = ns.dry_run
        result["_no_interactive"] = ns.no_interactive
        result["_keep_outputs"] = ns.keep_outputs
        if ns.run_name:
            result["_run_name"] = ns.run_name
        if ns.wandb_project:
            self.wandb_project = ns.wandb_project
        return result

    # -----------------------------------------------------------------------
    # Resolve values: apply overrides and callable defaults
    # -----------------------------------------------------------------------

    def _resolve_values(self, cli_args: dict, overrides: dict) -> dict:
        resolved = dict(cli_args)
        for p in self.params:
            if p.is_fixed:
                continue
            # Override from run(overrides=...) takes priority over CLI
            if p.dest in overrides:
                resolved[p.dest] = overrides[p.dest]
            # Apply callable defaults
            if resolved.get(p.dest) is None and p.default is not None:
                resolved[p.dest] = p.default() if callable(p.default) else p.default
            # Cast multi-value args to per-element types
            if p.types and resolved.get(p.dest) is not None:
                resolved[p.dest] = _cast_nargs(resolved[p.dest], p.types)
        return resolved

    # -----------------------------------------------------------------------
    # Interactive prompts
    # -----------------------------------------------------------------------

    def _prompt_missing(self, resolved: dict, interactive: bool) -> None:
        missing = [
            p for p in self.params
            if not p.is_fixed and p.type != "bool" and resolved.get(p.dest) is None
        ]

        if not missing:
            return

        if not interactive:
            names = [p.name for p in missing]
            print(f"Error: missing required params: {', '.join(names)}", file=sys.stderr)
            print("Run without -n for interactive mode, or pass them on the command line.", file=sys.stderr)
            sys.exit(2)

        for p in missing:
            if p.nargs is not None:
                resolved[p.dest] = self._prompt_nargs(p)
            else:
                resolved[p.dest] = self._prompt_single(p)

    def _prompt_single(self, p: Param) -> Any:
        label = p.help or p.name
        if p.choices:
            answer = questionary.select(f"{label}:", choices=p.choices).ask()
        elif p.type == "path":
            answer = questionary.path(f"{label}:").ask()
        else:
            answer = questionary.text(f"{label}:").ask()

        if answer is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)

        if p.type == "int":
            answer = int(answer)
        elif p.type == "float":
            answer = float(answer)
        return answer

    def _prompt_nargs(self, p: Param) -> list:
        labels = p.labels or [f"{p.name}[{i}]" for i in range(p.nargs)]
        element_types = p.types or [p.type] * p.nargs
        parts = []
        for label, etype in zip(labels, element_types):
            if etype == "path":
                answer = questionary.path(f"{p.name} {label}:").ask()
            else:
                answer = questionary.text(f"{p.name} {label}:").ask()
            if answer is None:
                print("Cancelled.", file=sys.stderr)
                sys.exit(1)
            parts.append(answer)
        return _cast_nargs(parts, element_types)

    # -----------------------------------------------------------------------
    # $output interpolation
    # -----------------------------------------------------------------------

    def _interpolate_output(self, resolved: dict, output_dir: Path) -> dict:
        """Return a copy of resolved with $output replaced in fixed-value params."""
        param_values = dict(resolved)
        out = str(output_dir)
        for p in self.params:
            if not p.is_fixed:
                continue
            if isinstance(p.value, list):
                param_values[p.dest] = [str(v).replace("$output", out) for v in p.value]
            else:
                val = str(p.value).replace("$output", out)
                if val != str(p.value):
                    # Value contained $output -- ensure parent dirs exist for the output file
                    Path(val).parent.mkdir(parents=True, exist_ok=True)
                param_values[p.dest] = val
        return param_values

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def _build_command(self, param_values: dict) -> list[str]:
        parts = list(self.command) if isinstance(self.command, list) else shlex.split(self.command)
        for p in self.params:
            val = param_values.get(p.dest)
            if val is None or (p.type == "bool" and not val):
                continue
            if p.type == "bool":
                parts.append(p.flag)
            elif isinstance(val, list):
                parts.append(p.flag)
                parts.extend(str(v) for v in val)
            else:
                parts.extend([p.flag, str(val)])
        return parts

    # -----------------------------------------------------------------------
    # Subprocess execution
    # -----------------------------------------------------------------------

    def _execute(self, cmd: list[str], output_dir: Path) -> tuple[int, float, str]:
        stdout_lines: list[str] = []
        lock = threading.Lock()

        with ExitStack() as stack:
            log_combined = stack.enter_context(open(output_dir / "run.log", "w"))
            log_stdout = stack.enter_context(open(output_dir / "stdout.log", "w"))
            log_stderr = stack.enter_context(open(output_dir / "stderr.log", "w"))

            run_env = {**os.environ, **self.env}

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
            )

            def stream_pipe(pipe, sys_stream, file_log, *, prefix="", capture=False):
                for raw_line in iter(pipe.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    sys_stream.write(line)
                    sys_stream.flush()
                    file_log.write(line)
                    file_log.flush()
                    with lock:
                        log_combined.write(prefix + line)
                        log_combined.flush()
                        if capture:
                            stdout_lines.append(line)
                pipe.close()

            t_out = threading.Thread(
                target=stream_pipe,
                args=(proc.stdout, sys.stdout, log_stdout),
                kwargs={"capture": True},
            )
            t_err = threading.Thread(
                target=stream_pipe,
                args=(proc.stderr, sys.stderr, log_stderr),
                kwargs={"prefix": "[stderr] "},
            )

            start = time.monotonic()
            t_out.start()
            t_err.start()

            try:
                proc.wait()
            except KeyboardInterrupt:
                print("\n[genai_runner] Ctrl-C received, terminating subprocess...")
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

            t_out.join()
            t_err.join()
            duration = time.monotonic() - start

        return proc.returncode, duration, "".join(stdout_lines)

    # -----------------------------------------------------------------------
    # File logging to W&B
    # -----------------------------------------------------------------------

    def _log_files(self, wb_run, param_values: dict, output_dir: Path, when: str) -> None:
        for p in self.params:
            if p.log_as is None or p.log_when != when:
                continue
            path_val = param_values.get(p.dest)
            if path_val is None:
                continue
            # For multi-value, the first element is the path
            if isinstance(path_val, list):
                path_val = path_val[0]
            path = Path(path_val)
            if not path.exists():
                print(f"[genai_runner] Warning: {path} not found, skipping upload")
                continue
            _upload_file(wb_run, path, p.log_as, label=p.name)

    def _log_extra_outputs(self, wb_run, output_dir: Path) -> None:
        def interpolate(s: str) -> str:
            return s.replace("$output", str(output_dir))

        for o in self.outputs:
            src = Path(interpolate(o.path))
            if not src.exists():
                print(f"[genai_runner] Warning: {src} not found, skipping")
                continue
            if o.copy_to:
                dst = Path(interpolate(o.copy_to))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                src = dst
            _upload_file(wb_run, src, o.log_as)

    # -----------------------------------------------------------------------
    # Metric extraction
    # -----------------------------------------------------------------------

    def _extract_metrics(self, wb_run, stdout_text: str) -> None:
        for m in self.metrics:
            matches = re.findall(m.pattern, stdout_text)
            if not matches:
                continue
            raw = matches[-1]  # last match wins
            if m.type == "float":
                try:
                    val: object = float(raw)
                except ValueError:
                    val = raw
            else:
                val = raw
            wb_run.summary[m.name] = val

    def _count_logged(self, media_type: str) -> int:
        """Count params and outputs that log as the given media type."""
        from_params = sum(1 for p in self.params if p.log_as == media_type and p.log_when == "after")
        from_outputs = sum(1 for o in self.outputs if o.log_as == media_type)
        return from_params + from_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cast_nargs(values: list, types: list[str]) -> list:
    """Cast each element in *values* according to the corresponding type string."""
    result = []
    for v, t in zip(values, types):
        caster = _PARAM_TYPE_MAP.get(t, str)
        result.append(caster(v))
    return result


def _upload_file(wb_run, path: Path, log_as: str, label: str | None = None) -> None:
    key = label or path.stem
    try:
        if log_as == "video":
            wb_run.log({key: wandb.Video(str(path))})
        elif log_as == "image":
            wb_run.log({key: wandb.Image(str(path))})
        elif log_as == "text":
            text = path.read_text(errors="replace")
            wb_run.log({key: wandb.Html(f"<pre>{text}</pre>")})
        elif log_as == "artifact":
            artifact = wandb.Artifact(f"{key}-{wb_run.id}", type=log_as)
            artifact.add_file(str(path))
            wb_run.log_artifact(artifact)
    except Exception as e:
        print(f"[genai_runner] Warning: failed to upload {path} as {log_as}: {e}")


def _collect_git_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        return {
            "repo": git("rev-parse", "--show-toplevel").rsplit("/", 1)[-1],
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _save_code_snapshot(wb_run, output_dir: Path, git_info: dict) -> None:
    """Save a full code snapshot: git archive + dirty diff."""
    if not git_info:
        return

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], capture_output=True, timeout=30,
        )

    # git archive: full snapshot of tracked files at HEAD
    archive_path = code_dir / "source.tar.gz"
    result = git("archive", "--format=tar.gz", "-o", str(archive_path), "HEAD")
    if result.returncode != 0:
        print(f"[genai_runner] Warning: git archive failed: {result.stderr.decode()}")
        return

    # Dirty diff (staged + unstaged vs HEAD)
    diff_path = None
    if git_info.get("dirty"):
        diff_result = git("diff", "HEAD")
        if diff_result.returncode == 0 and diff_result.stdout:
            diff_path = code_dir / "dirty.patch"
            diff_path.write_bytes(diff_result.stdout)

    # Upload as W&B artifact
    try:
        artifact = wandb.Artifact(f"code-{wb_run.id}", type="code")
        artifact.add_file(str(archive_path))
        if diff_path and diff_path.exists():
            artifact.add_file(str(diff_path))
        wb_run.log_artifact(artifact)
    except Exception as e:
        print(f"[genai_runner] Warning: failed to upload code snapshot: {e}")
