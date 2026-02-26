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
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, TextIO

import questionary
import wandb

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as _WBRun

ParamType = Literal[
    "str",
    "int",
    "float",
    "bool",
    "path",
    "path-image",
    "path-video",
    "path-artifact",
    "path-text",
]

_PARAM_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "path": str,
    "path-image": str,
    "path-video": str,
    "path-artifact": str,
    "path-text": str,
}


def _log_as_from_type(t: str) -> str | None:
    """Extract upload intent from a type string, e.g. 'path-video' -> 'video'."""
    if t.startswith("path-"):
        return t[5:]
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Param:
    """A CLI parameter for the model command.

    Args:
        name: Parameter name, used as argparse dest (underscored)
            and CLI flag (hyphenated).
        type: A ParamType or list of ParamType for multi-value
            flags.  Single: "str", "int", "float", "bool", "path",
            "path-image", "path-video", "path-artifact", "path-text".
            List: e.g. ["path-image", "float", "float"] for
            multi-value flags (nargs inferred).  The ``path-*``
            variants encode upload intent: ``path-video`` means
            "parse as a path, upload as video to W&B".
        default: Default value.  Can be a callable (called at
            resolve time).  Ignored for type="bool" (always False).
        choices: Allowed values (shown as select in TUI).
        help: Description shown in --help and TUI prompt.
        flag: Override the CLI flag (default: --<name with hyphens>).
        value: Fixed value with $output interpolation. Never
            prompted.  Can be a list for multi-value flags.
        labels: Names for each part, used in TUI prompts and --help
            metavar.  e.g. labels=["path", "start_frame", "strength"]
            prompts separately for each.
        hidden: If True, not shown in --help.
        log_when: "before" (input file) or "after" (output file).
            Auto-inferred from $output in value when type encodes
            upload intent (path-image, path-video, etc.).
    """

    name: str
    type: ParamType | list[ParamType] = "str"
    default: Any = None
    choices: list[str] | None = None
    help: str = ""
    flag: str | None = None
    value: Any = None
    labels: list[str] | None = None
    hidden: bool = False
    log_when: str | None = None

    def __post_init__(self) -> None:
        self._dest = self.name.replace("-", "_")
        if self.flag is None:
            self.flag = f"--{self.name.replace('_', '-')}"
        if self._primary_type == "bool" and self.default not in (None, False):
            print(
                f"[genai_runner] Warning: Param('{self.name}', type='bool')"
                f" has default={self.default!r} which is ignored"
                " (bool params always default to False)",
                file=sys.stderr,
            )
        if self.log_when is None:
            log_as = _log_as_from_type(self._primary_type)
            if log_as is not None:
                self.log_when = "after" if self._value_contains_output() else "before"

    def _value_contains_output(self) -> bool:
        """Check whether $output appears anywhere in self.value."""
        if self.value is None:
            return False
        if isinstance(self.value, list):
            return any("$output" in str(v) for v in self.value)
        return "$output" in str(self.value)

    @property
    def _primary_type(self) -> str:
        """The type string for single-value params, or first type for multi-value."""
        return self.type[0] if isinstance(self.type, list) else self.type

    @property
    def types(self) -> list[str] | None:
        """Per-element types for multi-value params, or None for single-value."""
        return list(self.type) if isinstance(self.type, list) else None

    @property
    def dest(self) -> str:
        return self._dest

    @property
    def nargs(self) -> int | None:
        return len(self.type) if isinstance(self.type, list) else None

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
        path: Absolute path, $output-relative path, or glob pattern
            (e.g. "debug/**/*.png").
        log_as: One of "video", "image", "artifact", "text", "zip".
            With globs: "zip" collects all matches into a zip and uploads as artifact;
            other values upload each matched file individually.
            For directories: "zip" zips the entire directory.
        copy_to: If set, copy the file here ($output interpolated) before logging.
            Not supported with glob patterns.
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


_RUNS_DIR = Path.home() / "genai_runs"


@dataclass
class Runner:
    command: str | list[str]
    params: list[Param] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    wandb_project: str | None = None
    group: str | None = None  # W&B run group for sweeps

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            self.command = shlex.split(self.command)
        self._cli_args = self._parse_cli_args()

    def run(self, overrides: dict[str, object] | None = None) -> None:
        """Execute the full run lifecycle."""
        resolved_values = self._resolve_values(self._cli_args, overrides or {})
        interactive = not resolved_values.pop("_no_interactive")
        dry_run = resolved_values.pop("_dry_run")
        run_name = resolved_values.pop("_run_name", None)

        # Prompt for missing params (interactive mode)
        self._prompt_missing(resolved_values, interactive=interactive)

        # Git info
        git_info = _collect_git_info()
        project = self.wandb_project or git_info.get("repo", "genai-runs")

        # Dry-run: show command without W&B or output dir
        if dry_run:
            param_values = self._interpolate_output(
                resolved,
                Path("/tmp/dry-run-output"),  # noqa: S108
            )
            cmd = self._build_command(param_values)
            print(f"[dry-run] Project: {project}")
            print(f"[dry-run] Command:\n  {shlex.join(cmd)}")
            return

        # W&B init
        run_tags = list(self.tags)
        if git_info.get("dirty"):
            run_tags.append("dirty-git")

        config = {
            **{
                f"param/{k}": v
                for k, v in resolved_values.items()
                if not k.startswith("_")
            },
            **{f"git/{k}": v for k, v in git_info.items()},
            "meta/hostname": os.uname().nodename,
            "meta/datetime": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "meta/command": shlex.join(self.command),
        }
        wb_run = wandb.init(
            project=project,
            name=run_name,
            group=self.group,
            tags=run_tags,
            save_code=True,
            config=config,
        )

        # Output dir
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M")
        dir_name = wb_run.name or wb_run.id or "run"
        output_dir = _RUNS_DIR / project / f"{timestamp}_{dir_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save code snapshot (git archive + dirty diff)
        _save_code_snapshot(wb_run, output_dir, git_info)

        # Interpolate $output in fixed-value params
        param_values = self._interpolate_output(resolved_values, output_dir)

        # Log input files (log_when == "before")
        self._log_files(wb_run, param_values, when="before")

        # Build command
        cmd = self._build_command(param_values)
        wb_run.config.update({"meta/full_command": shlex.join(cmd)})

        print(f"Output dir: {output_dir}")
        print(f"W&B run: {wb_run.url}")
        print(f"Command:\n  {shlex.join(cmd)}")
        print("=" * 60)

        # Execute
        exit_code, duration, stdout_text = self._execute(cmd, output_dir)

        # --- Post-run: never raise, always try to finish W&B run ---
        status = "failed" if exit_code != 0 else "success"
        summary = {
            "exit_code": exit_code,
            "duration_seconds": duration,
            "status": status,
            "output_dir": str(output_dir),
        }

        for step_name, step in [
            ("extract metrics", lambda: self._extract_metrics(wb_run, stdout_text)),
            (
                "log output files",
                lambda: self._log_files(wb_run, param_values, when="after"),
            ),
            ("log extra outputs", lambda: self._log_extra_outputs(wb_run, output_dir)),
            ("log run logs", lambda: self._log_run_logs(wb_run, output_dir)),
            ("tag failed run", lambda: _tag_failed(wb_run, exit_code, run_tags)),
            ("update W&B summary", lambda: wb_run.summary.update(summary)),
            ("finish W&B run", lambda: wb_run.finish(exit_code=exit_code)),
        ]:
            try:
                step()
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: {step_name} failed: {e}")

        print("=" * 60)
        print(f"Status: {status} (exit code {exit_code})")
        print(f"Duration: {duration:.1f}s")
        print(f"Output dir: {output_dir}")

    # -----------------------------------------------------------------------
    # CLI parsing
    # -----------------------------------------------------------------------

    def _parse_cli_args(self) -> dict:
        parser = argparse.ArgumentParser(
            description="genai_runner experiment launcher",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Built-in flags
        parser.add_argument(
            "--dry-run", action="store_true", help="Print command and exit"
        )
        parser.add_argument(
            "--no-interactive",
            action="store_true",
            help="Non-interactive mode; fail if required params are missing",
        )
        parser.add_argument("--run-name", default=None, help="Override W&B run name")
        parser.add_argument(
            "--wandb-project",
            default=None,
            help="Override W&B project name",
        )

        for p in self.params:
            if p.is_fixed:
                continue
            kwargs: dict = {
                "default": None,
                "help": argparse.SUPPRESS if p.hidden else (p.help or None),
            }
            if p._primary_type == "bool":
                kwargs.update(action="store_true", default=False)
            else:
                kwargs["type"] = _PARAM_TYPE_MAP.get(p._primary_type, str)
                if p.nargs is not None:
                    # Multi-value: argparse gets nargs=N, all as str
                    kwargs["nargs"] = p.nargs
                    kwargs["type"] = str
                    if p.labels and not p.hidden:
                        kwargs["metavar"] = tuple(p.labels)
                        kwargs["help"] = (
                            f"{p.help or ''} ({' '.join(p.labels)})"
                        ).strip()
            if p.choices:
                kwargs["choices"] = p.choices
            assert p.flag is not None
            parser.add_argument(p.flag, dest=p.dest, **kwargs)

        ns = parser.parse_args()
        result = {
            p.dest: getattr(ns, p.dest, None) for p in self.params if not p.is_fixed
        }
        result["_dry_run"] = ns.dry_run
        result["_no_interactive"] = ns.no_interactive
        result["_run_name"] = ns.run_name
        self.wandb_project = ns.wandb_project or self.wandb_project
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
            # Apply callable defaults (only if not overridden)
            elif resolved.get(p.dest) is None and p.default is not None:
                resolved[p.dest] = p.default() if callable(p.default) else p.default
            # Cast multi-value args to per-element types
            if p.types and resolved.get(p.dest) is not None:
                resolved[p.dest] = _cast_nargs(resolved[p.dest], p.types)
        return resolved

    # -----------------------------------------------------------------------
    # Interactive prompts
    # -----------------------------------------------------------------------

    def _prompt_missing(self, resolved: dict, *, interactive: bool) -> None:
        missing = [
            p
            for p in self.params
            if not p.is_fixed
            and p._primary_type != "bool"
            and resolved.get(p.dest) is None
        ]

        if not missing:
            return

        if not interactive:
            names = [p.name for p in missing]
            print(
                f"Error: missing required params: {', '.join(names)}",
                file=sys.stderr,
            )
            print(
                "Run without -n for interactive mode,"
                " or pass them on the command line.",
                file=sys.stderr,
            )
            sys.exit(2)

        for p in missing:
            if p.nargs is not None:
                resolved[p.dest] = self._prompt_nargs(p)
            else:
                resolved[p.dest] = self._prompt_single(p)

    def _prompt_single(self, p: Param) -> int | float | str:
        label = p.help or p.name
        if p.choices:
            answer = questionary.select(f"{label}:", choices=p.choices).ask()
        elif p._primary_type.startswith("path"):
            answer = questionary.path(f"{label}:").ask()
        else:
            answer = questionary.text(f"{label}:").ask()

        if answer is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)

        caster = _PARAM_TYPE_MAP.get(p._primary_type, str)
        return caster(answer)

    def _prompt_nargs(self, p: Param) -> list:
        assert p.nargs is not None
        labels = p.labels or [f"{p.name}[{i}]" for i in range(p.nargs)]
        element_types = p.types or [p._primary_type] * p.nargs
        parts = []
        for label, etype in zip(labels, element_types, strict=True):
            if etype.startswith("path"):
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
        result = dict(resolved)
        out = str(output_dir)
        for p in self.params:
            if not p.is_fixed:
                continue
            if isinstance(p.value, list):
                interpolated = [str(v).replace("$output", out) for v in p.value]
                if any("$output" in str(v) for v in p.value):
                    Path(interpolated[0]).parent.mkdir(parents=True, exist_ok=True)
                result[p.dest] = interpolated
            else:
                val = str(p.value).replace("$output", out)
                if "$output" in str(p.value):
                    Path(val).parent.mkdir(parents=True, exist_ok=True)
                result[p.dest] = val
        return result

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def _build_command(self, param_values: dict) -> list[str]:
        cmd = list(self.command)
        for p in self.params:
            val = param_values.get(p.dest)
            if val is None or (p._primary_type == "bool" and not val):
                continue
            assert p.flag is not None
            if p._primary_type == "bool":
                cmd.append(p.flag)
            elif isinstance(val, list):
                cmd.append(p.flag)
                cmd.extend(str(v) for v in val)
            else:
                cmd.extend([p.flag, str(val)])
        return cmd

    # -----------------------------------------------------------------------
    # Subprocess execution
    # -----------------------------------------------------------------------

    def _execute(self, cmd: list[str], output_dir: Path) -> tuple[int, float, str]:
        stdout_lines: list[str] = []
        lock = threading.Lock()

        def stream_pipe(
            pipe: IO[bytes],
            sys_stream: TextIO,
            file_log: TextIO,
            *,
            prefix: str = "",
            capture: bool = False,
        ) -> None:
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

        with ExitStack() as stack:
            log_combined = stack.enter_context((output_dir / "run.log").open("w"))
            log_stdout = stack.enter_context((output_dir / "stdout.log").open("w"))
            log_stderr = stack.enter_context((output_dir / "stderr.log").open("w"))

            run_env = {**os.environ, **self.env}

            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
            )

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
                # Close pipes so reader threads unblock
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

            t_out.join()
            t_err.join()
            duration = time.monotonic() - start

        return proc.returncode, duration, "".join(stdout_lines)

    # -----------------------------------------------------------------------
    # File logging to W&B
    # -----------------------------------------------------------------------

    def _log_files(self, wb_run: _WBRun, param_values: dict, when: str) -> None:
        for p in self.params:
            log_as = _log_as_from_type(p._primary_type)
            if log_as is None or p.log_when != when:
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
            _upload_file(wb_run, path, log_as, label=p.name)

    def _log_extra_outputs(self, wb_run: _WBRun, output_dir: Path) -> None:
        out = str(output_dir)
        zip_counter = 0
        for o in self.outputs:
            raw_path = o.path.replace("$output", out)
            is_glob = any(c in raw_path for c in ("*", "?", "["))
            path = Path(raw_path)

            if is_glob:
                base, pattern = _split_glob(raw_path)
                matches = sorted(base.glob(pattern))
            elif path.is_dir():
                if o.log_as != "zip":
                    print(
                        f"[genai_runner] Warning: {raw_path} is a directory,"
                        " use log_as='zip' or a glob pattern"
                    )
                    continue
                base = path
                matches = sorted(path.rglob("*"))
            else:
                _log_single_output(wb_run, path, o, out)
                continue

            # Glob or directory: upload matches
            if not matches:
                print(
                    f"[genai_runner] Warning: glob '{o.path}'"
                    " matched no files, skipping"
                )
                continue

            if o.log_as == "zip":
                zip_counter = _zip_and_upload(
                    wb_run, matches, base, output_dir, zip_counter
                )
            else:
                for m in matches:
                    if m.is_file():
                        _upload_file(wb_run, m, o.log_as)

    def _log_run_logs(self, wb_run: _WBRun, output_dir: Path) -> None:
        existing_logs = [
            output_dir / name
            for name in ("run.log", "stdout.log", "stderr.log")
            if (output_dir / name).exists()
        ]
        if existing_logs:
            artifact = wandb.Artifact(f"logs-{wb_run.id}", type="log")
            for f in existing_logs:
                artifact.add_file(str(f))
            wb_run.log_artifact(artifact)

    # -----------------------------------------------------------------------
    # Metric extraction
    # -----------------------------------------------------------------------

    def _extract_metrics(self, wb_run: _WBRun, stdout_text: str) -> None:
        for m in self.metrics:
            matches = re.findall(m.pattern, stdout_text)
            if not matches:
                continue
            raw = matches[-1]  # last match wins
            if m.type == "float":
                try:
                    wb_run.summary[m.name] = float(raw)
                except ValueError:
                    wb_run.summary[m.name] = raw
            else:
                wb_run.summary[m.name] = raw

    def _count_logged(self, media_type: str) -> int:
        """Count params and outputs that log as the given media type."""
        return sum(
            1 for p in self.params if _log_as_from_type(p._primary_type) == media_type
        ) + sum(1 for o in self.outputs if o.log_as == media_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cast_nargs(values: list, types: list[str]) -> list:
    """Cast each element in *values* according to the corresponding type string."""
    if len(values) != len(types):
        msg = f"Expected {len(types)} values, got {len(values)}: {values}"
        raise ValueError(msg)
    return [_PARAM_TYPE_MAP.get(t, str)(v) for v, t in zip(values, types, strict=True)]


def _split_glob(path_str: str) -> tuple[Path, str]:
    """Split 'dir/sub/**/*.png' into (Path('dir/sub'), '**/*.png')."""
    parts = Path(path_str).parts
    for i, part in enumerate(parts):
        if any(c in part for c in ("*", "?", "[")):
            base = Path(*parts[:i]) if i > 0 else Path()
            pattern = str(Path(*parts[i:]))
            return base, pattern
    # No glob chars found (shouldn't happen if caller checked)
    return Path(path_str).parent, Path(path_str).name


def _log_single_output(wb_run: _WBRun, path: Path, o: Output, out: str) -> None:
    """Handle a single (non-glob, non-directory) output file."""
    if not path.exists():
        print(f"[genai_runner] Warning: {path} not found, skipping")
        return
    if o.copy_to:
        dst = Path(o.copy_to.replace("$output", out))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        path = dst
    _upload_file(wb_run, path, o.log_as)


def _zip_and_upload(
    wb_run: _WBRun,
    matches: list[Path],
    base: Path,
    output_dir: Path,
    zip_counter: int,
) -> int:
    """Zip matched files and upload as artifact. Returns incremented zip_counter."""
    suffix = f"_{zip_counter}" if zip_counter else ""
    zip_name = base.name or "output"
    zip_path = output_dir / f"{zip_name}{suffix}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in matches:
            if m.is_file():
                zf.write(m, m.relative_to(base))
    _upload_file(wb_run, zip_path, "artifact")
    return zip_counter + 1


def _upload_file(
    wb_run: _WBRun,
    path: Path,
    log_as: str,
    label: str | None = None,
) -> None:
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
    except Exception as e:  # noqa: BLE001
        print(f"[genai_runner] Warning: failed to upload {path} as {log_as}: {e}")


def _tag_failed(wb_run: _WBRun, exit_code: int, run_tags: list[str]) -> None:
    """Set 'failed' tag on W&B run if exit code is non-zero."""
    if exit_code != 0:
        wb_run.tags = [*run_tags, "failed"]


def _collect_git_info() -> dict:
    def git(*args: str) -> str:
        return (
            subprocess.check_output(  # noqa: S603
                ["git", *args],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

    try:
        return {
            "repo": git("rev-parse", "--show-toplevel").rsplit("/", 1)[-1],
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": git("status", "--porcelain") != "",
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _save_code_snapshot(wb_run: _WBRun, output_dir: Path, git_info: dict) -> None:
    """Save a full code snapshot: git archive + dirty diff."""
    if not git_info:
        return

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            check=False,
            timeout=30,
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
    except Exception as e:  # noqa: BLE001
        print(f"[genai_runner] Warning: failed to upload code snapshot: {e}")
