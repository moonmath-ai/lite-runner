"""genai_runner -- Experiment runner for video diffusion models with W&B tracking."""

from __future__ import annotations

import argparse
import datetime
import os
import pprint
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


def _contains_output(val: object) -> bool:
    """Check whether $output appears in a string or any element of a list."""
    if isinstance(val, list):
        return any("$output" in str(v) for v in val)
    return "$output" in str(val)


# Sentinel for params the user explicitly skipped (typed '-' at the prompt).
_SKIP_INPUT = "-"


class _Unset:
    """Param value skipped by user during interactive prompting."""

    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()


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
        log_when: "before" (input file) or "after" (output file).
            Auto-inferred from $output in value when type encodes
            upload intent (path-image, path-video, etc.).
        hidden: If True, skip interactive prompting and use the
            default value.  The param still accepts CLI flags and
            is logged normally.  Requires a default.
    """

    name: str
    type: ParamType | list[ParamType] = "str"
    default: Any = None
    choices: list[str] | None = None
    help: str = ""
    flag: str | None = None
    value: Any = None
    labels: list[str] | None = None
    log_when: str | None = None
    hidden: bool = False

    def __post_init__(self) -> None:
        self._dest = self.name.replace("-", "_")
        if self.flag is None:
            self.flag = f"--{self.name.replace('_', '-')}"
        for t in self.type_list:
            if t == "bool" and isinstance(self.type, list):
                msg = (
                    "'bool' cannot appear in a multi-value"
                    f" type list for param '{self.name}'"
                )
                raise ValueError(msg)
            if t != "bool" and t not in _PARAM_TYPE_MAP:
                msg = f"Unknown param type '{t}' for param '{self.name}'"
                raise ValueError(msg)
        if self.hidden and self.default is None:
            msg = f"Param('{self.name}', hidden=True) requires a default"
            raise ValueError(msg)
        if self.type == "bool" and self.default not in (None, False):
            print(
                f"[genai_runner] Warning: Param('{self.name}', type='bool')"
                f" has default={self.default!r} which is ignored"
                " (bool params always default to False)",
                file=sys.stderr,
            )
        if self.log_when is None and any(_log_as_from_type(t) for t in self.type_list):
            self.log_when = "after" if self._value_contains_output() else "before"

    def _value_contains_output(self) -> bool:
        """Check whether $output appears anywhere in self.value."""
        return self.value is not None and _contains_output(self.value)

    @property
    def type_list(self) -> list[str]:
        """Types as a list -- single-value wrapped, multi-value as-is."""
        return list(self.type) if isinstance(self.type, list) else [self.type]

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

    def _argparse_kwargs(self) -> dict:
        """Build kwargs for argparse.add_argument."""
        kwargs: dict = {"dest": self.dest, "default": None, "help": self.help or None}
        if self.type == "bool":
            kwargs["action"] = "store_true"
            kwargs["default"] = False
        elif self.nargs is not None:
            kwargs["nargs"] = self.nargs
            kwargs["type"] = str
            if self.labels:
                kwargs["metavar"] = tuple(self.labels)
                kwargs["help"] = (
                    f"{self.help or ''} ({' '.join(self.labels)})"
                ).strip()
        else:
            kwargs["type"] = _PARAM_TYPE_MAP[self.type]
        if self.choices:
            kwargs["choices"] = self.choices
        return kwargs


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
    name: str | None = None
    copy_to: str | None = None


@dataclass
class Metric:
    """A value to extract from stdout via regex.

    Args:
        name: Metric name in W&B summary.
        pattern: Regex with one capture group.
        type: "float", "int", or "str".  Determines how the captured value is stored.
    """

    name: str
    pattern: str
    type: str = "float"


@dataclass(frozen=True)
class _RunFlags:
    dry_run: bool = False
    interactive: bool = True
    run_name: str | None = None
    wandb_project: str | None = None


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
        self.parsed_params, self.runner_flags = self._parse_cli_args()

    def run(self, overrides: dict[str, object] | None = None) -> None:
        """Execute the full run lifecycle."""
        runner_flags = self.runner_flags

        # Resolve params and prompt for missing ones
        resolved_params = self._resolve_params(self.parsed_params, overrides or {})
        resolved_params = self._prompt_params(
            resolved_params,
            self.parsed_params,
            overrides or {},
            interactive=runner_flags.interactive,
        )

        # Git info and project
        git_info = _collect_git_info()
        project = (
            runner_flags.wandb_project or self.wandb_project or git_info.get("repo")
        )
        if project is None:
            msg = (
                "Cannot determine project name:"
                " set wandb_project= or run from a git repo"
            )
            raise ValueError(msg)

        # Config
        config: dict[str, object] = {}
        for k, v in resolved_params.items():
            config[f"param/{k}"] = "<unset>" if v is UNSET else v
        for k, v in git_info.items():
            config[f"git/{k}"] = v
        timestamp = datetime.datetime.now(tz=datetime.UTC)
        config["meta/hostname"] = os.uname().nodename
        config["meta/datetime"] = timestamp.isoformat()
        config["meta/command"] = shlex.join(self.command)

        # W&B init
        if runner_flags.dry_run:
            print(f"[dry-run] Project: {project}")
            print(f"[dry-run] Run name: {runner_flags.run_name or '(auto)'}")
            print(f"[dry-run] Group: {self.group}")
            print(f"[dry-run] Tags: {self.tags}")
            print(f"[dry-run] Config:\n{pprint.pformat(config)}")
            run_name = runner_flags.run_name or "run"
            run_url = "<link to W&B run>"
        else:
            wb_run = wandb.init(
                project=project,
                name=runner_flags.run_name,
                group=self.group,
                tags=self.tags,
                save_code=True,
                config=config,
            )
            run_name = wb_run.name or wb_run.id or "run"
            run_url = wb_run.url

        # Output dir
        output_dir = (
            _RUNS_DIR / project / f"{timestamp.strftime('%Y%m%d_%H%M')}_{run_name}"
        )
        print(f"Output dir: {output_dir}")
        print(f"W&B run: {run_url}")
        if not runner_flags.dry_run:
            wb_run.config.update({"meta/output_dir": str(output_dir)})
            output_dir.mkdir(parents=True, exist_ok=True)

        # Save code snapshot (git archive + dirty diff)
        if not runner_flags.dry_run:
            try:
                _log_code_snapshot(wb_run, output_dir, git_info)
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: code snapshot failed: {e}")

        # Interpolate $output in param values
        interpolated_params = self._interpolate_output(resolved_params, output_dir)

        # Log input files (log_when == "before")
        if not runner_flags.dry_run:
            self._log_files(wb_run, interpolated_params, when="before")

        # Build command
        cmd = self._build_command(interpolated_params)
        print(f"Command:\n  {shlex.join(cmd)}")

        if runner_flags.dry_run:
            return

        # Execute
        print("=" * 60)
        exit_code, duration, stdout_text, aborted = self._execute(cmd, output_dir)
        print("=" * 60)

        # Post-run: never raise, always try to finish W&B run
        self._post_run(
            wb_run,
            interpolated_params,
            output_dir,
            exit_code,
            duration,
            stdout_text,
            self.tags,
            aborted=aborted,
        )

    def _post_run(
        self,
        wb_run: _WBRun,
        param_values: dict,
        output_dir: Path,
        exit_code: int,
        duration: float,
        stdout_text: str,
        run_tags: list[str],
        *,
        aborted: bool = False,
    ) -> None:
        if aborted:
            status = "aborted"
        elif exit_code == 0:
            status = "success"
        else:
            status = "failed"
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
            ("tag run status", lambda: _tag_status(wb_run, status, run_tags)),
            ("update W&B summary", lambda: wb_run.summary.update(summary)),
            ("finish W&B run", lambda: wb_run.finish(exit_code=exit_code)),
        ]:
            try:
                step()
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: {step_name} failed: {e}")

        print(f"Status: {status} (exit code {exit_code})")
        print(f"Duration: {duration:.1f}s")
        print(f"Output dir: {output_dir}")

    # -----------------------------------------------------------------------
    # CLI parsing
    # -----------------------------------------------------------------------

    def _parse_cli_args(self) -> tuple[dict, _RunFlags]:
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
            if not p.is_fixed:
                parser.add_argument(p.flag, **p._argparse_kwargs())

        ns = parser.parse_args()
        parsed_params = {
            p.name: getattr(ns, p.dest, None) for p in self.params if not p.is_fixed
        }
        runner_flags = _RunFlags(
            dry_run=ns.dry_run,
            interactive=not ns.no_interactive,
            run_name=ns.run_name,
            wandb_project=ns.wandb_project,
        )
        return parsed_params, runner_flags

    # -----------------------------------------------------------------------
    # Resolve values: apply overrides and callable defaults
    # -----------------------------------------------------------------------

    def _resolve_params(self, parsed_params: dict, overrides: dict) -> dict:
        resolved_params = dict(parsed_params)
        for p in self.params:
            if p.name in overrides:
                resolved_params[p.name] = overrides[p.name]
            elif p.is_fixed:
                resolved_params[p.name] = p.value() if callable(p.value) else p.value
            elif resolved_params.get(p.name) is not None:
                pass
            elif p.default is not None:
                resolved_params[p.name] = (
                    p.default() if callable(p.default) else p.default
                )
            # Cast multi-value args to per-element types
            val = resolved_params.get(p.name)
            if p.nargs is not None and val is not None and val is not UNSET:
                resolved_params[p.name] = _cast_nargs(
                    resolved_params[p.name], p.type_list
                )
        return resolved_params

    # -----------------------------------------------------------------------
    # Interactive prompts
    # -----------------------------------------------------------------------

    def _prompt_params(
        self,
        resolved_params: dict,
        parsed_params: dict,
        overrides: dict,
        *,
        interactive: bool,
    ) -> dict:
        resolved_params = dict(resolved_params)

        # Params eligible for prompting: non-fixed, non-bool,
        # not explicitly set via CLI or overrides
        promptable = [
            p
            for p in self.params
            if not p.is_fixed
            and not p.hidden
            and p.type != "bool"
            and p.name not in overrides
            and parsed_params.get(p.name) is None
        ]

        missing = [p for p in promptable if resolved_params.get(p.name) is None]

        if not interactive:
            if missing:
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
            return resolved_params

        for p in promptable:
            default = resolved_params.get(p.name)
            if p.nargs is not None:
                resolved_params[p.name] = self._prompt_nargs(p, default=default)
            else:
                resolved_params[p.name] = self._prompt_single(p, default=default)

        return resolved_params

    def _prompt_single(
        self, p: Param, default: object = None
    ) -> int | float | str | _Unset:
        label = p.help or p.name
        default_str = str(default) if default is not None else ""
        if p.choices:
            choices = [_SKIP_INPUT, *p.choices]
            default_choice = str(default) if default is not None else None
            answer = questionary.select(
                f"{label}:", choices=choices, default=default_choice
            ).ask()
        elif isinstance(p.type, str) and p.type.startswith("path"):
            answer = questionary.path(f"{label}:", default=default_str).ask()
        else:
            answer = questionary.text(f"{label}:", default=default_str).ask()

        if answer is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)

        if answer == _SKIP_INPUT:
            return UNSET

        caster = _PARAM_TYPE_MAP.get(p.type, str)
        return caster(answer)

    def _prompt_nargs(self, p: Param, default: object = None) -> list | _Unset:
        assert p.nargs is not None
        labels = p.labels or [f"{p.name}[{i}]" for i in range(p.nargs)]
        element_types = p.type_list
        defaults = default if isinstance(default, list) else [None] * p.nargs
        parts = []
        for label, etype, d in zip(labels, element_types, defaults, strict=True):
            default_str = str(d) if d is not None else ""
            if etype.startswith("path"):
                answer = questionary.path(
                    f"{p.name} {label}:", default=default_str
                ).ask()
            else:
                answer = questionary.text(
                    f"{p.name} {label}:", default=default_str
                ).ask()
            if answer is None:
                print("Cancelled.", file=sys.stderr)
                sys.exit(1)
            if answer == _SKIP_INPUT:
                return UNSET
            parts.append(answer)
        return _cast_nargs(parts, element_types)

    # -----------------------------------------------------------------------
    # $output interpolation
    # -----------------------------------------------------------------------

    def _interpolate_output(self, resolved: dict, output_dir: Path) -> dict:
        """Return a copy of resolved with $output replaced in any values."""
        result = dict(resolved)
        out = str(output_dir)
        for p in self.params:
            val = result.get(p.name)
            if val is None or val is UNSET or p.type == "bool":
                continue
            if isinstance(val, list):
                interpolated = [str(v).replace("$output", out) for v in val]
                result[p.name] = interpolated
            else:
                interpolated = str(val).replace("$output", out)
                result[p.name] = interpolated
        return result

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def _build_command(self, param_values: dict) -> list[str]:
        cmd = list(self.command)
        for p in self.params:
            val = param_values.get(p.name)
            if val is UNSET:
                continue
            assert val is not None
            assert p.flag is not None
            if p.type == "bool":
                if val:
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

    def _execute(
        self, cmd: list[str], output_dir: Path
    ) -> tuple[int, float, str, bool]:
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
            start = time.monotonic()

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

            aborted = False
            t_out.start()
            t_err.start()

            try:
                proc.wait()
            except KeyboardInterrupt:
                aborted = True
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

        return proc.returncode, duration, "".join(stdout_lines), aborted

    # -----------------------------------------------------------------------
    # File logging to W&B
    # -----------------------------------------------------------------------

    def _log_files(self, wb_run: _WBRun, param_values: dict, when: str) -> None:
        for p in self.params:
            if p.log_when != when:
                continue
            val = param_values.get(p.name)
            if val is None or val is UNSET:
                continue
            values = val if isinstance(val, list) else [val]
            for v, t in zip(values, p.type_list, strict=True):
                log_as = _log_as_from_type(t)
                if log_as is None:
                    continue
                path = Path(str(v))
                if not path.exists():
                    msg = f"File not found: {path} (param '{p.name}')"
                    raise FileNotFoundError(msg)
                _upload_file(wb_run, path, log_as, label=p.name)

    def _log_extra_outputs(self, wb_run: _WBRun, output_dir: Path) -> None:
        out = str(output_dir)
        seen_zips: set[str] = set()
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
                        " uploading each file individually"
                        " (use log_as='zip' to zip instead)"
                    )
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

            label = o.name or base.name or "output"
            if o.log_as == "zip":
                if label in seen_zips:
                    msg = (
                        f"Duplicate zip label '{label}':"
                        " set Output(name=...) to disambiguate"
                    )
                    raise ValueError(msg)
                seen_zips.add(label)
                _zip_and_upload(wb_run, matches, base, output_dir, label)
            else:
                for m in matches:
                    if m.is_file():
                        _upload_file(wb_run, m, o.log_as, label=label)

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
            elif m.type == "int":
                try:
                    wb_run.summary[m.name] = int(raw)
                except ValueError:
                    wb_run.summary[m.name] = raw
            else:
                wb_run.summary[m.name] = raw


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
        msg = f"Output file not found: {path}"
        raise FileNotFoundError(msg)
    if o.copy_to:
        dst = Path(o.copy_to.replace("$output", out))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        path = dst
    _upload_file(wb_run, path, o.log_as, label=o.name)


def _zip_and_upload(
    wb_run: _WBRun,
    matches: list[Path],
    base: Path,
    output_dir: Path,
    label: str,
) -> None:
    """Zip matched files and upload as artifact."""
    zip_path = output_dir / f"{label}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in matches:
            if m.is_file():
                zf.write(m, m.relative_to(base))
    _upload_file(wb_run, zip_path, "artifact", label=label)


def _upload_file(
    wb_run: _WBRun,
    path: Path,
    log_as: str,
    label: str | None = None,
) -> None:
    key = label or path.stem
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


def _tag_status(wb_run: _WBRun, status: str, run_tags: list[str]) -> None:
    """Add 'failed' or 'aborted' tag to W&B run when not successful."""
    if status != "success":
        wb_run.tags = [*run_tags, status]


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


def _log_code_snapshot(wb_run: _WBRun, output_dir: Path, git_info: dict) -> None:
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
        msg = f"git archive failed: {result.stderr.decode()}"
        raise RuntimeError(msg)

    # Dirty diff (staged + unstaged vs HEAD)
    diff_path = None
    if git_info.get("dirty"):
        diff_result = git("diff", "HEAD")
        if diff_result.returncode == 0 and diff_result.stdout:
            diff_path = code_dir / "dirty.patch"
            diff_path.write_bytes(diff_result.stdout)

    # Upload as W&B artifact
    artifact = wandb.Artifact(f"code-{wb_run.id}", type="code")
    artifact.add_file(str(archive_path))
    if diff_path and diff_path.exists():
        artifact.add_file(str(diff_path))
    wb_run.log_artifact(artifact)
