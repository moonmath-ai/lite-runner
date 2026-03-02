"""Runner orchestrator and helpers for genai_runner."""

from __future__ import annotations

import argparse
import copy
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
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TextIO

import questionary

from .backends import JsonBackend, LogBackend, WandbBackend
from .params import (
    _PARAM_TYPE_MAP,
    _SKIP_INPUT,
    UNSET,
    Metric,
    Output,
    Param,
    _log_as_from_type,
    _RunFlags,
    _Unset,
)

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
        self._resolved_params: dict | None = None
        self._overrides: dict | None = None
        self._filled: bool = False

    # -------------------------------------------------------------------
    # Helpers: merge overrides and check completeness
    # -------------------------------------------------------------------

    def _merge_overrides(
        self,
        overrides: dict[str, object] | None,
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        """Merge previous overrides, new overrides dict, and kwargs.

        Carries forward overrides from a previous resolve/override call,
        maps underscore kwargs to param names, and validates keys.
        """
        merged: dict[str, object] = {}
        if self._overrides is not None:
            merged.update(self._overrides)
        merged.update(overrides or {})
        name_by_dest = {p.dest: p.name for p in self.params}
        for k, v in kwargs.items():
            merged[name_by_dest.get(k, k)] = v

        valid_names = {p.name for p in self.params}
        unknown = set(merged) - valid_names
        if unknown:
            msg = f"Unknown param(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        return merged

    def _check_complete(self, resolved: dict) -> None:
        """Raise ValueError if any required params are still None."""
        missing = [
            p.name
            for p in self.params
            if not p.is_fixed and p.type != "bool" and resolved.get(p.name) is None
        ]
        if missing:
            msg = f"Missing required param(s): {', '.join(missing)}"
            raise ValueError(msg)

    # -------------------------------------------------------------------
    # Public param pipeline: resolve → fill
    # -------------------------------------------------------------------

    def resolve(
        self, overrides: dict[str, object] | None = None, **kwargs: object
    ) -> Runner:
        """Return a copy with overrides merged and params resolved.

        Priority chain: overrides > CLI args > callable defaults > fixed.
        Does NOT check completeness — missing params are OK at this stage.
        """
        merged = self._merge_overrides(overrides, kwargs)
        resolved = self._resolve_params(self.parsed_params, merged)

        new = copy.copy(self)
        new._resolved_params = resolved
        new._overrides = merged
        new._filled = False
        return new

    def fill(self) -> Runner:
        """Return a copy with missing params filled via interactive prompts.

        In non-interactive mode, raises SystemExit if required params
        are missing.  Must be called after resolve().
        """
        if self._resolved_params is None:
            msg = "fill() requires resolve() first"
            raise ValueError(msg)

        filled = self._prompt_params(
            self._resolved_params,
            self.parsed_params,
            self._overrides or {},
            interactive=self.runner_flags.interactive,
        )

        new = copy.copy(self)
        new._resolved_params = filled
        new._overrides = self._overrides
        new._filled = True
        return new

    def override(
        self, overrides: dict[str, object] | None = None, **kwargs: object
    ) -> Runner:
        """Return a copy with overrides pre-resolved and validated complete.

        Convenience method: resolve() + completeness check in one call.
        All params must be determined (via overrides, CLI, or defaults);
        raises ValueError if any required param is still missing.

        Example::

            r1 = runner.override(seed=42, prompt="a cat")
            r2 = runner.override(seed=99, prompt="a dog")
            # both validated ^^
            r1.run()
            r2.run()
        """
        r = self.resolve(overrides, **kwargs)
        self._check_complete(r._resolved_params)
        r._filled = True
        return r

    def run(self) -> None:
        """Execute the full run lifecycle."""
        runner_flags = self.runner_flags

        r = self
        if r._resolved_params is None:
            r = r.resolve()
        if not r._filled:
            r = r.fill()
        resolved_params = r._resolved_params
        overrides_dict = r._overrides or {}

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

        # Init WandbBackend first (needs to happen early to get run_name)
        wb_backend = None
        if not runner_flags.dry_run and not runner_flags.no_wandb:
            wb_backend = WandbBackend()
            wb_backend.init(
                project, runner_flags.run_name, self.group, self.tags, config
            )
            run_name = wb_backend.run_name
            run_url = wb_backend.run_url
        elif runner_flags.dry_run:
            run_name = runner_flags.run_name or "run"
            run_url = "(dry run)"
        else:
            run_name = runner_flags.run_name or "local"
            run_url = "(W&B disabled)"

        # Output dir
        output_dir = (
            _RUNS_DIR / project / f"{timestamp.strftime('%Y%m%d_%H%M')}_{run_name}"
        )

        # Augment config with output_dir and wandb info
        config["meta/output_dir"] = str(output_dir)
        if wb_backend is not None:
            config["wandb/name"] = run_name
            config["wandb/url"] = run_url
            wb_backend.update_config(
                {
                    "meta/output_dir": str(output_dir),
                    "wandb/name": run_name,
                    "wandb/url": run_url,
                }
            )

        # Dry run: print summary and return (no dirs, no execution)
        if runner_flags.dry_run:
            print(f"[dry-run] Project: {project}")
            print(f"[dry-run] Run name: {run_name}")
            print(f"[dry-run] Group: {self.group}")
            print(f"[dry-run] Tags: {self.tags}")
            print(f"[dry-run] Config:\n{pprint.pformat(config)}")
            interpolated_params = self._interpolate_output(resolved_params, output_dir)
            colored_cmd = self._format_command(
                interpolated_params, self.parsed_params, overrides_dict
            )
            print(f"Output dir: {output_dir}")
            print(f"Command:\n{colored_cmd}")
            # Show what files would be logged
            self._print_file_plan(interpolated_params)
            return

        # Create output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {output_dir}")
        print(f"W&B run: {run_url}")

        # JsonBackend is always active
        json_backend = JsonBackend(output_dir)
        json_backend.init(project, run_name, self.group, self.tags, config)

        # Assemble backends
        self._backends: list[LogBackend] = [json_backend]
        if wb_backend is not None:
            self._backends.append(wb_backend)

        # Save code snapshot (git archive + dirty diff)
        try:
            _log_code_snapshot(self._backends, output_dir, git_info)
        except Exception as e:  # noqa: BLE001
            print(f"[genai_runner] Warning: code snapshot failed: {e}")

        # Interpolate $output in param values
        interpolated_params = self._interpolate_output(resolved_params, output_dir)

        # Log table params (prompt, etc.)
        self._log_table_params(resolved_params)

        # Log input files (log_when == "before")
        self._log_files(interpolated_params, when="before")

        # Build command
        cmd = self._build_command(interpolated_params)
        for b in self._backends:
            b.update_config({"meta/full_command": shlex.join(cmd)})
        colored_cmd = self._format_command(
            interpolated_params, self.parsed_params, overrides_dict
        )
        print(f"Command:\n{colored_cmd}")

        # Execute
        print("=" * 60)
        exit_code, duration, stdout_text, aborted = self._execute(cmd, output_dir)
        print("=" * 60)

        # Post-run: never raise, always try to finish backends
        self._post_run(
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

        steps: list[tuple[str, object]] = [
            (
                "extract metrics",
                lambda: self._extract_metrics(stdout_text),
            ),
            (
                "log output files",
                lambda: self._log_files(param_values, when="after"),
            ),
            (
                "log extra outputs",
                lambda: self._log_extra_outputs(output_dir),
            ),
            (
                "log run logs",
                lambda: self._log_run_logs(output_dir),
            ),
            (
                "tag run status",
                lambda: (
                    [b.set_tags([*run_tags, status]) for b in self._backends]
                    if status != "success"
                    else None
                ),
            ),
            (
                "update summary",
                lambda: [b.set_summary(summary) for b in self._backends],
            ),
        ]

        for step_name, step in steps:
            try:
                step()
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: {step_name} failed: {e}")

        # Finish each backend individually so one failure doesn't block others
        for b in self._backends:
            try:
                b.finish(exit_code)
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: finish {type(b).__name__} failed: {e}")

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
        parser.add_argument(
            "--no-wandb",
            action="store_true",
            help="Skip W&B logging; still execute command and save outputs locally",
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
            no_wandb=ns.no_wandb,
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
            elif resolved_params.get(p.name) is None and p.default is not None:
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
    # Dry-run file plan
    # -----------------------------------------------------------------------

    def _print_file_plan(self, param_values: dict) -> None:
        """Print what files would be logged (for --dry-run)."""
        before = []
        after = []
        for p in self.params:
            val = param_values.get(p.name)
            if val is None or val is UNSET:
                continue
            if p.log_when is None:
                continue
            values = val if isinstance(val, list) else [val]
            for v, t in zip(values, p.type_list, strict=True):
                log_as = _log_as_from_type(t)
                if log_as is None:
                    continue
                entry = f"  {log_as}: {v} (param: {p.name})"
                if p.log_when == "before":
                    before.append(entry)
                else:
                    after.append(entry)
        for o in self.outputs:
            entry = f"  {o.log_as}: {o.path}"
            if o.name:
                entry += f" (name: {o.name})"
            after.append(entry)
        if before:
            print("Files to log (before run):")
            print("\n".join(before))
        if after:
            print("Files to log (after run):")
            print("\n".join(after))

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
                result[p.name] = [str(v).replace("$output", out) for v in val]
            else:
                result[p.name] = str(val).replace("$output", out)
        return result

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def _format_command(
        self,
        param_values: dict,
        parsed_params: dict,
        overrides: dict,
    ) -> str:
        """Build a colored command string for display."""
        _RST = "\033[0m"
        _BOLD = "\033[1m"
        _COLORS = {
            "flag": "\033[31m",  # red — prompted, required
            "flag+default": "\033[32m",  # green — prompted, kept default
            "cli": "\033[36m",  # cyan — from CLI, non-default value
            "cli+default": "\033[34m",  # blue — from CLI, kept default
            "value": "\033[33m",  # yellow — fixed value=
            "hidden": "\033[35m",  # magenta — hidden, default used
        }

        parts = [shlex.join(self.command)]
        for p in self.params:
            val = param_values.get(p.name)
            if val is UNSET:
                continue
            assert val is not None
            assert p.flag is not None

            if p.type == "bool" and not val:
                continue

            if p.is_fixed:
                kind = "value"
            elif p.hidden:
                kind = "hidden"
            elif p.name in overrides or parsed_params.get(p.name) is not None:
                kind = "cli+default" if self._is_default_value(p, val) else "cli"
            else:
                kind = "flag+default" if self._is_default_value(p, val) else "flag"

            color = _COLORS[kind]
            if p.type == "bool":
                parts.append(f"{color}{p.flag}{_RST}")
            else:
                values = val if isinstance(val, list) else [val]
                val_str = " ".join(shlex.quote(str(v)) for v in values)
                parts.append(f"{color}{p.flag} {_BOLD}{val_str}{_RST}")
        return " ".join(parts)

    @staticmethod
    def _is_default_value(p: Param, val: object) -> bool:
        """Check whether val matches the param's default."""
        if p.default is None:
            return False
        default = p.default() if callable(p.default) else p.default
        if isinstance(val, list) and isinstance(default, list):
            return [str(v) for v in val] == [str(d) for d in default]
        return str(val) == str(default)

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
            else:
                values = val if isinstance(val, list) else [val]
                cmd.append(p.flag)
                cmd.extend(str(v) for v in values)
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
            while True:
                chunk = pipe.read1(8192) if hasattr(pipe, "read1") else pipe.read(8192)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                sys_stream.write(text)
                sys_stream.flush()
                file_log.write(text)
                file_log.flush()
                with lock:
                    log_combined.write(prefix + text if prefix else text)
                    log_combined.flush()
                    if capture:
                        stdout_lines.append(text)
            pipe.close()

        with ExitStack() as stack:
            log_combined = stack.enter_context((output_dir / "run.log").open("w"))
            log_stdout = stack.enter_context((output_dir / "stdout.log").open("w"))
            log_stderr = stack.enter_context((output_dir / "stderr.log").open("w"))

            run_env = {**os.environ, **self.env}
            if "COLUMNS" not in run_env:
                with suppress(OSError):
                    run_env["COLUMNS"] = str(os.get_terminal_size().columns)

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
    # Table logging (prominent display of prompts etc.)
    # -----------------------------------------------------------------------

    def _log_table_params(self, resolved_params: dict) -> None:
        """Log params marked with table=True as a W&B Table."""
        rows = [
            [p.name, resolved_params.get(p.name)]
            for p in self.params
            if p.table and resolved_params.get(p.name) not in (None, UNSET)
        ]
        if rows:
            for b in self._backends:
                b.log_table("params", ["name", "value"], rows)

    # -----------------------------------------------------------------------
    # File logging
    # -----------------------------------------------------------------------

    def _log_files(self, param_values: dict, when: str) -> None:
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
                for b in self._backends:
                    b.log_file(path, log_as, key=p.name)

    def _log_extra_outputs(self, output_dir: Path) -> None:
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
                _log_single_output(self._backends, path, o, out)
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
                _zip_and_upload(self._backends, matches, base, output_dir, label)
            else:
                for m in matches:
                    if m.is_file():
                        for b in self._backends:
                            b.log_file(m, o.log_as, key=label)

    def _log_run_logs(self, output_dir: Path) -> None:
        files = [
            str(output_dir / name)
            for name in ("run.log", "stdout.log", "stderr.log")
            if (output_dir / name).exists()
        ]
        if files:
            for b in self._backends:
                b.log_artifact("logs", "log", files)

    # -----------------------------------------------------------------------
    # Metric extraction
    # -----------------------------------------------------------------------

    def _extract_metrics(self, stdout_text: str) -> None:
        casters = {"float": float, "int": int}
        for m in self.metrics:
            matches = re.findall(m.pattern, stdout_text)
            if not matches:
                continue
            raw = matches[-1]  # last match wins
            caster = casters.get(m.type)
            if caster is not None:
                try:
                    val = caster(raw)
                except ValueError:
                    val = raw
            else:
                val = raw
            for b in self._backends:
                b.set_metric(m.name, val)


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


def _log_single_output(
    backends: list[LogBackend], path: Path, o: Output, out: str
) -> None:
    """Handle a single (non-glob, non-directory) output file."""
    if not path.exists():
        msg = f"Output file not found: {path}"
        raise FileNotFoundError(msg)
    if o.copy_to:
        dst = Path(o.copy_to.replace("$output", out))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        path = dst
    key = o.name or path.stem
    for b in backends:
        b.log_file(path, o.log_as, key=key)


def _zip_and_upload(
    backends: list[LogBackend],
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
    for b in backends:
        b.log_file(zip_path, "artifact", key=label)


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


def _log_code_snapshot(
    backends: list[LogBackend], output_dir: Path, git_info: dict
) -> None:
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

    # Record in all backends
    files = [str(archive_path)]
    if diff_path and diff_path.exists():
        files.append(str(diff_path))
    for b in backends:
        b.log_artifact("code", "code", files)
