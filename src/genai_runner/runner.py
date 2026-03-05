"""Runner orchestrator and helpers for genai_runner."""

from __future__ import annotations

import argparse
import copy
import datetime
import gzip
import os
import pprint
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
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
_PREFIX = "\033[36mgenai-runner:\033[0m"


@dataclass
class Runner:
    """Experiment runner that wraps a model CLI with tracking.

    Declare params, outputs, and metrics, then call :meth:`run`.
    The runner handles CLI parsing, interactive prompts, W&B logging,
    subprocess execution, metric extraction, and file uploads.

    Pipeline methods (:meth:`parse_cli`, :meth:`override`,
    :meth:`resolve_defaults`, :meth:`ask_user`) each return a new
    Runner (immutable copies), so you can branch for sweeps::

        base = runner.parse_cli()
        base.override(seed=42).run()
        base.override(seed=99).run()

    :meth:`run` auto-calls any unapplied pipeline steps.

    Args:
        command: Shell command to run (str is split via shlex).
        params: CLI parameters declared via :class:`Param`.
        outputs: Extra output files declared via :class:`Output`.
        metrics: Regex patterns to extract from stdout via :class:`Metric`.
        tags: W&B run tags.
        env: Extra environment variables for the subprocess.
        wandb_project: W&B project name (default: git repo name).
        group: W&B run group for sweeps.
    """

    command: str | list[str]
    params: list[Param] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    env: dict[str, str | None] = field(default_factory=dict)
    wandb_project: str | None = None
    group: str | None = None  # W&B run group for sweeps

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            self.command = shlex.split(self.command)
        self._param_values: dict[str, object] = {}
        self._param_sources: dict[str, str] = {}
        self._run_flags: _RunFlags | None = None
        self._cli_explicit_flags: set[str] = set()
        self._cli_parsed: bool = False
        self._defaults_resolved: bool = False
        self._filled: bool = False
        self._backends: list[LogBackend] = []

    # -------------------------------------------------------------------
    # Public param pipeline
    # -------------------------------------------------------------------

    def parse_cli(self, argv: list[str] | None = None) -> Runner:
        """Parse CLI arguments and return a new Runner with values applied.

        Args:
            argv: Command-line arguments to parse.  ``None`` means
                ``sys.argv[1:]``.
        """
        parsed_params, run_flags, explicit_flags = self._parse_cli_args(argv)

        new = copy.copy(self)
        new._param_values = dict(self._param_values)
        new._param_sources = dict(self._param_sources)

        for name, val in parsed_params.items():
            if val is not None and new._param_sources.get(name) != "override":
                new._param_values[name] = val
                new._param_sources[name] = "cli"

        new._run_flags = run_flags
        new._cli_explicit_flags = explicit_flags
        new._cli_parsed = True
        new._defaults_resolved = False
        new._filled = False
        return new

    def override(self, **kwargs: object) -> Runner:
        """Return a copy with override values applied.

        Example::

            r1 = runner.override(seed=42, prompt="a cat")
            r2 = runner.override(seed=99, prompt="a dog")
            r1.run()
            r2.run()
        """
        name_by_dest = {p.dest: p.name for p in self.params}
        valid_names = {p.name for p in self.params}

        resolved: dict[str, object] = {}
        for k, v in kwargs.items():
            resolved[name_by_dest.get(k, k)] = v

        unknown = set(resolved) - valid_names
        if unknown:
            msg = f"Unknown param(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)

        new = copy.copy(self)
        new._param_values = dict(self._param_values)
        new._param_sources = dict(self._param_sources)
        new._param_values.update(resolved)
        for name in resolved:
            new._param_sources[name] = "override"
        return new

    def with_metadata(
        self,
        *,
        project: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> Runner:
        """Return a copy with updated project, group, or tags."""
        new = copy.copy(self)
        if project is not None:
            new.wandb_project = project
        if group is not None:
            new.group = group
        if tags is not None:
            new.tags = list(tags)
        return new

    def resolve_defaults(self) -> Runner:
        """Return a copy with defaults and fixed values applied.

        Params already set (by CLI or override) are not changed.
        """
        new = copy.copy(self)
        new._param_values = dict(self._param_values)
        new._param_sources = dict(self._param_sources)

        for p in self.params:
            if p.name in new._param_values:
                continue
            if p.is_fixed:
                new._param_values[p.name] = p.value() if callable(p.value) else p.value
                new._param_sources[p.name] = "fixed"
            elif p.type == "bool":
                new._param_values[p.name] = False
                new._param_sources[p.name] = "default"
            elif p.default is not None:
                new._param_values[p.name] = (
                    p.default() if callable(p.default) else p.default
                )
                new._param_sources[p.name] = "default"

        # Cast multi-value args to per-element types
        for p in self.params:
            val = new._param_values.get(p.name)
            if p.nargs is not None and val is not None and val is not UNSET:
                new._param_values[p.name] = _cast_nargs(val, p.type_list)

        new._defaults_resolved = True
        return new

    def ask_user(self, *, interactive: bool | None = None) -> Runner:
        """Return a copy with missing params filled via interactive prompts.

        In non-interactive mode, raises SystemExit if required params
        are missing.  Auto-calls :meth:`resolve_defaults` if needed.
        """
        r = self
        if not r._defaults_resolved:
            r = r.resolve_defaults()

        if interactive is None:
            interactive = r._run_flags.interactive if r._run_flags else True

        new = copy.copy(r)
        new._param_values = dict(r._param_values)
        new._param_sources = dict(r._param_sources)

        # Params eligible for prompting: non-fixed, non-bool,
        # not explicitly set via CLI or overrides
        promptable = [
            p
            for p in self.params
            if not p.is_fixed
            and p.prompt
            and p.type != "bool"
            and r._param_sources.get(p.name) not in ("cli", "override")
        ]

        missing = [p for p in promptable if p.name not in r._param_values]

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
            new._filled = True
            return new

        for p in promptable:
            default = new._param_values.get(p.name)
            if p.nargs is not None:
                val = self._prompt_nargs(p, default=default)
            else:
                val = self._prompt_single(p, default=default)
            new._param_values[p.name] = val
            new._param_sources[p.name] = "prompt"

        new._filled = True
        return new

    def _merge_run_flags(
        self,
        *,
        dry_run: bool | None = None,
        interactive: bool | None = None,
        no_wandb: bool | None = None,
        run_name: str | None = None,
    ) -> _RunFlags:
        """Merge run() kwargs with CLI flags, warning on contradictions."""
        from dataclasses import replace as _replace

        base = self._run_flags or _RunFlags()
        updates: dict[str, object] = {}
        for field_name, value in [
            ("dry_run", dry_run),
            ("interactive", interactive),
            ("no_wandb", no_wandb),
            ("run_name", run_name),
        ]:
            if value is None:
                continue
            if field_name in self._cli_explicit_flags:
                cli_val = getattr(base, field_name)
                if cli_val != value:
                    print(
                        f"[genai_runner] Warning: run({field_name}={value!r})"
                        f" overrides CLI flag (was {cli_val!r})",
                        file=sys.stderr,
                    )
            updates[field_name] = value

        return _replace(base, **updates) if updates else base

    def run(
        self,
        *,
        dry_run: bool | None = None,
        interactive: bool | None = None,
        no_wandb: bool | None = None,
        run_name: str | None = None,
    ) -> None:
        """Execute the full run lifecycle.

        Auto-calls :meth:`parse_cli`, :meth:`resolve_defaults`, and
        :meth:`ask_user` for any steps not yet applied.

        Keyword args override CLI flags (with warnings on contradiction):

        Args:
            dry_run: Print the command and exit without running.
            interactive: Prompt for missing params (default True).
            no_wandb: Skip W&B logging (still logs to JSON).
            run_name: Override the W&B run name.
        """
        gib = 1024 * 1024 * 1024
        minimal_free_space = 1 * gib
        # Find nearest existing ancestor for disk usage check
        check_path = _RUNS_DIR
        while not check_path.exists():
            check_path = check_path.parent
        if shutil.disk_usage(check_path).free < minimal_free_space:
            msg = (
                "Not enough free space on device. Minimal free space:"
                f" {minimal_free_space / gib:.2f} GiB. Available free space:"
                f" {shutil.disk_usage(check_path).free / gib:.2f} GiB"
            )
            print(f"{_PREFIX} Error: {msg}")
            sys.exit(1)

        r = self
        if not r._cli_parsed:
            r = r.parse_cli()

        flags = r._merge_run_flags(
            dry_run=dry_run,
            interactive=interactive,
            no_wandb=no_wandb,
            run_name=run_name,
        )

        if not r._defaults_resolved:
            r = r.resolve_defaults()
        if not r._filled:
            r = r.ask_user(interactive=flags.interactive)

        param_values = r._param_values
        param_sources = r._param_sources

        # Git info and project
        git_info = _collect_git_info()
        project = flags.wandb_project or r.wandb_project or git_info.get("repo")
        if project is None:
            msg = (
                "Cannot determine project name:"
                " set wandb_project= or run from a git repo"
            )
            raise ValueError(msg)

        # Config
        config: dict[str, object] = {}
        for k, v in param_values.items():
            config[f"param/{k}"] = "<unset>" if v is UNSET else v
        for k, v in git_info.items():
            config[f"git/{k}"] = v
        timestamp = datetime.datetime.now(tz=datetime.UTC)
        config["meta/hostname"] = os.uname().nodename
        config["meta/datetime"] = timestamp.isoformat()
        config["meta/command"] = shlex.join(r.command)

        # Init WandbBackend first (needs to happen early to get run_name)
        wb_backend = None
        if not flags.dry_run and not flags.no_wandb:
            wb_backend = WandbBackend()
            wb_backend.init(project, flags.run_name, r.group, r.tags, config)
            run_name = wb_backend.run_name
            run_url = wb_backend.run_url
        elif flags.dry_run:
            run_name = flags.run_name or "run"
            run_url = "(dry run)"
        else:
            run_name = flags.run_name or "local"
            run_url = "(W&B disabled)"

        # Output dir
        date_str = timestamp.strftime("%Y%m%d_%H%M")
        if r.group:
            dir_name = f"{date_str}_{r.group}_{run_name}"
        else:
            dir_name = f"{date_str}_{run_name}"
        output_dir = _RUNS_DIR / project / dir_name

        # Augment config with output_dir and wandb info
        config["meta/output_dir"] = str(output_dir)
        if wb_backend is not None:
            config["wandb/name"] = run_name
            config["wandb/url"] = run_url
            wb_backend.update_config({"meta/output_dir": str(output_dir)})

        # Dry run: print summary and return (no dirs, no execution)
        if flags.dry_run:
            print(f"[dry-run] Project: {project}")
            print(f"[dry-run] Run name: {run_name}")
            print(f"[dry-run] Group: {r.group}")
            print(f"[dry-run] Tags: {r.tags}")
            print(f"[dry-run] Config:\n{pprint.pformat(config)}")
            interpolated_params = r._interpolate_output(param_values, output_dir)
            colored_cmd = r._format_command(interpolated_params, param_sources)
            print(f"{_PREFIX} Output dir: {output_dir}")
            print(f"{_PREFIX} Command:\n{colored_cmd}")
            # Show what files would be logged
            r._print_file_plan(interpolated_params)
            return

        # Create output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{_PREFIX} Output dir: {output_dir}")

        # JsonBackend is always active
        json_backend = JsonBackend(output_dir)
        json_backend.init(project, run_name, r.group, r.tags, config)

        # Assemble backends
        r._backends = [json_backend]
        if wb_backend is not None:
            r._backends.append(wb_backend)

        # Save code snapshot (git archive + dirty diff)
        try:
            _log_code_snapshot(r._backends, output_dir, git_info)
        except Exception as e:  # noqa: BLE001
            print(f"[genai_runner] Warning: code snapshot failed: {e}")

        # Interpolate $output in param values
        interpolated_params = r._interpolate_output(param_values, output_dir)

        # Log table params (prompt, etc.)
        r._log_table_params(param_values)

        # Log input files (log_when == "before")
        r._log_files(interpolated_params, when="before")

        # Build command
        cmd = r._build_command(interpolated_params)
        for b in r._backends:
            b.update_config({"meta/full_command": shlex.join(cmd)})
        colored_cmd = r._format_command(interpolated_params, param_sources)
        print(f"{_PREFIX} Command:\n{colored_cmd}")

        # Execute
        print("=" * 60)
        exit_code, duration, stdout_text, aborted = r._execute(cmd, output_dir)
        print("=" * 60)

        # Post-run: never raise, always try to finish backends
        r._post_run(
            interpolated_params,
            output_dir,
            exit_code,
            duration,
            stdout_text,
            r.tags,
            aborted=aborted,
        )
        if aborted or exit_code:
            print("Aborting run due to aborted or failed exit code")
            sys.exit(1)

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
        """Run post-execution steps (metrics, file uploads, code snapshot).

        Each step is individually try-excepted so backends always finish.
        """
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

        print(f"{_PREFIX} Status: {status} (exit code {exit_code})")
        print(f"{_PREFIX} Duration: {duration:.1f}s")
        print(f"{_PREFIX} Output dir: {output_dir}")

    # -----------------------------------------------------------------------
    # CLI parsing
    # -----------------------------------------------------------------------

    def _parse_cli_args(
        self, argv: list[str] | None = None
    ) -> tuple[dict, _RunFlags, set[str]]:
        """Parse argv into (param_values, run_flags, explicit_flags).

        Returns a tuple of:
        - param values dict (name -> value, None for unset)
        - _RunFlags with built-in flag values
        - set of flag names explicitly passed on CLI (for contradiction warnings)
        """
        parser = argparse.ArgumentParser(
            description="genai_runner experiment launcher",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Built-in flags — default=None so we can detect explicit usage
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=None,
            help="Print command and exit",
        )
        parser.add_argument(
            "--no-interactive",
            action="store_true",
            default=None,
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
            default=None,
            help="Skip W&B logging; still execute command and save outputs locally",
        )

        for p in self.params:
            if not p.is_fixed:
                kw = p._argparse_kwargs()
                if p.type == "bool":
                    kw["default"] = None
                parser.add_argument(p.flag, **kw)

        ns = parser.parse_args(argv)
        parsed_params = {
            p.name: getattr(ns, p.dest, None) for p in self.params if not p.is_fixed
        }

        # Track which run flags were explicitly set on CLI
        explicit_flags: set[str] = set()
        if ns.dry_run is not None:
            explicit_flags.add("dry_run")
        if ns.no_interactive is not None:
            explicit_flags.add("interactive")
        if ns.no_wandb is not None:
            explicit_flags.add("no_wandb")
        if ns.run_name is not None:
            explicit_flags.add("run_name")
        if ns.wandb_project is not None:
            explicit_flags.add("wandb_project")

        run_flags = _RunFlags(
            dry_run=bool(ns.dry_run),
            interactive=not bool(ns.no_interactive),
            no_wandb=bool(ns.no_wandb),
            run_name=ns.run_name,
            wandb_project=ns.wandb_project,
        )
        return parsed_params, run_flags, explicit_flags

    # -----------------------------------------------------------------------
    # Interactive prompts (internal helpers)
    # -----------------------------------------------------------------------

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
        param_sources: dict,
    ) -> str:
        """Build a colored command string for display."""
        _RST = "\033[0m"
        _BOLD = "\033[1m"
        _COLORS = {
            "flag": "\033[31m",  # red — prompted, required
            "flag+default": "\033[32m",  # green — prompted, kept default
            "cli": "\033[36m",  # cyan — from CLI/override, non-default value
            "cli+default": "\033[34m",  # blue — from CLI/override, kept default
            "value": "\033[33m",  # yellow — fixed value=
            "no-prompt": "\033[35m",  # magenta — no-prompt, default used
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

            source = param_sources.get(p.name, "")
            if source == "fixed":
                kind = "value"
            elif not p.prompt:
                kind = "no-prompt"
            elif source in ("cli", "override"):
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
        """Build the subprocess argv from command + resolved param values."""
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
        """Run subprocess, stream stdout/stderr to terminal and log files.

        Returns (exit_code, duration_seconds, stdout_text, aborted).
        """
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
            for k, v in self.env.items():
                if v is None:
                    run_env.pop(k, None)
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
            "dirty": git("status", "--porcelain", "--ignore-submodules=none") != "",
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

    # git archive: full snapshot of tracked files at HEAD (including submodules)
    archive_path = code_dir / "source.tar.gz"
    tar_path = code_dir / "source.tar"
    result = git("archive", "--format=tar", "-o", str(tar_path), "HEAD")
    if result.returncode != 0:
        msg = f"git archive failed: {result.stderr.decode()}"
        raise RuntimeError(msg)

    # Archive each submodule and append into the tar
    sub_result = git(
        "submodule",
        "foreach",
        "--recursive",
        "--quiet",
        "echo $displaypath",
    )
    if sub_result.returncode == 0 and sub_result.stdout.strip():
        for sub_path in sub_result.stdout.decode().strip().splitlines():
            sub_path = sub_path.strip()
            if not sub_path:
                continue
            sub_tar = code_dir / "sub.tar"
            r = git(
                "-C",
                sub_path,
                "archive",
                "--format=tar",
                f"--prefix={sub_path}/",
                "-o",
                str(sub_tar),
                "HEAD",
            )
            if r.returncode == 0 and sub_tar.exists():
                # Append submodule tar entries into main tar
                with (
                    tarfile.open(tar_path, "a") as main_tf,
                    tarfile.open(sub_tar) as sub_tf,
                ):
                    for member in sub_tf:
                        main_tf.addfile(
                            member,
                            sub_tf.extractfile(member) if member.isfile() else None,
                        )
                sub_tar.unlink()

    # Compress the combined tar
    with open(tar_path, "rb") as f_in, gzip.open(archive_path, "wb") as f_out:
        f_out.writelines(f_in)
    tar_path.unlink()

    # Dirty diff (staged + unstaged vs HEAD, including submodules)
    diff_path = None
    if git_info.get("dirty"):
        diff_result = git("diff", "HEAD", "--submodule=diff")
        if diff_result.returncode == 0 and diff_result.stdout:
            diff_path = code_dir / "dirty.patch"
            diff_path.write_bytes(diff_result.stdout)

    # Record in all backends
    files = [str(archive_path)]
    if diff_path and diff_path.exists():
        files.append(str(diff_path))
    for b in backends:
        b.log_artifact("code", "code", files)
