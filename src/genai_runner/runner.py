"""Runner orchestrator and helpers for genai_runner."""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import IO, Self, TextIO

import git

from .backends import (
    DryRunBackend,
    JsonBackend,
    LogBackend,
    WandbBackend,
    extract_metrics,
    log_code_snapshot,
    log_extra_outputs,
    log_files,
    log_run_logs,
    log_table_params,
)
from .params import (
    UNSET,
    Metric,
    Output,
    Param,
)

RUNS_DIR = Path.home() / "genai_runs"
LOGGING_PREFIX = "\033[36mgenai-runner:\033[0m"


RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunFlags:
    dry_run: bool | None = None
    min_free_space_gib: float | None = None
    no_interactive: bool | None = None
    no_wandb: bool | None = None
    project: str | None = None
    run_name: str | None = None

    @classmethod
    def from_namespace(cls, ns: object) -> RunFlags:
        """Build RunFlags from an argparse namespace."""
        return cls(**{f.name: getattr(ns, f.name, None) for f in fields(cls)})

    def merge(self, **overrides: object) -> RunFlags:
        """Return a new RunFlags with *overrides* applied.

        ``None`` values in *overrides* are ignored (no change).
        Warns when an override contradicts a previously set flag.
        """
        updates: dict[str, object] = {}
        for field_name, value in overrides.items():
            if value is None:
                continue
            existing = getattr(self, field_name)
            if existing is not None and existing != value:
                print(
                    f"[genai_runner] Warning: run({field_name}={value!r})"
                    f" overrides CLI flag (was {existing!r})",
                    file=sys.stderr,
                )
            updates[field_name] = value
        return replace(self, **updates) if updates else self


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
        tags: run tags.
        env: Extra environment variables for the subprocess.
        project: project name (default: git repo name).
        run_group: run group for sweeps.
    """

    command: str | list[str]
    params: list[Param] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    env: dict[str, str | None] = field(default_factory=dict)
    project: str | None = None
    run_group: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            self.command = shlex.split(self.command)
        reserved = {f.name for f in fields(RunFlags)}
        for p in self.params:
            if p.name in reserved:
                msg = f"Param name {p.name!r} conflicts with built-in flag"
                raise ValueError(msg)
        self.params_by_name: dict[str, Param] = {p.name: p for p in self.params}
        self.param_values: dict[str, object] = {}
        self.param_sources: dict[str, str] = {}
        self.run_flags: RunFlags = RunFlags()
        self.cli_parsed: bool = False
        self.defaults_resolved: bool = False
        self.filled: bool = False

    # -------------------------------------------------------------------
    # Public param pipeline
    # -------------------------------------------------------------------

    def get_parser(self) -> argparse.ArgumentParser:
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
            "--min-free-space-gib",
            type=float,
            default=None,
            help="Minimum free disk space in GiB (default: 1.0)",
        )
        parser.add_argument(
            "--no-interactive",
            action="store_true",
            default=None,
            help="Non-interactive mode; fail if required params are missing",
        )
        parser.add_argument(
            "--no-wandb",
            action="store_true",
            default=None,
            help="Skip W&B logging; still execute command and save outputs locally",
        )
        parser.add_argument(
            "--project",
            default=None,
            help="Override project name",
        )
        parser.add_argument("--run-name", default=None, help="Override W&B run name")

        for param in self.params:
            if not param.is_fixed:
                param_kwargs = param.argparse_kwargs()
                if param.type == "bool":
                    # we need diffrentiating between explicit False and implicit False, so we set default to None
                    param_kwargs["default"] = None
                parser.add_argument(param.flag, **param_kwargs)

        return parser

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def with_metadata(
        self,
        *,
        project: str | None = None,
        run_group: str | None = None,
        tags: list[str] | None = None,
    ) -> Runner:
        """Return a copy with updated project, group, or tags."""
        new = self.copy()
        if project is not None:
            new.project = project
        if run_group is not None:
            new.run_group = run_group
        if tags is not None:
            new.tags = tags
        return new

    def override(self, **kwargs: object) -> Runner:
        """Return a copy with override values applied.

        Example::

            r1 = runner.override(seed=42, prompt="a cat")
            r2 = runner.override(seed=99, prompt="a dog")
            r1.run()
            r2.run()
        """
        dest_to_name = {p.dest: p.name for p in self.params}
        valid_param_names = set(dest_to_name.values())
        override_params = {dest_to_name.get(k, k): v for k, v in kwargs.items()}

        unknown = set(override_params) - valid_param_names
        if unknown:
            msg = f"Unknown param(s): {', '.join(sorted(unknown))}"
            raise ValueError(msg)

        new = self.copy()
        new.param_values.update(override_params)
        for name in override_params:
            new.param_sources[name] = "override"
        return new

    def parse_cli(self, argv: list[str] | None = None) -> Self:
        """Parse CLI arguments and return a Runner with values applied.

        Args:
            argv: Command-line arguments to parse.  ``None`` means
                ``sys.argv[1:]``.

        Returns:
            A new Runner with the values applied.
        """
        parser = self.get_parser()
        parsed_args = parser.parse_args(argv)
        parsed_params = {
            p.name: getattr(parsed_args, p.dest, None)
            for p in self.params
            if not p.is_fixed
        }

        new = self.copy()
        for name, val in parsed_params.items():
            if val is not None and new.param_sources.get(name) != "override":
                if self.params_by_name[name].nargs is not None:
                    val = self.params_by_name[name].cast_nargs(val)
                new.param_values[name] = val
                new.param_sources[name] = "cli"

        new.run_flags = RunFlags.from_namespace(parsed_args)
        new.cli_parsed = True
        return new

    def resolve_defaults(self) -> Runner:
        """Return a copy with defaults and fixed values applied.

        Params already set (by CLI or override) are not changed.
        """
        new = self.copy()
        for p in self.params:
            if p.name in new.param_values:
                continue
            if p.is_fixed:
                new.param_values[p.name] = p.value() if callable(p.value) else p.value
                new.param_sources[p.name] = "fixed"
            elif p.type == "bool":
                new.param_values[p.name] = False
                new.param_sources[p.name] = "default"
            elif p.default is not None:
                new.param_values[p.name] = (
                    p.default() if callable(p.default) else p.default
                )
                new.param_sources[p.name] = "default"
        new.defaults_resolved = True
        return new

    def ask_user(self, *, no_interactive: bool | None = None) -> Runner:
        """Return a copy with missing params filled via interactive prompts.

        In non-interactive mode, raises SystemExit if required params
        are missing.  Auto-calls :meth:`resolve_defaults` if needed.
        """
        new = self.copy() if self.defaults_resolved else self.resolve_defaults()

        if no_interactive is None:
            no_interactive = new.run_flags.no_interactive

        # Params eligible for prompting: non-fixed,
        # not explicitly set via CLI or overrides
        promptable = [
            p
            for p in self.params
            if not p.is_fixed
            and p.prompt
            and new.param_sources.get(p.name) not in ("cli", "override")
        ]

        missing = [p for p in promptable if p.name not in new.param_values]

        if no_interactive:
            if missing:
                names = [p.name for p in missing]
                print(
                    f"Error: missing required params: {', '.join(names)}",
                    file=sys.stderr,
                )
                print(
                    "Run without --no-interactive for interactive mode,"
                    " or pass them on the command line.",
                    file=sys.stderr,
                )
                sys.exit(2)
            new.filled = True
            return new

        for p in promptable:
            default = new.param_values.get(p.name)
            val = p.ask(default=default)
            new.param_values[p.name] = val
            new.param_sources[p.name] = "prompt"

        new.filled = True
        return new

    def check_disk_space(self, needed_gib: float) -> None:
        """Exit if the runs directory has less than *needed_gib* free."""
        gib = 1024 * 1024 * 1024
        needed = needed_gib * gib
        check_path = RUNS_DIR
        while not check_path.exists():
            check_path = check_path.parent
        free = shutil.disk_usage(check_path).free
        if free < needed:
            print(
                f"{LOGGING_PREFIX} Error: Not enough free space on device."
                f" Minimal free space: {needed_gib:.2f} GiB."
                f" Available free space: {free / gib:.2f} GiB",
            )
            sys.exit(1)  # TODO: all sys.exit should occur at run()

    def run(
        self,
        *,
        dry_run: bool | None = None,
        min_free_space_gib: float | None = None,
        no_interactive: bool | None = None,
        no_wandb: bool | None = None,
        project: str | None = None,
        run_name: str | None = None,
    ) -> None:
        """Execute the full run lifecycle.

        Auto-calls :meth:`parse_cli`, :meth:`resolve_defaults`, and
        :meth:`ask_user` for any steps not yet applied.

        Keyword args override CLI flags (with warnings on contradiction).
        """
        r = self
        if not r.cli_parsed:
            r = r.parse_cli()

        flags = r.run_flags.merge(
            dry_run=dry_run,
            min_free_space_gib=min_free_space_gib,
            no_interactive=no_interactive,
            no_wandb=no_wandb,
            project=project,
            run_name=run_name,
        )

        if flags.min_free_space_gib is not None:
            self.check_disk_space(flags.min_free_space_gib)

        if not r.defaults_resolved:
            r = r.resolve_defaults()
        if not r.filled:
            r = r.ask_user(no_interactive=flags.no_interactive)

        # Git info and project
        git_info = _collect_git_info()
        project = flags.project or r.project or git_info.get("repo")
        if project is None:
            msg = "Cannot determine project name: set project= or run from a git repo"
            raise ValueError(msg)

        # Config
        config: dict[str, object] = {}
        for k, v in r.param_values.items():
            config[f"param/{k}"] = "<unset>" if v is UNSET else v
        for k, v in git_info.items():
            config[f"git/{k}"] = v
        timestamp = datetime.datetime.now(tz=datetime.UTC)
        config["meta/hostname"] = os.uname().nodename
        config["meta/datetime"] = timestamp.isoformat()
        config["meta/command"] = shlex.join(r.command)

        # Init WandbBackend first (needs to happen early to get run_name)
        backend_classes = []
        if flags.dry_run:
            backend_classes.append(DryRunBackend)
        else:
            if not flags.no_wandb:
                backend_classes.append(WandbBackend)
            backend_classes.append(JsonBackend)

        run_name = flags.run_name
        backends = {}
        for backend_class in backend_classes:
            backends[backend_class] = backend_class(
                project=project,
                name=run_name,
                group=r.run_group,
                tags=r.tags,
                config=config,
            )
            run_name = run_name or backends[backend_class].run_name
            # TODO: dry run backend should default to "dry_run"
            # TODO: json backend should default to "local"
        backend_list = list(backends.values())

        # Output dir
        date_str = timestamp.strftime("%Y%m%d_%H%M")
        if r.run_group:
            dir_name = f"{date_str}_{r.run_group}_{run_name}"
        else:
            dir_name = f"{date_str}_{run_name}"
        output_dir = RUNS_DIR / project / dir_name
        config["meta/output_dir"] = str(output_dir)
        for backend in backend_list:
            backend.update_config({"meta/output_dir": config["meta/output_dir"]})

        # W&B url:
        config["wandb/url"] = (
            backends[WandbBackend].run_url if WandbBackend in backends else "(no wandb)"
        )
        for backend in backend_list:
            if not isinstance(backend, WandbBackend):
                backend.update_config({"wandb/url": config["wandb/url"]})

        # Create output dir
        if not flags.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{LOGGING_PREFIX} Output dir: {output_dir}")

        # Save code snapshot (git archive + dirty diff)
        try:
            log_code_snapshot(backend_list, output_dir, git_info)
        except Exception as e:  # noqa: BLE001
            print(f"[genai_runner] Warning: code snapshot failed: {e}")

        # Interpolate $output in param values
        interpolated_params = _interpolate_output(r.param_values, output_dir)

        # Log table params (prompt, etc.)
        log_table_params(backend_list, r.params, r.param_values)

        # Log input files (log_when == "before")
        log_files(backend_list, r.params, interpolated_params, when="before")

        # Build command
        cmd = r.build_command(interpolated_params)
        for b in backend_list:
            b.update_config({"meta/full_command": shlex.join(cmd)})
        colored = r.build_command(interpolated_params, r.param_sources)
        print(f"{LOGGING_PREFIX} Command:\n{' '.join(colored)}")

        # Execute
        print("=" * 60)
        if not flags.dry_run:
            exit_code, duration, stdout_text, aborted = r._execute(cmd, output_dir)
        else:
            exit_code = 0
            duration = 100
            stdout_text = "This is a dry run"
            aborted = False
        print("=" * 60)

        # Post-run: never raise, always try to finish backends
        r._post_run(
            backend_list,
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
        backends: list[LogBackend],
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
                lambda: extract_metrics(backends, self.metrics, stdout_text),
            ),
            (
                "log output files",
                lambda: log_files(backends, self.params, param_values, when="after"),
            ),
            (
                "log extra outputs",
                lambda: log_extra_outputs(backends, self.outputs, output_dir),
            ),
            (
                "log run logs",
                lambda: log_run_logs(backends, output_dir),
            ),
            (
                "tag run status",
                lambda: (
                    [b.set_tags([*run_tags, status]) for b in backends]
                    if status != "success"
                    else None
                ),
            ),
            (
                "update summary",
                lambda: [b.set_summary(summary) for b in backends],
            ),
        ]

        for step_name, step in steps:
            try:
                step()
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: {step_name} failed: {e}")

        # Finish each backend individually so one failure doesn't block others
        for b in backends:
            try:
                b.finish(exit_code)
            except Exception as e:  # noqa: BLE001
                print(f"[genai_runner] Warning: finish {type(b).__name__} failed: {e}")

        print(f"{LOGGING_PREFIX} Status: {status} (exit code {exit_code})")
        print(f"{LOGGING_PREFIX} Duration: {duration:.1f}s")
        print(f"{LOGGING_PREFIX} Output dir: {output_dir}")

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def build_command(
        self,
        param_values: dict,
        param_sources: dict | None = None,
    ) -> list[str]:
        """Build command as a token list.

        Without *param_sources*, returns plain tokens for subprocess.
        With *param_sources*, tokens are wrapped in ANSI color codes
        for display — join with ``" ".join()`` to print.
        """
        color = param_sources is not None
        cmd = self.command[:]
        for p in self.params:
            val = param_values.get(p.name)
            if val is UNSET:
                continue
            assert val is not None
            assert p.flag is not None
            if p.type == "bool" and not val:
                continue
            clr = _SOURCE_COLORS.get(param_sources.get(p.name, ""), "") if color else ""
            rst = _RST if color else ""
            bold = _BOLD if color else ""
            cmd.append(f"{clr}{p.flag}{rst}")
            if p.type == "bool":
                continue
            values = val if isinstance(val, list) else [val]
            if color:
                values = [shlex.quote(str(v)) for v in values]
            values = [f"{bold}{clr}{v}{rst}" for v in values]
            cmd.extend(values)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RST = "\033[0m"
_BOLD = "\033[1m"
_SOURCE_COLORS = {
    "prompt": "\033[31m",  # red — user prompted
    "default": "\033[32m",  # green — default value
    "cli": "\033[36m",  # cyan — from CLI
    "override": "\033[36m",  # cyan — from override
    "fixed": "\033[33m",  # yellow — fixed value=
}


def _interpolate_output(params: dict, output_dir: Path) -> dict:
    """Return a copy of *params* with $output replaced in string values."""
    out = str(output_dir)
    result = {}
    for k, v in params.items():
        if isinstance(v, list):
            result[k] = [
                x.replace("$output", out) if isinstance(x, str) else x for x in v
            ]
        elif isinstance(v, str):
            result[k] = v.replace("$output", out)
        else:
            result[k] = v
    return result


def _collect_git_info() -> dict:
    try:
        repo = git.Repo(search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return {}
    return {
        "repo": Path(repo.working_dir).name,
        "commit": repo.head.commit.hexsha,
        "branch": repo.active_branch.name if not repo.head.is_detached else "HEAD",
        "dirty": repo.is_dirty(untracked_files=True, submodules=True),
    }
