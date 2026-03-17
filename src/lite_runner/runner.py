"""Runner orchestrator and helpers for lite_runner."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import logging
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
from typing import IO, TYPE_CHECKING, ClassVar, TextIO

from typing_extensions import Self, override

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

import git

from .backends import (
    DryRunBackend,
    JsonBackend,
    LogBackend,
    LogFile,
    WandbBackend,
    collect_metrics,
    collect_param_files,
    collect_run_logs,
    prepare_code_archive,
    prepare_code_diff,
    prepare_extra_outputs,
)
from .params import (
    Metric,
    Output,
    Param,
    _contains_unset,
    is_seq,
)

PACKAGE_NAME = __name__.split(".")[0]
RUNS_DIR = Path.home() / "lite_runs"

logger = logging.getLogger(PACKAGE_NAME)

RUNS_DIR.mkdir(parents=True, exist_ok=True)


class ColorFormatter(logging.Formatter):
    """Formatter that colors the ``name:`` prefix by log level."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[36m",  # cyan
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    @override
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        formatted = super().format(record)
        return f"{color}{record.name}:{self.RESET} {formatted}"


def _ensure_logging() -> None:
    """Set up a default handler if none configured (besides NullHandler)."""
    root = logging.getLogger(PACKAGE_NAME)
    if not any(h for h in root.handlers if not isinstance(h, logging.NullHandler)):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ColorFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)


@dataclass(frozen=True)
class RunResult:
    """Result of a :meth:`Runner.run` call."""

    output_dir: Path
    exit_code: int
    duration: float
    run_name: str
    project: str
    config: dict[str, object]
    param_values: dict[str, object]
    param_sources: dict[str, str]


@dataclass(frozen=True)
class RunFlags:
    """CLI flags that control runner behavior (not model params)."""

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
        """Return a new RunFlags with non-None overrides applied.

        Warns when an override contradicts a previously set flag.
        """
        updates: dict[str, object] = {}
        for field_name, value in overrides.items():
            if value is None:
                continue
            existing = getattr(self, field_name)
            if existing is not None and existing != value:
                logger.warning(
                    "run(%s=%r) overrides CLI flag (was %r)",
                    field_name,
                    value,
                    existing,
                )
            updates[field_name] = value
        return replace(self, **updates) if updates else self  # type: ignore[arg-type]


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
        """Parse command string and validate param names."""
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
        """Build argparse parser with built-in flags and param flags."""
        parser = argparse.ArgumentParser(
            description="coda launcher with run tracking",
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
                    # differentiate explicit False from implicit False
                    param_kwargs["default"] = None
                assert param.flag is not None  # noqa: S101
                parser.add_argument(param.flag, **param_kwargs)  # type: ignore[arg-type]

        return parser

    def copy(self) -> Self:
        """Return a deep copy of this runner."""
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
                param = self.params_by_name[name]
                cast_val = param.cast_nargs(val) if param.nargs is not None else val
                new.param_values[name] = cast_val
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
                names = ", ".join(p.name for p in missing)
                msg = (
                    f"Missing required params: {names}."
                    " Run without --no-interactive for interactive mode,"
                    " or pass them on the command line."
                )
                raise ValueError(msg)
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
        """Raise OSError if the runs directory has less than *needed_gib* free."""
        gib = 1024 * 1024 * 1024
        needed = needed_gib * gib
        check_path = RUNS_DIR
        while not check_path.exists():
            check_path = check_path.parent
        free = shutil.disk_usage(check_path).free
        if free < needed:
            msg = (
                f"Not enough free space on device."
                f" Minimal free space: {needed_gib:.2f} GiB."
                f" Available free space: {free / gib:.2f} GiB"
            )
            raise OSError(msg)

    def run(
        self,
        *,
        dry_run: bool | None = None,
        min_free_space_gib: float | None = None,
        no_interactive: bool | None = None,
        no_wandb: bool | None = None,
        project: str | None = None,
        run_name: str | None = None,
    ) -> RunResult:
        """Execute the full run lifecycle.

        Auto-calls :meth:`parse_cli`, :meth:`resolve_defaults`, and
        :meth:`ask_user` for any steps not yet applied.

        Keyword args override CLI flags (with warnings on contradiction).

        Returns a :class:`RunResult` with output_dir, exit_code, etc.
        """
        _ensure_logging()
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

        try:
            if flags.min_free_space_gib is not None:
                self.check_disk_space(flags.min_free_space_gib)

            if not r.defaults_resolved:
                r = r.resolve_defaults()
            if not r.filled:
                r = r.ask_user(no_interactive=flags.no_interactive)
        except KeyboardInterrupt:
            sys.exit(1)
        except (ValueError, OSError) as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(2)

        # Git info and project
        git_info = _collect_git_info()
        repo_name = git_info.get("repo")
        assert repo_name is None or isinstance(repo_name, str)  # noqa: S101
        project = flags.project or r.project or repo_name
        if project is None:
            msg = "Cannot determine project name: set project= or run from a git repo"
            raise ValueError(msg)

        # Config
        config: dict[str, object] = {}
        for k, v in r.param_values.items():
            config[f"param/{k}"] = "<unset>" if _contains_unset(v) else v
        for k, v in git_info.items():
            config[f"git/{k}"] = v
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
        config["meta/hostname"] = os.uname().nodename
        config["meta/datetime"] = timestamp.isoformat()
        config["meta/command"] = shlex.join(r.command)

        # Init WandbBackend first (needs to happen early to get run_name)
        backend_classes: list[type[WandbBackend | JsonBackend | DryRunBackend]] = []
        if flags.dry_run:
            backend_classes.append(DryRunBackend)
        else:
            if not flags.no_wandb:
                backend_classes.append(WandbBackend)
            backend_classes.append(JsonBackend)

        run_name = flags.run_name
        backends: dict[type, WandbBackend | JsonBackend | DryRunBackend] = {}
        for backend_class in backend_classes:
            backends[backend_class] = backend_class(
                project=project,
                name=run_name,
                group=r.run_group,
                tags=r.tags,
                config=config,
            )
            run_name = run_name or backends[backend_class].run_name
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
        wb_backend = backends.get(WandbBackend)
        if isinstance(wb_backend, WandbBackend):
            config["wandb/url"] = wb_backend.run_url
        else:
            config["wandb/url"] = "(no wandb)"
        for backend in backend_list:
            if not isinstance(backend, WandbBackend):
                backend.update_config({"wandb/url": config["wandb/url"]})

        # Create output dir
        if not flags.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", output_dir)

        # Save code snapshot (git archive + dirty diff)
        pre_run_files = []
        for name, fn in [
            ("code archive", prepare_code_archive),
            ("code diff", prepare_code_diff),
        ]:
            try:
                pre_run_files.extend(fn(output_dir, dry_run=bool(flags.dry_run)))
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("%s failed: %s", name, e)

        # Interpolate $output in param values
        interpolated_params = _interpolate_output(r.param_values, output_dir)

        # Warn about non-existent input paths (not under $output)
        warn_missing_input_paths(r.params, interpolated_params, output_dir)

        # Collect input files (log_when == "before")
        input_files = collect_param_files(
            r.params,
            interpolated_params,
            when="before",
            dry_run=bool(flags.dry_run),
        )
        pre_run_files.extend(input_files)

        # Copy input files to output dir for local reproducibility
        if not flags.dry_run and input_files:
            input_dir = output_dir / "input"
            input_dir.mkdir(exist_ok=True)
            for f in input_files:
                try:
                    shutil.copy2(f.path, input_dir / f.path.name)
                except Exception as e:  # noqa: BLE001, PERF203
                    logger.warning("copy input file %s failed: %s", f.path, e)

        # Log pre-run files
        for b in backend_list:
            try:
                for f in pre_run_files:
                    b.log_file(f.path, f.log_as, f.key)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("%s pre-run logging failed: %s", type(b).__name__, e)

        # Build command
        cmd = r.build_command(interpolated_params)
        for b in backend_list:
            b.update_config({"meta/full_command": shlex.join(cmd)})
        logger.info("Command:\n%s", shlex.join(cmd))

        # Execute
        logger.info(
            "Run started at %s",
            datetime.datetime.now(tz=datetime.timezone.utc)
            .astimezone()
            .strftime("%H:%M:%S %Z"),
        )
        if not flags.dry_run:
            exit_code, duration, stdout_text, stderr_text, aborted = r.execute(
                cmd, output_dir
            )
        else:
            exit_code = 0
            duration = 100
            stdout_text = "This is a dry run"
            stderr_text = ""
            aborted = False

        logger.info("Run finished")

        # Post-run: never raise, always try to finish backends
        r.post_run(
            backend_list,
            interpolated_params,
            output_dir,
            exit_code,
            duration,
            stdout_text,
            stderr_text,
            r.tags,
            aborted=aborted,
            dry_run=bool(flags.dry_run),
        )
        result = RunResult(
            output_dir=output_dir,
            exit_code=exit_code,
            duration=duration,
            run_name=run_name or "run",
            project=project,
            config=config,
            param_values=r.param_values,
            param_sources=r.param_sources,
        )

        if aborted or exit_code:
            logger.error("Aborting run due to aborted or failed exit code")
            sys.exit(1)

        return result

    def post_run(
        self,
        backends: Sequence[LogBackend],
        param_values: dict[str, object],
        output_dir: Path,
        exit_code: int,
        duration: float,
        stdout_text: str,
        stderr_text: str,
        run_tags: list[str],
        *,
        aborted: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Run post-execution steps (metrics, file uploads, code snapshot).

        Collection/preparation is try-excepted per step so one failure
        doesn't skip others.  Then each backend gets all collected data,
        with per-backend error handling.
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

        # Collect metrics
        metrics = []
        try:
            metrics = collect_metrics(self.metrics, stdout_text + "\n" + stderr_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("extract metrics failed: %s", e)

        # Collect/prepare files
        files = []
        try:
            files.extend(
                collect_param_files(
                    self.params, param_values, when="after", dry_run=dry_run
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("output files failed: %s", e)

        # Log file hashes for param output files
        for f in files:
            try:
                sha = hashlib.sha256(f.path.read_bytes()).hexdigest()
                logger.info("%s  %s", sha, f.path)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("hash %s failed: %s", f.path, e)

        file_steps: list[tuple[str, Callable[[], list[LogFile]]]] = [
            (
                "extra outputs",
                lambda: prepare_extra_outputs(
                    self.outputs,
                    output_dir,
                    dry_run=dry_run,
                ),
            ),
            (
                "run logs",
                lambda: collect_run_logs(output_dir, dry_run=dry_run),
            ),
        ]
        for step_name, collector in file_steps:
            try:
                files.extend(collector())
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("%s failed: %s", step_name, e)

        # Send to each backend
        for b in backends:
            try:
                for name, value in metrics:
                    b.set_metric(name, value)
                for f in files:
                    b.log_file(f.path, f.log_as, f.key)
                b.set_summary(summary)
                if status != "success":
                    b.set_tags([*run_tags, status])
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("%s logging failed: %s", type(b).__name__, e)

        # Finish each backend individually
        for b in backends:
            try:
                b.finish(exit_code)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning("finish %s failed: %s", type(b).__name__, e)

        logger.info("Status: %s (exit code %s)", status, exit_code)
        logger.info("Duration: %s", datetime.timedelta(seconds=duration))
        logger.info("Output dir: %s", output_dir)

    # -----------------------------------------------------------------------
    # Build command
    # -----------------------------------------------------------------------

    def build_command(self, param_values: dict[str, object]) -> list[str]:
        """Build the subprocess command as a plain token list."""
        assert isinstance(self.command, list)  # noqa: S101
        cmd: list[str] = self.command[:]
        for p in self.params:
            val = param_values.get(p.name)
            if _contains_unset(val):
                continue
            if val is None or p.flag is None:
                msg = f"Param {p.name!r} missing value or flag in build_command"
                raise RuntimeError(msg)
            if p.type == "bool" and not val:
                continue
            cmd.append(p.flag)
            if p.type == "bool":
                continue
            val_list = val if is_seq(val) else [val]
            cmd.extend(str(v) for v in val_list)
        return cmd

    # -----------------------------------------------------------------------
    # Subprocess execution
    # -----------------------------------------------------------------------

    def execute(
        self, cmd: list[str], output_dir: Path
    ) -> tuple[int, float, str, str, bool]:
        """Run subprocess, stream stdout/stderr to terminal and log files.

        Returns (exit_code, duration_seconds, stdout_text, stderr_text, aborted).
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        lock = threading.Lock()

        def stream_pipe(
            pipe: IO[bytes],
            sys_stream: TextIO,
            file_log: TextIO,
            *,
            prefix: str = "",
            capture_list: list[str] | None = None,
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
                    if capture_list is not None:
                        capture_list.append(text)
            pipe.close()

        with ExitStack() as stack:
            log_combined = stack.enter_context((output_dir / "run.log").open("w"))
            log_stdout = stack.enter_context((output_dir / "stdout.log").open("w"))
            log_stderr = stack.enter_context((output_dir / "stderr.log").open("w"))

            run_env: dict[str, str] = {**os.environ}
            for k, v in self.env.items():
                if v is None:
                    run_env.pop(k, None)
                else:
                    run_env[k] = v
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
                kwargs={"capture_list": stdout_lines},
            )
            t_err = threading.Thread(
                target=stream_pipe,
                args=(proc.stderr, sys.stderr, log_stderr),
                kwargs={"prefix": "[stderr] ", "capture_list": stderr_lines},
            )

            aborted = False
            t_out.start()
            t_err.start()

            try:
                proc.wait()
            except KeyboardInterrupt:
                aborted = True
                logger.warning("Ctrl-C received, terminating subprocess...")
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

        return (
            proc.returncode,
            duration,
            "".join(stdout_lines),
            "".join(stderr_lines),
            aborted,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subst_output(v: object, out: str) -> object:
    if isinstance(v, str):
        v = v.replace("$output", out)
        if v.startswith("~"):
            v = str(Path(v).expanduser())
        return v
    return v


def _interpolate_output(
    params: dict[str, object], output_dir: Path
) -> dict[str, object]:
    """Return a copy of *params* with $output replaced in string values."""
    out = str(output_dir)
    return {
        k: [_subst_output(x, out) for x in v] if is_seq(v) else _subst_output(v, out)
        for k, v in params.items()
    }


def warn_missing_input_paths(
    params: list[Param],
    interpolated_params: Mapping[str, object],
    output_dir: Path,
) -> None:
    """Log a warning for each input path param whose file doesn't exist."""
    out_prefix = str(output_dir)
    for p in params:
        val = interpolated_params.get(p.name)
        if val is None or _contains_unset(val):
            continue
        types = p.type_list
        values = val if is_seq(val) else [val]
        for t, v in zip(types, values, strict=False):
            if not t.startswith("path"):
                continue
            s = str(v)
            if s.startswith(out_prefix):
                continue
            if not Path(s).exists():
                logger.warning("Input path does not exist: %s (param '%s')", s, p.name)


def _collect_git_info() -> dict[str, object]:
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
