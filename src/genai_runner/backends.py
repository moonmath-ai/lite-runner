"""Log backend protocol and implementations for genai_runner."""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import git

import wandb

from .params import UNSET, _log_as_from_type

if TYPE_CHECKING:
    from .params import Metric, Output, Param

VideoFormat = Literal["gif", "mp4", "webm", "ogg"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log item types (collect phase returns these)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogFile:
    """A file to log to backends (path + upload type + key)."""

    path: Path
    log_as: str
    key: str


# ---------------------------------------------------------------------------
# Backend protocol and implementations
# ---------------------------------------------------------------------------


class LogBackend(Protocol):
    """Protocol for logging backends."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict[str, object],
    ) -> None: ...

    @property
    def run_name(self) -> str: ...

    def update_config(self, updates: dict[str, object]) -> None: ...

    def log_file(self, path: Path, log_as: str, key: str) -> None: ...

    def set_metric(self, name: str, value: object) -> None: ...

    def set_summary(self, summary: dict[str, object]) -> None: ...

    def set_tags(self, tags: list[str]) -> None: ...

    def finish(self, exit_code: int) -> None: ...


class WandbBackend:
    """Log backend that sends data to Weights & Biases."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict[str, object],
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,
            tags=tags,
            save_code=True,
            config=config,
        )

    @property
    def run_name(self) -> str:
        return self.run.name or self.run.id or "run"

    @property
    def run_url(self) -> str:
        url = self.run.url
        assert url is not None
        return url

    def update_config(self, updates: dict[str, object]) -> None:
        self.run.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        if log_as == "artifact":
            self.run.log_artifact(path, name=f"{key}-{self.run.id}", type=key)
        elif log_as == "video":
            fmt = _video_format(path)
            self.run.log({key: wandb.Video(str(path), format=fmt)})
        elif log_as == "image":
            self.run.log({key: wandb.Image(str(path))})
        elif log_as == "text":
            text = path.read_text(errors="replace")
            self.run.log({key: wandb.Html(f"<pre>{text}</pre>")})
        else:
            msg = f"Invalid log_as: {log_as}"
            raise ValueError(msg)

    def set_metric(self, name: str, value: object) -> None:
        self.run.summary[name] = value

    def set_summary(self, summary: dict[str, object]) -> None:
        self.run.summary.update(summary)

    def set_tags(self, tags: list[str]) -> None:
        self.run.tags = tags

    def finish(self, exit_code: int) -> None:
        self.run.finish(exit_code=exit_code)


class JsonBackend:
    """Log backend that accumulates run info and writes run_info.json on finish."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict[str, object],
    ) -> None:
        self.metadata = {
            "project": project,
            "name": name or "(local)",
            "group": group,
            "tags": list(tags),
        }
        self.config: dict[str, object] = dict(config)
        self.metrics: dict[str, object] = {}
        self.summary: dict[str, object] = {}
        self.files_logged: list[dict[str, str]] = []

    @property
    def run_name(self) -> str:
        name = self.metadata["name"]
        assert isinstance(name, str)
        return name

    def update_config(self, updates: dict[str, object]) -> None:
        self.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        self.files_logged.append({"path": str(path), "log_as": log_as, "key": key})

    def set_metric(self, name: str, value: object) -> None:
        self.metrics[name] = value

    def set_summary(self, summary: dict[str, object]) -> None:
        assert not self.summary, "set_summary called twice"
        self.summary = summary

    def set_tags(self, tags: list[str]) -> None:
        self.metadata["tags"] = tags

    def finish(self, exit_code: int) -> None:
        output_dir_val = self.config["meta/output_dir"]
        assert isinstance(output_dir_val, str)
        output_dir = Path(output_dir_val)
        run_info = {
            "metadata": self.metadata,
            "config": self.config,
            "metrics": self.metrics,
            "summary": self.summary,
            "files_logged": self.files_logged,
            "exit_code": exit_code,
        }
        (output_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, default=str)
        )


_dry_run_logger = logging.getLogger(f"{__name__.split('.')[0]}.dry_run")


class DryRunBackend:
    """Log backend that does nothing."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict[str, object],
    ) -> None:
        self.run_name = name or "dry_run"
        _dry_run_logger.info("Project: %s", project)
        _dry_run_logger.info("Name: %s", self.run_name)
        _dry_run_logger.info("Group: %s", group)
        _dry_run_logger.info("Tags: %s", tags)
        _dry_run_logger.info("Config: %s", config)

    def update_config(self, updates: dict[str, object]) -> None:
        _dry_run_logger.info("Updating config: %s", updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        _dry_run_logger.info("Logging %s [%s]: %s", key, log_as, path)

    def set_metric(self, name: str, value: object) -> None:
        _dry_run_logger.info("Setting metric: %s to %s", name, value)

    def set_summary(self, summary: dict[str, object]) -> None:
        _dry_run_logger.info("Setting summary: %s", summary)

    def set_tags(self, tags: list[str]) -> None:
        _dry_run_logger.info("Setting tags: %s", tags)

    def finish(self, exit_code: int) -> None:
        _dry_run_logger.info("Finishing with exit code: %s", exit_code)


# ---------------------------------------------------------------------------
# Collectors (collect_*: non-mutating) and preparers (prepare_*: create files)
# ---------------------------------------------------------------------------

_METRIC_CASTERS = {"float": float, "int": int}


def collect_metrics(
    metrics: list[Metric],
    stdout_text: str,
) -> list[tuple[str, object]]:
    """Extract metrics from stdout via regex.

    Returns list of (name, value) pairs.
    """
    items: list[tuple[str, object]] = []
    for m in metrics:
        matches = re.findall(m.pattern, stdout_text)
        if not matches:
            continue
        raw = matches[-1]  # last match wins
        try:
            val = _METRIC_CASTERS[m.type](raw)
        except (KeyError, ValueError):
            val = raw
        items.append((m.name, val))
    return items


def collect_param_files(
    params: list[Param],
    param_values: dict[str, object],
    when: str,
    *,
    dry_run: bool = False,
) -> list[LogFile]:
    """Collect param files to log (before or after run).

    When *dry_run*, skips existence checks.
    """
    items: list[LogFile] = []
    for p in params:
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
            if not dry_run and not path.exists():
                msg = f"File not found: {path} (param '{p.name}')"
                raise FileNotFoundError(msg)
            items.append(LogFile(path, log_as, key=p.name))
    return items


def prepare_extra_outputs(
    outputs: list[Output],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> list[LogFile]:
    """Collect Output declarations (globs, directories, single files).

    Preparation steps (zip creation, file copying) happen here so they
    are isolated from the dispatch phase.

    When *dry_run*, skips existence checks, globbing, copying, and zip
    creation.  Returns LogFiles with the paths that *would* be used.
    """
    out = str(output_dir)
    seen_zips: set[str] = set()
    items: list[LogFile] = []

    for o in outputs:
        raw_path = o.path.replace("$output", out)
        is_glob = any(c in raw_path for c in ("*", "?", "["))
        path = Path(raw_path)

        if dry_run:
            if is_glob or raw_path.endswith("/"):
                label = o.name or Path(raw_path.rstrip("/")).name or "output"
                if o.log_as == "zip":
                    items.append(
                        LogFile(output_dir / f"{label}.zip", "artifact", key=label)
                    )
                else:
                    items.append(LogFile(path, o.log_as, key=label))
            else:
                # Single file
                key = o.name or path.stem
                items.append(LogFile(path, o.log_as, key=key))
            continue

        if is_glob:
            base, pattern = _split_glob(raw_path)
            matches = sorted(base.glob(pattern))
        elif path.is_dir():
            if o.log_as != "zip":
                logger.warning(
                    "%s is a directory, uploading each file individually"
                    " (use log_as='zip' to zip instead)",
                    raw_path,
                )
            base = path
            matches = sorted(path.rglob("*"))
        else:
            # Single file
            if not path.exists():
                msg = f"Output file not found: {path}"
                raise FileNotFoundError(msg)
            # Preparation: copy to destination if requested
            if o.copy_to:
                dst = Path(o.copy_to.replace("$output", out))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
                path = dst
            key = o.name or path.stem
            items.append(LogFile(path, o.log_as, key=key))
            continue

        # Glob or directory
        if not matches:
            logger.warning("glob '%s' matched no files, skipping", o.path)
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
            # Preparation: create zip archive
            zip_path = create_zip(matches, base, output_dir, label)
            items.append(LogFile(zip_path, "artifact", key=label))
        else:
            items.extend(
                LogFile(m, o.log_as, key=label) for m in matches if m.is_file()
            )

    return items


def collect_run_logs(output_dir: Path, *, dry_run: bool = False) -> list[LogFile]:
    """Collect run/stdout/stderr log files.

    When *dry_run*, returns all three log files without checking existence.
    """
    return [
        LogFile(path, "text", key=name)
        for name in ("run.log", "stdout.log", "stderr.log")
        if (path := output_dir / name).exists() or dry_run
    ]


# ---------------------------------------------------------------------------
# Preparation helpers (create files, return paths)
# ---------------------------------------------------------------------------


def create_zip(
    matches: list[Path],
    base: Path,
    output_dir: Path,
    label: str,
) -> Path:
    """Zip matched files and return the zip path."""
    zip_path = output_dir / f"{label}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in matches:
            if m.is_file():
                zf.write(m, m.relative_to(base))
    return zip_path


def _open_repo() -> git.Repo | None:
    """Open the enclosing git repo, or return None."""
    try:
        return git.Repo(search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return None


def create_repo_archive(output_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Create git archive (tar.gz) of tracked files at HEAD.

    Returns the archive path, or None if not in a git repo.
    When *dry_run*, returns the path without creating any files.
    """
    repo = _open_repo()
    if repo is None:
        return None

    archive_path = output_dir / "code" / "source.tar.gz"
    if dry_run:
        return archive_path

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)

    tar_path = code_dir / "source.tar"
    with tar_path.open("wb") as f:
        repo.archive(f, format="tar")

    # Archive each submodule and append into the tar
    for submodule in repo.submodules:
        try:
            sub_repo = submodule.module()
        except git.InvalidGitRepositoryError:
            continue
        sub_tar = code_dir / "sub.tar"
        with sub_tar.open("wb") as f:
            sub_repo.archive(f, format="tar", prefix=f"{submodule.path}/")
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
    with tar_path.open("rb") as f_in, gzip.open(archive_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tar_path.unlink()

    return archive_path


def create_repo_diff(output_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Create dirty diff patch file.

    Returns the patch path, or None if repo is clean or not in a git repo.
    When *dry_run*, checks dirty status (read-only) but skips writing.
    """
    repo = _open_repo()
    if repo is None:
        return None

    diff_text = repo.git.diff("HEAD", submodule="diff")
    if not diff_text:
        return None

    diff_path = output_dir / "code" / "dirty.patch"
    if dry_run:
        return diff_path

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)
    diff_path.write_text(diff_text)
    return diff_path


def prepare_code_archive(output_dir: Path, *, dry_run: bool = False) -> list[LogFile]:
    """Create code archive and return as LogFile list."""
    path = create_repo_archive(output_dir, dry_run=dry_run)
    if path:
        return [LogFile(path, "artifact", "code")]
    return []


def prepare_code_diff(output_dir: Path, *, dry_run: bool = False) -> list[LogFile]:
    """Create dirty diff patch and return as LogFile list."""
    path = create_repo_diff(output_dir, dry_run=dry_run)
    if path:
        return [LogFile(path, "artifact", "code-diff")]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VIDEO_FORMATS: dict[str, VideoFormat] = {
    "gif": "gif",
    "mp4": "mp4",
    "webm": "webm",
    "ogg": "ogg",
}


def _video_format(path: Path) -> VideoFormat:
    """Extract and validate video format from file extension."""
    fmt = path.suffix.lstrip(".")
    result = _VIDEO_FORMATS.get(fmt)
    if result is None:
        msg = f"Unsupported video format '{fmt}' for {path}"
        raise ValueError(msg)
    return result


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
