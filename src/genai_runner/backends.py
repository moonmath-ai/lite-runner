"""Log backend protocol and implementations for genai_runner."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import wandb

if TYPE_CHECKING:
    from .params import Metric, Output, Param


class LogBackend(Protocol):
    """Protocol for logging backends."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict,
    ) -> None: ...

    @property
    def run_name(self) -> str: ...

    def update_config(self, updates: dict) -> None: ...

    def log_file(self, path: Path, log_as: str, key: str) -> None: ...

    def set_metric(self, name: str, value: object) -> None: ...

    def set_summary(self, summary: dict) -> None: ...

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
        config: dict,
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
        return self.run.url

    def update_config(self, updates: dict) -> None:
        self.run.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        if log_as == "artifact":
            self.run.log_artifact(path, name=f"{key}-{self.run.id}", type=key)
        elif log_as == "video":
            fmt = path.suffix.lstrip(".")
            self.run.log({key: wandb.Video(str(path), format=fmt)})
        elif log_as == "image":
            self.run.log({key: wandb.Image(str(path))})
        elif log_as == "text":
            text = path.read_text(errors="replace")
            self.run.log({key: wandb.Html(f"<pre>{text}</pre>")})
        assert False, f"Invalid log_as: {log_as}"

    def set_metric(self, name: str, value: object) -> None:
        self.run.summary[name] = value

    def set_summary(self, summary: dict) -> None:
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
        config: dict,
    ) -> None:
        self.metadata = {
            "project": project,
            "name": name or "(local)",
            "group": group,
            "tags": list(tags),
        }
        self.config = dict(config)
        self.metrics = {}
        self.summary = {}
        self.files_logged = []

    @property
    def run_name(self) -> str:
        return self.metadata["name"]

    def update_config(self, updates: dict) -> None:
        self.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        self.files_logged.append({"path": str(path), "log_as": log_as, "key": key})

    def set_metric(self, name: str, value: object) -> None:
        self.metrics[name] = value

    def set_summary(self, summary: dict) -> None:
        self.summary = summary

    def set_tags(self, tags: list[str]) -> None:
        self.metadata["tags"] = tags

    def finish(self, exit_code: int) -> None:
        output_dir = Path(self.config["meta/output_dir"])
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


class DryRunBackend:
    """Log backend that does nothing."""

    def __init__(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict,
    ) -> None:
        self.run_name = name or "dry_run"
        print(f"[dry-run] Project: {project}")
        print(f"[dry-run] Name: {self.run_name}")
        print(f"[dry-run] Group: {group}")
        print(f"[dry-run] Tags: {tags}")
        print(f"[dry-run] Config: {config}")

    def update_config(self, updates: dict) -> None:
        print(f"[dry-run] Updating config: {updates}")

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        print(f"[dry-run] Logging file: {path} as {log_as} as {key}")

    def set_metric(self, name: str, value: object) -> None:
        print(f"[dry-run] Setting metric: {name} to {value}")

    def set_summary(self, summary: dict) -> None:
        print(f"[dry-run] Setting summary: {summary}")

    def set_tags(self, tags: list[str]) -> None:
        print(f"[dry-run] Setting tags: {tags}")

    def finish(self, exit_code: int) -> None:
        print(f"[dry-run] Finishing with exit code: {exit_code}")


# ---------------------------------------------------------------------------
# Multi-backend logging functions
# ---------------------------------------------------------------------------


def extract_metrics(
    backends: list[LogBackend],
    metrics: list[Metric],
    stdout_text: str,
) -> None:
    """Extract metrics from stdout via regex and record in all backends."""
    casters = {"float": float, "int": int}
    for m in metrics:
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
        for b in backends:
            b.set_metric(m.name, val)


def log_files(
    backends: list[LogBackend],
    params: list[Param],
    param_values: dict,
    when: str,
) -> None:
    """Log param files (before or after run) to all backends."""
    from .params import UNSET, _log_as_from_type

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
            if not path.exists():
                msg = f"File not found: {path} (param '{p.name}')"
                raise FileNotFoundError(msg)
            for b in backends:
                b.log_file(path, log_as, key=p.name)


def log_extra_outputs(
    backends: list[LogBackend],
    outputs: list[Output],
    output_dir: Path,
) -> None:
    """Log Output declarations (globs, directories, single files) to all backends."""
    out = str(output_dir)
    seen_zips: set[str] = set()
    for o in outputs:
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
            log_single_output(backends, path, o, out)
            continue

        # Glob or directory: upload matches
        if not matches:
            print(f"[genai_runner] Warning: glob '{o.path}' matched no files, skipping")
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
            zip_and_upload(backends, matches, base, output_dir, label)
        else:
            for m in matches:
                if m.is_file():
                    for b in backends:
                        b.log_file(m, o.log_as, key=label)


def log_run_logs(backends: list[LogBackend], output_dir: Path) -> None:
    """Upload run/stdout/stderr logs as an artifact to all backends."""
    files = [
        str(output_dir / name)
        for name in ("run.log", "stdout.log", "stderr.log")
        if (output_dir / name).exists()
    ]
    for file in files:
        for b in backends:
            b.log_file(file, "text", file.name)


def log_single_output(
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


def zip_and_upload(
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


def log_code_snapshot(
    backends: list[LogBackend], output_dir: Path, git_info: dict
) -> None:
    """Save a full code snapshot: git archive + dirty diff."""
    import git

    if not git_info:
        return

    try:
        repo = git.Repo(search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # git archive: full snapshot of tracked files at HEAD
    archive_path = code_dir / "source.tar.gz"
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
        f_out.writelines(f_in)
    tar_path.unlink()
    for b in backends:
        b.log_file(archive_path, "artifact", "code")

    # Dirty diff (staged + unstaged vs HEAD, including submodules)
    if git_info.get("dirty"):
        diff_text = repo.git.diff("HEAD", submodule="diff")
        assert diff_text
        diff_path = code_dir / "dirty.patch"
        diff_path.write_text(diff_text)
        for b in backends:
            b.log_file(diff_path, "artifact", "code-diff")


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
