"""Log backend protocol and implementations for genai_runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import wandb


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

    def log_artifact(self, name: str, type: str, files: list[str]) -> None: ...

    def set_metric(self, name: str, value: object) -> None: ...

    def set_summary(self, summary: dict) -> None: ...

    def set_tags(self, tags: list[str]) -> None: ...

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        """Log a table of values for prominent display."""
        ...

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
        if log_as == "video":
            fmt = path.suffix.lstrip(".")
            self.run.log({key: wandb.Video(str(path), format=fmt)})
        elif log_as == "image":
            self.run.log({key: wandb.Image(str(path))})
        elif log_as == "text":
            text = path.read_text(errors="replace")
            self.run.log({key: wandb.Html(f"<pre>{text}</pre>")})
        elif log_as == "artifact":
            artifact = wandb.Artifact(f"{key}-{self.run.id}", type=log_as)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def log_artifact(self, name: str, type: str, files: list[str]) -> None:
        artifact = wandb.Artifact(f"{name}-{self.run.id}", type=type)
        for f in files:
            artifact.add_file(f)
        self.run.log_artifact(artifact)

    def set_metric(self, name: str, value: object) -> None:
        self.run.summary[name] = value

    def set_summary(self, summary: dict) -> None:
        self.run.summary.update(summary)

    def set_tags(self, tags: list[str]) -> None:
        self.run.tags = tags

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        table = wandb.Table(columns=columns, data=data)
        self.run.log({key: table})

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
        self.tables = {}

    @property
    def run_name(self) -> str:
        return self.metadata["name"]

    def update_config(self, updates: dict) -> None:
        self.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        self.files_logged.append({"path": str(path), "log_as": log_as, "key": key})

    def log_artifact(self, name: str, type: str, files: list[str]) -> None:
        self.files_logged.append({"type": type, "files": files})

    def set_metric(self, name: str, value: object) -> None:
        self.metrics[name] = value

    def set_summary(self, summary: dict) -> None:
        self.summary = summary

    def set_tags(self, tags: list[str]) -> None:
        self.metadata["tags"] = tags

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        self.tables[key] = {
            "columns": columns,
            "data": data,
        }

    def finish(self, exit_code: int) -> None:
        output_dir = Path(self.config["meta/output_dir"])
        run_info = {
            "metadata": self.metadata,
            "config": self.config,
            "metrics": self.metrics,
            "summary": self.summary,
            "files_logged": self.files_logged,
            "tables": self.tables,
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
        print(f"[dry-run] Project: {project}")
        print(f"[dry-run] Name: {name}")
        print(f"[dry-run] Group: {group}")
        print(f"[dry-run] Tags: {tags}")
        print(f"[dry-run] Config: {config}")

    @property
    def run_name(self) -> str:
        return "dry_run"

    def update_config(self, updates: dict) -> None:
        print(f"[dry-run] Updating config: {updates}")

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        print(f"[dry-run] Logging file: {path} as {log_as} as {key}")

    def log_artifact(self, name: str, type: str, files: list[str]) -> None:
        print(f"[dry-run] Logging artifact: {name} as {type} with files: {files}")

    def set_metric(self, name: str, value: object) -> None:
        print(f"[dry-run] Setting metric: {name} to {value}")

    def set_summary(self, summary: dict) -> None:
        print(f"[dry-run] Setting summary: {summary}")

    def set_tags(self, tags: list[str]) -> None:
        print(f"[dry-run] Setting tags: {tags}")

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        print(
            f"[dry-run] Logging table: {key} with columns: {columns} and data: {data}"
        )

    def finish(self, exit_code: int) -> None:
        print(f"[dry-run] Finishing with exit code: {exit_code}")
