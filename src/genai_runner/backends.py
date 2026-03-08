"""Log backend protocol and implementations for genai_runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class LogBackend(Protocol):
    """Protocol for logging backends."""

    @property
    def run_name(self) -> str: ...

    @property
    def run_url(self) -> str: ...

    def init(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict,
    ) -> None: ...

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

    def __init__(self) -> None:
        self._run: Any = None
        self._wandb: Any = None

    @property
    def run_name(self) -> str:
        return self._run.name or self._run.id or "run"

    @property
    def run_url(self) -> str:
        return self._run.url

    def init(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=name,
            group=group,
            tags=tags,
            save_code=True,
            config=config,
        )

    def update_config(self, updates: dict) -> None:
        self._run.config.update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        if log_as == "video":
            fmt = path.suffix.lstrip(".")
            self._run.log({key: self._wandb.Video(str(path), format=fmt)})
        elif log_as == "image":
            self._run.log({key: self._wandb.Image(str(path))})
        elif log_as == "text":
            text = path.read_text(errors="replace")
            self._run.log({key: self._wandb.Html(f"<pre>{text}</pre>")})
        elif log_as == "artifact":
            artifact = self._wandb.Artifact(f"{key}-{self._run.id}", type=log_as)
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)

    def log_artifact(self, name: str, type: str, files: list[str]) -> None:
        artifact = self._wandb.Artifact(f"{name}-{self._run.id}", type=type)
        for f in files:
            artifact.add_file(f)
        self._run.log_artifact(artifact)

    def set_metric(self, name: str, value: object) -> None:
        self._run.summary[name] = value

    def set_summary(self, summary: dict) -> None:
        self._run.summary.update(summary)

    def set_tags(self, tags: list[str]) -> None:
        self._run.tags = tags

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        table = self._wandb.Table(columns=columns, data=data)
        self._run.log({key: table})

    def finish(self, exit_code: int) -> None:
        self._run.finish(exit_code=exit_code)


class JsonBackend:
    """Log backend that accumulates run info and writes run_info.json on finish."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self.run_info: dict[str, Any] = {
            "config": {},
            "tags": [],
            "group": None,
            "metrics": {},
            "summary": {},
            "files_logged": [],
        }

    @property
    def run_name(self) -> str:
        return "local"

    @property
    def run_url(self) -> str:
        return "(local)"

    def init(
        self,
        project: str,
        name: str | None,
        group: str | None,
        tags: list[str],
        config: dict,
    ) -> None:
        self.run_info["config"] = dict(config)
        self.run_info["tags"] = list(tags)
        self.run_info["group"] = group

    def update_config(self, updates: dict) -> None:
        self.run_info["config"].update(updates)

    def log_file(self, path: Path, log_as: str, key: str) -> None:
        self.run_info["files_logged"].append(
            {"path": str(path), "log_as": log_as, "key": key}
        )

    def log_artifact(self, name: str, type: str, files: list[str]) -> None:
        self.run_info["files_logged"].append({"type": type, "files": files})

    def set_metric(self, name: str, value: object) -> None:
        self.run_info["metrics"][name] = value

    def set_summary(self, summary: dict) -> None:
        self.run_info["summary"] = summary

    def set_tags(self, tags: list[str]) -> None:
        self.run_info["tags"] = tags

    def log_table(self, key: str, columns: list[str], data: list[list[object]]) -> None:
        self.run_info.setdefault("tables", {})[key] = {
            "columns": columns,
            "data": data,
        }

    def finish(self, exit_code: int) -> None:
        (self._output_dir / "run_info.json").write_text(
            json.dumps(self.run_info, indent=2, default=str)
        )
