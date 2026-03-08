"""Data classes and type system for genai_runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

import questionary

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


# Sentinel for params the user explicitly skipped (typed '-' at the prompt).
_SKIP_INPUT = "-"


class _Unset:
    """Param value skipped by user during interactive prompting."""

    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()


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
        prompt: If False, skip interactive prompting and use the
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
    prompt: bool = True

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
        if not self.prompt and self.default is None:
            msg = f"Param('{self.name}', prompt=False) requires a default"
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
        if self.value is None:
            return False
        values = self.value if isinstance(self.value, list) else [self.value]
        return any("$output" in str(v) for v in values)

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

    def argparse_kwargs(self) -> dict:
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

    def cast_nargs(self, values: list) -> list:
        """Cast each element in *values* according to this param's type list."""
        types = self.type_list
        if len(values) != len(types):
            msg = f"Expected {len(types)} values, got {len(values)}: {values}"
            raise ValueError(msg)
        return [
            _PARAM_TYPE_MAP.get(t, str)(v) for v, t in zip(values, types, strict=True)
        ]

    def ask(self, default: object = None) -> Any:
        """Interactively prompt the user for this param's value."""
        if self.type == "bool":
            return self._prompt_bool(default)
        if self.nargs is not None:
            return self._prompt_nargs(default)
        return self._prompt_single(default)

    def _prompt_bool(self, default: object = None) -> bool:
        label = self.help or self.name
        answer = questionary.confirm(f"{label}:", default=bool(default)).ask()
        if answer is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)
        return answer

    def _prompt_single(self, default: object = None) -> int | float | str | _Unset:
        label = self.help or self.name
        default_str = str(default) if default is not None else ""
        if self.choices:
            choices = [_SKIP_INPUT, *self.choices]
            default_choice = str(default) if default is not None else None
            answer = questionary.select(
                f"{label}:", choices=choices, default=default_choice
            ).ask()
        elif isinstance(self.type, str) and self.type.startswith("path"):
            answer = questionary.path(f"{label}:", default=default_str).ask()
        else:
            answer = questionary.text(f"{label}:", default=default_str).ask()

        if answer is None:
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)

        if answer == _SKIP_INPUT:
            return UNSET

        caster = _PARAM_TYPE_MAP.get(self.type, str)
        return caster(answer)

    def _prompt_nargs(self, default: object = None) -> list | _Unset:
        assert self.nargs is not None
        labels = self.labels or [f"{self.name}[{i}]" for i in range(self.nargs)]
        element_types = self.type_list
        defaults = default if isinstance(default, list) else [None] * self.nargs
        parts = []
        for label, etype, d in zip(labels, element_types, defaults, strict=True):
            default_str = str(d) if d is not None else ""
            if etype.startswith("path"):
                answer = questionary.path(
                    f"{self.name} {label}:", default=default_str
                ).ask()
            else:
                answer = questionary.text(
                    f"{self.name} {label}:", default=default_str
                ).ask()
            if answer is None:
                print("Cancelled.", file=sys.stderr)
                sys.exit(1)
            if answer == _SKIP_INPUT:
                return UNSET
            parts.append(answer)
        return self.cast_nargs(parts)


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
