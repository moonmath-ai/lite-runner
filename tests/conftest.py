"""Shared fixtures and helpers for lite_runner tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lite_runner import Param, Runner

_FAKE_GIT_INFO = {
    "repo": "test-repo",
    "commit": "abc",
    "branch": "main",
    "dirty": False,
}


def _mock_wb_run(**overrides: Any) -> MagicMock:  # noqa: ANN401
    defaults = {
        "summary": {},
        "name": "test-run-42",
        "id": "abc123",
        "url": "https://wandb.test/run",
        "tags": [],
        "config": {},
    }
    run = MagicMock()
    for attr, value in {**defaults, **overrides}.items():
        setattr(run, attr, value)
    return run


@pytest.fixture(autouse=True)
def _clean_argv() -> Generator[None]:
    """Ensure parse_cli() sees clean argv when auto-called by run()."""
    with patch("sys.argv", ["prog"]):
        yield


def _make_runner(
    params: list[Param] | None = None, command: str = "echo hello", **kwargs: Any  # noqa: ANN401
) -> Runner:
    return Runner(command=command, params=params or [], **kwargs)
