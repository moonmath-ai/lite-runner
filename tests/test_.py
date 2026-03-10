import importlib.metadata

import lite_runner


def test_version() -> None:
    assert importlib.metadata.version("lite_runner") == lite_runner.__version__
