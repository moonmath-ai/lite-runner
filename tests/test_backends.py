"""Tests for genai_runner.backends."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_runner import JsonBackend, Metric, Output, Param, UNSET
from genai_runner.backends import (
    _split_glob,
    extract_metrics,
    log_extra_outputs,
    log_files,
    log_table_params,
)


# ---------------------------------------------------------------------------
# _split_glob
# ---------------------------------------------------------------------------


def test_split_glob_star():
    base, pattern = _split_glob("/tmp/output/frames/*.jpg")
    assert base == Path("/tmp/output/frames")
    assert pattern == "*.jpg"


def test_split_glob_doublestar():
    base, pattern = _split_glob("debug/**/*.png")
    assert base == Path("debug")
    assert pattern == "**/*.png"


def test_split_glob_question():
    base, pattern = _split_glob("/tmp/file?.txt")
    assert base == Path("/tmp")
    assert pattern == "file?.txt"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def test_metric_float():
    metrics = [Metric("skip_pct", pattern=r"skipped=([\d.]+)%")]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    extract_metrics([json_backend], metrics, "some output\nskipped=32.8%\ndone")
    assert json_backend.metrics["skip_pct"] == 32.8


def test_metric_str():
    metrics = [Metric("status", pattern=r"final: (\w+)", type="str")]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    extract_metrics([json_backend], metrics, "final: completed")
    assert json_backend.metrics["status"] == "completed"


def test_metric_last_match_wins():
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    extract_metrics([json_backend], metrics, "x=1.0\nx=2.0\nx=3.0")
    assert json_backend.metrics["val"] == 3.0


def test_metric_no_match():
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    extract_metrics([json_backend], metrics, "no matches here")
    assert "val" not in json_backend.metrics


def test_extract_metrics_with_json_backend():
    """Metrics are extracted into JsonBackend."""
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    extract_metrics([json_backend], metrics, "x=3.14")
    assert json_backend.metrics["val"] == 3.14


# ---------------------------------------------------------------------------
# log_files
# ---------------------------------------------------------------------------


def test_log_files_skips_unset(tmp_path):
    """log_files skips params whose value is UNSET."""
    params = [Param("img", type="path-image")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    # Should not raise or try to upload
    log_files([json_backend], params, {"img": UNSET}, when="before")
    assert json_backend.files_logged == []


# ---------------------------------------------------------------------------
# Output glob + zip
# ---------------------------------------------------------------------------


def test_log_extra_outputs_glob(tmp_path):
    """Glob pattern expands and uploads each matched file."""
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "001.png").write_text("a")
    (tmp_path / "frames" / "002.png").write_text("b")
    (tmp_path / "frames" / "skip.txt").write_text("c")

    outputs = [Output("$output/frames/*.png", log_as="image")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    log_extra_outputs([json_backend], outputs, tmp_path)
    # Should log 2 png files, not the txt
    logged = [f for f in json_backend.files_logged if f.get("log_as") == "image"]
    assert len(logged) == 2


def test_log_extra_outputs_glob_zip(tmp_path):
    """Glob + log_as='zip' creates a zip archive."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "b.pt").write_text("tensor2")

    outputs = [Output("$output/debug/*.pt", log_as="zip")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    log_extra_outputs([json_backend], outputs, tmp_path)

    # Check zip was created
    zip_path = tmp_path / "debug.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["a.pt", "b.pt"]


def test_log_extra_outputs_dir_zip(tmp_path):
    """Directory with log_as='zip' zips entire directory."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "sub").mkdir()
    (tmp_path / "debug" / "sub" / "b.pt").write_text("tensor2")

    outputs = [Output("$output/debug", log_as="zip")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    log_extra_outputs([json_backend], outputs, tmp_path)

    zip_path = tmp_path / "debug.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        assert "a.pt" in names
        assert "sub/b.pt" in names


def test_log_extra_outputs_glob_no_match(tmp_path, capsys):
    """Glob with no matches prints a warning."""
    outputs = [Output("$output/nope/*.png", log_as="image")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    log_extra_outputs([json_backend], outputs, tmp_path)
    assert "matched no files" in capsys.readouterr().out


def test_log_extra_outputs_single_file(tmp_path):
    """Non-glob single file still works (regression)."""
    (tmp_path / "meta.json").write_text("{}")

    outputs = [Output("$output/meta.json", log_as="artifact")]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    log_extra_outputs([json_backend], outputs, tmp_path)
    logged = [f for f in json_backend.files_logged if f.get("log_as") == "artifact"]
    assert len(logged) == 1


def test_log_extra_outputs_duplicate_zip_raises(tmp_path):
    """Two zip outputs with same implicit label should raise."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("x")
    (tmp_path / "debug" / "b.png").write_text("y")

    outputs = [
        Output("$output/debug/*.pt", log_as="zip"),
        Output("$output/debug/*.png", log_as="zip"),
    ]
    json_backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": str(tmp_path)},
    )
    with pytest.raises(ValueError, match="Duplicate zip label 'debug'"):
        log_extra_outputs([json_backend], outputs, tmp_path)


# ---------------------------------------------------------------------------
# Table param logging
# ---------------------------------------------------------------------------


def test_table_param_logged_to_json_backend():
    """Params with table=True are logged via log_table to JsonBackend."""
    params = [Param("prompt", table=True), Param("seed", type="int", default=42)]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    log_table_params([json_backend], params, {"prompt": "a cat", "seed": 42})
    table = json_backend.tables["params"]
    assert table["columns"] == ["name", "value"]
    assert ["prompt", "a cat"] in table["data"]
    # seed has table=False, should NOT appear
    assert all(row[0] != "seed" for row in table["data"])


def test_table_param_skips_unset():
    """UNSET and None table params are excluded from the table."""
    params = [Param("prompt", table=True), Param("neg", table=True)]
    json_backend = JsonBackend(
        project="test", name=None, group=None, tags=[], config={}
    )
    log_table_params([json_backend], params, {"prompt": "a cat", "neg": None})
    table = json_backend.tables["params"]
    assert len(table["data"]) == 1
    assert table["data"][0] == ["prompt", "a cat"]


def test_table_param_no_table_params_skips():
    """No log_table call when no params have table=True."""
    params = [Param("seed", type="int", default=42)]
    backend = MagicMock()
    log_table_params([backend], params, {"seed": 42})
    backend.log_table.assert_not_called()
