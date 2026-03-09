"""Tests for genai_runner.backends."""

import zipfile
from pathlib import Path

import pytest

from genai_runner import UNSET, Metric, Output, Param
from genai_runner.backends import (
    _split_glob,
    collect_metrics,
    collect_param_files,
    collect_run_logs,
    prepare_extra_outputs,
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
# Metric extraction (collect_metrics)
# ---------------------------------------------------------------------------


def test_metric_float():
    metrics = [Metric("skip_pct", pattern=r"skipped=([\d.]+)%")]
    items = collect_metrics(metrics, "some output\nskipped=32.8%\ndone")
    assert items == [("skip_pct", 32.8)]


def test_metric_str():
    metrics = [Metric("status", pattern=r"final: (\w+)", type="str")]
    items = collect_metrics(metrics, "final: completed")
    assert items == [("status", "completed")]


def test_metric_last_match_wins():
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    items = collect_metrics(metrics, "x=1.0\nx=2.0\nx=3.0")
    assert items == [("val", 3.0)]


def test_metric_no_match():
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    items = collect_metrics(metrics, "no matches here")
    assert items == []


def test_collect_metrics_multiple():
    """Multiple metrics are collected independently."""
    metrics = [
        Metric("val", pattern=r"x=([\d.]+)"),
        Metric("status", pattern=r"final: (\w+)", type="str"),
    ]
    items = collect_metrics(metrics, "x=3.14\nfinal: done")
    assert items == [("val", 3.14), ("status", "done")]


# ---------------------------------------------------------------------------
# collect_param_files
# ---------------------------------------------------------------------------


def test_collect_param_files_skips_unset():
    """collect_param_files skips params whose value is UNSET."""
    params = [Param("img", type="path-image")]
    items = collect_param_files(params, {"img": UNSET}, when="before")
    assert items == []


# ---------------------------------------------------------------------------
# Output glob + zip (prepare_extra_outputs)
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_glob(tmp_path):
    """Glob pattern expands and collects each matched file."""
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "001.png").write_text("a")
    (tmp_path / "frames" / "002.png").write_text("b")
    (tmp_path / "frames" / "skip.txt").write_text("c")

    outputs = [Output("$output/frames/*.png", log_as="image")]
    items = prepare_extra_outputs(outputs, tmp_path)
    image_items = [i for i in items if i.log_as == "image"]
    assert len(image_items) == 2


def test_prepare_extra_outputs_glob_zip(tmp_path):
    """Glob + log_as='zip' creates a zip archive."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "b.pt").write_text("tensor2")

    outputs = [Output("$output/debug/*.pt", log_as="zip")]
    items = prepare_extra_outputs(outputs, tmp_path)

    # Check zip was created
    zip_path = tmp_path / "debug.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["a.pt", "b.pt"]

    # Item points to the zip
    assert len(items) == 1
    assert items[0].path == zip_path
    assert items[0].log_as == "artifact"


def test_prepare_extra_outputs_dir_zip(tmp_path):
    """Directory with log_as='zip' zips entire directory."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "sub").mkdir()
    (tmp_path / "debug" / "sub" / "b.pt").write_text("tensor2")

    outputs = [Output("$output/debug", log_as="zip")]
    items = prepare_extra_outputs(outputs, tmp_path)

    zip_path = tmp_path / "debug.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        assert "a.pt" in names
        assert "sub/b.pt" in names

    assert len(items) == 1
    assert items[0].path == zip_path


def test_prepare_extra_outputs_glob_no_match(tmp_path, capsys):
    """Glob with no matches prints a warning."""
    outputs = [Output("$output/nope/*.png", log_as="image")]
    items = prepare_extra_outputs(outputs, tmp_path)
    assert items == []
    assert "matched no files" in capsys.readouterr().out


def test_prepare_extra_outputs_single_file(tmp_path):
    """Non-glob single file still works (regression)."""
    (tmp_path / "meta.json").write_text("{}")

    outputs = [Output("$output/meta.json", log_as="artifact")]
    items = prepare_extra_outputs(outputs, tmp_path)
    artifact_items = [i for i in items if i.log_as == "artifact"]
    assert len(artifact_items) == 1


def test_prepare_extra_outputs_duplicate_zip_raises(tmp_path):
    """Two zip outputs with same implicit label should raise."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("x")
    (tmp_path / "debug" / "b.png").write_text("y")

    outputs = [
        Output("$output/debug/*.pt", log_as="zip"),
        Output("$output/debug/*.png", log_as="zip"),
    ]
    with pytest.raises(ValueError, match="Duplicate zip label 'debug'"):
        prepare_extra_outputs(outputs, tmp_path)


# ---------------------------------------------------------------------------
# collect_run_logs
# ---------------------------------------------------------------------------


def test_collect_run_logs(tmp_path):
    """Collects existing log files from output dir."""
    (tmp_path / "run.log").write_text("combined")
    (tmp_path / "stdout.log").write_text("out")
    # stderr.log missing — should be skipped

    items = collect_run_logs(tmp_path)
    assert len(items) == 2
    keys = {i.key for i in items}
    assert keys == {"run.log", "stdout.log"}
    assert all(i.log_as == "text" for i in items)


def test_collect_run_logs_empty(tmp_path):
    """No log files → empty list."""
    items = collect_run_logs(tmp_path)
    assert items == []
