"""Tests for lite_runner.backends."""

import datetime
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lite_runner import UNSET, Metric, Output, Param
from lite_runner.backends import (
    DryRunBackend,
    JsonBackend,
    _split_glob,
    _video_format,
    collect_metrics,
    collect_param_files,
    collect_run_logs,
    create_repo_archive,
    create_repo_diff,
    prepare_code_archive,
    prepare_code_diff,
    prepare_extra_outputs,
)

# ---------------------------------------------------------------------------
# _split_glob
# ---------------------------------------------------------------------------


def test_split_glob_star() -> None:
    base, pattern = _split_glob("/fake/output/frames/*.jpg")
    assert base == Path("/fake/output/frames")
    assert pattern == "*.jpg"


def test_split_glob_doublestar() -> None:
    base, pattern = _split_glob("debug/**/*.png")
    assert base == Path("debug")
    assert pattern == "**/*.png"


def test_split_glob_question() -> None:
    base, pattern = _split_glob("/fake/file?.txt")
    assert base == Path("/fake")
    assert pattern == "file?.txt"


# ---------------------------------------------------------------------------
# Metric extraction (collect_metrics)
# ---------------------------------------------------------------------------


def test_metric_float() -> None:
    metrics = [Metric("skip_pct", pattern=r"skipped=([\d.]+)%")]
    items = collect_metrics(metrics, "some output\nskipped=32.8%\ndone")
    assert items == [("skip_pct", 32.8)]


def test_metric_str() -> None:
    metrics = [Metric("status", pattern=r"final: (\w+)", type="str")]
    items = collect_metrics(metrics, "final: completed")
    assert items == [("status", "completed")]


def test_metric_last_match_wins() -> None:
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    items = collect_metrics(metrics, "x=1.0\nx=2.0\nx=3.0")
    assert items == [("val", 3.0)]


def test_metric_no_match() -> None:
    metrics = [Metric("val", pattern=r"x=([\d.]+)")]
    items = collect_metrics(metrics, "no matches here")
    assert items == []


def test_collect_metrics_multiple() -> None:
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


def test_collect_param_files_skips_unset() -> None:
    """collect_param_files skips params whose value is UNSET."""
    params = [Param("img", type="path-image")]
    items = collect_param_files(params, {"img": UNSET}, when="before")
    assert items == []


# ---------------------------------------------------------------------------
# Output glob + zip (prepare_extra_outputs)
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_glob(tmp_path: Path) -> None:
    """Glob pattern expands and collects each matched file."""
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "001.png").write_text("a")
    (tmp_path / "frames" / "002.png").write_text("b")
    (tmp_path / "frames" / "skip.txt").write_text("c")

    outputs = [Output("$output/frames/*.png", log_as="image")]
    items = prepare_extra_outputs(outputs, tmp_path)
    image_items = [i for i in items if i.log_as == "image"]
    assert len(image_items) == 2


def test_prepare_extra_outputs_glob_zip(tmp_path: Path) -> None:
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


def test_prepare_extra_outputs_dir_zip(tmp_path: Path) -> None:
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


def test_prepare_extra_outputs_glob_no_match(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Glob with no matches logs a warning."""
    outputs = [Output("$output/nope/*.png", log_as="image")]
    with caplog.at_level("WARNING", logger="lite_runner"):
        items = prepare_extra_outputs(outputs, tmp_path)
    assert items == []
    assert "matched no files" in caplog.text


def test_prepare_extra_outputs_single_file(tmp_path: Path) -> None:
    """Non-glob single file still works (regression)."""
    (tmp_path / "meta.json").write_text("{}")

    outputs = [Output("$output/meta.json", log_as="artifact")]
    items = prepare_extra_outputs(outputs, tmp_path)
    artifact_items = [i for i in items if i.log_as == "artifact"]
    assert len(artifact_items) == 1


def test_prepare_extra_outputs_duplicate_zip_raises(tmp_path: Path) -> None:
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


def test_collect_run_logs(tmp_path: Path) -> None:
    """Collects existing log files from output dir."""
    (tmp_path / "run.log").write_text("combined")
    (tmp_path / "stdout.log").write_text("out")
    # stderr.log missing — should be skipped

    items = collect_run_logs(tmp_path)
    assert len(items) == 2
    keys = {i.key for i in items}
    assert keys == {"run.log", "stdout.log"}
    assert all(i.log_as == "text" for i in items)


def test_collect_run_logs_empty(tmp_path: Path) -> None:
    """No log files → empty list."""
    items = collect_run_logs(tmp_path)
    assert items == []


def test_collect_run_logs_dry_run(tmp_path: Path) -> None:
    """Dry run returns all three log files even when they don't exist."""
    items = collect_run_logs(tmp_path, dry_run=True)
    assert len(items) == 3
    keys = {i.key for i in items}
    assert keys == {"run.log", "stdout.log", "stderr.log"}


# ---------------------------------------------------------------------------
# collect_param_files — file not found
# ---------------------------------------------------------------------------


def test_collect_param_files_file_not_found() -> None:
    """Raises FileNotFoundError when the file doesn't exist."""
    params = [Param("img", type="path-image")]
    with pytest.raises(FileNotFoundError, match="File not found"):
        collect_param_files(params, {"img": "/fake/photo.jpg"}, when="before")


def test_collect_param_files_dry_run_skips_existence() -> None:
    """Dry run skips file existence check."""
    params = [Param("img", type="path-image")]
    items = collect_param_files(
        params, {"img": "/fake/photo.jpg"}, when="before", dry_run=True
    )
    assert len(items) == 1
    assert items[0].log_as == "image"


def test_collect_param_files_skips_none() -> None:
    """Params with None value are skipped."""
    params = [Param("img", type="path-image")]
    items = collect_param_files(params, {"img": None}, when="before")
    assert items == []


def test_collect_param_files_skips_wrong_when() -> None:
    """Params with different log_when are skipped."""
    params = [Param("out", type="path-video", value="$output/vid.mp4")]
    items = collect_param_files(params, {"out": "/fake/vid.mp4"}, when="before")
    assert items == []


def test_collect_param_files_collects_existing(tmp_path: Path) -> None:
    """Collects existing file."""
    img = tmp_path / "photo.jpg"
    img.write_text("fake image")
    params = [Param("img", type="path-image")]
    items = collect_param_files(params, {"img": str(img)}, when="before")
    assert len(items) == 1
    assert items[0].log_as == "image"
    assert items[0].key == "img"


def test_collect_param_files_multi_value(tmp_path: Path) -> None:
    """Multi-value param collects only path-typed elements."""
    img = tmp_path / "photo.jpg"
    img.write_text("fake image")
    params = [
        Param(
            "x",
            type=["path-image", "float", "float"],
            labels=["img", "start", "strength"],
        )
    ]
    items = collect_param_files(params, {"x": [str(img), 0.0, 0.8]}, when="before")
    assert len(items) == 1
    assert items[0].log_as == "image"


# ---------------------------------------------------------------------------
# prepare_extra_outputs — single file not found
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_single_file_not_found(tmp_path: Path) -> None:
    """Single file that doesn't exist raises FileNotFoundError."""
    outputs = [Output("$output/missing.json", log_as="artifact")]
    with pytest.raises(FileNotFoundError, match="Output file not found"):
        prepare_extra_outputs(outputs, tmp_path)


# ---------------------------------------------------------------------------
# prepare_extra_outputs — copy_to
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_copy_to(tmp_path: Path) -> None:
    """Output with copy_to copies the file and logs the destination."""
    src = tmp_path / "src_data.json"
    src.write_text('{"key": "value"}')

    outputs = [
        Output(
            str(src),
            log_as="artifact",
            copy_to="$output/copied_data.json",
        )
    ]
    items = prepare_extra_outputs(outputs, tmp_path)
    assert len(items) == 1
    dst = tmp_path / "copied_data.json"
    assert dst.exists()
    assert dst.read_text() == '{"key": "value"}'
    assert items[0].path == dst


# ---------------------------------------------------------------------------
# prepare_extra_outputs — directory with non-zip log_as (warning)
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_dir_non_zip_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Directory with log_as != 'zip' logs a warning and uploads files."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.png").write_text("img1")
    (tmp_path / "debug" / "b.png").write_text("img2")

    outputs = [Output("$output/debug", log_as="image")]
    with caplog.at_level("WARNING", logger="lite_runner"):
        items = prepare_extra_outputs(outputs, tmp_path)
    assert "directory" in caplog.text
    assert len(items) == 2
    assert all(i.log_as == "image" for i in items)


# ---------------------------------------------------------------------------
# prepare_extra_outputs — dry_run paths
# ---------------------------------------------------------------------------


def test_prepare_extra_outputs_dry_run_glob(tmp_path: Path) -> None:
    """Dry run with glob returns placeholder LogFile."""
    outputs = [Output("$output/frames/*.png", log_as="image")]
    items = prepare_extra_outputs(outputs, tmp_path, dry_run=True)
    assert len(items) == 1
    assert items[0].log_as == "image"


def test_prepare_extra_outputs_dry_run_glob_zip(tmp_path: Path) -> None:
    """Dry run with glob + zip returns zip LogFile."""
    outputs = [Output("$output/frames/*.pt", log_as="zip", name="frames")]
    items = prepare_extra_outputs(outputs, tmp_path, dry_run=True)
    assert len(items) == 1
    assert items[0].log_as == "artifact"
    assert items[0].path == tmp_path / "frames.zip"


def test_prepare_extra_outputs_dry_run_single_file(tmp_path: Path) -> None:
    """Dry run with single file returns LogFile without existence check."""
    outputs = [Output("$output/meta.json", log_as="artifact")]
    items = prepare_extra_outputs(outputs, tmp_path, dry_run=True)
    assert len(items) == 1
    assert items[0].log_as == "artifact"


def test_prepare_extra_outputs_dry_run_directory(tmp_path: Path) -> None:
    """Dry run with trailing slash returns placeholder LogFile."""
    outputs = [Output("$output/debug/", log_as="image")]
    items = prepare_extra_outputs(outputs, tmp_path, dry_run=True)
    assert len(items) == 1
    assert items[0].key == "debug"


# ---------------------------------------------------------------------------
# _video_format
# ---------------------------------------------------------------------------


def test_video_format_valid() -> None:
    assert _video_format(Path("video.mp4")) == "mp4"
    assert _video_format(Path("animation.gif")) == "gif"
    assert _video_format(Path("clip.webm")) == "webm"
    assert _video_format(Path("sound.ogg")) == "ogg"


def test_video_format_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported video format"):
        _video_format(Path("image.png"))


# ---------------------------------------------------------------------------
# _split_glob — no glob chars fallback
# ---------------------------------------------------------------------------


def test_split_glob_no_glob_chars() -> None:
    """Path with no glob chars returns (parent, name)."""
    base, pattern = _split_glob("/fake/output/file.txt")
    assert base == Path("/fake/output")
    assert pattern == "file.txt"


# ---------------------------------------------------------------------------
# JsonBackend
# ---------------------------------------------------------------------------


def test_json_backend_set_summary_twice_raises() -> None:
    backend = JsonBackend(
        project="test",
        name="run-1",
        group=None,
        tags=[],
        config={"meta/output_dir": "/fake/output"},
    )
    backend.set_summary({"status": "success"})
    with pytest.raises(RuntimeError, match="set_summary called twice"):
        backend.set_summary({"status": "failed"})


def test_json_backend_run_name() -> None:
    backend = JsonBackend(
        project="test",
        name="my-run",
        group=None,
        tags=[],
        config={"meta/output_dir": "/fake/output"},
    )
    assert backend.run_name == "my-run"


def test_json_backend_run_name_default() -> None:
    backend = JsonBackend(
        project="test",
        name=None,
        group=None,
        tags=[],
        config={"meta/output_dir": "/fake/output"},
    )
    assert backend.run_name == "(local)"


def test_json_backend_full_lifecycle(tmp_path: Path) -> None:
    """Test JsonBackend write lifecycle: config, metrics, files, summary, finish."""
    backend = JsonBackend(
        project="test",
        name="run-1",
        group="sweep",
        tags=["v1"],
        config={"meta/output_dir": str(tmp_path)},
    )
    backend.update_config({"param/seed": 42})
    backend.set_metric("loss", 0.5)
    backend.log_file(Path("/fake/video.mp4"), "video", "output")
    backend.set_summary({"status": "success", "exit_code": 0})
    backend.set_tags(["v1", "failed"])
    backend.finish(exit_code=0)

    run_info = json.loads((tmp_path / "run_info.json").read_text())
    assert run_info["config"]["param/seed"] == 42
    assert run_info["metrics"]["loss"] == 0.5
    assert len(run_info["files_logged"]) == 1
    assert run_info["summary"]["status"] == "success"
    assert run_info["metadata"]["tags"] == ["v1", "failed"]
    assert run_info["exit_code"] == 0


# ---------------------------------------------------------------------------
# DryRunBackend
# ---------------------------------------------------------------------------


def test_dry_run_backend_operations(caplog: pytest.LogCaptureFixture) -> None:
    """DryRunBackend logs all operations without error."""
    with caplog.at_level("INFO", logger="lite_runner"):
        backend = DryRunBackend(
            project="test",
            name="dry-run-1",
            group=None,
            tags=["v1"],
            config={"param/seed": 42},
        )
        assert backend.run_name == "dry-run-1"
        backend.update_config({"param/lr": 0.001})
        backend.log_file(Path("/fake/video.mp4"), "video", "output")
        backend.set_metric("loss", 0.5)
        backend.set_summary({"status": "success"})
        backend.set_tags(["v1", "failed"])
        backend.finish(exit_code=0)


def test_dry_run_backend_default_name() -> None:
    backend = DryRunBackend(project="test", name=None, group=None, tags=[], config={})
    assert backend.run_name == "dry_run"


# ---------------------------------------------------------------------------
# Metric int type
# ---------------------------------------------------------------------------


def test_metric_int() -> None:
    """Metric with type='int' casts to int."""
    metrics = [Metric("count", pattern=r"count=(\d+)", type="int")]
    items = collect_metrics(metrics, "count=42")
    assert items == [("count", 42)]


# ---------------------------------------------------------------------------
# Metric timedelta type
# ---------------------------------------------------------------------------


def test_metric_timedelta_hms() -> None:
    """Metric with type='timedelta' parses HH:MM:SS.ddd to seconds."""
    metrics = [Metric("elapsed", pattern=r"time=([\d:.]+)", type="timedelta")]
    items = collect_metrics(metrics, "time=1:02:03.5")
    assert items == [("elapsed", 3723.5)]


def test_metric_timedelta_ms() -> None:
    """Metric with type='timedelta' parses MM:SS to seconds."""
    metrics = [Metric("elapsed", pattern=r"time=([\d:.]+)", type="timedelta")]
    items = collect_metrics(metrics, "time=05:30")
    assert items == [("elapsed", 330.0)]


def test_metric_timedelta_seconds_only() -> None:
    """Metric with type='timedelta' parses bare SS to seconds."""
    metrics = [Metric("elapsed", pattern=r"time=([\d:.]+)", type="timedelta")]
    items = collect_metrics(metrics, "time=42.75")
    assert items == [("elapsed", 42.75)]


# ---------------------------------------------------------------------------
# Metric time type (timestamp → seconds since run start)
# ---------------------------------------------------------------------------


def _local_t0(
    h: int = 12,
    m: int = 0,
    s: int = 0,
) -> datetime.datetime:
    """Build a run_started_at in local tz for deterministic tests."""
    local_tz = datetime.datetime.now().astimezone().tzinfo
    return datetime.datetime(2026, 1, 1, h, m, s, tzinfo=local_tz)


def test_metric_time_iso() -> None:
    """Full ISO datetime with explicit tz."""
    t0 = datetime.datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=datetime.timezone.utc,
    )
    metrics = [Metric("t", pattern=r"at=(\S+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=2026-01-01T12:01:30+00:00",
        run_started_at=t0,
    )
    assert items == [("t", 90.0)]


def test_metric_time_naive_iso() -> None:
    """Naive ISO datetime is interpreted as local time."""
    t0 = _local_t0()
    metrics = [Metric("t", pattern=r"at=(\S+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=2026-01-01T12:00:45",
        run_started_at=t0,
    )
    assert items == [("t", 45.0)]


def test_metric_time_hms() -> None:
    """Wall-clock HH:MM:SS."""
    t0 = _local_t0()
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=12:01:30",
        run_started_at=t0,
    )
    assert items == [("t", 90.0)]


def test_metric_time_ms() -> None:
    """Wall-clock MM:SS (no hours)."""
    t0 = _local_t0(h=0)
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=05:30",
        run_started_at=t0,
    )
    assert items == [("t", 330.0)]


def test_metric_time_s() -> None:
    """Wall-clock SS (seconds only)."""
    t0 = _local_t0(h=0)
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=42",
        run_started_at=t0,
    )
    assert items == [("t", 42.0)]


def test_metric_time_hms_frac() -> None:
    """Wall-clock HH:MM:SS.fff."""
    t0 = _local_t0()
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=12:00:01.500",
        run_started_at=t0,
    )
    assert items == [("t", 1.5)]


def test_metric_time_ms_frac() -> None:
    """Wall-clock MM:SS.fff (no hours)."""
    t0 = _local_t0(h=0)
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=01:02.500",
        run_started_at=t0,
    )
    assert items == [("t", 62.5)]


def test_metric_time_s_frac() -> None:
    """Wall-clock SS.fff (seconds with fractional)."""
    t0 = _local_t0(h=0)
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        "at=42.750",
        run_started_at=t0,
    )
    assert items == [("t", 42.75)]


def test_metric_time_hms_with_tz() -> None:
    """Wall-clock HH:MM:SS.fff with timezone abbreviation."""
    t0 = _local_t0()
    local_tz_name = t0.strftime("%Z")
    metrics = [Metric("t", pattern=r"at=(.+)", type="time")]
    items = collect_metrics(
        metrics,
        f"at=12:00:30.000 {local_tz_name}",
        run_started_at=t0,
    )
    assert items == [("t", 30.0)]


def test_metric_time_without_run_started_at_falls_back() -> None:
    """type='time' without run_started_at stores raw string."""
    metrics = [Metric("t", pattern=r"at=(\S+)", type="time")]
    items = collect_metrics(metrics, "at=2026-01-01T12:00:00+00:00")
    assert items == [("t", "2026-01-01T12:00:00+00:00")]


# ---------------------------------------------------------------------------
# create_repo_archive / create_repo_diff
# ---------------------------------------------------------------------------


def test_create_repo_archive_no_repo() -> None:
    """Returns None when not in a git repo."""
    with patch("lite_runner.backends._open_repo", return_value=None):
        result = create_repo_archive(Path("/fake/output"))
    assert result is None


def test_create_repo_archive_dry_run() -> None:
    """Dry run returns path without creating files."""
    fake_dir = Path("/fake/output")
    mock_repo = MagicMock()
    with patch("lite_runner.backends._open_repo", return_value=mock_repo):
        result = create_repo_archive(fake_dir, dry_run=True)
    assert result == fake_dir / "code" / "source.tar.gz"
    mock_repo.archive.assert_not_called()


def test_create_repo_diff_no_repo() -> None:
    """Returns None when not in a git repo."""
    with patch("lite_runner.backends._open_repo", return_value=None):
        result = create_repo_diff(Path("/fake/output"))
    assert result is None


def test_create_repo_diff_clean_repo() -> None:
    """Returns None when repo is clean (no diff)."""
    mock_repo = MagicMock()
    mock_repo.git.diff.return_value = ""
    with patch("lite_runner.backends._open_repo", return_value=mock_repo):
        result = create_repo_diff(Path("/fake/output"))
    assert result is None


def test_create_repo_diff_dry_run() -> None:
    """Dry run returns path when repo is dirty."""
    fake_dir = Path("/fake/output")
    mock_repo = MagicMock()
    mock_repo.git.diff.return_value = "some diff content"
    with patch("lite_runner.backends._open_repo", return_value=mock_repo):
        result = create_repo_diff(fake_dir, dry_run=True)
    assert result == fake_dir / "code" / "dirty.patch"


def test_create_repo_diff_writes_patch(tmp_path: Path) -> None:
    """Creates patch file when repo is dirty."""
    mock_repo = MagicMock()
    mock_repo.git.diff.return_value = "diff --git a/file.py b/file.py\n+new line"
    with patch("lite_runner.backends._open_repo", return_value=mock_repo):
        result = create_repo_diff(tmp_path)
    assert result == tmp_path / "code" / "dirty.patch"
    assert result.exists()
    assert "new line" in result.read_text()


# ---------------------------------------------------------------------------
# prepare_code_archive / prepare_code_diff
# ---------------------------------------------------------------------------


def test_prepare_code_archive_no_repo() -> None:
    with patch("lite_runner.backends._open_repo", return_value=None):
        items = prepare_code_archive(Path("/fake/output"))
    assert items == []


def test_prepare_code_diff_returns_logfile(tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.git.diff.return_value = "some diff"
    with patch("lite_runner.backends._open_repo", return_value=mock_repo):
        items = prepare_code_diff(tmp_path)
    assert len(items) == 1
    assert items[0].log_as == "artifact"
    assert items[0].key == "code-diff"


def test_prepare_code_diff_clean_repo() -> None:
    with patch("lite_runner.backends._open_repo", return_value=None):
        items = prepare_code_diff(Path("/fake/output"))
    assert items == []


def test_prepare_extra_outputs_with_name(tmp_path: Path) -> None:
    """Output.name overrides the default key."""
    (tmp_path / "data.json").write_text("{}")
    outputs = [Output("$output/data.json", log_as="artifact", name="my-data")]
    items = prepare_extra_outputs(outputs, tmp_path)
    assert items[0].key == "my-data"
