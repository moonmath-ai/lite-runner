"""Tests for genai_runner."""

import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_runner import (
    Metric,
    Output,
    Param,
    Runner,
    _collect_git_info,
    _log_as_from_type,
    _split_glob,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_GIT_INFO = {
    "repo": "test-repo",
    "commit": "abc",
    "branch": "main",
    "dirty": False,
}


def _mock_wb_run(**overrides) -> MagicMock:
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
def _clean_argv():
    """Ensure Runner.__post_init__ sees a clean sys.argv in every test."""
    with patch("sys.argv", ["prog"]):
        yield


def _make_runner(params=None, **kwargs):
    return Runner(
        command=kwargs.pop("command", "echo hello"), params=params or [], **kwargs
    )


# ---------------------------------------------------------------------------
# _log_as_from_type
# ---------------------------------------------------------------------------


def test_log_as_from_type_path_image():
    assert _log_as_from_type("path-image") == "image"


def test_log_as_from_type_path_video():
    assert _log_as_from_type("path-video") == "video"


def test_log_as_from_type_path_artifact():
    assert _log_as_from_type("path-artifact") == "artifact"


def test_log_as_from_type_path_text():
    assert _log_as_from_type("path-text") == "text"


def test_log_as_from_type_plain_path():
    assert _log_as_from_type("path") is None


def test_log_as_from_type_str():
    assert _log_as_from_type("str") is None


def test_log_as_from_type_int():
    assert _log_as_from_type("int") is None


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
# Param
# ---------------------------------------------------------------------------


def test_param_dest_normalizes_hyphens():
    assert Param("debug-output").dest == "debug_output"


def test_param_dest_preserves_underscores():
    assert Param("debug_output").dest == "debug_output"


def test_param_flag_from_hyphens():
    assert Param("debug-output").flag == "--debug-output"


def test_param_flag_from_underscores():
    assert Param("output_path").flag == "--output-path"


def test_param_flag_explicit_override():
    assert Param("out", flag="-o").flag == "-o"


def test_param_is_fixed_with_value():
    assert Param("out", value="$output/video.mp4").is_fixed is True


def test_param_is_fixed_without_value():
    assert Param("prompt").is_fixed is False


def test_param_needs_prompt_no_default():
    assert Param("prompt").needs_prompt is True


def test_param_needs_prompt_with_default():
    assert Param("seed", default=42).needs_prompt is False


def test_param_needs_prompt_fixed():
    assert Param("out", value="$output/x.mp4").needs_prompt is False


def test_param_log_when_inferred_after():
    assert (
        Param("out", value="$output/video.mp4", type="path-video").log_when == "after"
    )


def test_param_log_when_inferred_before():
    assert Param("image", type="path-image").log_when == "before"


def test_param_log_when_explicit_overrides():
    assert (
        Param(
            "out", value="$output/x.mp4", type="path-video", log_when="before"
        ).log_when
        == "before"
    )


def test_param_log_when_list_value():
    assert (
        Param(
            "img",
            value=["$output/img.jpg", "0", "0.8"],
            type=["path-image", "float", "float"],
        ).log_when
        == "after"
    )


def test_param_log_when_none_without_upload_type():
    assert Param("prompt").log_when is None


def test_param_log_when_none_plain_path():
    """Plain 'path' type without upload suffix does not set log_when."""
    assert Param("config", type="path").log_when is None


def test_param_type_list_infers_nargs():
    p = Param(
        "image",
        type=["path-image", "float", "float"],
        labels=["path", "start", "strength"],
    )
    assert p.nargs == 3
    assert p.types == ["path-image", "float", "float"]
    assert p.labels == ["path", "start", "strength"]


def test_param_nargs_none_without_type_list():
    assert Param("prompt").nargs is None


def test_param_primary_type_single():
    assert Param("seed", type="int")._primary_type == "int"


def test_param_primary_type_list():
    p = Param("img", type=["path-image", "float", "float"])
    assert p._primary_type == "path-image"


def test_param_types_property_single():
    assert Param("seed", type="int").types is None


def test_param_types_property_list():
    assert Param("img", type=["path", "float"]).types == ["path", "float"]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parse_basic_args():
    with patch("sys.argv", ["prog", "--prompt", "a cat", "--seed", "99"]):
        runner = _make_runner([Param("prompt"), Param("seed", type="int", default=42)])
    assert runner.parsed_params["prompt"] == "a cat"
    assert runner.parsed_params["seed"] == 99


def test_parse_bool_flag():
    with patch("sys.argv", ["prog", "--verbose"]):
        runner = _make_runner([Param("verbose", type="bool")])
    assert runner.parsed_params["verbose"] is True


def test_parse_bool_flag_absent():
    with patch("sys.argv", ["prog"]):
        runner = _make_runner([Param("verbose", type="bool")])
    assert runner.parsed_params["verbose"] is False


def test_fixed_params_not_in_argparse():
    with patch("sys.argv", ["prog", "--prompt", "hi"]):
        runner = _make_runner(
            [Param("prompt"), Param("output-path", value="$output/video.mp4")]
        )
    assert "prompt" in runner.parsed_params
    assert "output_path" not in runner.parsed_params


def test_parse_choices():
    with patch("sys.argv", ["prog", "--mode", "quality"]):
        runner = _make_runner(
            [Param("mode", choices=["fast", "quality"], default="fast")]
        )
    assert runner.parsed_params["mode"] == "quality"


def test_parse_type_list():
    with patch("sys.argv", ["prog", "--image", "photo.jpg", "0", "0.8"]):
        runner = _make_runner(
            [
                Param(
                    "image",
                    type=["path", "float", "float"],
                    labels=["path", "start", "strength"],
                )
            ]
        )
    # argparse returns strings; casting happens in _resolve_values
    assert runner.parsed_params["image"] == ["photo.jpg", "0", "0.8"]


def test_parse_types_with_spaces_in_path():
    with patch(
        "sys.argv", ["prog", "--image", "path/to something/img.jpg", "0", "0.8"]
    ):
        runner = _make_runner([Param("image", type=["path", "float", "float"])])
    assert runner.parsed_params["image"] == ["path/to something/img.jpg", "0", "0.8"]


def test_builtin_flags():
    with patch("sys.argv", ["prog", "--dry-run", "--no-interactive"]):
        runner = _make_runner()
    assert runner.runner_flags.dry_run is True
    assert runner.runner_flags.interactive is False


def test_wandb_project_override():
    with patch("sys.argv", ["prog", "--wandb-project", "my-project"]):
        runner = _make_runner()
    assert runner.runner_flags.wandb_project == "my-project"


def test_unknown_param_type_raises():
    with pytest.raises(ValueError, match="Unknown param type 'banana'"):
        Runner(command="echo", params=[Param("x", type="banana")])


# ---------------------------------------------------------------------------
# Resolve values
# ---------------------------------------------------------------------------


def test_overrides_take_priority():
    runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
    resolved = runner._resolve_values({"seed": 99}, overrides={"seed": 777})
    assert resolved["seed"] == 777


def test_callable_default():
    runner = Runner(
        command="echo", params=[Param("path", default=lambda: "/computed/path")]
    )
    resolved = runner._resolve_values({"path": None}, overrides={})
    assert resolved["path"] == "/computed/path"


def test_cli_value_beats_default():
    runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
    resolved = runner._resolve_values({"seed": 99}, overrides={})
    assert resolved["seed"] == 99


def test_resolve_casts_type_list():
    runner = Runner(
        command="echo",
        params=[Param("image", type=["path", "int", "float"])],
    )
    resolved = runner._resolve_values(
        {"image": ["photo.jpg", "5", "0.8"]}, overrides={}
    )
    assert resolved["image"] == ["photo.jpg", 5, 0.8]


def test_resolve_casts_default_list():
    runner = Runner(
        command="echo",
        params=[
            Param(
                "image",
                type=["path", "float", "float"],
                default=["img.jpg", "0", "0.8"],
            ),
        ],
    )
    resolved = runner._resolve_values({"image": None}, overrides={})
    assert resolved["image"] == ["img.jpg", 0.0, 0.8]


def test_resolve_fixed_callable():
    runner = Runner(
        command="echo",
        params=[Param("out", value=lambda: "/computed")],
    )
    resolved = runner._resolve_values({}, overrides={})
    assert resolved["out"] == "/computed"


def test_resolve_fixed_override():
    runner = Runner(
        command="echo",
        params=[Param("out", value="$output/video.mp4")],
    )
    resolved = runner._resolve_values({}, overrides={"out": "/override/video.mp4"})
    assert resolved["out"] == "/override/video.mp4"


# ---------------------------------------------------------------------------
# Prompt missing
# ---------------------------------------------------------------------------


def test_no_interactive_exits_on_missing():
    runner = Runner(command="echo", params=[Param("prompt")])
    with pytest.raises(SystemExit, match="2"):
        runner._prompt_missing({"prompt": None}, interactive=False)


def test_no_missing_params_is_noop():
    runner = Runner(command="echo", params=[Param("seed", default=42)])
    runner._prompt_missing({"seed": 42}, interactive=False)


def test_interactive_fills_from_questionary():
    runner = Runner(command="echo", params=[Param("prompt")])
    resolved = {"prompt": None}
    with patch("genai_runner.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        runner._prompt_missing(resolved, interactive=True)
    assert resolved["prompt"] == "a cat"


def test_interactive_select_for_choices():
    runner = Runner(command="echo", params=[Param("mode", choices=["fast", "slow"])])
    resolved = {"mode": None}
    with patch("genai_runner.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "fast"
        runner._prompt_missing(resolved, interactive=True)
    assert resolved["mode"] == "fast"


def test_interactive_type_list_prompts_each_part():
    runner = Runner(
        command="echo",
        params=[
            Param(
                "image",
                type=["path-image", "float", "float"],
                labels=["path", "start", "strength"],
            ),
        ],
    )
    resolved = {"image": None}
    answers = iter(["0", "0.8"])
    with patch("genai_runner.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "photo.jpg"
        mock_q.text.return_value.ask.side_effect = lambda: next(answers)
        runner._prompt_missing(resolved, interactive=True)
    # After prompting, types are cast: str, float, float
    assert resolved["image"] == ["photo.jpg", 0.0, 0.8]


def test_interactive_path_image_uses_path_widget():
    """path-image type should use questionary.path() widget."""
    runner = Runner(command="echo", params=[Param("img", type="path-image")])
    resolved = {"img": None}
    with patch("genai_runner.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "/tmp/photo.jpg"
        runner._prompt_missing(resolved, interactive=True)
    mock_q.path.assert_called_once()
    assert resolved["img"] == "/tmp/photo.jpg"


def test_interactive_cancel_exits():
    runner = Runner(command="echo", params=[Param("prompt")])
    with patch("genai_runner.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(SystemExit, match="1"):
            runner._prompt_missing({"prompt": None}, interactive=True)


# ---------------------------------------------------------------------------
# Interpolate output
# ---------------------------------------------------------------------------


def test_interpolate_replaces_output_in_string(tmp_path):
    runner = Runner(command="echo", params=[Param("out", value="$output/video.mp4")])
    result = runner._interpolate_output({"out": "$output/video.mp4"}, tmp_path)
    assert result["out"] == f"{tmp_path}/video.mp4"


def test_interpolate_replaces_output_in_list(tmp_path):
    runner = Runner(
        command="echo", params=[Param("img", value=["$output/img.jpg", "0", "0.8"])]
    )
    result = runner._interpolate_output(
        {"img": ["$output/img.jpg", "0", "0.8"]}, tmp_path
    )
    assert result["img"] == [f"{tmp_path}/img.jpg", "0", "0.8"]


def test_interpolate_non_output_unchanged(tmp_path):
    runner = Runner(command="echo", params=[Param("config", value="/etc/config.toml")])
    result = runner._interpolate_output({"config": "/etc/config.toml"}, tmp_path)
    assert result["config"] == "/etc/config.toml"


def test_interpolate_preserves_resolved_params(tmp_path):
    runner = Runner(command="echo", params=[Param("out", value="$output/video.mp4")])
    result = runner._interpolate_output(
        {"out": "$output/video.mp4", "prompt": "a cat", "seed": 42}, tmp_path
    )
    assert result["prompt"] == "a cat"
    assert result["seed"] == 42


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------


def test_build_basic_command():
    runner = Runner(
        command="python generate.py",
        params=[Param("prompt"), Param("seed", type="int")],
    )
    assert runner._build_command({"prompt": "a cat", "seed": 42}) == [
        "python",
        "generate.py",
        "--prompt",
        "a cat",
        "--seed",
        "42",
    ]


def test_build_bool_flag():
    runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
    assert runner._build_command({"verbose": True}) == ["run.py", "--verbose"]


def test_build_bool_flag_false_omitted():
    runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
    assert runner._build_command({"verbose": False}) == ["run.py"]


def test_build_multi_value_flag():
    runner = Runner(
        command="run.py", params=[Param("image", value=["photo.jpg", "0", "0.8"])]
    )
    assert runner._build_command({"image": ["photo.jpg", "0", "0.8"]}) == [
        "run.py",
        "--image",
        "photo.jpg",
        "0",
        "0.8",
    ]


def test_build_none_values_omitted():
    runner = Runner(
        command="run.py", params=[Param("prompt"), Param("seed", type="int")]
    )
    assert runner._build_command({"prompt": "hi", "seed": None}) == [
        "run.py",
        "--prompt",
        "hi",
    ]


def test_build_custom_flag():
    runner = Runner(command="run.py", params=[Param("out", flag="-o")])
    assert runner._build_command({"out": "/tmp/x"}) == ["run.py", "-o", "/tmp/x"]


def test_build_command_as_list():
    runner = Runner(command=["python", "-m", "my model"], params=[Param("prompt")])
    assert runner._build_command({"prompt": "a cat"}) == [
        "python",
        "-m",
        "my model",
        "--prompt",
        "a cat",
    ]


def test_build_command_string_with_quotes():
    runner = Runner(command='python "my script.py"', params=[])
    assert runner._build_command({}) == ["python", "my script.py"]


def test_build_type_list_from_cli():
    runner = Runner(
        command="run.py", params=[Param("image", type=["path", "float", "float"])]
    )
    assert runner._build_command({"image": ["photo.jpg", 0.0, 0.8]}) == [
        "run.py",
        "--image",
        "photo.jpg",
        "0.0",
        "0.8",
    ]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def test_metric_float():
    runner = Runner(
        command="echo", metrics=[Metric("skip_pct", pattern=r"skipped=([\d.]+)%")]
    )
    wb_run = _mock_wb_run()
    runner._extract_metrics(wb_run, "some output\nskipped=32.8%\ndone")
    assert wb_run.summary["skip_pct"] == 32.8


def test_metric_str():
    runner = Runner(
        command="echo", metrics=[Metric("status", pattern=r"final: (\w+)", type="str")]
    )
    wb_run = _mock_wb_run()
    runner._extract_metrics(wb_run, "final: completed")
    assert wb_run.summary["status"] == "completed"


def test_metric_last_match_wins():
    runner = Runner(command="echo", metrics=[Metric("val", pattern=r"x=([\d.]+)")])
    wb_run = _mock_wb_run()
    runner._extract_metrics(wb_run, "x=1.0\nx=2.0\nx=3.0")
    assert wb_run.summary["val"] == 3.0


def test_metric_no_match():
    runner = Runner(command="echo", metrics=[Metric("val", pattern=r"x=([\d.]+)")])
    wb_run = _mock_wb_run()
    runner._extract_metrics(wb_run, "no matches here")
    assert "val" not in wb_run.summary


# ---------------------------------------------------------------------------
# Execute subprocess
# ---------------------------------------------------------------------------


def test_execute_captures_stdout_and_stderr(tmp_path):
    runner = Runner(command="echo")
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr)",
    ]
    exit_code, duration, stdout_text = runner._execute(cmd, tmp_path)

    assert exit_code == 0
    assert duration > 0
    assert "out" in stdout_text
    assert (tmp_path / "stdout.log").read_text().strip() == "out"
    assert "err" in (tmp_path / "stderr.log").read_text()
    run_log = (tmp_path / "run.log").read_text()
    assert "out" in run_log
    assert "[stderr] err" in run_log


def test_execute_nonzero_exit_code(tmp_path):
    runner = Runner(command="echo")
    exit_code, _, _ = runner._execute(
        [sys.executable, "-c", "import sys; sys.exit(42)"], tmp_path
    )
    assert exit_code == 42


def test_execute_env_vars_passed(tmp_path):
    runner = Runner(command="echo", env={"MY_TEST_VAR": "hello123"})
    cmd = [sys.executable, "-c", "import os; print(os.environ['MY_TEST_VAR'])"]
    exit_code, _, stdout_text = runner._execute(cmd, tmp_path)
    assert exit_code == 0
    assert "hello123" in stdout_text


# ---------------------------------------------------------------------------
# Dry run (integration)
# ---------------------------------------------------------------------------


def test_dry_run_prints_command_no_wandb(capsys):
    with patch(
        "sys.argv",
        ["prog", "--dry-run", "--prompt", "test", "--no-interactive"],
    ):
        runner = Runner(
            command="python gen.py",
            params=[
                Param("prompt"),
                Param("seed", type="int", default=42),
                Param("output-path", value="$output/video.mp4", type="path-video"),
            ],
            tags=["v1"],
        )
    with patch("genai_runner._collect_git_info", return_value=_FAKE_GIT_INFO):
        runner.run()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "--prompt test" in captured.out
    assert "--seed 42" in captured.out
    assert "$output/video.mp4" in captured.out
    assert "Run name: (auto)" in captured.out
    assert "Tags: ['v1']" in captured.out


def test_no_project_raises_valueerror():
    """Runner with no wandb_project and no git repo raises ValueError."""
    with patch("sys.argv", ["prog", "--no-interactive"]):
        runner = Runner(command="echo", params=[])
    with patch("genai_runner._collect_git_info", return_value={}):
        with pytest.raises(ValueError, match="Cannot determine project name"):
            runner.run()


# ---------------------------------------------------------------------------
# Full run (integration with mocked W&B)
# ---------------------------------------------------------------------------


def test_full_run_with_mocked_wandb(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    with patch("sys.argv", ["prog", "--no-interactive"]):
        runner = Runner(
            command=(
                f"{sys.executable} -c"
                " \"import sys; print('hello');"
                " print('err', file=sys.stderr)\""
            ),
            params=[
                Param("output-path", value="$output/video.mp4", type="path-video"),
            ],
            metrics=[Metric("val", pattern=r"no-match")],
        )

    with (
        patch("genai_runner.wandb", mock_wb),
        patch("genai_runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner._save_code_snapshot"),
        patch("genai_runner._RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run()

    mock_wb.init.assert_called_once()
    assert mock_wb.init.call_args[1]["project"] == "test-repo"
    assert mock_wb.init.call_args[1]["save_code"] is True
    assert mock_wb.init.call_args[1]["group"] is None
    assert wb_run.summary["status"] == "success"
    assert wb_run.summary["exit_code"] == 0
    assert wb_run.summary["duration_seconds"] > 0
    assert (tmp_path / "genai_runs" / "test-repo").exists()
    wb_run.finish.assert_called_once_with(exit_code=0)


def test_full_run_explicit_group(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    with patch("sys.argv", ["prog", "--no-interactive"]):
        runner = Runner(
            command=f"{sys.executable} -c \"print('ok')\"",
            params=[],
            group="my-sweep",
        )

    with (
        patch("genai_runner.wandb", mock_wb),
        patch("genai_runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner._save_code_snapshot"),
        patch("genai_runner._RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run()

    assert mock_wb.init.call_args[1]["group"] == "my-sweep"


# ---------------------------------------------------------------------------
# count_logged with path-* types
# ---------------------------------------------------------------------------


def test_count_logged_path_video():
    runner = Runner(
        command="echo",
        params=[
            Param("out", value="$output/video.mp4", type="path-video"),
            Param("img", type="path-image"),
        ],
        outputs=[Output("extra.mp4", log_as="video")],
    )
    assert runner._count_logged("video") == 2  # param + output
    assert runner._count_logged("image") == 1


def test_count_logged_type_list():
    runner = Runner(
        command="echo",
        params=[Param("img", type=["path-image", "float", "float"])],
    )
    assert runner._count_logged("image") == 1


# ---------------------------------------------------------------------------
# Output glob + zip
# ---------------------------------------------------------------------------


def test_log_extra_outputs_glob(tmp_path):
    """Glob pattern expands and uploads each matched file."""
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "001.png").write_text("a")
    (tmp_path / "frames" / "002.png").write_text("b")
    (tmp_path / "frames" / "skip.txt").write_text("c")

    runner = Runner(
        command="echo",
        outputs=[Output("$output/frames/*.png", log_as="image")],
    )
    wb_run = _mock_wb_run()
    with patch("genai_runner.wandb"):
        runner._log_extra_outputs(wb_run, tmp_path)
    # Should upload 2 png files, not the txt
    assert wb_run.log.call_count == 2


def test_log_extra_outputs_glob_zip(tmp_path):
    """Glob + log_as='zip' creates a zip archive."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "b.pt").write_text("tensor2")

    runner = Runner(
        command="echo",
        outputs=[Output("$output/debug/*.pt", log_as="zip")],
    )
    wb_run = _mock_wb_run()
    with patch("genai_runner.wandb"):
        runner._log_extra_outputs(wb_run, tmp_path)

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

    runner = Runner(
        command="echo",
        outputs=[Output("$output/debug", log_as="zip")],
    )
    wb_run = _mock_wb_run()
    runner._log_extra_outputs(wb_run, tmp_path)

    zip_path = tmp_path / "debug.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        assert "a.pt" in names
        assert "sub/b.pt" in names


def test_log_extra_outputs_glob_no_match(tmp_path, capsys):
    """Glob with no matches prints a warning."""
    runner = Runner(
        command="echo",
        outputs=[Output("$output/nope/*.png", log_as="image")],
    )
    wb_run = _mock_wb_run()
    runner._log_extra_outputs(wb_run, tmp_path)
    assert "matched no files" in capsys.readouterr().out


def test_log_extra_outputs_single_file(tmp_path):
    """Non-glob single file still works (regression)."""
    (tmp_path / "meta.json").write_text("{}")

    runner = Runner(
        command="echo",
        outputs=[Output("$output/meta.json", log_as="artifact")],
    )
    wb_run = _mock_wb_run()
    runner._log_extra_outputs(wb_run, tmp_path)
    wb_run.log_artifact.assert_called_once()


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------


def test_git_info_returns_expected_keys():
    outputs = [
        b"/home/user/genai-runner\n",
        b"abc123def456\n",
        b"main\n",
        b"",
    ]
    with patch("subprocess.check_output", side_effect=outputs):
        info = _collect_git_info()
    assert info == {
        "repo": "genai-runner",
        "commit": "abc123def456",
        "branch": "main",
        "dirty": False,
    }


def test_git_info_dirty_flag():
    outputs = [
        b"/home/user/genai-runner\n",
        b"abc123\n",
        b"main\n",
        b" M src/file.py\n",
    ]
    with patch("subprocess.check_output", side_effect=outputs):
        assert _collect_git_info()["dirty"] is True


def test_git_info_empty_outside_repo():
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")
    ):
        assert _collect_git_info() == {}
