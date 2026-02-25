"""Tests for genai_runner."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_runner import Metric, Output, Param, Runner, _collect_git_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUILTIN_FLAGS = {"_dry_run": False, "_no_interactive": True, "_keep_outputs": False}

_FAKE_GIT_INFO = {"repo": "test-repo", "commit": "abc", "branch": "main", "dirty": False}


def _mock_wb_run(**overrides) -> MagicMock:
    run = MagicMock()
    run.summary = {}
    run.name = overrides.get("name", "test-run-42")
    run.id = overrides.get("id", "abc123")
    run.url = overrides.get("url", "https://wandb.test/run")
    run.tags = overrides.get("tags", [])
    run.config = overrides.get("config", {})
    return run


def _make_runner(params=None, **kwargs):
    return Runner(command=kwargs.pop("command", "echo hello"), params=params or [], **kwargs)


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
    assert Param("out", value="$output/video.mp4", log_as="video").log_when == "after"


def test_param_log_when_inferred_before():
    assert Param("image", type="path", log_as="image").log_when == "before"


def test_param_log_when_explicit_overrides():
    assert Param("out", value="$output/x.mp4", log_as="video", log_when="before").log_when == "before"


def test_param_log_when_list_value():
    assert Param("img", value=["$output/img.jpg", "0", "0.8"], log_as="image").log_when == "after"


def test_param_log_when_none_without_log_as():
    assert Param("prompt").log_when is None


def test_param_types_infers_nargs():
    p = Param("image", types=["path", "float", "float"], labels=["path", "start", "strength"])
    assert p.nargs == 3
    assert p.types == ["path", "float", "float"]
    assert p.labels == ["path", "start", "strength"]


def test_param_nargs_none_without_types():
    assert Param("prompt").nargs is None


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_parse_basic_args():
    runner = _make_runner([Param("prompt"), Param("seed", type="int", default=42)])
    with patch("sys.argv", ["prog", "--prompt", "a cat", "--seed", "99"]):
        args = runner._parse_cli_args()
    assert args["prompt"] == "a cat"
    assert args["seed"] == 99


def test_parse_bool_flag():
    runner = _make_runner([Param("verbose", type="bool")])
    with patch("sys.argv", ["prog", "--verbose"]):
        assert runner._parse_cli_args()["verbose"] is True


def test_parse_bool_flag_absent():
    runner = _make_runner([Param("verbose", type="bool")])
    with patch("sys.argv", ["prog"]):
        assert runner._parse_cli_args()["verbose"] is False


def test_fixed_params_not_in_argparse():
    runner = _make_runner([Param("prompt"), Param("output-path", value="$output/video.mp4")])
    with patch("sys.argv", ["prog", "--prompt", "hi"]):
        args = runner._parse_cli_args()
    assert "prompt" in args
    assert "output_path" not in args


def test_parse_choices():
    runner = _make_runner([Param("mode", choices=["fast", "quality"], default="fast")])
    with patch("sys.argv", ["prog", "--mode", "quality"]):
        assert runner._parse_cli_args()["mode"] == "quality"


def test_parse_types():
    runner = _make_runner([Param("image", types=["path", "float", "float"], labels=["path", "start", "strength"])])
    with patch("sys.argv", ["prog", "--image", "photo.jpg", "0", "0.8"]):
        args = runner._parse_cli_args()
    # argparse returns strings; casting happens in _resolve_values
    assert args["image"] == ["photo.jpg", "0", "0.8"]


def test_parse_types_with_spaces_in_path():
    runner = _make_runner([Param("image", types=["path", "float", "float"])])
    with patch("sys.argv", ["prog", "--image", "path/to something/img.jpg", "0", "0.8"]):
        args = runner._parse_cli_args()
    assert args["image"] == ["path/to something/img.jpg", "0", "0.8"]


def test_builtin_flags():
    runner = _make_runner()
    with patch("sys.argv", ["prog", "--dry-run", "-n", "--keep-outputs"]):
        args = runner._parse_cli_args()
    assert args["_dry_run"] is True
    assert args["_no_interactive"] is True
    assert args["_keep_outputs"] is True


def test_wandb_project_override():
    runner = _make_runner()
    with patch("sys.argv", ["prog", "--wandb-project", "my-project"]):
        runner._parse_cli_args()
    assert runner.wandb_project == "my-project"


# ---------------------------------------------------------------------------
# Resolve values
# ---------------------------------------------------------------------------

def test_overrides_take_priority():
    runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
    resolved = runner._resolve_values({"seed": 99, **_BUILTIN_FLAGS}, overrides={"seed": 777})
    assert resolved["seed"] == 777


def test_callable_default():
    runner = Runner(command="echo", params=[Param("path", default=lambda: "/computed/path")])
    resolved = runner._resolve_values({"path": None, **_BUILTIN_FLAGS}, overrides={})
    assert resolved["path"] == "/computed/path"


def test_cli_value_beats_default():
    runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
    resolved = runner._resolve_values({"seed": 99, **_BUILTIN_FLAGS}, overrides={})
    assert resolved["seed"] == 99


def test_resolve_casts_types():
    runner = Runner(command="echo", params=[
        Param("image", types=["path", "int", "float"]),
    ])
    resolved = runner._resolve_values(
        {"image": ["photo.jpg", "5", "0.8"], **_BUILTIN_FLAGS}, overrides={},
    )
    assert resolved["image"] == ["photo.jpg", 5, 0.8]


def test_resolve_casts_default_list():
    runner = Runner(command="echo", params=[
        Param("image", types=["path", "float", "float"], default=["img.jpg", "0", "0.8"]),
    ])
    resolved = runner._resolve_values({"image": None, **_BUILTIN_FLAGS}, overrides={})
    assert resolved["image"] == ["img.jpg", 0.0, 0.8]


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


def test_interactive_types_prompts_each_part():
    runner = Runner(command="echo", params=[
        Param("image", types=["path", "float", "float"], labels=["path", "start", "strength"]),
    ])
    resolved = {"image": None}
    answers = iter(["0", "0.8"])
    with patch("genai_runner.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "photo.jpg"
        mock_q.text.return_value.ask.side_effect = lambda: next(answers)
        runner._prompt_missing(resolved, interactive=True)
    # After prompting, types are cast: str, float, float
    assert resolved["image"] == ["photo.jpg", 0.0, 0.8]


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
    result = runner._interpolate_output({}, tmp_path)
    assert result["out"] == f"{tmp_path}/video.mp4"


def test_interpolate_replaces_output_in_list(tmp_path):
    runner = Runner(command="echo", params=[Param("img", value=["$output/img.jpg", "0", "0.8"])])
    result = runner._interpolate_output({}, tmp_path)
    assert result["img"] == [f"{tmp_path}/img.jpg", "0", "0.8"]


def test_interpolate_non_output_unchanged(tmp_path):
    runner = Runner(command="echo", params=[Param("config", value="/etc/config.toml")])
    result = runner._interpolate_output({}, tmp_path)
    assert result["config"] == "/etc/config.toml"


def test_interpolate_preserves_resolved_params(tmp_path):
    runner = Runner(command="echo", params=[Param("out", value="$output/video.mp4")])
    result = runner._interpolate_output({"prompt": "a cat", "seed": 42}, tmp_path)
    assert result["prompt"] == "a cat"
    assert result["seed"] == 42


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def test_build_basic_command():
    runner = Runner(command="python generate.py", params=[Param("prompt"), Param("seed", type="int")])
    assert runner._build_command({"prompt": "a cat", "seed": 42}) == [
        "python", "generate.py", "--prompt", "a cat", "--seed", "42",
    ]


def test_build_bool_flag():
    runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
    assert runner._build_command({"verbose": True}) == ["run.py", "--verbose"]


def test_build_bool_flag_false_omitted():
    runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
    assert runner._build_command({"verbose": False}) == ["run.py"]


def test_build_multi_value_flag():
    runner = Runner(command="run.py", params=[Param("image", value=["photo.jpg", "0", "0.8"])])
    assert runner._build_command({"image": ["photo.jpg", "0", "0.8"]}) == [
        "run.py", "--image", "photo.jpg", "0", "0.8",
    ]


def test_build_none_values_omitted():
    runner = Runner(command="run.py", params=[Param("prompt"), Param("seed", type="int")])
    assert runner._build_command({"prompt": "hi", "seed": None}) == ["run.py", "--prompt", "hi"]


def test_build_custom_flag():
    runner = Runner(command="run.py", params=[Param("out", flag="-o")])
    assert runner._build_command({"out": "/tmp/x"}) == ["run.py", "-o", "/tmp/x"]


def test_build_command_as_list():
    runner = Runner(command=["python", "-m", "my model"], params=[Param("prompt")])
    assert runner._build_command({"prompt": "a cat"}) == [
        "python", "-m", "my model", "--prompt", "a cat",
    ]


def test_build_command_string_with_quotes():
    runner = Runner(command='python "my script.py"', params=[])
    assert runner._build_command({}) == ["python", "my script.py"]


def test_build_types_from_cli():
    runner = Runner(command="run.py", params=[Param("image", types=["path", "float", "float"])])
    assert runner._build_command({"image": ["photo.jpg", 0.0, 0.8]}) == [
        "run.py", "--image", "photo.jpg", "0.0", "0.8",
    ]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def test_metric_float():
    runner = Runner(command="echo", metrics=[Metric("skip_pct", pattern=r"skipped=([\d.]+)%")])
    wb_run = _mock_wb_run()
    runner._extract_metrics(wb_run, "some output\nskipped=32.8%\ndone")
    assert wb_run.summary["skip_pct"] == 32.8


def test_metric_str():
    runner = Runner(command="echo", metrics=[Metric("status", pattern=r"final: (\w+)", type="str")])
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
    cmd = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
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
    exit_code, _, _ = runner._execute([sys.executable, "-c", "import sys; sys.exit(42)"], tmp_path)
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
    runner = Runner(
        command="python gen.py",
        params=[
            Param("prompt"),
            Param("seed", type="int", default=42),
            Param("output-path", value="$output/video.mp4"),
        ],
    )
    with patch("sys.argv", ["prog", "--dry-run", "--prompt", "test", "-n"]):
        runner.run()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "--prompt test" in captured.out
    assert "--seed 42" in captured.out
    assert "--output-path /tmp/dry-run-output/video.mp4" in captured.out


# ---------------------------------------------------------------------------
# Full run (integration with mocked W&B)
# ---------------------------------------------------------------------------

def test_full_run_with_mocked_wandb(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    runner = Runner(
        command=f"{sys.executable} -c \"import sys; print('hello'); print('err', file=sys.stderr)\"",
        params=[Param("output-path", value="$output/video.mp4")],
        metrics=[Metric("val", pattern=r"no-match")],
    )

    with (
        patch("sys.argv", ["prog", "-n"]),
        patch("genai_runner.wandb", mock_wb),
        patch("genai_runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner._save_code_snapshot"),
        patch("genai_runner.Path.home", return_value=tmp_path),
    ):
        runner.run()

    mock_wb.init.assert_called_once()
    assert mock_wb.init.call_args[1]["project"] == "test-repo"
    assert mock_wb.init.call_args[1]["save_code"] is True
    assert wb_run.summary["status"] == "success"
    assert wb_run.summary["exit_code"] == 0
    assert wb_run.summary["duration_seconds"] > 0
    assert (tmp_path / "genai_runs" / "test-repo").exists()
    wb_run.finish.assert_called_once_with(exit_code=0)


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------

def test_git_info_returns_expected_keys():
    outputs = iter([
        b"/home/user/genai-runner\n",
        b"abc123def456\n",
        b"main\n",
        b"",
    ])
    with patch("subprocess.check_output", side_effect=lambda *a, **kw: next(outputs)):
        info = _collect_git_info()
    assert info == {"repo": "genai-runner", "commit": "abc123def456", "branch": "main", "dirty": False}


def test_git_info_dirty_flag():
    outputs = iter([
        b"/home/user/genai-runner\n",
        b"abc123\n",
        b"main\n",
        b" M src/file.py\n",
    ])
    with patch("subprocess.check_output", side_effect=lambda *a, **kw: next(outputs)):
        assert _collect_git_info()["dirty"] is True


def test_git_info_empty_outside_repo():
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
        assert _collect_git_info() == {}
