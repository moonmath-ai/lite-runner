"""Tests for lite_runner.runner."""

import json
import re
import sys
from unittest.mock import MagicMock, patch

import git
import pytest
from conftest import _FAKE_GIT_INFO, _make_runner, _mock_wb_run

from lite_runner import UNSET, Metric, Param, Runner
from lite_runner.runner import _collect_git_info, _interpolate_output

# ---------------------------------------------------------------------------
# CLI parsing (via parse_cli)
# ---------------------------------------------------------------------------


def test_parse_basic_args():
    runner = _make_runner(params=[Param("prompt"), Param("seed", type="int")])
    r = runner.parse_cli(["--prompt", "a cat", "--seed", "42"])
    assert r.param_values["prompt"] == "a cat"
    assert r.param_values["seed"] == 42
    assert r.cli_parsed


def test_parse_bool_flag():
    runner = _make_runner(params=[Param("turbo", type="bool")])
    r = runner.parse_cli(["--turbo"])
    assert r.param_values["turbo"] is True


def test_parse_bool_flag_absent():
    runner = _make_runner(params=[Param("turbo", type="bool")])
    r = runner.parse_cli([])
    assert "turbo" not in r.param_values


def test_fixed_params_not_in_argparse():
    runner = _make_runner(
        params=[
            Param("prompt"),
            Param("output", value="$output/video.mp4"),
        ],
    )
    parser = runner.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "x"])


def test_parse_choices():
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    r = runner.parse_cli(["--mode", "fast"])
    assert r.param_values["mode"] == "fast"


def test_parse_type_list():
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path-image", "float", "float"],
                labels=["path", "start", "strength"],
            ),
        ],
    )
    r = runner.parse_cli(["--image", "photo.jpg", "0", "0.8"])
    assert r.param_values["image"] == ["photo.jpg", 0.0, 0.8]


def test_parse_types_with_spaces_in_path():
    runner = _make_runner(params=[Param("config", type="path")])
    r = runner.parse_cli(["--config", "/path/with spaces/config.yaml"])
    assert r.param_values["config"] == "/path/with spaces/config.yaml"


def test_builtin_flags():
    runner = _make_runner()
    r = runner.parse_cli(["--dry-run", "--no-interactive"])
    assert r.run_flags.dry_run is True
    assert r.run_flags.no_interactive is True


def test_project_override():
    runner = _make_runner()
    r = runner.parse_cli(["--project", "my-project"])
    assert r.run_flags.project == "my-project"


def test_param_name_conflicts_with_builtin_flag():
    with pytest.raises(ValueError, match="conflicts with built-in flag"):
        Runner(command="echo", params=[Param("dry_run")])


# ---------------------------------------------------------------------------
# resolve_defaults
# ---------------------------------------------------------------------------


def test_resolve_defaults_applies_defaults():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.resolve_defaults()
    assert r.param_values["seed"] == 42
    assert r.param_sources["seed"] == "default"


def test_resolve_defaults_callable():
    runner = _make_runner(params=[Param("ts", default=lambda: "now")])
    r = runner.resolve_defaults()
    assert r.param_values["ts"] == "now"


def test_resolve_defaults_fixed():
    runner = _make_runner(params=[Param("out", value="/fixed")])
    r = runner.resolve_defaults()
    assert r.param_values["out"] == "/fixed"
    assert r.param_sources["out"] == "fixed"


def test_resolve_defaults_fixed_callable():
    runner = _make_runner(params=[Param("ts", value=lambda: "now")])
    r = runner.resolve_defaults()
    assert r.param_values["ts"] == "now"


def test_resolve_defaults_bool_false():
    runner = _make_runner(params=[Param("turbo", type="bool")])
    r = runner.resolve_defaults()
    assert r.param_values["turbo"] is False


def test_resolve_defaults_does_not_overwrite_cli():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"]).resolve_defaults()
    assert r.param_values["seed"] == 99


def test_resolve_defaults_does_not_overwrite_override():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=77).resolve_defaults()
    assert r.param_values["seed"] == 77


def test_resolve_defaults_casts_type_list():
    runner = _make_runner(
        params=[Param("x", type=["float", "float"])],
    )
    r = runner.parse_cli(["--x", "1.0", "2.0"]).resolve_defaults()
    assert r.param_values["x"] == [1.0, 2.0]


def test_resolve_defaults_casts_default_list():
    runner = _make_runner(
        params=[
            Param(
                "x",
                type=["path", "float"],
                default=["a.jpg", "0.5"],
                labels=["path", "val"],
            ),
        ],
    )
    r = runner.resolve_defaults()
    assert r.param_values["x"] == ["a.jpg", "0.5"]


# ---------------------------------------------------------------------------
# override
# ---------------------------------------------------------------------------


def test_override_returns_new_runner():
    runner = _make_runner(
        params=[Param("seed", type="int", default=42)],
    )
    r = runner.override(seed=99)
    assert r is not runner
    assert r.param_values["seed"] == 99
    assert "seed" not in runner.param_values


def test_override_kwarg_underscore_to_hyphen():
    runner = _make_runner(params=[Param("my-param")])
    r = runner.override(my_param="val")
    assert r.param_values["my-param"] == "val"


def test_override_unknown_param_raises():
    runner = _make_runner(params=[Param("seed", type="int")])
    with pytest.raises(ValueError, match="Unknown param"):
        runner.override(bad_param=99)


def test_override_chained():
    runner = _make_runner(
        params=[Param("seed", type="int"), Param("prompt")],
    )
    r = runner.override(seed=42, prompt="a cat")
    assert r.param_values["seed"] == 42
    assert r.param_values["prompt"] == "a cat"


def test_override_preserves_cli_args():
    runner = _make_runner(
        params=[Param("seed", type="int"), Param("prompt")],
    )
    r = runner.parse_cli(["--seed", "42"])
    r2 = r.override(prompt="a cat")
    assert r2.param_values["seed"] == 42
    assert r2.param_values["prompt"] == "a cat"


def test_override_fixed_params_included():
    runner = _make_runner(params=[Param("out", value="/fixed")])
    r = runner.resolve_defaults()
    assert r.param_values["out"] == "/fixed"


def test_override_run_skips_prompting(tmp_path):
    runner = Runner(
        command=f"{sys.executable} -c \"print('hello')\"",
        params=[
            Param("prompt"),
            Param("seed", type="int", default=42),
        ],
    )
    with (
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.override(prompt="a cat").run(
            no_interactive=True,
            no_wandb=True,
        )


# ---------------------------------------------------------------------------
# with_metadata
# ---------------------------------------------------------------------------


def test_with_metadata_project():
    runner = _make_runner()
    r = runner.with_metadata(project="new-proj")
    assert r.project == "new-proj"
    assert runner.project is None


def test_with_metadata_group():
    runner = _make_runner()
    r = runner.with_metadata(run_group="sweep-1")
    assert r.run_group == "sweep-1"
    assert runner.run_group is None


def test_with_metadata_tags():
    runner = _make_runner()
    r = runner.with_metadata(tags=["v1", "test"])
    assert r.tags == ["v1", "test"]
    assert runner.tags == []


def test_with_metadata_partial():
    runner = _make_runner()
    r = runner.with_metadata(project="proj")
    assert r.project == "proj"
    assert r.run_group is None


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------


def test_ask_user_fills_from_questionary():
    runner = _make_runner(params=[Param("prompt")])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        r = runner.ask_user()
    assert r.param_values["prompt"] == "a cat"
    assert r.param_sources["prompt"] == "prompt"
    assert r.filled


def test_ask_user_non_interactive_raises_on_missing():
    runner = _make_runner(params=[Param("prompt")])
    with pytest.raises(ValueError, match="Missing required params"):
        runner.ask_user(no_interactive=True)


def test_ask_user_non_interactive_ok_with_defaults():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.ask_user(no_interactive=True)
    assert r.param_values["seed"] == 42
    assert r.filled


def test_ask_user_select_for_choices():
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "fast"
        r = runner.ask_user()
    assert r.param_values["mode"] == "fast"


def test_ask_user_type_list_prompts_each_part():
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path-image", "float", "float"],
                labels=["path", "start", "strength"],
            ),
        ],
    )
    answers = iter(["0", "0.8"])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "photo.jpg"
        mock_q.text.return_value.ask.side_effect = lambda: next(answers)
        r = runner.ask_user()
    # After prompting, types are cast: str, float, float
    assert r.param_values["image"] == ["photo.jpg", 0.0, 0.8]


def test_ask_user_path_image_uses_path_widget():
    """path-image type should use questionary.path() widget."""
    runner = _make_runner(params=[Param("img", type="path-image")])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "/fake/photo.jpg"
        r = runner.ask_user()
    mock_q.path.assert_called_once()
    assert r.param_values["img"] == "/fake/photo.jpg"


def test_ask_user_cancel_raises():
    """Cancelling a text prompt raises KeyboardInterrupt."""
    runner = _make_runner(params=[Param("prompt")])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            runner.ask_user()


def test_ask_user_prompts_default_param():
    """Params with defaults are prompted with default pre-filled."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "99"
        r = runner.ask_user()
    mock_q.text.assert_called_once_with("seed:", default="42")
    assert r.param_values["seed"] == 99


def test_ask_user_skips_cli_provided_param():
    """Params explicitly provided on CLI are not prompted."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"])
    with patch("lite_runner.params.questionary") as mock_q:
        r = r.ask_user()
    mock_q.text.assert_not_called()
    assert r.param_values["seed"] == 99


def test_ask_user_skips_overridden_param():
    """Params set via overrides are not prompted."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=77)
    with patch("lite_runner.params.questionary") as mock_q:
        r = r.ask_user()
    mock_q.text.assert_not_called()
    assert r.param_values["seed"] == 77


def test_ask_user_skips_no_prompt_param():
    """Params with prompt=False are not prompted; they use their default."""
    runner = _make_runner(
        params=[
            Param("prompt"),
            Param("threshold", type="float", default=-3.0, prompt=False),
        ],
    )
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        r = runner.ask_user()
    # Only prompt should be prompted, not threshold
    assert mock_q.text.call_count == 1
    assert r.param_values["prompt"] == "a cat"
    assert r.param_values["threshold"] == -3.0


def test_no_prompt_param_accepts_cli_flag():
    """Params with prompt=False can still be overridden from CLI."""
    runner = _make_runner(
        [Param("threshold", type="float", default=-3.0, prompt=False)]
    )
    r = runner.parse_cli(["--threshold", "-5.0"])
    assert r.param_values["threshold"] == -5.0
    assert r.param_sources["threshold"] == "cli"


# ---------------------------------------------------------------------------
# UNSET in commands and config
# ---------------------------------------------------------------------------


def test_build_command_skips_unset():
    """UNSET params are omitted from the command."""
    runner = _make_runner(
        params=[Param("prompt"), Param("mode", choices=["fast", "slow"])],
    )
    cmd = runner.build_command({"prompt": "a cat", "mode": UNSET})
    assert cmd == ["echo", "hello", "--prompt", "a cat"]


def test_build_command_skips_unset_after_copy():
    """UNSET survives Runner.copy() (deepcopy) and is still omitted."""
    runner = _make_runner(
        params=[Param("prompt"), Param("mode", choices=["fast", "slow"])],
    )
    runner.param_values = {"prompt": "a cat", "mode": UNSET}
    copied = runner.copy()
    cmd = copied.build_command(copied.param_values)
    assert cmd == ["echo", "hello", "--prompt", "a cat"]
    assert "<unset>" not in " ".join(cmd)


def test_config_logs_unset_as_marker():
    """Skipped params appear as '<unset>' in the config dict."""
    resolved = {"prompt": "test", "mode": UNSET}
    config: dict[str, object] = {}
    for k, v in resolved.items():
        config[f"param/{k}"] = "<unset>" if v is UNSET else v
    assert config["param/prompt"] == "test"
    assert config["param/mode"] == "<unset>"


# ---------------------------------------------------------------------------
# Interpolate output
# ---------------------------------------------------------------------------


def test_interpolate_preserves_unset(tmp_path):
    """UNSET values pass through interpolation unchanged."""
    result = _interpolate_output({"out": UNSET}, tmp_path)
    assert result["out"] is UNSET


def test_interpolate_preserves_bool(tmp_path):
    """Bool values are not stringified by interpolation."""
    result = _interpolate_output({"on": True, "off": False}, tmp_path)
    assert result["on"] is True
    assert result["off"] is False


def test_interpolate_replaces_output_in_string(tmp_path):
    result = _interpolate_output({"out": "$output/video.mp4"}, tmp_path)
    assert result["out"] == f"{tmp_path}/video.mp4"


def test_interpolate_replaces_output_in_list(tmp_path):
    result = _interpolate_output({"img": ["$output/img.jpg", 0, 0.8]}, tmp_path)
    assert result["img"] == [f"{tmp_path}/img.jpg", 0, 0.8]


def test_interpolate_non_output_unchanged(tmp_path):
    result = _interpolate_output({"config": "/etc/config.toml"}, tmp_path)
    assert result["config"] == "/etc/config.toml"


def test_interpolate_preserves_resolved_params(tmp_path):
    result = _interpolate_output(
        {"out": "$output/video.mp4", "prompt": "a cat", "seed": 42}, tmp_path
    )
    assert result["out"] == f"{tmp_path}/video.mp4"
    assert result["prompt"] == "a cat"
    assert result["seed"] == 42


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------


def test_build_basic_command():
    runner = Runner(
        command="python generate.py",
        params=[
            Param("prompt"),
            Param("seed", type="int", default=42),
        ],
    )
    cmd = runner.build_command({"prompt": "a cat", "seed": 42})
    assert cmd == [
        "python",
        "generate.py",
        "--prompt",
        "a cat",
        "--seed",
        "42",
    ]


def test_build_bool_flag():
    runner = _make_runner(params=[Param("turbo", type="bool")])
    cmd = runner.build_command({"turbo": True})
    assert cmd == ["echo", "hello", "--turbo"]


def test_build_bool_flag_false_omitted():
    runner = _make_runner(params=[Param("turbo", type="bool")])
    cmd = runner.build_command({"turbo": False})
    assert cmd == ["echo", "hello"]


def test_build_multi_value_flag():
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path-image", "float", "float"],
                labels=["path", "start", "strength"],
            ),
        ],
    )
    cmd = runner.build_command({"image": ["photo.jpg", 0, 0.8]})
    assert cmd == ["echo", "hello", "--image", "photo.jpg", "0", "0.8"]


def test_build_custom_flag():
    runner = _make_runner(params=[Param("x", flag="-x")])
    cmd = runner.build_command({"x": "val"})
    assert cmd == ["echo", "hello", "-x", "val"]


def test_build_command_as_list():
    runner = Runner(
        command=["python", "-u", "gen.py"],
        params=[Param("seed", type="int")],
    )
    cmd = runner.build_command({"seed": 42})
    assert cmd == ["python", "-u", "gen.py", "--seed", "42"]


def test_build_command_string_with_quotes():
    runner = Runner(command='echo "hello world"', params=[])
    cmd = runner.build_command({})
    assert cmd == ["echo", "hello world"]


def test_build_type_list_from_cli():
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path-image", "float", "float"],
                labels=["path", "start", "strength"],
            ),
        ],
    )
    r = runner.parse_cli(["--image", "photo.jpg", "0", "0.8"])
    cmd = runner.build_command(r.param_values)
    assert cmd == ["echo", "hello", "--image", "photo.jpg", "0.0", "0.8"]


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
    exit_code, duration, stdout_text, aborted = runner.execute(cmd, tmp_path)

    assert exit_code == 0
    assert aborted is False
    assert duration > 0
    assert "out" in stdout_text
    assert (tmp_path / "stdout.log").read_text().strip() == "out"
    assert "err" in (tmp_path / "stderr.log").read_text()
    run_log = (tmp_path / "run.log").read_text()
    assert "out" in run_log
    assert "[stderr] err" in run_log


def test_execute_nonzero_exit_code(tmp_path):
    runner = Runner(command="echo")
    exit_code, _, _, _ = runner.execute(
        [sys.executable, "-c", "import sys; sys.exit(42)"], tmp_path
    )
    assert exit_code == 42


def test_execute_env_vars_passed(tmp_path):
    runner = Runner(command="echo", env={"MY_TEST_VAR": "hello123"})
    cmd = [sys.executable, "-c", "import os; print(os.environ['MY_TEST_VAR'])"]
    exit_code, _, stdout_text, _ = runner.execute(cmd, tmp_path)
    assert exit_code == 0
    assert "hello123" in stdout_text


def test_execute_env_none_removes_var(tmp_path):
    runner = Runner(command="echo", env={"FOO": "bar", "PATH": None})
    cmd = [
        sys.executable,
        "-c",
        "import os,json; print(json.dumps("
        "{'FOO':os.environ.get('FOO'),'PATH':os.environ.get('PATH')}))",
    ]
    exit_code, _, stdout_text, _ = runner.execute(cmd, tmp_path)
    assert exit_code == 0
    result = json.loads(stdout_text.strip().splitlines()[-1])
    assert result["FOO"] == "bar"
    assert result["PATH"] is None


# ---------------------------------------------------------------------------
# run() kwargs and _merge_run_flags
# ---------------------------------------------------------------------------


def test_run_kwargs_override_defaults():
    """run(dry_run=True) works without CLI flags."""
    runner = Runner(
        command="python gen.py",
        params=[Param("prompt", default="test")],
        tags=["v1"],
    )
    with patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO):
        runner.run(dry_run=True, no_interactive=True)


def test_run_kwargs_warn_on_contradiction(caplog):
    """run() warns when kwargs contradict explicit CLI flags."""
    runner = _make_runner()
    r = runner.parse_cli(["--no-interactive"])
    with caplog.at_level("WARNING", logger="lite_runner"):
        flags = r.run_flags.merge(no_interactive=False)
    assert flags.no_interactive is False  # kwarg wins
    assert "no_interactive" in caplog.text


def test_run_kwargs_no_warn_on_default():
    """No warning when kwarg matches CLI default (not explicitly passed)."""
    runner = _make_runner()
    r = runner.parse_cli([])
    flags = r.run_flags.merge(no_interactive=True)
    assert flags.no_interactive is True  # kwarg wins, no warning


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------


def test_check_disk_space_passes_when_enough():
    runner = Runner(command="echo", params=[])
    # Should not raise — there's definitely more than 0.001 GiB free
    runner.check_disk_space(0.001)


def test_check_disk_space_raises_when_not_enough():
    runner = Runner(command="echo", params=[])
    with pytest.raises(OSError, match="Not enough free space"):
        runner.check_disk_space(999_999)


# ---------------------------------------------------------------------------
# Dry run (integration)
# ---------------------------------------------------------------------------


def test_dry_run_prints_command_no_wandb(caplog):
    runner = Runner(
        command="python gen.py",
        params=[
            Param("prompt"),
            Param("seed", type="int", default=42),
            Param("output-path", value="$output/video.mp4", type="path-video"),
        ],
        tags=["v1"],
    )
    r = runner.parse_cli(["--prompt", "test"])
    with (
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        caplog.at_level("INFO", logger="lite_runner"),
    ):
        r.run(dry_run=True, no_interactive=True)
    out = re.sub(r"\033\[[0-9;]*m", "", caplog.text)
    assert "dry_run" in out
    assert "--prompt test" in out
    assert "--seed 42" in out
    assert "$output/video.mp4" in out
    assert "Tags: ['v1']" in out


def test_no_project_raises_valueerror():
    """Runner with no wandb_project and no git repo raises ValueError."""
    runner = Runner(command="echo", params=[])
    with (
        patch("lite_runner.runner._collect_git_info", return_value={}),
        pytest.raises(ValueError, match="Cannot determine project name"),
    ):
        runner.run(no_interactive=True)


# ---------------------------------------------------------------------------
# Full run (integration with mocked W&B)
# ---------------------------------------------------------------------------


def test_full_run_with_mocked_wandb(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

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
        patch("lite_runner.backends.wandb", mock_wb),
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.run(no_interactive=True)

    mock_wb.init.assert_called_once()
    assert mock_wb.init.call_args[1]["project"] == "test-repo"
    assert mock_wb.init.call_args[1]["save_code"] is True
    assert mock_wb.init.call_args[1]["group"] is None
    assert wb_run.summary["status"] == "success"
    assert wb_run.summary["exit_code"] == 0
    assert wb_run.summary["duration_seconds"] > 0
    assert (tmp_path / "lite_runs" / "test-repo").exists()
    wb_run.finish.assert_called_once_with(exit_code=0)


def test_run_project_kwarg_flows_to_backend(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    runner = Runner(
        command=f"{sys.executable} -c \"print('ok')\"",
        params=[Param("seed", type="int", default=42)],
    )
    with (
        patch("lite_runner.backends.wandb", mock_wb),
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.run(no_interactive=True, project="custom-proj")

    assert mock_wb.init.call_args[1]["project"] == "custom-proj"


def test_full_run_explicit_group(tmp_path):
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    runner = Runner(
        command=f"{sys.executable} -c \"print('ok')\"",
        params=[],
        run_group="my-sweep",
    )

    with (
        patch("lite_runner.backends.wandb", mock_wb),
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.run(no_interactive=True)

    assert mock_wb.init.call_args[1]["group"] == "my-sweep"


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------


def test_git_info_returns_expected_keys():
    mock_repo = MagicMock()
    mock_repo.working_dir = "/home/user/lite-runner"
    mock_repo.head.commit.hexsha = "abc123def456"
    mock_repo.head.is_detached = False
    mock_repo.active_branch.name = "main"
    mock_repo.is_dirty.return_value = False
    with patch("lite_runner.runner.git.Repo", return_value=mock_repo):
        info = _collect_git_info()
    assert info == {
        "repo": "lite-runner",
        "commit": "abc123def456",
        "branch": "main",
        "dirty": False,
    }


def test_git_info_dirty_flag():
    mock_repo = MagicMock()
    mock_repo.working_dir = "/home/user/lite-runner"
    mock_repo.head.commit.hexsha = "abc123"
    mock_repo.head.is_detached = False
    mock_repo.active_branch.name = "main"
    mock_repo.is_dirty.return_value = True
    with patch("lite_runner.runner.git.Repo", return_value=mock_repo):
        assert _collect_git_info()["dirty"] is True


def test_git_info_empty_outside_repo():
    with patch(
        "lite_runner.runner.git.Repo",
        side_effect=git.InvalidGitRepositoryError,
    ):
        assert _collect_git_info() == {}


# ---------------------------------------------------------------------------
# --no-wandb flag
# ---------------------------------------------------------------------------


def test_no_wandb_flag_parsed():
    runner = _make_runner()
    r = runner.parse_cli(["--no-wandb"])
    assert r.run_flags.no_wandb is True


def test_no_wandb_flag_default():
    runner = _make_runner()
    r = runner.parse_cli([])
    assert bool(r.run_flags.no_wandb) is False


def test_full_run_no_wandb(tmp_path):
    """--no-wandb skips W&B but still runs command and writes run_info.json."""
    runner = Runner(
        command=(
            f"{sys.executable} -c \"import sys; print('hello'); print('x=42.0')\""
        ),
        params=[],
        metrics=[Metric("val", pattern=r"x=([\d.]+)")],
        tags=["v1"],
        run_group="test-group",
    )

    with (
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.run(no_wandb=True, no_interactive=True)

    # Find the output dir
    project_dir = tmp_path / "lite_runs" / "test-repo"
    assert project_dir.exists()
    run_dirs = list(project_dir.iterdir())
    assert len(run_dirs) == 1
    output_dir = run_dirs[0]
    assert "local" in output_dir.name

    # run_info.json should exist
    run_info_path = output_dir / "run_info.json"
    assert run_info_path.exists()
    run_info = json.loads(run_info_path.read_text())

    # Check config
    assert run_info["config"]["git/repo"] == "test-repo"
    assert run_info["config"]["meta/output_dir"] == str(output_dir)

    # Check metrics extracted
    assert run_info["metrics"]["val"] == 42.0

    # Check summary
    assert run_info["summary"]["status"] == "success"
    assert run_info["summary"]["exit_code"] == 0
    assert run_info["summary"]["duration_seconds"] > 0

    # Check tags and group
    assert "v1" in run_info["metadata"]["tags"]
    assert run_info["metadata"]["group"] == "test-group"

    # Logs should exist
    assert (output_dir / "stdout.log").exists()
    assert (output_dir / "run.log").exists()

    # wandb url should indicate no wandb
    assert run_info["config"]["wandb/url"] == "(no wandb)"


def test_full_run_with_wandb_also_writes_run_info(tmp_path):
    """Even with W&B enabled, run_info.json is written."""
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    runner = Runner(
        command=f"{sys.executable} -c \"print('ok')\"",
        params=[],
    )

    with (
        patch("lite_runner.backends.wandb", mock_wb),
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.run(no_interactive=True)

    # W&B should have been called
    mock_wb.init.assert_called_once()

    # run_info.json should also exist
    project_dir = tmp_path / "lite_runs" / "test-repo"
    run_dirs = list(project_dir.iterdir())
    assert len(run_dirs) == 1
    run_info_path = run_dirs[0] / "run_info.json"
    assert run_info_path.exists()
    run_info = json.loads(run_info_path.read_text())
    assert run_info["summary"]["status"] == "success"
    # wandb info should be in config
    assert run_info["config"]["wandb/url"] == "https://wandb.test/run"


def test_no_wandb_failed_run(tmp_path):
    """--no-wandb with a failing command records failure status and tags."""
    runner = Runner(
        command=f'{sys.executable} -c "import sys; sys.exit(1)"',
        params=[],
        tags=["v1"],
    )

    with (
        patch("lite_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
        pytest.raises(SystemExit, match="1"),
    ):
        runner.run(no_wandb=True, no_interactive=True)

    project_dir = tmp_path / "lite_runs" / "test-repo"
    run_dirs = list(project_dir.iterdir())
    output_dir = run_dirs[0]
    run_info = json.loads((output_dir / "run_info.json").read_text())

    assert run_info["summary"]["status"] == "failed"
    assert run_info["summary"]["exit_code"] == 1
    assert "failed" in run_info["metadata"]["tags"]


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------


def test_sources_cli():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"])
    assert r.param_sources["seed"] == "cli"


def test_sources_override():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=99)
    assert r.param_sources["seed"] == "override"


def test_sources_default():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.resolve_defaults()
    assert r.param_sources["seed"] == "default"


def test_sources_fixed():
    runner = _make_runner(params=[Param("out", value="/fixed")])
    r = runner.resolve_defaults()
    assert r.param_sources["out"] == "fixed"


def test_sources_prompt():
    runner = _make_runner(params=[Param("prompt")])
    with patch("lite_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        r = runner.ask_user()
    assert r.param_sources["prompt"] == "prompt"


def test_override_does_not_overwrite_on_parse_cli():
    """parse_cli() preserves earlier override values."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=777).parse_cli(["--seed", "10"])
    assert r.param_values["seed"] == 777
    assert r.param_sources["seed"] == "override"
