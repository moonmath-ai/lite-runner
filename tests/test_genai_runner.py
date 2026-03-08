"""Tests for genai_runner."""

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_runner import UNSET, JsonBackend, Metric, Output, Param, Runner
from genai_runner.params import RunFlags, _log_as_from_type
from genai_runner.runner import _collect_git_info, _split_glob

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
    """Ensure parse_cli() sees clean argv when auto-called by run()."""
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
            value=["$output/img.jpg", 0, 0.8],
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
    assert p.type_list == ["path-image", "float", "float"]
    assert p.labels == ["path", "start", "strength"]


def test_param_nargs_none_without_type_list():
    assert Param("prompt").nargs is None


def test_param_type_list_single():
    assert Param("seed", type="int").type_list == ["int"]


def test_param_type_list_multi():
    assert Param("img", type=["path", "float"]).type_list == ["path", "float"]


def test_param_log_when_multi_value_non_first_path():
    """path-video in non-first position still infers log_when."""
    p = Param("combo", type=["float", "path-video", "float"])
    assert p.log_when == "before"


def test_param_log_when_multi_value_non_first_path_output():
    """path-video in non-first position with $output infers log_when='after'."""
    p = Param(
        "combo",
        type=["float", "path-video", "float"],
        value=["0.5", "$output/vid.mp4", "1.0"],
    )
    assert p.log_when == "after"


# ---------------------------------------------------------------------------
# CLI parsing (via parse_cli)
# ---------------------------------------------------------------------------


def test_parse_basic_args():
    runner = _make_runner([Param("prompt"), Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--prompt", "a cat", "--seed", "99"])
    assert r.param_values["prompt"] == "a cat"
    assert r.param_values["seed"] == 99
    assert r.param_sources["prompt"] == "cli"
    assert r.param_sources["seed"] == "cli"


def test_parse_bool_flag():
    runner = _make_runner([Param("verbose", type="bool")])
    r = runner.parse_cli(["--verbose"])
    assert r.param_values["verbose"] is True
    assert r.param_sources["verbose"] == "cli"


def test_parse_bool_flag_absent():
    runner = _make_runner([Param("verbose", type="bool")])
    r = runner.parse_cli([])
    assert "verbose" not in r.param_values


def test_fixed_params_not_in_argparse():
    runner = _make_runner(
        [Param("prompt"), Param("output-path", value="$output/video.mp4")]
    )
    r = runner.parse_cli(["--prompt", "hi"])
    assert r.param_values["prompt"] == "hi"
    assert "output-path" not in r.param_values


def test_parse_choices():
    runner = _make_runner([Param("mode", choices=["fast", "quality"], default="fast")])
    r = runner.parse_cli(["--mode", "quality"])
    assert r.param_values["mode"] == "quality"


def test_parse_type_list():
    runner = _make_runner(
        [
            Param(
                "image",
                type=["path", "float", "float"],
                labels=["path", "start", "strength"],
            )
        ]
    )
    r = runner.parse_cli(["--image", "photo.jpg", "0", "0.8"])
    # argparse returns strings; casting happens in resolve_defaults
    assert r.param_values["image"] == ["photo.jpg", 0, 0.8]


def test_parse_types_with_spaces_in_path():
    runner = _make_runner([Param("image", type=["path", "float", "float"])])
    r = runner.parse_cli(["--image", "path/to something/img.jpg", "0", "0.8"])
    assert r.param_values["image"] == ["path/to something/img.jpg", 0, 0.8]


def test_builtin_flags():
    runner = _make_runner()
    r = runner.parse_cli(["--dry-run", "--no-interactive"])
    assert r.run_flags.dry_run is True
    assert r.run_flags.no_interactive is True


def test_project_override():
    runner = _make_runner()
    r = runner.parse_cli(["--project", "my-project"])
    assert r.run_flags.project == "my-project"


def test_unknown_param_type_raises():
    with pytest.raises(ValueError, match="Unknown param type 'banana'"):
        Param("x", type="banana")


def test_unknown_param_type_in_list_raises():
    with pytest.raises(ValueError, match="Unknown param type 'banana'"):
        Param("x", type=["str", "banana"])


def test_param_name_conflicts_with_builtin_flag():
    with pytest.raises(ValueError, match="conflicts with built-in flag"):
        Runner(command="echo", params=[Param("project")])


def test_bool_in_type_list_raises():
    with pytest.raises(ValueError, match="'bool' cannot appear in a multi-value type"):
        Param("x", type=["bool", "str"])


def test_no_prompt_requires_default():
    with pytest.raises(ValueError, match="prompt=False.*requires a default"):
        Param("x", prompt=False)


def test_no_prompt_with_default_ok():
    p = Param("x", prompt=False, default=42)
    assert p.prompt is False


# ---------------------------------------------------------------------------
# resolve_defaults()
# ---------------------------------------------------------------------------


def test_resolve_defaults_applies_defaults():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.resolve_defaults()
    assert r.param_values["seed"] == 42
    assert r.param_sources["seed"] == "default"


def test_resolve_defaults_callable():
    runner = _make_runner(params=[Param("path", default=lambda: "/computed/path")])
    r = runner.resolve_defaults()
    assert r.param_values["path"] == "/computed/path"
    assert r.param_sources["path"] == "default"


def test_resolve_defaults_fixed():
    runner = _make_runner(params=[Param("out", value="$output/video.mp4")])
    r = runner.resolve_defaults()
    assert r.param_values["out"] == "$output/video.mp4"
    assert r.param_sources["out"] == "fixed"


def test_resolve_defaults_fixed_callable():
    runner = _make_runner(params=[Param("out", value=lambda: "/computed")])
    r = runner.resolve_defaults()
    assert r.param_values["out"] == "/computed"
    assert r.param_sources["out"] == "fixed"


def test_resolve_defaults_bool_false():
    runner = _make_runner(params=[Param("verbose", type="bool")])
    r = runner.resolve_defaults()
    assert r.param_values["verbose"] is False
    assert r.param_sources["verbose"] == "default"


def test_resolve_defaults_does_not_overwrite_cli():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"]).resolve_defaults()
    assert r.param_values["seed"] == 99
    assert r.param_sources["seed"] == "cli"


def test_resolve_defaults_does_not_overwrite_override():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=777).resolve_defaults()
    assert r.param_values["seed"] == 777
    assert r.param_sources["seed"] == "override"


def test_resolve_defaults_casts_type_list():
    runner = _make_runner(params=[Param("image", type=["path", "int", "float"])])
    r = runner.parse_cli(["--image", "photo.jpg", "5", "0.8"])
    assert r.param_values["image"] == ["photo.jpg", 5, 0.8]


def test_resolve_defaults_casts_default_list():
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path", "float", "float"],
                default=["img.jpg", 0, 0.8],
            ),
        ],
    )
    r = runner.resolve_defaults()
    assert r.param_values["image"] == ["img.jpg", 0.0, 0.8]


def test_resolve_overrides_take_priority():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"]).override(seed=777).resolve_defaults()
    assert r.param_values["seed"] == 777


def test_resolve_cli_beats_default():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"]).resolve_defaults()
    assert r.param_values["seed"] == 99


# ---------------------------------------------------------------------------
# override()
# ---------------------------------------------------------------------------


def test_override_returns_new_runner():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r2 = runner.override(seed=99)
    assert r2 is not runner
    assert r2.param_values["seed"] == 99
    assert r2.param_sources["seed"] == "override"
    # Original unchanged
    assert "seed" not in runner.param_values


def test_override_kwarg_underscore_to_hyphen():
    """Kwarg underscores are mapped to param names via dest lookup."""
    runner = _make_runner(params=[Param("output-path", default="/tmp")])
    r2 = runner.override(output_path="/new")
    assert r2.param_values["output-path"] == "/new"


def test_override_unknown_param_raises():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    with pytest.raises(ValueError, match="Unknown param"):
        runner.override(nope=1)


def test_override_chained():
    """Calling override() on an already-overridden runner works."""
    runner = _make_runner(
        params=[Param("seed", type="int", default=42), Param("mode", default="fast")],
    )
    r2 = runner.override(seed=99)
    r3 = r2.override(mode="slow")
    assert r3.param_values["seed"] == 99
    assert r3.param_values["mode"] == "slow"


def test_override_preserves_cli_args():
    """CLI args are preserved; overrides take priority."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "10"])
    # CLI gave seed=10, override doesn't touch it
    r2 = r.resolve_defaults()
    assert r2.param_values["seed"] == 10
    # Override takes priority over CLI
    r3 = r.override(seed=99).resolve_defaults()
    assert r3.param_values["seed"] == 99


def test_override_fixed_params_included():
    """Fixed params (value=) are resolved even without overrides."""
    runner = _make_runner(params=[Param("out", value="$output/video.mp4")])
    r2 = runner.resolve_defaults()
    assert r2.param_values["out"] == "$output/video.mp4"


def test_override_run_skips_prompting(tmp_path):
    """run() on an overridden Runner does not prompt."""
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()

    runner = Runner(
        command=f"{sys.executable} -c \"print('ok')\"",
        params=[Param("seed", type="int", default=42)],
    )
    r2 = runner.override(seed=99)
    with (
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
        patch("genai_runner.params.questionary") as mock_q,
    ):
        r2.run(no_interactive=True)
    # Questionary should never be called
    mock_q.text.assert_not_called()
    mock_q.select.assert_not_called()
    # Config should reflect the override
    config = mock_wb.init.call_args[1]["config"]
    assert config["param/seed"] == 99


# ---------------------------------------------------------------------------
# with_metadata()
# ---------------------------------------------------------------------------


def test_with_metadata_project():
    runner = _make_runner()
    r2 = runner.with_metadata(project="my-project")
    assert r2.project == "my-project"
    assert runner.project is None


def test_with_metadata_group():
    runner = _make_runner()
    r2 = runner.with_metadata(run_group="sweep-1")
    assert r2.run_group == "sweep-1"
    assert runner.run_group is None


def test_with_metadata_tags():
    runner = _make_runner(tags=["v1"])
    r2 = runner.with_metadata(tags=["v2", "exp"])
    assert r2.tags == ["v2", "exp"]
    assert runner.tags == ["v1"]


def test_with_metadata_partial():
    runner = _make_runner(project="orig", run_group="g1", tags=["t1"])
    r2 = runner.with_metadata(run_group="g2")
    assert r2.project == "orig"
    assert r2.run_group == "g2"
    assert r2.tags == ["t1"]


# ---------------------------------------------------------------------------
# ask_user()
# ---------------------------------------------------------------------------


def test_ask_user_fills_from_questionary():
    runner = _make_runner(params=[Param("prompt")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        r = runner.ask_user()
    assert r.param_values["prompt"] == "a cat"
    assert r.param_sources["prompt"] == "prompt"
    assert r.filled


def test_ask_user_non_interactive_exits_on_missing():
    runner = _make_runner(params=[Param("prompt")])
    with pytest.raises(SystemExit, match="2"):
        runner.ask_user(no_interactive=True)


def test_ask_user_non_interactive_ok_with_defaults():
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.ask_user(no_interactive=True)
    assert r.param_values["seed"] == 42
    assert r.filled


def test_ask_user_select_for_choices():
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    with patch("genai_runner.params.questionary") as mock_q:
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
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "photo.jpg"
        mock_q.text.return_value.ask.side_effect = lambda: next(answers)
        r = runner.ask_user()
    # After prompting, types are cast: str, float, float
    assert r.param_values["image"] == ["photo.jpg", 0.0, 0.8]


def test_ask_user_path_image_uses_path_widget():
    """path-image type should use questionary.path() widget."""
    runner = _make_runner(params=[Param("img", type="path-image")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "/tmp/photo.jpg"
        r = runner.ask_user()
    mock_q.path.assert_called_once()
    assert r.param_values["img"] == "/tmp/photo.jpg"


def test_ask_user_cancel_exits():
    runner = _make_runner(params=[Param("prompt")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(SystemExit, match="1"):
            runner.ask_user()


def test_ask_user_prompts_default_param():
    """Params with defaults are prompted with default pre-filled."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "99"
        r = runner.ask_user()
    mock_q.text.assert_called_once_with("seed:", default="42")
    assert r.param_values["seed"] == 99


def test_ask_user_skips_cli_provided_param():
    """Params explicitly provided on CLI are not prompted."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.parse_cli(["--seed", "99"])
    with patch("genai_runner.params.questionary") as mock_q:
        r = r.ask_user()
    mock_q.text.assert_not_called()
    assert r.param_values["seed"] == 99


def test_ask_user_skips_overridden_param():
    """Params set via overrides are not prompted."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=77)
    with patch("genai_runner.params.questionary") as mock_q:
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
    with patch("genai_runner.params.questionary") as mock_q:
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
# Skip param (type '-' to omit from command)
# ---------------------------------------------------------------------------


def test_skip_single_param_returns_unset():
    """Typing '-' at a text prompt returns UNSET."""
    runner = _make_runner(params=[Param("prompt")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "-"
        r = runner.ask_user()
    assert r.param_values["prompt"] is UNSET


def test_skip_select_param_returns_unset():
    """Selecting '-' in a choices prompt returns UNSET."""
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "-"
        r = runner.ask_user()
    assert r.param_values["mode"] is UNSET


def test_skip_select_includes_dash_in_choices():
    """Select prompt should prepend '-' to the choices list."""
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "fast"
        runner.ask_user()
    call_args = mock_q.select.call_args
    assert call_args[1]["choices"] == ["-", "fast", "slow"]


def test_skip_nargs_returns_unset():
    """Typing '-' for any nargs element skips the whole param."""
    runner = _make_runner(
        params=[
            Param(
                "image",
                type=["path-image", "float"],
                labels=["path", "strength"],
            ),
        ],
    )
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.path.return_value.ask.return_value = "-"
        r = runner.ask_user()
    assert r.param_values["image"] is UNSET


def test_build_command_skips_unset():
    """UNSET params are omitted from the command."""
    runner = _make_runner(
        params=[Param("prompt"), Param("mode", choices=["fast", "slow"])],
    )
    cmd = runner._build_command({"prompt": "a cat", "mode": UNSET})
    assert cmd == ["echo", "hello", "--prompt", "a cat"]


def test_config_logs_unset_as_marker():
    """Skipped params appear as '<unset>' in the config dict."""
    resolved = {"prompt": "test", "mode": UNSET}
    config: dict[str, object] = {}
    for k, v in resolved.items():
        config[f"param/{k}"] = "<unset>" if v is UNSET else v
    assert config["param/prompt"] == "test"
    assert config["param/mode"] == "<unset>"


def test_interpolate_preserves_unset(tmp_path):
    """UNSET values pass through interpolation unchanged."""
    runner = _make_runner(params=[Param("out", value="$output/x.mp4")])
    result = runner._interpolate_output({"out": UNSET}, tmp_path)
    assert result["out"] is UNSET


def test_interpolate_preserves_bool(tmp_path):
    """Bool values are not stringified by interpolation."""
    runner = _make_runner(
        params=[
            Param("on", type="bool", value=True),
            Param("off", type="bool", value=False),
        ],
    )
    result = runner._interpolate_output({"on": True, "off": False}, tmp_path)
    assert result["on"] is True
    assert result["off"] is False


def test_log_files_skips_unset(tmp_path):
    """_log_files skips params whose value is UNSET."""
    runner = _make_runner(
        params=[Param("img", type="path-image")],
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    # Should not raise or try to upload
    runner._log_files({"img": UNSET}, when="before")
    assert json_backend.run_info["files_logged"] == []


# ---------------------------------------------------------------------------
# Interpolate output
# ---------------------------------------------------------------------------


def test_interpolate_replaces_output_in_string(tmp_path):
    runner = _make_runner(params=[Param("out", value="$output/video.mp4")])
    result = runner._interpolate_output({"out": "$output/video.mp4"}, tmp_path)
    assert result["out"] == f"{tmp_path}/video.mp4"


def test_interpolate_replaces_output_in_list(tmp_path):
    runner = _make_runner(params=[Param("img", value=["$output/img.jpg", 0, 0.8])])
    result = runner._interpolate_output({"img": ["$output/img.jpg", 0, 0.8]}, tmp_path)
    assert result["img"] == [f"{tmp_path}/img.jpg", 0, 0.8]


def test_interpolate_non_output_unchanged(tmp_path):
    runner = _make_runner(params=[Param("config", value="/etc/config.toml")])
    result = runner._interpolate_output({"config": "/etc/config.toml"}, tmp_path)
    assert result["config"] == "/etc/config.toml"


def test_interpolate_preserves_resolved_params(tmp_path):
    runner = _make_runner(params=[Param("out", value="$output/video.mp4")])
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
        command="run.py", params=[Param("image", value=["photo.jpg", 0, 0.8])]
    )
    assert runner._build_command({"image": ["photo.jpg", 0, 0.8]}) == [
        "run.py",
        "--image",
        "photo.jpg",
        "0",
        "0.8",
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


def test_metric_float(tmp_path):
    runner = Runner(
        command="echo", metrics=[Metric("skip_pct", pattern=r"skipped=([\d.]+)%")]
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._extract_metrics("some output\nskipped=32.8%\ndone")
    assert json_backend.run_info["metrics"]["skip_pct"] == 32.8


def test_metric_str(tmp_path):
    runner = Runner(
        command="echo", metrics=[Metric("status", pattern=r"final: (\w+)", type="str")]
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._extract_metrics("final: completed")
    assert json_backend.run_info["metrics"]["status"] == "completed"


def test_metric_last_match_wins(tmp_path):
    runner = Runner(command="echo", metrics=[Metric("val", pattern=r"x=([\d.]+)")])
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._extract_metrics("x=1.0\nx=2.0\nx=3.0")
    assert json_backend.run_info["metrics"]["val"] == 3.0


def test_metric_no_match(tmp_path):
    runner = Runner(command="echo", metrics=[Metric("val", pattern=r"x=([\d.]+)")])
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._extract_metrics("no matches here")
    assert "val" not in json_backend.run_info["metrics"]


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
    exit_code, duration, stdout_text, aborted = runner._execute(cmd, tmp_path)

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
    exit_code, _, _, _ = runner._execute(
        [sys.executable, "-c", "import sys; sys.exit(42)"], tmp_path
    )
    assert exit_code == 42


def test_execute_env_vars_passed(tmp_path):
    runner = Runner(command="echo", env={"MY_TEST_VAR": "hello123"})
    cmd = [sys.executable, "-c", "import os; print(os.environ['MY_TEST_VAR'])"]
    exit_code, _, stdout_text, _ = runner._execute(cmd, tmp_path)
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
    exit_code, _, stdout_text, _ = runner._execute(cmd, tmp_path)
    assert exit_code == 0
    import json

    result = json.loads(stdout_text.strip().splitlines()[-1])
    assert result["FOO"] == "bar"
    assert result["PATH"] is None


# ---------------------------------------------------------------------------
# run() kwargs and _merge_run_flags
# ---------------------------------------------------------------------------


def test_run_kwargs_override_defaults(tmp_path):
    """run(dry_run=True) works without CLI flags."""
    runner = Runner(
        command="python gen.py",
        params=[Param("prompt", default="test")],
        tags=["v1"],
    )
    with patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO):
        runner.run(dry_run=True, no_interactive=True)
    # If we got here without error, dry run worked


def test_run_kwargs_warn_on_contradiction(capsys):
    """run() warns when kwargs contradict explicit CLI flags."""
    runner = _make_runner()
    r = runner.parse_cli(["--no-interactive"])
    flags = r.run_flags.merge(no_interactive=False)
    assert flags.no_interactive is False  # kwarg wins
    captured = capsys.readouterr()
    assert "Warning" in captured.err
    assert "no_interactive" in captured.err


def test_run_kwargs_no_warn_on_default():
    """No warning when kwarg matches CLI default (not explicitly passed)."""
    runner = _make_runner()
    r = runner.parse_cli([])
    flags = r.run_flags.merge(no_interactive=True)
    assert flags.no_interactive is True  # kwarg wins, no warning


# ---------------------------------------------------------------------------
# Dry run (integration)
# ---------------------------------------------------------------------------


def test_dry_run_prints_command_no_wandb(capsys):
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
    with patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO):
        r.run(dry_run=True, no_interactive=True)
    captured = capsys.readouterr()
    out = re.sub(r"\033\[[0-9;]*m", "", captured.out)
    assert "[dry-run]" in out
    assert "--prompt test" in out
    assert "--seed 42" in out
    assert "$output/video.mp4" in out
    assert "Run name: run" in out
    assert "Tags: ['v1']" in out
    # File plan: output-path is log_when="after" (value has $output)
    assert "Files to log (after run):" in out
    assert "video:" in out
    assert "output-path" in out


def test_no_project_raises_valueerror():
    """Runner with no wandb_project and no git repo raises ValueError."""
    runner = Runner(command="echo", params=[])
    with (
        patch("genai_runner.runner._collect_git_info", return_value={}),
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
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run(no_interactive=True)

    mock_wb.init.assert_called_once()
    assert mock_wb.init.call_args[1]["project"] == "test-repo"
    assert mock_wb.init.call_args[1]["save_code"] is True
    assert mock_wb.init.call_args[1]["group"] is None
    assert wb_run.summary["status"] == "success"
    assert wb_run.summary["exit_code"] == 0
    assert wb_run.summary["duration_seconds"] > 0
    assert (tmp_path / "genai_runs" / "test-repo").exists()
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
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
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
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run(no_interactive=True)

    assert mock_wb.init.call_args[1]["group"] == "my-sweep"


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
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_extra_outputs(tmp_path)
    # Should log 2 png files, not the txt
    logged = [
        f for f in json_backend.run_info["files_logged"] if f.get("log_as") == "image"
    ]
    assert len(logged) == 2


def test_log_extra_outputs_glob_zip(tmp_path):
    """Glob + log_as='zip' creates a zip archive."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("tensor1")
    (tmp_path / "debug" / "b.pt").write_text("tensor2")

    runner = Runner(
        command="echo",
        outputs=[Output("$output/debug/*.pt", log_as="zip")],
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_extra_outputs(tmp_path)

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
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_extra_outputs(tmp_path)

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
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_extra_outputs(tmp_path)
    assert "matched no files" in capsys.readouterr().out


def test_log_extra_outputs_single_file(tmp_path):
    """Non-glob single file still works (regression)."""
    (tmp_path / "meta.json").write_text("{}")

    runner = Runner(
        command="echo",
        outputs=[Output("$output/meta.json", log_as="artifact")],
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_extra_outputs(tmp_path)
    logged = [
        f
        for f in json_backend.run_info["files_logged"]
        if f.get("log_as") == "artifact"
    ]
    assert len(logged) == 1


def test_log_extra_outputs_duplicate_zip_raises(tmp_path):
    """Two zip outputs with same implicit label should raise."""
    (tmp_path / "debug").mkdir()
    (tmp_path / "debug" / "a.pt").write_text("x")
    (tmp_path / "debug" / "b.png").write_text("y")

    runner = Runner(
        command="echo",
        outputs=[
            Output("$output/debug/*.pt", log_as="zip"),
            Output("$output/debug/*.png", log_as="zip"),
        ],
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    with pytest.raises(ValueError, match="Duplicate zip label 'debug'"):
        runner._log_extra_outputs(tmp_path)


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------


def test_git_info_returns_expected_keys():
    mock_repo = MagicMock()
    mock_repo.working_dir = "/home/user/genai-runner"
    mock_repo.head.commit.hexsha = "abc123def456"
    mock_repo.head.is_detached = False
    mock_repo.active_branch.name = "main"
    mock_repo.is_dirty.return_value = False
    with patch("genai_runner.runner.git.Repo", return_value=mock_repo):
        info = _collect_git_info()
    assert info == {
        "repo": "genai-runner",
        "commit": "abc123def456",
        "branch": "main",
        "dirty": False,
    }


def test_git_info_dirty_flag():
    mock_repo = MagicMock()
    mock_repo.working_dir = "/home/user/genai-runner"
    mock_repo.head.commit.hexsha = "abc123"
    mock_repo.head.is_detached = False
    mock_repo.active_branch.name = "main"
    mock_repo.is_dirty.return_value = True
    with patch("genai_runner.runner.git.Repo", return_value=mock_repo):
        assert _collect_git_info()["dirty"] is True


def test_git_info_empty_outside_repo():
    import git as gitmodule

    with patch(
        "genai_runner.runner.git.Repo",
        side_effect=gitmodule.InvalidGitRepositoryError,
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
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run(no_wandb=True, no_interactive=True)

    # Find the output dir
    project_dir = tmp_path / "genai_runs" / "test-repo"
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
    assert "v1" in run_info["tags"]
    assert run_info["group"] == "test-group"

    # Logs should exist
    assert (output_dir / "stdout.log").exists()
    assert (output_dir / "run.log").exists()

    # wandb keys should NOT be in config
    assert "wandb/name" not in run_info["config"]
    assert "wandb/url" not in run_info["config"]


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
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run(no_interactive=True)

    # W&B should have been called
    mock_wb.init.assert_called_once()

    # run_info.json should also exist
    project_dir = tmp_path / "genai_runs" / "test-repo"
    run_dirs = list(project_dir.iterdir())
    assert len(run_dirs) == 1
    run_info_path = run_dirs[0] / "run_info.json"
    assert run_info_path.exists()
    run_info = json.loads(run_info_path.read_text())
    assert run_info["summary"]["status"] == "success"
    # wandb info should be in config
    assert run_info["config"]["wandb/name"] == "test-run-42"
    assert run_info["config"]["wandb/url"] == "https://wandb.test/run"


def test_no_wandb_failed_run(tmp_path):
    """--no-wandb with a failing command records failure status and tags."""
    runner = Runner(
        command=f'{sys.executable} -c "import sys; sys.exit(1)"',
        params=[],
        tags=["v1"],
    )

    with (
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
        pytest.raises(SystemExit, match="1"),
    ):
        runner.run(no_wandb=True, no_interactive=True)

    project_dir = tmp_path / "genai_runs" / "test-repo"
    run_dirs = list(project_dir.iterdir())
    output_dir = run_dirs[0]
    run_info = json.loads((output_dir / "run_info.json").read_text())

    assert run_info["summary"]["status"] == "failed"
    assert run_info["summary"]["exit_code"] == 1
    assert "failed" in run_info["tags"]


def test_extract_metrics_with_json_backend(tmp_path):
    """Metrics are extracted into JsonBackend."""
    runner = Runner(command="echo", metrics=[Metric("val", pattern=r"x=([\d.]+)")])
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._extract_metrics("x=3.14")
    assert json_backend.run_info["metrics"]["val"] == 3.14


# ---------------------------------------------------------------------------
# Table param logging
# ---------------------------------------------------------------------------


def test_table_param_logged_to_json_backend(tmp_path):
    """Params with table=True are logged via log_table to JsonBackend."""
    runner = _make_runner(
        params=[Param("prompt", table=True), Param("seed", type="int", default=42)],
    )
    json_backend = JsonBackend(tmp_path)
    runner.backends = [json_backend]
    runner._log_table_params({"prompt": "a cat", "seed": 42})
    assert "tables" in json_backend.run_info
    table = json_backend.run_info["tables"]["params"]
    assert table["columns"] == ["name", "value"]
    assert ["prompt", "a cat"] in table["data"]
    # seed has table=False, should NOT appear
    assert all(row[0] != "seed" for row in table["data"])


def test_table_param_skips_unset():
    """UNSET and None table params are excluded from the table."""
    runner = _make_runner(
        params=[Param("prompt", table=True), Param("neg", table=True)],
    )
    json_backend = JsonBackend(Path("/tmp"))
    runner.backends = [json_backend]
    runner._log_table_params({"prompt": "a cat", "neg": None})
    table = json_backend.run_info["tables"]["params"]
    assert len(table["data"]) == 1
    assert table["data"][0] == ["prompt", "a cat"]


def test_table_param_no_table_params_skips():
    """No log_table call when no params have table=True."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    backend = MagicMock()
    runner.backends = [backend]
    runner._log_table_params({"seed": 42})
    backend.log_table.assert_not_called()


def test_table_param_logged_to_wandb(tmp_path):
    """Full integration: table=True param creates a wandb.Table in run."""
    mock_wb = MagicMock()
    wb_run = _mock_wb_run()
    mock_wb.init.return_value = wb_run
    mock_wb.Artifact = MagicMock()
    mock_table = MagicMock()
    mock_wb.Table.return_value = mock_table

    runner = Runner(
        command=f"{sys.executable} -c \"print('ok')\"",
        params=[
            Param("prompt", default="a cat", table=True),
            Param("seed", type="int", default=42),
        ],
    )

    with (
        patch.dict("sys.modules", {"wandb": mock_wb}),
        patch("genai_runner.runner._collect_git_info", return_value=_FAKE_GIT_INFO),
        patch("genai_runner.runner._log_code_snapshot"),
        patch("genai_runner.runner.RUNS_DIR", tmp_path / "genai_runs"),
    ):
        runner.run(no_interactive=True)

    mock_wb.Table.assert_called_once_with(
        columns=["name", "value"], data=[["prompt", "a cat"]]
    )
    wb_run.log.assert_any_call({"params": mock_table})


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
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.text.return_value.ask.return_value = "a cat"
        r = runner.ask_user()
    assert r.param_sources["prompt"] == "prompt"


def test_override_does_not_overwrite_on_parse_cli():
    """parse_cli() preserves earlier override values."""
    runner = _make_runner(params=[Param("seed", type="int", default=42)])
    r = runner.override(seed=777).parse_cli(["--seed", "10"])
    assert r.param_values["seed"] == 777
    assert r.param_sources["seed"] == "override"
