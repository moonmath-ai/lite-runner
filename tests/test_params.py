"""Tests for genai_runner.params."""

from unittest.mock import patch

import pytest
from conftest import _make_runner

from genai_runner import UNSET, Param
from genai_runner.params import _log_as_from_type

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
# Param
# ---------------------------------------------------------------------------


def test_param_dest_normalizes_hyphens():
    p = Param("my-param")
    assert p.dest == "my_param"


def test_param_dest_preserves_underscores():
    p = Param("my_param")
    assert p.dest == "my_param"


def test_param_flag_from_hyphens():
    p = Param("my-param")
    assert p.flag == "--my-param"


def test_param_flag_from_underscores():
    p = Param("my_param")
    assert p.flag == "--my-param"


def test_param_flag_explicit_override():
    p = Param("x", flag="-x")
    assert p.flag == "-x"


def test_param_is_fixed_with_value():
    p = Param("x", value="fixed")
    assert p.is_fixed


def test_param_is_fixed_without_value():
    p = Param("x")
    assert not p.is_fixed


def test_param_log_when_inferred_after():
    p = Param("out", type="path-video", value="$output/vid.mp4")
    assert p.log_when == "after"


def test_param_log_when_inferred_before():
    p = Param("input", type="path-video")
    assert p.log_when == "before"


def test_param_log_when_explicit_overrides():
    p = Param(
        "input",
        type="path-video",
        value="$output/vid.mp4",
        log_when="before",
    )
    assert p.log_when == "before"


def test_param_log_when_list_value():
    p = Param(
        "img",
        type=["path-image", "float", "float"],
        value=["$output/img.jpg", 0, 0.8],
        labels=["path", "start", "strength"],
    )
    assert p.log_when == "after"


def test_param_log_when_none_without_upload_type():
    p = Param("seed", type="int", default=42)
    assert p.log_when is None


def test_param_log_when_none_plain_path():
    """path (no sub-type) has no upload intent, so log_when stays None."""
    p = Param("config", type="path")
    assert p.log_when is None


def test_param_type_list_infers_nargs():
    p = Param(
        "image",
        type=["path-image", "float", "float"],
        labels=["path", "start", "strength"],
    )
    assert p.nargs == 3
    assert p.type_list == ["path-image", "float", "float"]
    kwargs = p.argparse_kwargs()
    assert kwargs["nargs"] == 3
    assert kwargs["type"] is str


def test_param_nargs_none_without_type_list():
    p = Param("prompt")
    assert p.nargs is None


def test_param_type_list_single():
    p = Param("seed", type="int")
    assert p.type_list == ["int"]


def test_param_type_list_multi():
    p = Param("x", type=["float", "float"])
    assert p.type_list == ["float", "float"]


def test_param_log_when_multi_value_non_first_path():
    p = Param("x", type=["float", "path-image"], labels=["val", "img"])
    assert p.log_when == "before"


def test_param_log_when_multi_value_non_first_path_output():
    p = Param(
        "x",
        type=["float", "path-image"],
        value=[0.5, "$output/img.jpg"],
        labels=["val", "img"],
    )
    assert p.log_when == "after"


def test_unknown_param_type_raises():
    with pytest.raises(ValueError, match="Unknown param type"):
        Param("x", type="badtype")


def test_unknown_param_type_in_list_raises():
    with pytest.raises(ValueError, match="Unknown param type"):
        Param("x", type=["float", "badtype"])


def test_bool_in_type_list_raises():
    with pytest.raises(ValueError, match=r"bool.*cannot.*multi-value"):
        Param("x", type=["bool", "float"])


def test_no_prompt_requires_default():
    with pytest.raises(ValueError, match="requires a default"):
        Param("x", prompt=False)


def test_no_prompt_with_default_ok():
    p = Param("x", prompt=False, default="val")
    assert p.prompt is False


# ---------------------------------------------------------------------------
# Bool param prompting
# ---------------------------------------------------------------------------


def test_ask_user_prompts_bool_param():
    """Bool params are prompted via questionary.confirm()."""
    runner = _make_runner(params=[Param("turbo", type="bool")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.confirm.return_value.ask.return_value = True
        r = runner.ask_user()
    mock_q.confirm.assert_called_once_with("turbo:", default=False)
    assert r.param_values["turbo"] is True
    assert r.param_sources["turbo"] == "prompt"


def test_ask_user_bool_false():
    """Bool param answered False is logged."""
    runner = _make_runner(params=[Param("turbo", type="bool")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.confirm.return_value.ask.return_value = False
        r = runner.ask_user()
    assert r.param_values["turbo"] is False


def test_ask_user_bool_skips_cli_provided():
    """Bool param set via CLI is not re-prompted."""
    runner = _make_runner(params=[Param("turbo", type="bool")])
    r = runner.parse_cli(["--turbo"])
    with patch("genai_runner.params.questionary") as mock_q:
        r = r.ask_user()
    mock_q.confirm.assert_not_called()
    assert r.param_values["turbo"] is True


def test_ask_user_bool_cancel_exits():
    """Cancelling a bool prompt exits."""
    runner = _make_runner(params=[Param("turbo", type="bool")])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.confirm.return_value.ask.return_value = None
        with pytest.raises(SystemExit, match="1"):
            runner.ask_user()


def test_bool_param_logged_in_config():
    """Bool param value appears in run config (via JsonBackend)."""
    runner = _make_runner(params=[Param("turbo", type="bool")])
    r = runner.parse_cli(["--turbo"])
    r = r.resolve_defaults()
    assert r.param_values["turbo"] is True


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
    """Typing '-' at a select prompt returns UNSET."""
    runner = _make_runner(params=[Param("mode", choices=["fast", "slow"])])
    with patch("genai_runner.params.questionary") as mock_q:
        mock_q.select.return_value.ask.return_value = "-"
        r = runner.ask_user()
    assert r.param_values["mode"] is UNSET


def test_skip_select_includes_dash_in_choices():
    """Select prompt includes '-' as the first choice."""
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
