"""Tests for genai_runner."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_runner import Metric, Output, Param, Runner, _collect_git_info


# ---------------------------------------------------------------------------
# Param tests
# ---------------------------------------------------------------------------

class TestParam:
    def test_dest_from_name_with_hyphens(self):
        p = Param("debug-output")
        assert p.dest == "debug_output"

    def test_dest_from_name_with_underscores(self):
        p = Param("debug_output")
        assert p.dest == "debug_output"

    def test_flag_auto_generated_from_hyphens(self):
        p = Param("debug-output")
        assert p.flag == "--debug-output"

    def test_flag_auto_generated_from_underscores(self):
        p = Param("output_path")
        assert p.flag == "--output-path"

    def test_flag_explicit_override(self):
        p = Param("out", flag="-o")
        assert p.flag == "-o"

    def test_is_fixed_with_value(self):
        p = Param("out", value="$output/video.mp4")
        assert p.is_fixed is True

    def test_is_fixed_without_value(self):
        p = Param("prompt")
        assert p.is_fixed is False

    def test_needs_prompt_no_default(self):
        p = Param("prompt")
        assert p.needs_prompt is True

    def test_needs_prompt_with_default(self):
        p = Param("seed", default=42)
        assert p.needs_prompt is False

    def test_needs_prompt_fixed(self):
        p = Param("out", value="$output/x.mp4")
        assert p.needs_prompt is False

    def test_log_when_auto_inferred_after(self):
        p = Param("out", value="$output/video.mp4", log_as="video")
        assert p.log_when == "after"

    def test_log_when_auto_inferred_before(self):
        p = Param("image", type="path", log_as="image")
        assert p.log_when == "before"

    def test_log_when_explicit(self):
        p = Param("out", value="$output/x.mp4", log_as="video", log_when="before")
        assert p.log_when == "before"

    def test_log_when_list_value(self):
        p = Param("img", value=["$output/img.jpg", "0", "0.8"], log_as="image")
        assert p.log_when == "after"

    def test_log_when_none_without_log_as(self):
        p = Param("prompt")
        assert p.log_when is None


# ---------------------------------------------------------------------------
# Runner: CLI parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def _make_runner(self, params=None):
        return Runner(command="echo hello", params=params or [])

    def test_parse_basic_args(self):
        runner = self._make_runner([
            Param("prompt"),
            Param("seed", type="int", default=42),
        ])
        with patch("sys.argv", ["prog", "--prompt", "a cat", "--seed", "99"]):
            args = runner._parse_cli_args()
        assert args["prompt"] == "a cat"
        assert args["seed"] == 99

    def test_parse_bool_flag(self):
        runner = self._make_runner([Param("verbose", type="bool")])
        with patch("sys.argv", ["prog", "--verbose"]):
            args = runner._parse_cli_args()
        assert args["verbose"] is True

    def test_parse_bool_flag_absent(self):
        runner = self._make_runner([Param("verbose", type="bool")])
        with patch("sys.argv", ["prog"]):
            args = runner._parse_cli_args()
        assert args["verbose"] is False

    def test_fixed_params_not_in_argparse(self):
        runner = self._make_runner([
            Param("prompt"),
            Param("output-path", value="$output/video.mp4"),
        ])
        with patch("sys.argv", ["prog", "--prompt", "hi"]):
            args = runner._parse_cli_args()
        assert "prompt" in args
        assert "output_path" not in args

    def test_choices(self):
        runner = self._make_runner([
            Param("mode", choices=["fast", "quality"], default="fast"),
        ])
        with patch("sys.argv", ["prog", "--mode", "quality"]):
            args = runner._parse_cli_args()
        assert args["mode"] == "quality"

    def test_builtin_flags(self):
        runner = self._make_runner()
        with patch("sys.argv", ["prog", "--dry-run", "-n", "--keep-outputs"]):
            args = runner._parse_cli_args()
        assert args["_dry_run"] is True
        assert args["_no_interactive"] is True
        assert args["_keep_outputs"] is True

    def test_wandb_project_override(self):
        runner = self._make_runner()
        with patch("sys.argv", ["prog", "--wandb-project", "my-project"]):
            runner._parse_cli_args()
        assert runner.wandb_project == "my-project"


# ---------------------------------------------------------------------------
# Runner: resolve values
# ---------------------------------------------------------------------------

class TestResolveValues:
    def test_overrides_take_priority(self):
        runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
        cli_args = {"seed": 99, "_dry_run": False, "_no_interactive": True, "_keep_outputs": False}
        resolved = runner._resolve_values(cli_args, overrides={"seed": 777})
        assert resolved["seed"] == 777

    def test_callable_default(self):
        runner = Runner(command="echo", params=[Param("path", default=lambda: "/computed/path")])
        cli_args = {"path": None, "_dry_run": False, "_no_interactive": True, "_keep_outputs": False}
        resolved = runner._resolve_values(cli_args, overrides={})
        assert resolved["path"] == "/computed/path"

    def test_cli_value_beats_default(self):
        runner = Runner(command="echo", params=[Param("seed", type="int", default=42)])
        cli_args = {"seed": 99, "_dry_run": False, "_no_interactive": True, "_keep_outputs": False}
        resolved = runner._resolve_values(cli_args, overrides={})
        assert resolved["seed"] == 99


# ---------------------------------------------------------------------------
# Runner: prompt missing
# ---------------------------------------------------------------------------

class TestPromptMissing:
    def test_no_interactive_exits_on_missing(self):
        runner = Runner(command="echo", params=[Param("prompt")])
        resolved = {"prompt": None}
        with pytest.raises(SystemExit) as exc_info:
            runner._prompt_missing(resolved, interactive=False)
        assert exc_info.value.code == 2

    def test_no_missing_params_is_noop(self):
        runner = Runner(command="echo", params=[Param("seed", default=42)])
        resolved = {"seed": 42}
        runner._prompt_missing(resolved, interactive=False)  # should not raise

    def test_interactive_fills_from_questionary(self):
        runner = Runner(command="echo", params=[Param("prompt")])
        resolved = {"prompt": None}
        with patch("genai_runner.questionary") as mock_q:
            mock_q.text.return_value.ask.return_value = "a cat"
            runner._prompt_missing(resolved, interactive=True)
        assert resolved["prompt"] == "a cat"

    def test_interactive_select_for_choices(self):
        runner = Runner(command="echo", params=[Param("mode", choices=["fast", "slow"])])
        resolved = {"mode": None}
        with patch("genai_runner.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "fast"
            runner._prompt_missing(resolved, interactive=True)
        assert resolved["mode"] == "fast"

    def test_interactive_cancel_exits(self):
        runner = Runner(command="echo", params=[Param("prompt")])
        resolved = {"prompt": None}
        with patch("genai_runner.questionary") as mock_q:
            mock_q.text.return_value.ask.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                runner._prompt_missing(resolved, interactive=True)
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Runner: interpolate output
# ---------------------------------------------------------------------------

class TestInterpolateOutput:
    def test_replaces_output_in_string_value(self, tmp_path):
        runner = Runner(command="echo", params=[Param("out", value="$output/video.mp4")])
        resolved = {}
        result = runner._interpolate_output(resolved, tmp_path)
        assert result["out"] == f"{tmp_path}/video.mp4"
        assert (tmp_path / "video.mp4").parent.exists()

    def test_replaces_output_in_list_value(self, tmp_path):
        runner = Runner(command="echo", params=[
            Param("img", value=["$output/img.jpg", "0", "0.8"]),
        ])
        result = runner._interpolate_output({}, tmp_path)
        assert result["img"] == [f"{tmp_path}/img.jpg", "0", "0.8"]

    def test_non_output_value_unchanged(self, tmp_path):
        runner = Runner(command="echo", params=[
            Param("config", value="/etc/config.toml"),
        ])
        result = runner._interpolate_output({}, tmp_path)
        assert result["config"] == "/etc/config.toml"

    def test_resolved_params_preserved(self, tmp_path):
        runner = Runner(command="echo", params=[
            Param("out", value="$output/video.mp4"),
        ])
        resolved = {"prompt": "a cat", "seed": 42}
        result = runner._interpolate_output(resolved, tmp_path)
        assert result["prompt"] == "a cat"
        assert result["seed"] == 42


# ---------------------------------------------------------------------------
# Runner: build command
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_basic_command(self):
        runner = Runner(command="python generate.py", params=[
            Param("prompt"),
            Param("seed", type="int"),
        ])
        cmd = runner._build_command({"prompt": "a cat", "seed": 42})
        assert cmd == ["python", "generate.py", "--prompt", "a cat", "--seed", "42"]

    def test_bool_flag(self):
        runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
        cmd = runner._build_command({"verbose": True})
        assert cmd == ["run.py", "--verbose"]

    def test_bool_flag_false_omitted(self):
        runner = Runner(command="run.py", params=[Param("verbose", type="bool")])
        cmd = runner._build_command({"verbose": False})
        assert cmd == ["run.py"]

    def test_multi_value_flag(self):
        runner = Runner(command="run.py", params=[
            Param("image", value=["photo.jpg", "0", "0.8"]),
        ])
        cmd = runner._build_command({"image": ["photo.jpg", "0", "0.8"]})
        assert cmd == ["run.py", "--image", "photo.jpg", "0", "0.8"]

    def test_none_values_omitted(self):
        runner = Runner(command="run.py", params=[
            Param("prompt"),
            Param("seed", type="int"),
        ])
        cmd = runner._build_command({"prompt": "hi", "seed": None})
        assert cmd == ["run.py", "--prompt", "hi"]

    def test_custom_flag(self):
        runner = Runner(command="run.py", params=[
            Param("out", flag="-o"),
        ])
        cmd = runner._build_command({"out": "/tmp/x"})
        assert cmd == ["run.py", "-o", "/tmp/x"]


# ---------------------------------------------------------------------------
# Runner: metric extraction
# ---------------------------------------------------------------------------

class TestMetricExtraction:
    def test_float_metric(self):
        runner = Runner(command="echo", metrics=[
            Metric("skip_pct", pattern=r"skipped=([\d.]+)%"),
        ])
        wb_run = MagicMock()
        wb_run.summary = {}
        runner._extract_metrics(wb_run, "some output\nskipped=32.8%\ndone")
        assert wb_run.summary["skip_pct"] == 32.8

    def test_str_metric(self):
        runner = Runner(command="echo", metrics=[
            Metric("status", pattern=r"final: (\w+)", type="str"),
        ])
        wb_run = MagicMock()
        wb_run.summary = {}
        runner._extract_metrics(wb_run, "final: completed")
        assert wb_run.summary["status"] == "completed"

    def test_last_match_wins(self):
        runner = Runner(command="echo", metrics=[
            Metric("val", pattern=r"x=([\d.]+)"),
        ])
        wb_run = MagicMock()
        wb_run.summary = {}
        runner._extract_metrics(wb_run, "x=1.0\nx=2.0\nx=3.0")
        assert wb_run.summary["val"] == 3.0

    def test_no_match_no_summary(self):
        runner = Runner(command="echo", metrics=[
            Metric("val", pattern=r"x=([\d.]+)"),
        ])
        wb_run = MagicMock()
        wb_run.summary = {}
        runner._extract_metrics(wb_run, "no matches here")
        assert "val" not in wb_run.summary


# ---------------------------------------------------------------------------
# Runner: execute subprocess
# ---------------------------------------------------------------------------

class TestExecute:
    def test_captures_stdout_and_stderr(self, tmp_path):
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

    def test_nonzero_exit_code(self, tmp_path):
        runner = Runner(command="echo")
        cmd = [sys.executable, "-c", "import sys; sys.exit(42)"]
        exit_code, _, _ = runner._execute(cmd, tmp_path)
        assert exit_code == 42

    def test_env_vars_passed(self, tmp_path):
        runner = Runner(command="echo", env={"MY_TEST_VAR": "hello123"})
        cmd = [sys.executable, "-c", "import os; print(os.environ['MY_TEST_VAR'])"]
        exit_code, _, stdout_text = runner._execute(cmd, tmp_path)
        assert exit_code == 0
        assert "hello123" in stdout_text


# ---------------------------------------------------------------------------
# Runner: dry run (integration)
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_prints_command_no_wandb(self, capsys):
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
# Runner: full run (integration with mocked W&B)
# ---------------------------------------------------------------------------

class TestFullRun:
    def test_full_run_with_mocked_wandb(self, tmp_path, capsys):
        mock_wb_run = MagicMock()
        mock_wb_run.name = "test-run-42"
        mock_wb_run.id = "abc123"
        mock_wb_run.url = "https://wandb.test/run"
        mock_wb_run.tags = []
        mock_wb_run.summary = {}
        mock_wb_run.config = {}

        runner = Runner(
            command=f"{sys.executable} -c \"import sys; print('hello'); print('err', file=sys.stderr)\"",
            params=[
                Param("output-path", value="$output/video.mp4"),
            ],
            metrics=[Metric("val", pattern=r"no-match")],
        )

        with patch("sys.argv", ["prog", "-n"]), \
             patch("genai_runner.wandb") as mock_wandb, \
             patch("genai_runner._collect_git_info", return_value={"repo": "test-repo", "commit": "abc", "branch": "main", "dirty": False}), \
             patch("genai_runner._save_code_snapshot"), \
             patch("genai_runner.Path.home", return_value=tmp_path):
            mock_wandb.init.return_value = mock_wb_run
            mock_wandb.Artifact = MagicMock()
            runner.run()

        # Check W&B was initialized
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-repo"
        assert call_kwargs["save_code"] is True

        # Check summary was set
        assert mock_wb_run.summary["status"] == "success"
        assert mock_wb_run.summary["exit_code"] == 0
        assert mock_wb_run.summary["duration_seconds"] > 0

        # Check output dir was created
        output_dir = tmp_path / "genai_runs" / "test-repo"
        assert output_dir.exists()

        # Check finish was called
        mock_wb_run.finish.assert_called_once_with(exit_code=0)


# ---------------------------------------------------------------------------
# git info
# ---------------------------------------------------------------------------

class TestCollectGitInfo:
    def test_returns_dict_in_git_repo(self):
        info = _collect_git_info()
        # We're in the genai-runner repo
        if info:
            assert "repo" in info
            assert "commit" in info
            assert "branch" in info
            assert "dirty" in info

    def test_returns_empty_dict_outside_git(self):
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            info = _collect_git_info()
        assert info == {}
