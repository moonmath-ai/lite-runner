# LiteRunner

[![Tests][tests-badge]][tests-link]
[![codecov][codecov-badge]][codecov-link]
[![Made Using tsvikas/python-template][template-badge]][template-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![PRs Welcome][prs-welcome-badge]][prs-welcome-link]

## Overview

Runner for generative models with local and W&B tracking.

Write a small Python script per model that declares params, outputs, and metrics.
`lite-runner` handles the rest: CLI parsing, interactive prompts for missing values,
subprocess execution, stdout/stderr capture, metric extraction, file uploads to W&B,
and code snapshots for reproducibility.

## Quick start

Create a `run.py` for your model:

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lite-runner @ git+https://github.com/moonmath-ai/LiteRunner"]
# ///
from lite_runner import Runner, Param, Metric

runner = Runner(
    command="python generate.py",
    params=[
        Param("prompt", help="Text prompt"),
        Param("seed", type="int", default=42),
        Param("output-path", value="$output/video.mp4", type="path-video"),
    ],
    metrics=[
        Metric("loss", pattern=r"loss=([\d.]+)"),
    ],
)

if __name__ == "__main__":
    runner.run()
```

Then run it (requires [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
chmod +x run.py
./run.py --prompt "a cat walking"           # interactive TUI fills missing params
./run.py --prompt "a cat" --no-interactive  # non-interactive, fail if missing
./run.py --prompt "a cat" --dry-run         # print command, don't run
```

## What it does

Each `runner.run()` call:

1. Parses CLI args (all params are optional in argparse; missing ones trigger TUI prompts)
1. Creates an output directory at `~/lite_runs/<project>/<timestamp>_<run_name>/`
1. Inits a W&B run and logs all params, git info, and host metadata
1. Saves a code snapshot (git archive + dirty diff) as a W&B artifact
1. Builds and runs the subprocess, streaming stdout/stderr to terminal and log files
1. Extracts metrics from stdout via regex
1. Uploads output files (videos, images, artifacts) to W&B
1. Logs duration, exit code, and status to W&B summary

## Param

<!-- blacken-docs:off -->

```python
Param("name")                               # basic string param
Param("seed", type="int", default=42)       # typed with default
Param("mode", choices=["fast", "quality"])  # select from choices
Param("image", type="path-image")           # file input, uploaded to W&B before run
Param(
    "output-path",
    value="$output/video.mp4",              # fixed value, $output interpolated
    type="path-video",
)                                           # uploaded to W&B after run
Param(
    "input-image",
    type=["path-image", "float", "float"],  # multi-value flag
    labels=["img", "start", "strength"],
)                                           # each part prompted separately in TUI
```

<!-- blacken-docs:on -->

- `value=` makes a param fixed (never prompted, not in CLI)
- `default=` can be a callable (called at prompt time to compute the default)
- `$output` in value is replaced with the run's output directory
- `type="path-*"` encodes upload intent:
  - `"path-video"` — upload as video to W&B
  - `"path-image"` — upload as image
  - `"path-artifact"` — upload as artifact
  - `"path-text"` — upload as text
  - `"path"` — file path, no auto-upload
- `log_when=` auto-inferred: `"before"` for inputs, `"after"` for `$output` paths
- `type=[...]` gives per-element types for multi-value flags (nargs inferred from length)

## Output

For files the model writes to uncontrolled locations:

```python
Output("model_metadata.json", log_as="artifact", copy_to="$output/model_metadata.json")
```

Supports glob patterns and directory zipping:

<!-- blacken-docs:off -->

```python
Output("debug/**/*.png", log_as="image")      # upload each matched png
Output("debug/", log_as="image")              # upload each file in directory
Output("debug/", log_as="zip")                # zip entire directory, upload as artifact
Output("$output/frames/*.jpg", log_as="zip")  # zip glob matches into archive
```

<!-- blacken-docs:on -->

## Metric

Extract values from stdout:

```python
Metric("loss", pattern=r"loss=([\d.]+)")
Metric("status", pattern=r"status: (\w+)", type="str")
```

Last match wins. Stored in `wandb.run.summary`.

## Sweeps

Loop with `override()`. Runs are grouped in W&B for easy comparison:

```python
runner = Runner(
    command="python gen.py",
    params=[...],
    run_group="lr-sweep",  # groups all runs together in W&B UI
)
for lr in [1e-3, 1e-4, 1e-5]:
    runner.override(learning_rate=lr).run(no_interactive=True)
```

Each call creates a separate W&B run, all grouped under the same `group`.

You can also update metadata per-run:

```python
runner.override(seed=42).with_metadata(tags=["baseline"]).run()
```

## Runner options

<!-- blacken-docs:off -->

```python
Runner(
    command="python gen.py",            # str or list[str] (list avoids shell splitting)
    params=[...],
    outputs=[...],
    metrics=[...],
    tags=["experiment-1"],              # W&B run tags
    env={"CUDA_VISIBLE_DEVICES": "0"},  # extra env vars for subprocess
    project="my-project",               # default: git repo name
    run_group="my-sweep",               # W&B run group for sweeps (None = no grouping)
)
```

<!-- blacken-docs:on -->

## Pipeline API

Each method returns a new `Runner` (immutable copies), so you can branch:

<!-- blacken-docs:off -->

```python
base = runner.parse_cli()    # parse sys.argv
r1 = base.override(seed=42)  # override params by name
r2 = base.override(seed=99)
r1.run()                     # auto-resolves defaults & prompts
r2.run()
```

<!-- blacken-docs:on -->

Methods:

| Method                                       | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| `parse_cli(argv)`                            | Parse CLI args (default: `sys.argv[1:]`)      |
| `override(**kwargs)`                         | Set param values by name                      |
| `with_metadata(project=, run_group=, tags=)` | Update W&B metadata                           |
| `resolve_defaults()`                         | Apply defaults and fixed values               |
| `ask_user(no_interactive=)`                  | Prompt for missing values                     |
| `run(...)`                                   | Auto-calls any unapplied steps, then executes |

`run()` accepts kwargs `dry_run`, `min_free_space_gib`, `no_interactive`, `no_wandb`, `project`, `run_name` as alternatives to CLI flags.

## Built-in CLI flags

| Flag                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `--dry-run`              | Print command and exit                        |
| `--min-free-space-gib N` | Minimum free disk space in GiB (default: 1.0) |
| `--no-interactive`       | Fail if required params missing               |
| `--no-wandb`             | Skip W&B logging (still logs to JSON)         |
| `--run-name NAME`        | Override W&B run name                         |
| `--project NAME`         | Override project name                         |

## What gets logged to W&B

| Location                | Content                                             |
| ----------------------- | --------------------------------------------------- |
| `run.config["param/*"]` | All param values                                    |
| `run.config["git/*"]`   | commit, branch, repo, dirty                         |
| `run.config["meta/*"]`  | hostname, datetime, command                         |
| `run.summary`           | exit_code, duration_seconds, status, metrics        |
| Artifacts               | Log files, code snapshot, artifact-type outputs     |
| Media                   | Videos and images from `path-*` type params/outputs |

## Contributing

Interested in contributing?
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guideline.

[codecov-badge]: https://codecov.io/gh/moonmath-ai/LiteRunner/graph/badge.svg
[codecov-link]: https://codecov.io/gh/moonmath-ai/LiteRunner
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]: https://github.com/moonmath-ai/LiteRunner/discussions
[prs-welcome-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
[prs-welcome-link]: https://opensource.guide/how-to-contribute/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[template-badge]: https://img.shields.io/badge/%F0%9F%9A%80_Made_Using-tsvikas%2Fpython--template-gold
[template-link]: https://github.com/tsvikas/python-template
[tests-badge]: https://github.com/moonmath-ai/LiteRunner/actions/workflows/ci.yml/badge.svg
[tests-link]: https://github.com/moonmath-ai/LiteRunner/actions/workflows/ci.yml
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv-link]: https://github.com/astral-sh/uv
