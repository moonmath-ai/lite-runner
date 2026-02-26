# genai-runner

Experiment runner for video diffusion models with W&B tracking.

Write a small Python script per model that declares params, outputs, and metrics.
`genai-runner` handles the rest: CLI parsing, interactive prompts for missing values,
subprocess execution, stdout/stderr capture, metric extraction, file uploads to W&B,
and code snapshots for reproducibility.

## Install

```bash
pip install genai-runner@git+https://github.com/tsvikas/genai-runner
```

Or as a uv [script dependency](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies):

```python
# /// script
# dependencies = ["genai-runner @ git+https://github.com/tsvikas/genai-runner"]
# ///
```

## Quick start

Create a `run.py` for your model:

```python
# /// script
# dependencies = ["genai-runner @ git+https://github.com/tsvikas/genai-runner"]
# ///
from genai_runner import Runner, Param, Metric

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

Then run it:

```bash
python run.py --prompt "a cat walking"          # interactive TUI fills missing params
python run.py --prompt "a cat" --no-interactive  # non-interactive, fail if missing
python run.py --prompt "a cat" --dry-run        # print command, don't run
```

## What it does

Each `runner.run()` call:

1. Parses CLI args (all params are optional in argparse; missing ones trigger TUI prompts)
2. Creates an output directory at `~/genai_runs/<project>/<timestamp>_<run_name>/`
3. Inits a W&B run and logs all params, git info, and host metadata
4. Saves a code snapshot (git archive + dirty diff) as a W&B artifact
5. Builds and runs the subprocess, streaming stdout/stderr to terminal and log files
6. Extracts metrics from stdout via regex
7. Uploads output files (videos, images, artifacts) to W&B
8. Logs duration, exit code, and status to W&B summary

## Param

```python
Param("name")                                          # basic string param
Param("seed", type="int", default=42)                  # typed with default
Param("mode", choices=["fast", "quality"])             # select from choices
Param("image", type="path-image")                      # file input, uploaded to W&B before run
Param("output-path", value="$output/video.mp4",        # fixed value, $output interpolated
      type="path-video")                               # uploaded to W&B after run
Param("input-image", type=["path-image", "float", "float"],  # multi-value flag
      labels=["img", "start", "strength"])                    # each part prompted separately in TUI
```

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

```python
Output("debug/**/*.png", log_as="image")        # upload each matched png
Output("debug/", log_as="zip")                   # zip entire directory, upload as artifact
Output("$output/frames/*.jpg", log_as="zip")     # zip glob matches into archive
```

## Metric

Extract values from stdout:

```python
Metric("loss", pattern=r"loss=([\d.]+)")           # float (default)
Metric("status", pattern=r"status: (\w+)", type="str")
```

Last match wins. Stored in `wandb.run.summary`.

## Sweeps

Loop with overrides. Runs are grouped in W&B for easy comparison:

```python
runner = Runner(
    command="python gen.py",
    params=[...],
    group="lr-sweep",           # groups all runs together in W&B UI
)
for lr in [1e-3, 1e-4, 1e-5]:
    runner.run(overrides={"learning_rate": lr})
```

Each call creates a separate W&B run, all grouped under the same `group`.

## Runner options

```python
Runner(
    command="python gen.py",          # str or list[str] (list avoids shell splitting)
    params=[...],
    outputs=[...],
    metrics=[...],
    tags=["experiment-1"],            # W&B run tags
    env={"CUDA_VISIBLE_DEVICES": "0"},  # extra env vars for subprocess
    wandb_project="my-project",       # default: git repo name
    group="my-sweep",                 # W&B run group for sweeps (None = no grouping)
)
```

## Built-in CLI flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Print command and exit |
| `--no-interactive` | Fail if required params missing |
| `--run-name NAME` | Override W&B run name |
| `--wandb-project NAME` | Override W&B project |

## What gets logged to W&B

| Location | Content |
|----------|---------|
| `run.config["param/*"]` | All param values |
| `run.config["git/*"]` | commit, branch, repo, dirty |
| `run.config["meta/*"]` | hostname, datetime, command |
| `run.summary` | exit_code, duration_seconds, status, metrics |
| Artifacts | Log files, code snapshot, artifact-type outputs |
| Media | Videos and images from `path-*` type params/outputs |
