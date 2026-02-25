# genai-runner

Experiment runner for video diffusion models with W&B tracking.

Write a small Python script per model that declares params, outputs, and metrics.
`genai-runner` handles the rest: CLI parsing, interactive prompts for missing values,
subprocess execution, stdout/stderr capture, metric extraction, file uploads to W&B,
and code snapshots for reproducibility.

## Install

```bash
pip install genai-runner
# or as a uv script dependency:
# /// script
# dependencies = ["genai-runner @ git+https://github.com/YOU/genai-runner"]
# ///
```

## Quick start

Create a `run.py` for your model:

```python
from genai_runner import Runner, Param, Output, Metric

runner = Runner(
    command="python generate.py",
    params=[
        Param("prompt", help="Text prompt"),
        Param("seed", type="int", default=42),
        Param("output-path", value="$output/video.mp4", log_as="video"),
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
python run.py --prompt "a cat" -n               # non-interactive, fail if missing
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
Param("name")                                    # basic string param
Param("seed", type="int", default=42)            # typed with default
Param("mode", choices=["fast", "quality"])        # select from choices
Param("image", type="path", log_as="image")      # file input, uploaded to W&B before run
Param("output-path", value="$output/video.mp4",  # fixed value, $output interpolated
      log_as="video")                             # uploaded to W&B after run
Param("image", types=["path", "float", "float"], # multi-value flag: --image photo.jpg 0 0.8
      labels=["path", "start", "strength"])       # each part prompted separately in TUI
```

- `value=` makes a param fixed (never prompted, not in CLI)
- `$output` in value is replaced with the run's output directory
- `log_as=` uploads the file to W&B (`"video"`, `"image"`, `"artifact"`, `"text"`)
- `log_when=` auto-inferred: `"before"` for inputs, `"after"` for `$output` paths
- `types=` gives per-element types for multi-value flags (nargs inferred from length)

## Output

For files the model writes to uncontrolled locations:

```python
Output("model_metadata.json", log_as="artifact", copy_to="$output/model_metadata.json")
```

## Metric

Extract values from stdout:

```python
Metric("loss", pattern=r"loss=([\d.]+)")           # float (default)
Metric("status", pattern=r"status: (\w+)", type="str")
```

Last match wins. Stored in `wandb.run.summary`.

## Sweeps

Loop with overrides:

```python
for lr in [1e-3, 1e-4, 1e-5]:
    runner.run(overrides={"learning_rate": lr})
```

Each call creates a separate W&B run.

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
)
```

## Built-in CLI flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Print command and exit |
| `-n` / `--no-interactive` | Fail if required params missing |
| `--keep-outputs` | Reminder that outputs are kept |
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
| Media | Videos and images from `log_as` params/outputs |
