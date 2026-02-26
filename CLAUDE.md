# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
uv sync                          # install deps (including dev group)
uv run pytest                    # run all tests
uv run pytest -k test_name       # run a single test by name
uv run pytest -x                 # stop on first failure
ruff format .                    # format
ruff check .                     # lint
```

Python >=3.11 required. Build backend is hatchling.

## Architecture

Single-package library (`src/genai_runner/__init__.py`, ~960 lines) with four core abstractions:

- **`Param`** — Declares a CLI parameter.
  Type string encodes both parsing and W&B upload intent (e.g. `"path-video"` means file path that gets uploaded as video).
  Multi-value via `type=[...]`.
  Values resolve through a priority chain: CLI args > `overrides` > callable defaults > fixed `value=`.
- **`Output`** — Declares output files not tied to a Param (glob patterns, directories, zips).
  Processed after subprocess completes.
- **`Metric`** — Regex pattern applied to stdout; last match wins.
  Stored in `wandb.run.summary`.
- **`Runner`** — Orchestrator.
  Parses CLI, prompts for missing values (via `questionary`),
  inits W&B, creates output dir,
  runs subprocess with stdout/stderr threading, then does post-run logging (metrics, files, code snapshot).
  Post-run steps are individually try-excepted so W&B always finishes.

`$output` is a placeholder interpolated to `~/genai_runs/<project>/<timestamp>_<run_name>/` at runtime.

The `_UNSET` sentinel marks params the user explicitly skipped (typed `-` at prompt) — these are omitted from the built command.

## Key Design Decisions

- **Decoupled from models**: User writes a `run.py` per model that configures a `Runner`. The runner only knows about CLI wrapping, not model internals.
- **Interactive-first**: Missing params trigger TUI prompts by default; `--no-interactive` for CI/sweeps.
- **Never-fail post-run**: Each post-run step catches exceptions and warns, ensuring W&B run always completes.

## Testing

Tests are in `tests/test_genai_runner.py`. W&B is mocked in integration tests.
The test file covers: param validation, CLI parsing, value resolution, interactive prompts, command building, metric extraction, subprocess execution, output logging, and full run lifecycle.
