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

Three modules under `src/genai_runner/`:

- **`params.py`** — Data classes and type system: `Param`, `Output`, `Metric`, `_RunFlags`, type constants, `UNSET` sentinel.
- **`backends.py`** — `LogBackend` protocol, `WandbBackend`, `JsonBackend`.
- **`runner.py`** — `Runner` orchestrator and module-level helpers.
- **`__init__.py`** — Re-exports the public API.

Four core abstractions:

- **`Param`** — Declares a CLI parameter.
  Type string encodes both parsing and W&B upload intent (e.g. `"path-video"` means file path that gets uploaded as video).
  Multi-value via `type=[...]`.
- **`Output`** — Declares output files not tied to a Param (glob patterns, directories, zips).
  Processed after subprocess completes.
- **`Metric`** — Regex pattern applied to stdout; last match wins.
  Stored in `wandb.run.summary`.
- **`Runner`** — Orchestrator with an immutable pipeline API.
  Each pipeline method returns a new Runner via `copy.copy`:
  `parse_cli()` → `override()` → `resolve_defaults()` → `ask_user()`.
  `run()` auto-calls any steps not yet applied.
  Values are tracked in `_param_values` with source tags in `_param_sources`
  (`"cli"`, `"override"`, `"default"`, `"fixed"`, `"prompt"`).
  Post-run steps are individually try-excepted so W&B always finishes.

`$output` is a placeholder interpolated to `~/genai_runs/<project>/<timestamp>_<run_name>/` at runtime.

The `UNSET` sentinel marks params the user explicitly skipped (typed `-` at prompt) — these are omitted from the built command.

## Key Design Decisions

- **Decoupled from models**: User writes a `run.py` per model that configures a `Runner`. The runner only knows about CLI wrapping, not model internals.
- **Interactive-first**: Missing params trigger TUI prompts by default; `--no-interactive` or `run(interactive=False)` for CI/sweeps.
- **Never-fail post-run**: Each post-run step catches exceptions and warns, ensuring W&B run always completes.

## Testing

Tests are in `tests/test_genai_runner.py`. W&B is mocked in integration tests.
The test file covers: param validation, CLI parsing, value resolution, interactive prompts, command building, metric extraction, subprocess execution, output logging, and full run lifecycle.
