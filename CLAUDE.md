# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
uv sync                          # install deps (dev, test, typing groups by default)
uv run just prepare              # one-time: install pre-commit hooks
uv run pytest                    # run all tests (includes doctests via --doctest-modules)
uv run pytest -k test_name       # run a single test by name
uv run pytest -x                 # stop on first failure
uv run ruff format .             # format
uv run ruff check .              # lint
uv run mypy                      # type check (strict mode)
```

If you changed package metadata or entry points, use `uv run just test` instead of `uv run pytest` — it forces a fresh hatchling rebuild via `--reinstall-package`.

Python >=3.10 required. Build backend is hatchling. Version is derived from git tags via hatch-vcs (`_version.py` is auto-generated — don't edit it).

Ruff is configured with `select = ["ALL"]` — nearly all lint rules are enabled. Pytest runs with `--reverse` ordering, coverage enabled, and `filterwarnings = ["error"]` so new warnings fail tests.

## Architecture

Four modules under `src/lite_runner/`:

- **`params.py`** — Data classes and type system: `Param`, `Output`, `Metric`, type constants, `UNSET` sentinel.
- **`backends.py`** — `LogBackend` protocol, `WandbBackend`, `JsonBackend`, `DryRunBackend`. Collect/prepare functions for metrics, files, code snapshots.
- **`runner.py`** — `Runner` orchestrator, `RunFlags`, and module-level helpers.
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
  Each pipeline method returns a new Runner via `copy.deepcopy`:
  `parse_cli()` → `override()` → `resolve_defaults()` → `ask_user()`.
  `run()` auto-calls any steps not yet applied.
  Values are tracked in `param_values` with source tags in `param_sources`
  (`"cli"`, `"override"`, `"default"`, `"fixed"`, `"prompt"`).
  Post-run steps are individually try-excepted so W&B always finishes.

`$output` is a placeholder interpolated to `~/lite_runs/<project>/<timestamp>_<run_name>/` at runtime.

The `UNSET` sentinel marks params the user explicitly skipped (typed `-` at prompt) — these are omitted from the built command.

## Key Design Decisions

- **Decoupled from models**: User writes a `run.py` per model that configures a `Runner`. The runner only knows about CLI wrapping, not model internals.
- **Interactive-first**: Missing params trigger TUI prompts by default; `--no-interactive` or `run(no_interactive=True)` for CI/sweeps.
- **Never-fail post-run**: Each post-run step catches exceptions and warns, ensuring W&B run always completes.

## Testing

Tests are split across `tests/test_runner.py`, `tests/test_params.py`, `tests/test_backends.py`, `tests/test_.py`, and `tests/conftest.py`. W&B is mocked in integration tests.
Tests cover: param validation, CLI parsing, value resolution, interactive prompts, command building, metric extraction, subprocess execution, output logging, and full run lifecycle.

## Keeping docs in sync

When changing the public API (Param fields, Runner methods, CLI flags), update:

- **This file** (`CLAUDE.md`) — Architecture section above
- **`README.md`** — Quick start, Param docs, Pipeline API, CLI flags table, Sweeps example
- **`examples/`** — `run_example.py`, `run_sweep.py`, `run_ltx2.py` use the public API
- **Docstrings** — `Runner` class docstring, `Param` class docstring, public method docstrings
