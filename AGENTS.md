# Agent Instructions

## Python Environment

- Always prefer the repo-local virtual environment at `.venv`.
- When running Python tools in this repository, use executables from `.venv/bin/` when available.
- Prefer `.venv/bin/python -m pytest` over `uv run` for tests unless the user explicitly asks for `uv`.
- Prefer `.venv/bin/python` for scripts and module entrypoints unless the user explicitly asks for a different runner.
