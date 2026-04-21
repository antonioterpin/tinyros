# Contributing

This page summarizes the development workflow and software-engineering
practices adopted in this codebase.

- [Development setup](#development-setup)
- [Quality gates](#quality-gates)
- [Testing a feature](#testing-a-feature)
- [Preparing for a PR](#preparing-for-a-pr)

We use [uv](https://docs.astral.sh/) to manage the virtual environment and dependencies.
If you do not have it yet, install it with:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Development setup

Provision the local environment with development tools:
```sh
uv sync --extra dev
```

Run the example application:
```sh
uv run python main.py
```

Run tests:
```sh
uv run pytest
```

## Quality gates

Pre-commit enforces the coding style:
```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Hooks:

- **Ruff** -- linting, docstring checks, modern typing, unused imports
- **Black** -- formatting
- **Basedpyright** -- type checking (pre-push stage)
- **Pytest** -- tests (pre-push stage)
- **Standard hooks** -- YAML validation, trailing whitespace, end-of-file fixes

Run everything manually:
```sh
uv run pre-commit run --all-files
uv run pre-commit run --hook-stage push --all-files
```

## Testing a feature

Follow the [testing standards](../standards/testing.md) and the
[feature workflow](../workflows/feature.md). Write tests before code.

Run a subset:
```sh
uv run pytest tests/test_transport.py
```

Run benchmarks (opt-in):
```sh
uv run pytest -m run_explicitly tests/benchmark/
```

## Preparing for a PR

Follow the [version-control standard](../standards/version-control.md):

- Squash your branch into a single, conventional-commits-formatted commit.
- Do not push to `main` or `dev`; push your feature branch and open a PR.
- Configure the commit template once:
  ```sh
  git config commit.template .gitmessage
  ```
