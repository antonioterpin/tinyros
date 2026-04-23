# TinyROS documentation hub

Single source of truth for all TinyROS documentation. Start here.

---

## Architecture

| Document | What it covers |
|---|---|
| [overview.md](guides/architecture/overview.md) | Top-level layout, core concepts, where to make changes |
| [transport.md](guides/architecture/transport.md) | Wire protocol, framing, shared-memory fast path, threading |

---

## Guides

| Document | What it covers |
|---|---|
| [contributing.md](guides/contributing.md) | Development setup, quality gates, PR preparation |
| [agent-development.md](guides/agent-development.md) | Using TinyROS with Claude Code, Antigravity, and custom agents |

---

## Standards

One file per topic -- all code must satisfy these:

| Standard | Topic |
|---|---|
| [environment-tooling.md](standards/environment-tooling.md) | Always use `uv run`; virtual-env discipline |
| [code-quality.md](standards/code-quality.md) | Pre-commit gates, quality checklist |
| [code-clarity.md](standards/code-clarity.md) | Naming, comments, header capitalization |
| [code-organization.md](standards/code-organization.md) | Package layout, logging scope naming |
| [typing-docstrings.md](standards/typing-docstrings.md) | PEP 585/604 typing, Google-style docstrings |
| [testing.md](standards/testing.md) | TDD, Arrange/Act/Assert, benchmark gating |
| [api-design.md](standards/api-design.md) | Public API contracts, over-engineering |
| [change-scope.md](standards/change-scope.md) | What belongs in a single change |
| [version-control.md](standards/version-control.md) | Commit messages, branch naming |
| [linting-formatting.md](standards/linting-formatting.md) | Ruff config as single source of truth |
| [exploration-validation.md](standards/exploration-validation.md) | Scratch scripts and API validation |

---

## Workflows

Step-by-step procedures for common tasks:

| Workflow | When to use |
|---|---|
| [orientation.md](workflows/orientation.md) | First time on the project, or returning after a long absence |
| [feature.md](workflows/feature.md) | Implementing a new feature (TDD) |
| [bugfix.md](workflows/bugfix.md) | Fixing a reported bug |
| [refactor.md](workflows/refactor.md) | Improving structure without changing behavior |
| [api-validation.md](workflows/api-validation.md) | Exploring external APIs |
| [docs.md](workflows/docs.md) | Documentation-only changes |

---

## Agent personas

| Persona | Purpose |
|---|---|
| [agents/implementer.md](agents/implementer.md) | Guidance for the implementing agent role |
| [agents/reviewer.md](agents/reviewer.md) | Guidance for the reviewing agent role |

---

## Quick reference

```bash
# Run tests
uv run pytest

# Lint and format
uv run pre-commit run --all-files

# Type check
uv run pre-commit run --hook-stage push --all-files

# Run the example application
uv run python main.py

# Benchmarks (opt-in)
uv run pytest -m run_explicitly tests/benchmark/
```
