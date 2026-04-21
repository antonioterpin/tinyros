# Agent-driven development guide

This guide explains how to use the TinyROS documentation and workflows
with different agentic systems (Claude Code, GitHub Copilot, Antigravity, custom agents).

## Overview

TinyROS is optimized for agent-driven development. All rules, workflows,
and procedures are centralized in the `docs/` folder, and thin wrappers in
`.agent/`, `.github/`, and `.claude/` reference them.

## Infrastructure at a glance

| System | Primary entry point | Canonical rule source |
|---|---|---|
| Claude Code | `.claude/CLAUDE.md` | `docs/standards/` |
| GitHub Copilot | `.github/copilot-instructions.md` (optional) | `docs/standards/` |
| Antigravity | `.agent/rules/rules.md` + `.agent/workflows/*` | `docs/standards/` |
| Custom agents | Integrator-defined | `docs/standards/` |

`docs/standards/` is the single source of truth for policy. Wrappers only
define how agents discover and trigger workflow behavior.

## Using with Claude Code

Claude Code reads its configuration from:

1. `.claude/CLAUDE.md` -- project instructions, references `docs/standards/`
2. `docs/` -- full documentation
3. `.agent/` -- workflow wrappers

Typical flow:

```
User: "I want to replace the subscriber FIFO. Please follow the feature workflow."

Claude Code will:
1. Read .claude/CLAUDE.md for project context
2. Reference docs/workflows/feature.md for step-by-step procedure
3. Follow docs/standards/ for code quality rules
```

## Using with Antigravity

Antigravity reads its configuration from:

1. `.agent/rules/rules.md` (always-on project rules)
2. `.agent/workflows/` (thin workflow wrappers)

Slash commands map directly to workflow files:

- `/feature`, `/bugfix`, `/refactor`, `/api-validation`, `/docs`, `/orientation`

## Making changes to documentation

1. **Make changes in `docs/`** -- single source of truth.
2. **All agents will automatically see the updates** via the wrappers.
3. **Do not duplicate content** in `.agent/`, `.github/`, or `.claude/`.

## Workflow quick reference

| Workflow | When | Location |
|---|---|---|
| Feature | New feature, TDD | [docs/workflows/feature.md](../workflows/feature.md) |
| Bugfix | Regression / bug | [docs/workflows/bugfix.md](../workflows/bugfix.md) |
| Refactor | Structure, no behavior change | [docs/workflows/refactor.md](../workflows/refactor.md) |
| API validation | External lib exploration | [docs/workflows/api-validation.md](../workflows/api-validation.md) |
| Docs | Docs-only change | [docs/workflows/docs.md](../workflows/docs.md) |
| Orientation | First time / returning | [docs/workflows/orientation.md](../workflows/orientation.md) |
