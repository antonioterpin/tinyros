---
description: First-time orientation before making changes
---

Use when working on the project for the first time, or returning after a long absence. Goal: build a mental model of the system before touching code.

1. **Read the architecture overview.**
   [`docs/guides/architecture/overview.md`](../guides/architecture/overview.md) covers the system overview and module responsibilities. Pay attention to:
   - The two layers: transport (the wire protocol) and nodes (the user-facing pub/sub API).
   - The static-network-config philosophy: every connection is declared upfront in the YAML topology.
   - Nodes are both publishers *and* servers (RPC callbacks bound from the config).

2. **Understand the transport model.**
   [`docs/guides/architecture/transport.md`](../guides/architecture/transport.md) explains the wire protocol, framing, and the shared-memory side-channel for large numpy payloads.

3. **Review the standards.**
   Skim [`docs/standards/`](../standards/) before writing code -- in particular [`code-organization.md`](../standards/code-organization.md) and [`linting-formatting.md`](../standards/linting-formatting.md).

4. **Confirm the environment works.**
   Run the test suite:
   ```bash
   uv run pytest
   ```
   All tests should pass before you write a single line.

5. **Pick the right workflow for your task.**
   | Task | Workflow |
   |---|---|
   | New feature | [`feature.md`](feature.md) |
   | Bug fix | [`bugfix.md`](bugfix.md) |
   | Refactor | [`refactor.md`](refactor.md) |
   | External API exploration | [`api-validation.md`](api-validation.md) |
   | Documentation only | [`docs.md`](docs.md) |

**Done criteria:**
- You can describe what `src/tinyros/node.py` and `src/tinyros/transport.py` do without looking them up.
- You know which module owns the code you intend to change.
- `uv run pytest` passes.
