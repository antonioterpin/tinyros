# Code organization & readability

## Code style

- Prefer **clear, explicit code** over clever or compact code.
- Avoid introducing new abstractions unless they reduce complexity.
- Match existing patterns in the codebase before inventing new ones.

## Package layout

- Library code lives under `src/tinyros/`.
- Tests live under `tests/`, mirroring `src/tinyros/` so modules map directly to test folders.
- Benchmarks live under `tests/benchmark/`, excluded by default and opted in via `-m run_explicitly`.

## Logging scope naming

All logging goes through `goggles`. Scopes passed to `get_logger()` MUST:

- Be dot-separated.
- Be rooted at the top-level package (`tinyros`).
- Go from coarse to fine.
- Stop at a meaningful sub-component boundary -- do not include class, file, or algorithm names.

Examples:

| File | Scope |
|---|---|
| `src/tinyros/node.py` | `tinyros.node` |
| `src/tinyros/transport.py` | `tinyros.transport` |
| `tests/benchmark/test_publish_fn_speed.py` | `tinyros.benchmark` |
