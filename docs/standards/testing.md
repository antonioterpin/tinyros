# Testing policy

## Test framework

- **Always use `pytest`** for testing.

## Test structure

- Test directories **mirror `src/tinyros/`** so you can find tests by module path.
- **Each test folder has a `README.md`** explaining what the tests inside cover.
  Write the README *before* writing the tests (docs-first approach).
- **Each `__init__.py`** in test folders has a module docstring summarizing
  the folder's scope.
- **Keep test files small and focused.** One file per concern or layer -- e.g.,
  `test_network_config.py`, `test_transport_framing.py`, `test_node_pubsub.py`.
- Each test file has a **module-level docstring** explaining what behavior it verifies.

## Test design

- **Functionality-driven tests only.** Test behavior and contracts, not
  implementation details. Drop tests that just verify a constructor sets an
  attribute or that a type check fires.
- Use **multiple small unit tests** rather than one large test.
- Add **integration tests** when behavior spans multiple processes or
  crosses the transport boundary (e.g., publisher -> subscriber).

## Docs-first workflow

When adding tests to a new area:

1. **Write a `README.md`** in the target folder listing what will be tested.
2. **Write the test file** with a module docstring explaining the tested behavior.
3. Each test function gets a **one-line docstring** saying what contract it verifies.
4. Implement the tests, run them, iterate.

## Testing best practices

- Prefer `pytest.mark.parametrize` over repeated tests.
- **Always check `tests/conftest.py`** for existing fixtures before adding new ones.
- **Always include a descriptive message in `assert` statements** to explain
  what exactly failed and why (e.g., `assert x == y, f"Expected {y}, got {x}"`).
- Use `pytest.mark.skip(reason=...)` for tests that need adaptation, with a
  clear reason explaining what changed.
- **Benchmarks are gated** behind `pytest.mark.run_explicitly`: they only
  run when selected with `-m run_explicitly` to keep the default suite fast.

## Running tests

```bash
# Full unit suite (default -- excludes run_explicitly)
uv run pytest

# A specific module
uv run pytest tests/test_transport.py

# Benchmarks only
uv run pytest -m run_explicitly tests/benchmark/
```
