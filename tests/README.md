# Test suite

Mirrors `src/tinyros/` so each test file maps to the module it covers.

## Layout

| File | Covers |
|---|---|
| `test_network_config.py` | `tinyros.node.TinyNetworkConfig` -- parsing from dict, node lookups, publisher / subscriber queries, error cases. |
| `test_transport_framing.py` | `tinyros.transport` framing helpers -- `_pack_oob` / `_unpack_oob` round-trips for scalars, strings, and numpy arrays with out-of-band buffers. |
| `test_transport_rpc.py` | `tinyros.transport.TinyServer` and `TinyClient` end-to-end -- small-payload RPC, large-payload shm side-channel, graceful close, error propagation. |
| `test_node.py` | `tinyros.TinyNode` pub/sub integration -- callback binding, multi-subscriber fanout, shutdown semantics. |

## Benchmarks

`benchmark/` contains latency / throughput comparisons against portal and
ROS 2. They are gated behind the `run_explicitly` pytest mark so the
default suite stays fast.

```bash
uv run pytest -m run_explicitly tests/benchmark/
```

### Optional benchmark dependencies

| Suite | Path | Extra dependency | Install |
|---|---|---|---|
| TinyROS (native) | `benchmark/tinyros/` | none (core install is enough) | — |
| Portal comparison | `benchmark/portal/`, `benchmark/test_publish_fn_speed.py` | [`portal`](https://pypi.org/project/portal/) | `uv sync --extra portal` or `pip install -e '.[portal]'` |
| ROS 2 comparison | `benchmark/ros2/` | ROS 2 Humble (conda) | see [`benchmark/ros2/ROS2.md`](benchmark/ros2/ROS2.md) |

The portal benchmark tests call `pytest.importorskip("portal")`, so
running them without the `[portal]` extra yields a clean skip rather
than a collection error. The ROS 2 directory is excluded from default
collection via `norecursedirs` in `pyproject.toml`.

To run only the portal parity benchmarks:

```bash
uv sync --extra portal
uv run pytest -m run_explicitly tests/benchmark/portal/
```

## Conventions

- Every test function has a one-line docstring stating what contract it
  verifies.
- Every `assert` carries a descriptive message explaining what failed.
- Network-using tests allocate ephemeral ports with the `free_port`
  fixture (see `conftest.py`) to avoid cross-test collisions.
- Tests that spin up `TinyNode`s always call `shutdown()` in a `finally`
  clause to keep ports and shared-memory blocks clean.
