# Benchmarks

Latency and throughput benchmarks for TinyROS, with parity baselines against
[`portal`](https://pypi.org/project/portal/) and ROS 2.

The benchmark suite lives in [`tests/benchmark/`](../../tests/benchmark/) and is
gated behind the `run_explicitly` pytest mark so the default test suite stays
fast.

```bash
uv run pytest -m run_explicitly tests/benchmark/
```

---

## Suites

| Suite | Path | Optional dependency | Install |
|---|---|---|---|
| TinyROS (native) | [`tests/benchmark/tinyros/`](../../tests/benchmark/tinyros/) | none | core install is enough |
| Portal parity | [`tests/benchmark/portal/`](../../tests/benchmark/portal/), [`tests/benchmark/test_publish_fn_speed.py`](../../tests/benchmark/test_publish_fn_speed.py) | [`portal`](https://pypi.org/project/portal/) | `uv sync --extra portal` |
| ROS 2 baseline | [`tests/benchmark/ros2/`](../../tests/benchmark/ros2/) | ROS 2 Humble (conda) | see [ROS 2 baseline](#ros-2-baseline) |

The portal suites call `pytest.importorskip("portal")`, so running them without
the `[portal]` extra yields a clean skip rather than a collection error. The
ROS 2 directory is excluded from default collection via `norecursedirs` in
`pyproject.toml`.

---

## TinyROS native suite

### Two-process round-trip benchmark

[`tests/benchmark/tinyros/test_speed_interprocess.py`](../../tests/benchmark/tinyros/test_speed_interprocess.py)

Spawns a subscriber process and a publisher process (the pytest worker),
measures per-message round-trip latency across a small matrix of payload types
and sizes, and reports min / median / p50 / p95 / p99 / max / mean / stdev.

**Design** -- mirrors the goggles `examples/105_benchmark.py` design:

- every payload category is isolated in its own process pair so the transport
  state does not leak between cases;
- scalars, strings, bytes, and ndarray sweeps (CPU only) are covered;
- ndarray sizes span both the inline (< 64 KiB) and the shared-memory
  (>= 64 KiB) code paths, so the shm fast path can be observed kicking in.

**Correctness instrumentation** (added on top of the latency bench):

- every iteration `i` builds a payload that encodes `i` as a *pattern* in a
  type-specific way (leading decimal digits for strings/bytes, `arr.flat[0]`
  for ndarrays, the value itself for scalars);
- the subscriber's callback decodes `i` from the incoming payload, increments
  a monotonic counter, and returns `(i, counter)`;
- the publisher asserts both fields on every reply -- so per-message content
  correctness *and* end-to-end delivery count are verified without adding
  significant measurement overhead (the assertions run after `future.result()`,
  outside the timed region).

**Run:**

```bash
uv run pytest -m run_explicitly \
    tests/benchmark/tinyros/test_speed_interprocess.py -s
```

### CPU/GPU payload sweep

[`tests/benchmark/tinyros/test_speed_tinyros.py`](../../tests/benchmark/tinyros/test_speed_tinyros.py)

Benchmarks tinyros message passing speed across CPU and GPU payloads.

**Run:**

```bash
uv run pytest -m run_explicitly tests/benchmark/tinyros/test_speed_tinyros.py
```

---

## Portal parity suite

Comparison runs that use the same harness against [`portal`](https://pypi.org/project/portal/).

### Install

Requires the optional `[portal]` extra:

```bash
uv sync --extra portal
# or
pip install -e '.[portal]'
```

### CPU/GPU payload sweep

[`tests/benchmark/portal/test_speed_portal.py`](../../tests/benchmark/portal/test_speed_portal.py)

Mirrors the TinyROS CPU/GPU sweep against portal's transport.

```bash
uv run pytest -m run_explicitly tests/benchmark/portal/test_speed_portal.py
```

### `publish()` latency under a slow subscriber

[`tests/benchmark/test_publish_fn_speed.py`](../../tests/benchmark/test_publish_fn_speed.py)

Measures `publish()` latency in the publisher worker while a separate
subscriber worker sleeps inside its callback.

- Worker A: slow subscriber (sleep inside callback)
- Worker B: fast publisher measuring `publish()` latency

Despite the `test_` filename this is a runnable script, not a pytest test.
When collected by pytest it is skipped cleanly if `portal` is not installed
(only `portal.Process` is used, to spawn the subscriber worker).

**Run:**

```bash
GOGGLES_PORT=8374 uv run python -m \
    tests.benchmark.test_publish_fn_speed \
    --num-msgs 15000 --sleep-ms 100 --wandb
```

---

## ROS 2 baseline

[`tests/benchmark/ros2/`](../../tests/benchmark/ros2/) measures end-to-end
message latency using **ROS 2** (publisher -> subscriber) for different
payload sizes. It is intended as a baseline for comparison with the TinyROS
and portal suites above.

### Install ROS 2

Install the dependencies **in the order below**:

#### Create a dedicated conda environment

```bash
conda create -n ros2-bench python=3.10
conda activate ros2-bench
```

#### Install Python requirements for the benchmark

```bash
pip install -r tests/benchmark/ros2/requirements.txt
```

#### Install ROS 2

Add the following channels to the environment:

```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
```

Then install ROS 2:

```bash
conda install ros-humble-desktop
conda deactivate
conda activate ros2-bench
```

### Run

```bash
python -m tests.benchmark.ros2.test_runner_singleprocess
python -m tests.benchmark.ros2.test_runner_multiprocess
```
