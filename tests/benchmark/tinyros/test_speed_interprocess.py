r"""Two-process TinyROS transport benchmark.

Spawns a subscriber process and a publisher process (the pytest worker),
measures per-message round-trip latency across a small matrix of
payload types and sizes, and reports min / median / p50 / p95 / p99 /
max / mean / stdev.

Mirrors the goggles ``examples/105_benchmark.py`` design:

- every payload category is isolated in its own process pair so the
  transport state does not leak between cases;
- scalars, strings, bytes, and ndarray sweeps (CPU only) are covered;
- ndarray sizes span both the inline (< 64 KiB) and the shared-memory
  (>= 64 KiB) code paths, so we can see the shm fast path kick in.

Run with::

    uv run pytest -m run_explicitly \
        tests/benchmark/tinyros/test_speed_interprocess.py -s
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import socket
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tinyros import (
    TinyNetworkConfig,
    TinyNode,
    TinyNodeDescription,
    TinySubscription,
)

RESULTS_DIR = Path(__file__).parent / "results_ipc"
CSV_DIR = RESULTS_DIR / "csv"

REPETITIONS = 2000
WARMUP = 50
SLEEP_BETWEEN_ITERS_S = 0.0
CONNECT_WAIT_S = 1.0


def _get_free_port() -> int:
    """Ask the kernel for a free loopback port.

    Returns:
        A TCP port that was free at the moment the function ran.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _build_config(pub_port: int, sub_port: int) -> dict:
    """Build the dict the subscriber process will use to configure its node.

    Args:
        pub_port: Port the publisher will bind on.
        sub_port: Port the subscriber will bind on.

    Returns:
        A plain-Python config dict (picklable across processes).
    """
    return {
        "pub_port": pub_port,
        "sub_port": sub_port,
    }


def _make_network(pub_port: int, sub_port: int) -> TinyNetworkConfig:
    """Construct the two-node topology shared by pub and sub.

    Args:
        pub_port: Publisher's TCP port.
        sub_port: Subscriber's TCP port.

    Returns:
        Immutable network config wiring ``pub -> sub`` on ``topic``.
    """
    return TinyNetworkConfig(
        nodes={
            "pub": TinyNodeDescription(port=pub_port, host="127.0.0.1"),
            "sub": TinyNodeDescription(port=sub_port, host="127.0.0.1"),
        },
        connections={
            "pub": {
                "topic": [TinySubscription(actor="sub", cb_name="on_msg")],
            },
        },
    )


class _BenchSub(TinyNode):
    """Minimal subscriber whose callback returns ``None``."""

    def on_msg(self, _msg: Any) -> None:
        """Discard the payload; we only time the round trip.

        Args:
            _msg: Forwarded message (unused).
        """
        return None


def _subscriber_entry(
    cfg_dict: dict,
    ready_evt: mp.synchronize.Event,
    stop_evt: mp.synchronize.Event,
) -> None:
    """Subprocess target that brings up the subscriber and waits to stop.

    Args:
        cfg_dict: Picklable config dict (see :func:`_build_config`).
        ready_evt: Set once the subscriber's server is bound and listening.
        stop_evt: When set, the subscriber shuts down and exits.
    """
    cfg = _make_network(cfg_dict["pub_port"], cfg_dict["sub_port"])
    sub = _BenchSub("sub", cfg, bind_host="127.0.0.1")
    ready_evt.set()
    try:
        stop_evt.wait()
    finally:
        sub.shutdown()


def _percentile(values: list[float], q: float) -> float:
    """Return the ``q``-quantile (``0 <= q <= 1``) of ``values``.

    Args:
        values: Samples to compute the percentile from.
        q: Quantile in ``[0, 1]``.

    Returns:
        Interpolated percentile value; 0 if ``values`` is empty.
    """
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
    return sorted(values)[k]


def _stats(lat_ms: list[float]) -> dict[str, float]:
    """Compute a fixed set of summary statistics from latency samples.

    Args:
        lat_ms: Per-call latencies in milliseconds.

    Returns:
        Mapping with keys ``min``, ``median``, ``mean``, ``std``, ``p50``,
        ``p95``, ``p99``, ``max``.
    """
    return {
        "min": min(lat_ms),
        "median": statistics.median(lat_ms),
        "mean": statistics.mean(lat_ms),
        "std": statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0,
        "p50": _percentile(lat_ms, 0.50),
        "p95": _percentile(lat_ms, 0.95),
        "p99": _percentile(lat_ms, 0.99),
        "max": max(lat_ms),
    }


def _print_row(label: str, nbytes: int, stats: dict[str, float]) -> None:
    """Print a single summary row to stdout.

    Args:
        label: Short name of the payload category.
        nbytes: Approximate payload size in bytes.
        stats: Output of :func:`_stats`.
    """
    print(
        f"{label:<22} {nbytes:>10} "
        f"{stats['min']:>8.3f} {stats['p50']:>8.3f} "
        f"{stats['mean']:>8.3f} {stats['p95']:>8.3f} "
        f"{stats['p99']:>8.3f} {stats['max']:>8.3f}"
    )


_PAYLOADS: list[tuple[str, Callable[[], Any], int]] = [
    ("float", lambda: 3.14159, 8),
    ("str_64B", lambda: "x" * 64, 64),
    ("str_1KB", lambda: "x" * 1024, 1024),
    ("bytes_4KB", lambda: b"x" * 4096, 4096),
    ("ndarray_1x1_f32", lambda: np.zeros((1, 1), dtype=np.float32), 4),
    (
        "ndarray_16x16_f32",
        lambda: np.zeros((16, 16), dtype=np.float32),
        16 * 16 * 4,
    ),
    (
        "ndarray_64x64_f32",
        lambda: np.zeros((64, 64), dtype=np.float32),
        64 * 64 * 4,
    ),
    (
        "ndarray_128x128_f32",  # inline (< 64 KiB)
        lambda: np.zeros((128, 128), dtype=np.float32),
        128 * 128 * 4,
    ),
    (
        "ndarray_256x256_f32",  # shm path (>= 64 KiB)
        lambda: np.zeros((256, 256), dtype=np.float32),
        256 * 256 * 4,
    ),
    (
        "ndarray_512x512_f32",
        lambda: np.zeros((512, 512), dtype=np.float32),
        512 * 512 * 4,
    ),
    (
        "ndarray_1024x1024_f32",
        lambda: np.zeros((1024, 1024), dtype=np.float32),
        1024 * 1024 * 4,
    ),
]


def _run_case(label: str, build_payload: Callable[[], Any]) -> list[float]:
    """Run one payload category in isolated pub / sub processes.

    The subscriber lives in a spawned process; the publisher lives here
    (the pytest worker). Measuring ``publish -> future.result()`` gives
    the round-trip cost of the wire plus one callback dispatch.

    Args:
        label: Short name of the payload category.
        build_payload: Zero-arg factory that produces a fresh payload.

    Returns:
        Per-iteration latencies in milliseconds (length ``REPETITIONS``).
    """
    pub_port = _get_free_port()
    sub_port = _get_free_port()
    cfg_dict = _build_config(pub_port, sub_port)

    ctx = mp.get_context("spawn")
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()
    sub_proc = ctx.Process(
        target=_subscriber_entry,
        args=(cfg_dict, ready_evt, stop_evt),
        daemon=False,
    )
    sub_proc.start()

    if not ready_evt.wait(timeout=10.0):
        sub_proc.terminate()
        sub_proc.join(timeout=5.0)
        raise AssertionError(f"[{label}] subscriber never signalled ready")

    time.sleep(CONNECT_WAIT_S)

    pub = TinyNode(
        "pub",
        _make_network(pub_port, sub_port),
        bind_host="127.0.0.1",
    )
    try:
        for _ in range(WARMUP):
            fut = pub.publish("topic", build_payload())[0]
            fut.result(timeout=5.0)

        latencies_ms: list[float] = []
        for _ in range(REPETITIONS):
            payload = build_payload()
            t0 = time.perf_counter()
            fut = pub.publish("topic", payload)[0]
            fut.result(timeout=10.0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if SLEEP_BETWEEN_ITERS_S > 0:
                time.sleep(SLEEP_BETWEEN_ITERS_S)
        return latencies_ms
    finally:
        pub.shutdown()
        stop_evt.set()
        sub_proc.join(timeout=10.0)
        if sub_proc.is_alive():
            sub_proc.terminate()
            sub_proc.join(timeout=5.0)


def _save_csv(
    rows: list[tuple[str, int, dict[str, float]]], out_path: Path
) -> None:
    """Persist per-case summary stats as CSV for offline analysis.

    Args:
        rows: ``(label, nbytes, stats)`` tuples to serialize.
        out_path: Destination CSV path; parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "nbytes",
                "min_ms",
                "median_ms",
                "mean_ms",
                "std_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
            ]
        )
        for label, nbytes, stats in rows:
            writer.writerow(
                [
                    label,
                    nbytes,
                    f"{stats['min']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['p50']:.6f}",
                    f"{stats['p95']:.6f}",
                    f"{stats['p99']:.6f}",
                    f"{stats['max']:.6f}",
                ]
            )


@pytest.mark.run_explicitly
def test_interprocess_latency_matrix() -> None:
    """Produce a latency table for every payload category in its own process."""
    os.makedirs(CSV_DIR, exist_ok=True)
    rows: list[tuple[str, int, dict[str, float]]] = []
    header_fmt = (
        f"{'case':<22} {'bytes':>10} "
        f"{'min':>8} {'p50':>8} {'mean':>8} "
        f"{'p95':>8} {'p99':>8} {'max':>8}"
    )
    print()
    print("=== TinyROS two-process round-trip latency (ms) ===")
    print(f"repetitions={REPETITIONS}, warmup={WARMUP}, host=127.0.0.1")
    print(header_fmt)
    print("-" * len(header_fmt))
    for label, build_payload, nbytes in _PAYLOADS:
        lat_ms = _run_case(label, build_payload)
        stats = _stats(lat_ms)
        _print_row(label, nbytes, stats)
        rows.append((label, nbytes, stats))
    _save_csv(rows, CSV_DIR / "interprocess_latency.csv")
    assert len(rows) == len(_PAYLOADS), (
        f"expected one row per payload, got {len(rows)}"
    )
