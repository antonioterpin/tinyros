r"""Two-process TinyROS transport benchmark.

Spawns a subscriber process and a publisher process (the pytest worker),
measures per-message round-trip latency across a small matrix of payload
types and sizes, and reports min / median / p50 / p95 / p99 / max / mean
/ stdev.

Mirrors the goggles ``examples/105_benchmark.py`` design:

- every payload category is isolated in its own process pair so the
  transport state does not leak between cases;
- scalars, strings, bytes, and ndarray sweeps (CPU only) are covered;
- ndarray sizes span both the inline (< 64 KiB) and the shared-memory
  (>= 64 KiB) code paths, so we can see the shm fast path kick in.

Correctness instrumentation (added on top of the latency bench):

- every iteration ``i`` builds a payload that encodes ``i`` as a
  *pattern* in a type-specific way (leading decimal digits for
  strings/bytes, ``arr.flat[0]`` for ndarrays, the value itself for
  scalars);
- the subscriber's callback decodes ``i`` from the incoming payload,
  increments a monotonic counter, and returns ``(i, counter)``;
- the publisher asserts both fields on every reply -- so we verify
  per-message content correctness *and* end-to-end delivery count
  without adding significant measurement overhead (the assertions run
  after ``future.result()``, outside the timed region).

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
from multiprocessing.synchronize import Event as MpEvent
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

# Width of the decimal index prefix embedded in string / bytes payloads.
# 10 digits is enough for REPETITIONS <= 9_999_999_999 and keeps the
# prefix small compared to the smallest string payload (64 B).
_INDEX_PREFIX_LEN = 10


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
                "topic": (TinySubscription(actor="sub", cb_name="on_msg"),),
            },
        },
    )


# --- Pattern builders and extractors --------------------------------------
#
# Every builder takes an iteration index and returns a payload that
# encodes the index in a type-specific way. The matching extractor
# recovers the index on the receiver side. The sub process uses a single
# extractor that dispatches on Python type, so we don't need to ship the
# per-case extractor across the process boundary.


def _build_float(i: int) -> float:
    """Build a Python float whose value equals ``i``.

    Args:
        i: Iteration index.

    Returns:
        ``float(i)``; roundtrips exactly for any i << 2**53.
    """
    return float(i)


def _build_str(i: int, n_bytes: int) -> str:
    """Build an ASCII string of length ``n_bytes`` starting with ``i``.

    Args:
        i: Iteration index (written as a zero-padded decimal prefix).
        n_bytes: Total string length.

    Returns:
        ``f"{i:010d}" + "x" * pad``.
    """
    prefix = f"{i:0{_INDEX_PREFIX_LEN}d}"
    pad = max(0, n_bytes - _INDEX_PREFIX_LEN)
    return prefix + ("x" * pad)


def _build_bytes(i: int, n_bytes: int) -> bytes:
    """Build a byte string of length ``n_bytes`` starting with ``i``.

    Args:
        i: Iteration index (written as a zero-padded decimal prefix).
        n_bytes: Total byte length.

    Returns:
        ``f"{i:010d}".encode() + b"x" * pad``.
    """
    prefix = f"{i:0{_INDEX_PREFIX_LEN}d}".encode()
    pad = max(0, n_bytes - _INDEX_PREFIX_LEN)
    return prefix + (b"x" * pad)


def _build_ndarray(i: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Build a zero ndarray with ``flat[0] = i``.

    Args:
        i: Iteration index, stored in the first flat element.
        shape: Output shape.
        dtype: Output dtype.

    Returns:
        A fresh ndarray with its first flat element set to ``i``.
    """
    arr = np.zeros(shape, dtype=dtype)
    arr.flat[0] = i
    return arr


def _extract_index(msg: Any) -> int:
    """Recover the iteration index encoded by the builders.

    Runs inside the subscriber process. The dispatch is on the runtime
    Python type so a single callback handles every payload category.

    Args:
        msg: Received payload.

    Returns:
        The integer index embedded in the payload.

    Raises:
        TypeError: If ``msg`` is not one of the expected categories.
    """
    if isinstance(msg, (bool, int)):
        return int(msg)
    if isinstance(msg, float):
        return int(msg)
    if isinstance(msg, str):
        return int(msg[:_INDEX_PREFIX_LEN])
    if isinstance(msg, (bytes, bytearray)):
        return int(bytes(msg[:_INDEX_PREFIX_LEN]).decode())
    if isinstance(msg, np.ndarray):
        return int(msg.flat[0])
    raise TypeError(f"unsupported payload type: {type(msg).__name__}")


class _BenchSub(TinyNode):
    """Subscriber that verifies and counts incoming messages.

    On each call the subscriber decodes the iteration index from the
    payload, increments its monotonic receive counter, and returns
    ``(index, counter)``. The publisher asserts on both fields for
    every reply, which verifies:

    - that the payload was received byte-identically (the encoded
      pattern round-trips), and
    - that no message was dropped or reordered (``counter`` equals
      ``index + 1`` for every reply).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the node and the receive counter.

        Args:
            *args: Forwarded to :class:`TinyNode`.
            **kwargs: Forwarded to :class:`TinyNode`.
        """
        self._count = 0
        super().__init__(*args, **kwargs)

    def on_msg(self, msg: Any) -> tuple[int, int]:
        """Decode, count, and ack a received message.

        Args:
            msg: Forwarded payload.

        Returns:
            ``(decoded_index, post_increment_count)``.
        """
        idx = _extract_index(msg)
        self._count += 1
        return idx, self._count


def _subscriber_entry(
    cfg_dict: dict,
    ready_evt: MpEvent,
    stop_evt: MpEvent,
) -> None:
    """Subprocess target that brings up the subscriber and waits to stop.

    Args:
        cfg_dict: Picklable config dict (see :func:`_build_config`).
        ready_evt: Set once the subscriber's server is bound and listening.
        stop_evt: When set, the subscriber shuts down and exits.
    """
    cfg = _make_network(cfg_dict["pub_port"], cfg_dict["sub_port"])
    sub = _BenchSub(name="sub", network_config=cfg, bind_host="127.0.0.1")
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


def _print_row(
    label: str, nbytes: int, stats: dict[str, float], delivered: int
) -> None:
    """Print a single summary row to stdout.

    Args:
        label: Short name of the payload category.
        nbytes: Approximate payload size in bytes.
        stats: Output of :func:`_stats`.
        delivered: Number of correctly-received messages for the case.
    """
    print(
        f"{label:<22} {nbytes:>10} {delivered:>6}/{REPETITIONS:<6} "
        f"{stats['min']:>7.3f} {stats['p50']:>7.3f} "
        f"{stats['mean']:>7.3f} {stats['p95']:>7.3f} "
        f"{stats['p99']:>7.3f} {stats['max']:>7.3f}"
    )


# Builders take the iteration index and return a payload encoding it.
_PAYLOADS: list[tuple[str, Callable[[int], Any], int]] = [
    ("float", _build_float, 8),
    ("str_64B", lambda i: _build_str(i, 64), 64),
    ("str_1KB", lambda i: _build_str(i, 1024), 1024),
    ("bytes_4KB", lambda i: _build_bytes(i, 4096), 4096),
    (
        "ndarray_1x1_f32",
        lambda i: _build_ndarray(i, (1, 1), np.dtype(np.float32)),
        4,
    ),
    (
        "ndarray_16x16_f32",
        lambda i: _build_ndarray(i, (16, 16), np.dtype(np.float32)),
        16 * 16 * 4,
    ),
    (
        "ndarray_64x64_f32",
        lambda i: _build_ndarray(i, (64, 64), np.dtype(np.float32)),
        64 * 64 * 4,
    ),
    (
        "ndarray_128x128_f32",  # inline path (< 64 KiB at this shape/dtype)
        lambda i: _build_ndarray(i, (128, 128), np.dtype(np.float32)),
        128 * 128 * 4,
    ),
    (
        "ndarray_256x256_f32",  # shm path (>= 64 KiB)
        lambda i: _build_ndarray(i, (256, 256), np.dtype(np.float32)),
        256 * 256 * 4,
    ),
    (
        "ndarray_512x512_f32",
        lambda i: _build_ndarray(i, (512, 512), np.dtype(np.float32)),
        512 * 512 * 4,
    ),
    (
        "ndarray_1024x1024_f32",
        lambda i: _build_ndarray(i, (1024, 1024), np.dtype(np.float32)),
        1024 * 1024 * 4,
    ),
]


def _run_case(
    label: str, build_payload: Callable[[int], Any]
) -> tuple[list[float], int]:
    """Run one payload category in isolated pub / sub processes.

    Every iteration ``i`` sends a payload encoding ``i``, awaits the
    future, and asserts that:

    - the subscriber-decoded index equals ``i`` (content correctness),
    - the subscriber's monotonic counter equals ``i + 1`` (no drop,
      no reorder).

    Args:
        label: Short name of the payload category.
        build_payload: Factory that produces a fresh payload encoding
            its integer argument.

    Returns:
        ``(latencies_ms, delivered)``. ``delivered`` is always
        ``REPETITIONS`` when every assertion held; we still return it
        so the caller can print it.
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
    delivered = 0
    try:
        # Warm-up: sub's counter starts at 0 and ticks on every received
        # message, so iteration indices must stay monotonic across
        # warmup + measured phase.
        for i in range(WARMUP):
            fut = pub.publish("topic", build_payload(i))[0]
            got_idx, got_count = fut.result(timeout=5.0)
            assert (
                got_idx == i
            ), f"[{label}] warmup idx mismatch: sent {i}, got {got_idx}"
            assert got_count == i + 1, (
                f"[{label}] warmup count mismatch: "
                f"expected {i + 1}, got {got_count}"
            )

        latencies_ms: list[float] = []
        for step in range(REPETITIONS):
            i = WARMUP + step
            payload = build_payload(i)
            t0 = time.perf_counter()
            fut = pub.publish("topic", payload)[0]
            got_idx, got_count = fut.result(timeout=10.0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            # Assertions run *after* the timing window; they verify
            # per-message correctness and running delivery count.
            assert (
                got_idx == i
            ), f"[{label}] step {step}: expected idx {i}, got {got_idx}"
            assert got_count == i + 1, (
                f"[{label}] step {step}: expected count {i + 1}, " f"got {got_count}"
            )
            delivered += 1
            if SLEEP_BETWEEN_ITERS_S > 0:
                time.sleep(SLEEP_BETWEEN_ITERS_S)
        return latencies_ms, delivered
    finally:
        pub.shutdown()
        stop_evt.set()
        sub_proc.join(timeout=10.0)
        if sub_proc.is_alive():
            sub_proc.terminate()
            sub_proc.join(timeout=5.0)


def _save_csv(
    rows: list[tuple[str, int, int, dict[str, float]]], out_path: Path
) -> None:
    """Persist per-case summary stats as CSV for offline analysis.

    Args:
        rows: ``(label, nbytes, delivered, stats)`` tuples to serialize.
        out_path: Destination CSV path; parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "nbytes",
                "delivered",
                "expected",
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
        for label, nbytes, delivered, stats in rows:
            writer.writerow(
                [
                    label,
                    nbytes,
                    delivered,
                    REPETITIONS,
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
    """Run every payload category and verify delivery + content correctness."""
    os.makedirs(CSV_DIR, exist_ok=True)
    rows: list[tuple[str, int, int, dict[str, float]]] = []
    header_fmt = (
        f"{'case':<22} {'bytes':>10} {'delivered':>13} "
        f"{'min':>7} {'p50':>7} {'mean':>7} "
        f"{'p95':>7} {'p99':>7} {'max':>7}"
    )
    print()
    print("=== TinyROS two-process round-trip latency (ms) ===")
    print(f"repetitions={REPETITIONS}, warmup={WARMUP}, host=127.0.0.1")
    print(header_fmt)
    print("-" * len(header_fmt))
    for label, build_payload, nbytes in _PAYLOADS:
        lat_ms, delivered = _run_case(label, build_payload)
        stats = _stats(lat_ms)
        _print_row(label, nbytes, stats, delivered)
        rows.append((label, nbytes, delivered, stats))
        assert (
            delivered == REPETITIONS
        ), f"[{label}] delivery mismatch: got {delivered}/{REPETITIONS}"
    _save_csv(rows, CSV_DIR / "interprocess_latency.csv")
    assert len(rows) == len(_PAYLOADS), f"expected one row per payload, got {len(rows)}"
