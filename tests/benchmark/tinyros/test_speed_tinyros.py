"""Benchmark TinyROS message-passing throughput across CPU and GPU payloads.

See ``docs/guides/benchmarks.md`` for run instructions.
"""

import csv
import os
import socket
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tinyros import (
    TinyNetworkConfig,
    TinyNode,
    TinyNodeDescription,
    TinySubscription,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IMG_DIR = os.path.join(RESULTS_DIR, "images")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

REPETITIONS = 1000
VISUALIZE = True
WARMUP = 10
SLEEP_BETWEEN_ITERS_S = 1e-3


def save_latency_plot(
    *,
    lat_ms: Sequence[float],
    out_path: str | os.PathLike,
    title: str,
    shape: tuple[int, int],
    pub_hw: str,
    sub_hw: str,
    nbytes: int,
    dpi: int = 150,
    show_p50_p95: bool = True,
) -> None:
    """Save a PNG plot of latency (ms) over iterations."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(lat_ms, dtype=np.float64)
    x = np.arange(len(y), dtype=np.int32)

    fig = plt.figure()
    plt.plot(x, y)  # no explicit colors
    plt.xlabel("Iteration")
    plt.ylabel("Latency [ms]")

    header = (
        f"{title}\n"
        f"{pub_hw} -> {sub_hw} | shape={shape[0]}x{shape[1]} | bytes={nbytes}"
    )
    plt.title(header)

    if show_p50_p95 and len(y) > 0:
        median = float(np.percentile(y, 50))
        mean = float(np.mean(y))
        p95 = float(np.percentile(y, 95))
        plt.axhline(median, linestyle="--", linewidth=1, color="red")
        plt.axhline(mean, linestyle="--", linewidth=1, color="green")
        plt.axhline(p95, linestyle="--", linewidth=1)
        plt.legend(
            [
                "latency",
                f"median={median:.3f}ms",
                f"mean={mean:.3f}ms",
                f"p95={p95:.3f}ms",
            ],
            loc="best",
        )
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def get_free_port() -> int:
    """Find a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def wait_port_free(port: int, *, timeout_s: float = 2.0) -> None:
    """Wait until the given port is free (closed)."""
    t0 = time.perf_counter()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            ok = s.connect_ex(("127.0.0.1", port))  # 0 = open, !=0 = closed
        if ok != 0:
            return
        if time.perf_counter() - t0 > timeout_s:
            raise AssertionError(f"Port {port} still open after {timeout_s}s")
        time.sleep(0.01)


class SinkNode(TinyNode):
    """Subscriber node with optional GPU staging."""

    def __init__(
        self,
        name: str,
        network: TinyNetworkConfig,
        *,
        sub_hw: str = "cpu",
    ) -> None:
        """Initialize the sink node."""
        self.sub_hw = sub_hw
        self.recv_ts: list[float] = []
        super().__init__(name, network)

    def on_msg(self, msg: np.ndarray) -> None:
        """Handle incoming message."""

        if self.sub_hw == "gpu":
            # host -> device transfer
            arr = jax.device_put(msg, jax.devices("gpu")[0])
            jax.block_until_ready(arr)

        t = time.perf_counter()
        self.recv_ts.append(t)


@pytest.mark.run_explicitly
@pytest.mark.parametrize("sub_hw", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
def test_latency_cpu_gpu_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payload_factory: Callable,
    shape: tuple[int, int],
    sub_hw: str,
) -> None:
    """Test latency of tinyros message passing with CPU/GPU payloads."""
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    monkeypatch.setattr("atexit.register", lambda *a, **k: None)

    meta = payload_factory(shape)
    payload = meta["payload"]
    pub_hw = meta["pub_hw"]
    nbytes = meta["bytes"]

    # skip impossible GPU cases early
    if sub_hw == "gpu" or pub_hw == "gpu":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("JAX GPU backend not available")

    sub_port = get_free_port()
    pub_port = get_free_port()

    net = TinyNetworkConfig(
        nodes={
            "pub": TinyNodeDescription(port=pub_port, host="localhost"),
            "sub": TinyNodeDescription(port=sub_port, host="localhost"),
        },
        connections={
            "pub": {"topic": (TinySubscription(actor="sub", cb_name="on_msg"),)}
        },
    )

    sub = SinkNode("sub", net, sub_hw=sub_hw)
    pub = TinyNode("pub", net)

    time.sleep(1)

    latencies: list[float] = []

    time.sleep(3.0)  # allow some time for connections to establish

    # warm-up
    for _ in range(WARMUP):
        if pub_hw == "gpu":
            payload_host = np.asarray(payload)
        else:
            payload_host = payload
        pub.publish("topic", payload_host)
        while len(sub.recv_ts) <= len(latencies):
            time.sleep(1e-5)
        sub.recv_ts.clear()
        time.sleep(SLEEP_BETWEEN_ITERS_S)

    for _ in range(REPETITIONS):
        t0 = time.perf_counter()

        # device -> host if needed
        if pub_hw == "gpu":
            payload_host = np.asarray(payload)
        else:
            payload_host = payload

        futures = pub.publish("topic", payload_host)
        for future in futures:
            future.result()

        # wait for receive
        while len(sub.recv_ts) <= len(latencies):
            time.sleep(1e-5)

        t1 = sub.recv_ts[len(latencies)]
        latencies.append(t1 - t0)
        time.sleep(SLEEP_BETWEEN_ITERS_S)  # avoid overwhelming the subscriber

    lat_ms = [x * 1e3 for x in latencies]

    # Save per-iteration plot
    if VISUALIZE:
        img_path = os.path.join(
            IMG_DIR,
            f"tinyros_latency_trace_{pub_hw}_to_{sub_hw}_{shape[0]}x{shape[1]}.png",
        )
        save_latency_plot(
            lat_ms=lat_ms,
            out_path=img_path,
            title="TinyROS latency trace",
            shape=shape,
            pub_hw=pub_hw,
            sub_hw=sub_hw,
            nbytes=nbytes,
        )

    stats = {
        "min": min(lat_ms),
        "max": max(lat_ms),
        "mean": statistics.mean(lat_ms),
        "std": statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0,
        "median": statistics.median(lat_ms),
        "p95_best": statistics.quantiles(lat_ms, n=20)[0],  # 5th percentile
        "p95_worst": statistics.quantiles(lat_ms, n=20)[18],  # 95th percentile
    }

    csv_path = os.path.join(
        CSV_DIR,
        f"tinyros_latency_{pub_hw}_to_{sub_hw}.csv",
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "pub_hw",
                    "sub_hw",
                    "height",
                    "width",
                    "bytes",
                    "min_ms",
                    "max_ms",
                    "mean_ms",
                    "std_ms",
                    "median_ms",
                    "p95_best_ms",
                    "p95_worst_ms",
                ]
            )
        w.writerow(
            [
                pub_hw,
                sub_hw,
                shape[0],
                shape[1],
                nbytes,
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["std"],
                stats["median"],
                stats["p95_best"],
                stats["p95_worst"],
            ]
        )

    assert len(latencies) == REPETITIONS, "Did not receive all messages"

    # shut down nodes
    pub.shutdown()
    sub.shutdown()

    # give time for sockets to close
    wait_port_free(pub_port)
    wait_port_free(sub_port)
