"""Benchmark ROS2 latency between publisher and subscriber in separate processes."""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import statistics
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Tuple
from collections.abc import Sequence

import jax
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from tests.benchmark.ros2.test_publisher_node import LatencyPublisher
from tests.benchmark.ros2.test_subscriber_node import LatencySubscriber

WARMUP = 10
SLEEP_BETWEEN_ITERS_S = 1e-3
VISUALIZE = True
GPU_PUB = "0"
GPU_SUB = "1"

REPETITIONS = 1000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_mp")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
IMG_DIR = os.path.join(RESULTS_DIR, "images")

SHAPES = [
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
]

PUB_HWS = ["cpu", "gpu"]
SUB_HWS = ["cpu", "gpu"]

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_PUB


def _spin_subscriber(*, sub_hw: str, ack_q: mp.Queue, stop_ev: Event) -> None:
    """Subscriber process: spin and push callback timestamps to ack_q."""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SUB if sub_hw == "gpu" else ""
    rclpy.init()
    node = LatencySubscriber(sub_hw=sub_hw, ack_q=ack_q)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        while rclpy.ok() and not stop_ev.is_set():
            ex.spin_once(timeout_sec=1e-5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _wait_for_match(pub: LatencyPublisher, *, timeout_s: float = 2.0) -> None:
    """Wait until publisher sees at least one subscription (DDS match)."""
    t0 = time.perf_counter()
    while pub.publisher.get_subscription_count() == 0:
        if time.perf_counter() - t0 > timeout_s:
            # Not fatal, but usually indicates discovery issues
            return
        time.sleep(0.1)


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
) -> None:
    """Save a PNG plot of latency (ms) over iterations."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(lat_ms, dtype=np.float64)
    x = np.arange(len(y), dtype=np.int32)

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Latency [ms]")

    plt.title(
        f"{title}\n"
        f"{pub_hw} -> {sub_hw} | "
        f"shape={shape[0]}x{shape[1]} | bytes={nbytes}"
    )

    if len(y) > 0:
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


def run_once_mp(
    *,
    shape: tuple[int, int],
    pub_hw: str,
    sub_hw: str,
) -> dict[str, float] | None:
    """Run a single multiprocess latency benchmark."""
    # Skip impossible GPU cases early
    if pub_hw == "gpu" or sub_hw == "gpu":
        try:
            jax.devices("gpu")
        except RuntimeError:
            print("Skipping GPU case: JAX GPU backend not available")
            return None

    ctx = mp.get_context("spawn")
    ack_q: mp.Queue = ctx.Queue(maxsize=REPETITIONS * 2)
    stop_ev: Event = ctx.Event()

    sub_p = ctx.Process(
        target=_spin_subscriber,
        kwargs={
            "sub_hw": sub_hw,
            "ack_q": ack_q,
            "stop_ev": stop_ev,
        },
        daemon=True,
    )
    sub_p.start()

    # Publisher lives in THIS process
    rclpy.init()
    pub = LatencyPublisher(pub_hw=pub_hw)

    ex = SingleThreadedExecutor()
    ex.add_node(pub)

    arr = np.zeros(shape, dtype=np.float32)
    nbytes = arr.nbytes
    if pub_hw == "gpu":
        arr = jax.device_put(arr, jax.devices("gpu")[0])
        jax.block_until_ready(arr)

    latencies: list[float] = []

    try:
        _wait_for_match(pub, timeout_s=2.0)

        # Warm-up
        for _ in range(WARMUP):
            ex.spin_once(timeout_sec=0.0)
            pub.publish_array(arr)
            ack_q.get(timeout=5.0)  # drain one ack
            pub.send_ts.clear()
            time.sleep(SLEEP_BETWEEN_ITERS_S)

        for _ in range(REPETITIONS):
            # Spin a bit to process DDS events (discovery, etc.)
            ex.spin_once(timeout_sec=0.0)

            # Publish array
            pub.publish_array(arr)

            # Wait for receive ack from subscriber callback (t1)
            t1 = ack_q.get(timeout=10.0)
            latencies.append(t1 - pub.send_ts[-1])
            # avoid overwhelming the subscriber
            time.sleep(SLEEP_BETWEEN_ITERS_S)

    finally:
        # stop subscriber process
        stop_ev.set()
        sub_p.join(timeout=2.0)
        if sub_p.is_alive():
            sub_p.kill()

        pub.destroy_node()
        rclpy.shutdown()

    lat_ms = [x * 1e3 for x in latencies]

    if VISUALIZE:
        img_path = os.path.join(
            IMG_DIR,
            f"ros2_latency_trace_mp_{pub_hw}_to_{sub_hw}_{shape[0]}x{shape[1]}.png",
        )
        save_latency_plot(
            lat_ms=lat_ms,
            out_path=img_path,
            title="ROS2 Latency Benchmark (Multiprocess)",
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

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    csv_path = os.path.join(
        CSV_DIR,
        f"ros2_latency_mp_{pub_hw}_to_{sub_hw}.csv",
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

    return stats


if __name__ == "__main__":
    for pub_hw in PUB_HWS:
        for sub_hw in SUB_HWS:
            for shape in SHAPES:
                print(f"Running ROS2 MP latency: {pub_hw} -> {sub_hw}, shape={shape}")
                run_once_mp(shape=shape, pub_hw=pub_hw, sub_hw=sub_hw)
