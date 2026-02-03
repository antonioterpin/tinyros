"""Benchmark ROS2 latency between publisher and subscriber nodes."""

import csv
import os
import statistics
import time
from pathlib import Path
from typing import Sequence, Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from benchmark_ros2.publisher_node import LatencyPublisher
from benchmark_ros2.subscriber_node import LatencySubscriber

REPETITIONS = 1000

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results",
)
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
WARMUP = 10
SLEEP_BETWEEN_ITERS_S = 1e-3
VISUALIZE = True


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
        plt.legend(["latency",
                    f"median={median:.3f}ms",
                    f"mean={mean:.3f}ms",
                    f"p95={p95:.3f}ms"],
                   loc="best",
                   )

    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_once(
    *,
    shape: Tuple[int, int],
    pub_hw: str,
    sub_hw: str,
) -> dict[str, float] | None:
    """Run a single latency benchmark between publisher and subscriber.

    Args:
        shape: Shape of the array to send.
        pub_hw: Publisher hardware ("cpu" or "gpu").
        sub_hw: Subscriber hardware ("cpu" or "gpu").

    Returns:
        A dictionary with latency statistics in microseconds. If the
        benchmark could not be run (e.g., due to missing GPU), returns None.
    """
    rclpy.init()

    if pub_hw == "gpu" or sub_hw == "gpu":
        try:
            jax.devices("gpu")
        except Exception:
            print("Skipping GPU case: JAX GPU not available")
            return None

    pub = LatencyPublisher(pub_hw=pub_hw)
    sub = LatencySubscriber(sub_hw=sub_hw)

    executor = SingleThreadedExecutor()
    executor.add_node(pub)
    executor.add_node(sub)

    arr = np.zeros(shape, dtype=np.float32)
    nbytes = arr.nbytes
    if pub_hw == "gpu":
        arr = jax.device_put(arr, jax.devices("gpu")[0])
        jax.block_until_ready(arr)

    latencies = []

    time.sleep(3.0)  # Allow some time for DDS discovery

    # Warm-up
    for _ in range(WARMUP):
        pub.publish_array(arr)

        while len(sub.recv_ts) < len(pub.send_ts):
            executor.spin_once(timeout_sec=1e-5)

        # clear state so warm-up does not contaminate measurements
        sub.recv_ts.clear()
        pub.send_ts.clear()
        time.sleep(SLEEP_BETWEEN_ITERS_S)

    try:
        for _ in range(REPETITIONS):
            pub.publish_array(arr)

            while len(sub.recv_ts) < len(pub.send_ts):
                executor.spin_once(timeout_sec=1e-5)

            latencies.append(
                sub.recv_ts[-1] - pub.send_ts[-1]
            )

            # avoid overwhelming the subscriber
            time.sleep(SLEEP_BETWEEN_ITERS_S)

    finally:
        pub.destroy_node()
        sub.destroy_node()
        rclpy.shutdown()

    lat_ms = [x * 1e3 for x in latencies]

    if VISUALIZE:
        img_path = os.path.join(
            IMG_DIR,
            f"ros2_latency_trace_{pub_hw}_to_{sub_hw}_{shape[0]}x{shape[1]}.png",
        )

        save_latency_plot(
            lat_ms=lat_ms,
            out_path=img_path,
            title="ROS2 latency trace",
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
        "p95_best": statistics.quantiles(lat_ms, n=20)[0],    # 5th percentile
        "p95_worst": statistics.quantiles(lat_ms, n=20)[18],  # 95th percentile
    }

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    csv_path = os.path.join(
        CSV_DIR,
        f"ros2_latency_{pub_hw}_to_{sub_hw}.csv",
    )

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "pub_hw", "sub_hw",
                "height", "width", "bytes",
                "min_ms", "max_ms",
                "mean_ms", "std_ms",
                "median_ms",
                "p95_best_ms", "p95_worst_ms",
            ])
        w.writerow([
            pub_hw, sub_hw,
            shape[0], shape[1], nbytes,
            stats["min"], stats["max"],
            stats["mean"], stats["std"],
            stats["median"],
            stats["p95_best"], stats["p95_worst"],
        ])

    return stats


if __name__ == "__main__":
    for pub_hw in PUB_HWS:
        for sub_hw in SUB_HWS:
            for shape in SHAPES:
                print(
                    f"Running ROS latency: {pub_hw} -> {sub_hw}, "
                    f"shape={shape}"
                )
                run_once(
                    shape=shape,
                    pub_hw=pub_hw,
                    sub_hw=sub_hw,
                )
