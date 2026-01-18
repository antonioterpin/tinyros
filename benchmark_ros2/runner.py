"""Benchmark ROS2 latency between publisher and subscriber nodes."""

import csv
import os
import statistics
from typing import Tuple

import jax
import numpy as np
import rclpy

from benchmark_ros2.publisher_node import LatencyPublisher
from benchmark_ros2.subscriber_node import LatencySubscriber

REPETITIONS = 1000

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results",
)


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

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(pub)
    executor.add_node(sub)

    arr = np.zeros(shape, dtype=np.float32)
    nbytes = arr.nbytes

    latencies = []

    try:
        for _ in range(REPETITIONS):
            pub.publish_array(arr)

            while len(sub.recv_ts) < len(pub.send_ts):
                executor.spin_once(timeout_sec=0.001)

            latencies.append(
                sub.recv_ts[-1] - pub.send_ts[-1]
            )

    finally:
        pub.destroy_node()
        sub.destroy_node()
        rclpy.shutdown()

    lat_us = [x * 1e6 for x in latencies]

    stats = {
        "min": min(lat_us),
        "max": max(lat_us),
        "mean": statistics.mean(lat_us),
        "std": statistics.stdev(lat_us) if len(lat_us) > 1 else 0.0,
        "median": statistics.median(lat_us),
        "p95_best": statistics.quantiles(lat_us, n=20)[0],    # 5th percentile
        "p95_worst": statistics.quantiles(lat_us, n=20)[18],  # 95th percentile
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(
        RESULTS_DIR,
        f"ros_latency_{pub_hw}_to_{sub_hw}.csv",
    )

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "pub_hw", "sub_hw",
                "height", "width", "bytes",
                "min_us", "max_us",
                "mean_us", "std_us",
                "median_us",
                "p95_best_us", "p95_worst_us",
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
