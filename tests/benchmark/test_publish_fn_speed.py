"""Measure ``publish()`` latency while a separate subscriber worker blocks.

Despite the ``test_`` filename this is a runnable script, not a pytest test;
it is skipped cleanly if the optional ``[portal]`` extra is not installed.

See ``docs/guides/benchmarks.md`` for install and run instructions.
"""

from __future__ import annotations

import argparse
import datetime
import time
from statistics import mean, median, stdev

import goggles as gg
import numpy as np
import pytest

portal = pytest.importorskip(
    "portal",
    reason="Install the `[portal]` extra to run the portal-based benchmark.",
)

from tinyros import TinyNetworkConfig, TinyNode  # noqa: E402

logger = gg.get_logger(
    "tinyros.benchmark",
    scope="tinyros.benchmark",
    with_metrics=True,
)

gg.attach(
    gg.ConsoleHandler(
        name="tinyros.console",
        level=gg.INFO,
    ),
    scopes=["tinyros"],
)

# Global variable: latency ranges in ms (must be sorted)
# Example: [1.0, 10.0] creates ranges: <1.0, [1.0-10.0), >=10.0
LATENCY_RANGES = [
    0.02,
    0.04,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
]


def validate_ranges(ranges: list) -> None:
    """Validate that ranges are sorted in ascending order.

    Args:
        ranges: List of latency thresholds in ms.

    Raises:
        ValueError: If ranges are not sorted.
    """
    if ranges != sorted(ranges):
        raise ValueError(
            f"LATENCY_RANGES must be sorted in ascending order. Got: {ranges}"
        )


_NETWORK_CONFIG = {
    "nodes": {
        "publisher": {"port": 9546, "host": "localhost"},
        "slow_sub": {"port": 9430, "host": "localhost"},
    },
    "connections": {
        "publisher": {
            "topic0": [
                {
                    "actor": "slow_sub",
                    "cb_name": "on_topic0",
                }
            ],
        },
    },
}

NETWORK_CONFIG = TinyNetworkConfig.load_from_config(config=_NETWORK_CONFIG)


class SlowSubscriber(TinyNode):
    """Subscriber node with slow callback."""

    def __init__(self, *, sleep_ms: float) -> None:
        """Initialize subscriber node."""
        super().__init__(name="slow_sub", network_config=NETWORK_CONFIG)
        self.sleep_s = sleep_ms / 1000.0

    def on_topic0(self, msg: float) -> float:
        """Callback for topic0 that simulates slow processing."""
        time.sleep(self.sleep_s)
        return msg + 1


class FastPublisher(TinyNode):
    """Publisher node measuring publish latency."""

    def __init__(self, *, num_msgs: int, delay: float = 0.0) -> None:
        """Initialize publisher node."""
        super().__init__(name="publisher", network_config=NETWORK_CONFIG)
        self.delay = delay
        self.num_msgs = num_msgs

    def run(self) -> list[float]:
        """Publish messages and measure latency."""
        times: list[float] = []
        start_time = time.perf_counter()

        logger.info("Starting publish benchmark")
        for i in range(self.num_msgs):
            value = float(np.random.randn())

            t0 = time.perf_counter()
            self.publish("topic0", value)
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

            if i % 500 == 0:
                print(f"Published {i}/{self.num_msgs} messages")

            time.sleep(self.delay)  # Optional delay between publishes

        # Compute statistics
        logger.info("\n\n")
        logger.info("=== Publish Latency Statistics ===")
        logger.info(f"Total measurements: {len(times)}")
        logger.info(f"Total time: {(time.perf_counter() - start_time):.2f} seconds")

        if len(times) > 0:
            min_latency = min(times)
            max_latency = max(times)
            mean_latency = mean(times)
            median_latency = median(times)
            stdev_latency = stdev(times) if len(times) > 1 else 0.0

            logger.info(f"Min latency: {min_latency:.6f} ms")
            logger.info(f"Max latency: {max_latency:.6f} ms")
            logger.info(f"Mean latency: {mean_latency:.6f} ms")
            logger.info(f"Median latency: {median_latency:.6f} ms")
            logger.info(f"Std dev: {stdev_latency:.6f} ms")

            # Classify latencies into ranges
            latency_counters = [0] * (len(LATENCY_RANGES) + 1)
            for latency in times:
                # Default to last bucket (out of range)
                range_idx = len(LATENCY_RANGES)
                for i, threshold in enumerate(LATENCY_RANGES):
                    if latency < threshold:
                        range_idx = i
                        break
                latency_counters[range_idx] += 1

            logger.info("\n\n")
            logger.info("=== Latency Range Distribution ===")
            for i, threshold in enumerate(LATENCY_RANGES):
                if i == 0:
                    logger.info(f"  < {threshold:.3f} ms: {latency_counters[i]}")
                else:
                    logger.info(
                        f"  [{LATENCY_RANGES[i-1]:.3f}, {threshold:.3f}) ms: {latency_counters[i]}"
                    )
            logger.info(f"  >= {LATENCY_RANGES[-1]:.3f} ms: {latency_counters[-1]}")

        return times


def worker_slow_sub(sleep_ms: float) -> None:
    """Worker process for slow subscriber."""
    SlowSubscriber(sleep_ms=sleep_ms)
    while True:
        time.sleep(1)


def worker_fast_pub(num_msgs: int, delay: float = 0.0) -> list[float]:
    """Worker process for fast publisher."""
    pub = FastPublisher(num_msgs=num_msgs, delay=delay)
    return pub.run()


def main(*, num_msgs: int, sleep_ms: float, delay: float = 0.0) -> list[float]:
    """Master process that launches workers."""
    slow = portal.Process(
        worker_slow_sub,
        sleep_ms,
        name="slow_sub_worker",
        start=True,
    )

    logger.info(f"Slow subscriber worker started pid={slow.pid}")

    try:
        # small delay so subscriber is ready
        time.sleep(1.0)

        times = worker_fast_pub(num_msgs=num_msgs, delay=delay)

        logger.info(f"Fast publisher completed with {len(times)} measurements")

        return times
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return []
    finally:
        slow.kill(timeout=5)
        logger.info("Workers terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyROS communication benchmark")
    parser.add_argument(
        "--num-msgs",
        type=int,
        default=5000,
        help="Number of publish calls",
    )
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=100.0,
        help="Subscriber callback sleep in ms",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between publish calls in seconds",
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gg.attach(
        gg.WandBHandler(
            project="tinyros-benchmarks",
            config={
                "experiment": "benchmark_communication",
                "num_msgs": args.num_msgs,
                "sleep_ms": args.sleep_ms,
                "run": timestamp,
            },
        ),
        scopes=["tinyros"],
    )

    # Validate ranges
    validate_ranges(LATENCY_RANGES)

    logger.info("Starting TinyROS communication benchmark")

    times = main(
        num_msgs=args.num_msgs,
        sleep_ms=args.sleep_ms,
        delay=args.delay,
    )

    if args.wandb:
        for i, dt in enumerate(times):
            logger.scalar(
                name="publish_latency_ms",
                value=dt,
                step=i,
            )

    gg.finish()
