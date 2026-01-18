import csv
import os
import socket
import statistics
import time
from typing import Callable

import jax
import numpy as np
import pytest

from tinyros import (TinyNetworkConfig, TinyNode, TinyNodeDescription,
                     TinySubscription)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

REPETITIONS = 1000


def get_free_port() -> int:
    """Find a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


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
            arr = jax.device_put(np.asarray(msg), jax.devices("gpu")[0])
            jax.block_until_ready(arr)

        t = time.perf_counter()
        self.recv_ts.append(t)


@pytest.mark.run_explicitly
@pytest.mark.parametrize("sub_hw", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1), (2, 2), (4, 4), (8, 8), (16, 16),
        (32, 32), (64, 64), (128, 128),
        (256, 256), (512, 512), (1024, 1024)
    ]
)
def test_latency_cpu_gpu_payloads(
        payload_factory: Callable,
        shape: tuple[int, int],
        sub_hw: str
) -> None:
    """Test latency of tinyros message passing with CPU/GPU payloads."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
            "pub": {
                "topic": [TinySubscription(actor="sub", cb_name="on_msg")]
            }
        },
    )

    sub = SinkNode("sub", net, sub_hw=sub_hw)
    pub = TinyNode("pub", net)

    time.sleep(1)

    latencies: list[float] = []

    for _ in range(REPETITIONS):
        t0 = time.perf_counter()

        # device -> host if needed
        if pub_hw == "gpu":
            jax.block_until_ready(payload)
            payload_host = np.asarray(payload)
        else:
            payload_host = payload

        pub.publish("topic", payload_host)

        # wait for receive
        while len(sub.recv_ts) <= len(latencies):
            time.sleep(0.0001)

        t1 = sub.recv_ts[len(latencies)]
        latencies.append(t1 - t0)

    lat_us = [x * 1e6 for x in latencies]

    stats = {
        "min": min(lat_us),
        "max": max(lat_us),
        "mean": statistics.mean(lat_us),
        "std": statistics.stdev(lat_us) if len(lat_us) > 1 else 0.0,
        "median": statistics.median(lat_us),
        "p95_best": statistics.quantiles(lat_us, n=20)[0],   # 5th percentile
        "p95_worst": statistics.quantiles(lat_us, n=20)[18],  # 95th percentile
    }

    csv_path = os.path.join(
        RESULTS_DIR,
        f"tinyros_latency_{pub_hw}_to_{sub_hw}.csv",
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

    assert len(latencies) == REPETITIONS
