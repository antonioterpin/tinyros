"""ROS 2 subscriber node for latency benchmarking."""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

import jax
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image

VALID_HW = ("cpu", "gpu")


class LatencySubscriber(Node):
    """ROS 2 subscriber that records receive timestamps.

    Optionally stages received data to GPU to include host->device cost.
    Optionally pushes receive timestamps to a multiprocessing Queue to support
    multiprocess benchmarking.
    """

    def __init__(
        self,
        sub_hw: str = "cpu",
        *,
        ack_q: mp.Queue | None = None,
    ):
        """Initialize the subscriber node."""
        super().__init__("latency_subscriber")

        if sub_hw not in VALID_HW:
            raise ValueError(f"Invalid sub_hw: {sub_hw}. Must be one of {VALID_HW}.")
        self.sub_hw = sub_hw
        self.recv_ts: list[float] = []
        self.ack_q = ack_q

        self.create_subscription(
            Image,
            "latency_topic",
            self.on_msg,
            10,
        )

    def on_msg(self, msg: Image) -> None:
        """Callback function for received messages."""
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

        if self.sub_hw == "gpu":
            dev = jax.device_put(arr, jax.devices("gpu")[0])
            jax.block_until_ready(dev)

        t1 = time.perf_counter()
        self.recv_ts.append(t1)

        if self.ack_q is not None:
            # Portal-like: notify runner that message has been received
            self.ack_q.put(t1)
