"""ROS 2 publisher node for latency benchmarking (cv_bridge)."""

import time

import jax
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

VALID_HW = ("cpu", "gpu")


class LatencyPublisher(Node):
    """ROS 2 publisher that sends messages and records send timestamps."""

    def __init__(self, pub_hw: str = "cpu"):
        """Initialize the publisher node."""
        super().__init__("latency_publisher")

        if pub_hw not in VALID_HW:
            raise ValueError(f"Invalid pub_hw: {pub_hw}. Must be one of {VALID_HW}.")
        self.pub_hw = pub_hw

        self.publisher = self.create_publisher(Image, "latency_topic", 10)
        self.bridge = CvBridge()
        self.send_ts: list[float] = []

    def publish_array(self, arr: np.ndarray | jax.Array) -> None:
        """Publish the given array and record the send timestamp."""
        t = time.perf_counter()

        if self.pub_hw == "gpu":
            arr = np.asarray(jax.device_get(arr))  # GPU -> CPU

        # cv_bridge packing
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="32FC1")

        self.publisher.publish(msg)
        self.send_ts.append(t)
