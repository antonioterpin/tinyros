"""ROS 2 subscriber node for latency benchmarking."""

import time

import jax
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

VALID_HW = ("cpu", "gpu")


class LatencySubscriber(Node):
    """ROS 2 subscriber that records receive timestamps.

    Optionally stages received data to GPU to include host->device cost.
    """

    def __init__(self, sub_hw: str = "cpu"):
        """Initialize the subscriber node."""
        super().__init__("latency_subscriber")

        if sub_hw not in VALID_HW:
            raise ValueError(
                f"Invalid sub_hw: {sub_hw}. Must be one of {VALID_HW}.")
        self.sub_hw = sub_hw
        self.recv_ts: list[float] = []

        self.create_subscription(
            Float32MultiArray,
            "latency_topic",
            self.on_msg,
            10,
        )

    def on_msg(self, msg: Float32MultiArray) -> None:
        """Callback function for received messages."""
        arr = np.asarray(msg.data, dtype=np.float32)

        if self.sub_hw == "gpu":
            dev = jax.device_put(arr, jax.devices("gpu")[0])
            jax.block_until_ready(dev)

        self.recv_ts.append(time.perf_counter())


def main() -> None:
    """Main function to run the subscriber node."""
    rclpy.init()
    node = LatencySubscriber(sub_hw="cpu")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
