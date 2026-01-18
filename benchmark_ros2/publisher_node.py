"""ROS 2 publisher node for latency benchmarking."""

import time

import jax
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

VALID_HW = ("cpu", "gpu")


class LatencyPublisher(Node):
    """ROS 2 publisher that sends messages and records send timestamps."""

    def __init__(self, pub_hw: str = "cpu"):
        """Initialize the publisher node."""
        super().__init__("latency_publisher")

        if pub_hw not in VALID_HW:
            raise ValueError(
                f"Invalid pub_hw: {pub_hw}. Must be one of {VALID_HW}.")
        self.pub_hw = pub_hw

        self.publisher = self.create_publisher(
            Float32MultiArray,
            "latency_topic",
            10,
        )

        self.send_ts: list[float] = []

    def publish_array(self, arr: np.ndarray) -> None:
        """Publish the given array and record the send timestamp."""
        if self.pub_hw == "gpu":
            dev = jax.device_put(arr, jax.devices("gpu")[0])
            jax.block_until_ready(dev)
            arr = np.asarray(dev)

        msg = Float32MultiArray()
        msg.data = arr.ravel().tolist()

        t = time.perf_counter()
        self.publisher.publish(msg)
        self.send_ts.append(t)


def main() -> None:
    """Main function to run the publisher node."""
    rclpy.init()
    node = LatencyPublisher(pub_hw="cpu")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
