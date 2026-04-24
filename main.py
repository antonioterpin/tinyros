"""TinyROS example application.

Demonstrates a multi-process robotics application using TinyROS with:

- Scalar publisher at 0.5 Hz
- Image publisher at 1 Hz
- Control processor that aggregates data and publishes actuation at
  0.5 Hz
- Actuation feedback with noise

Each actor is implemented as a class inheriting from :class:`TinyNode`.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import yaml

from tinyros import TinyNetworkConfig, TinyNode

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
)
_logger = logging.getLogger("tinyros.example")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "network_config.yaml")
with open(_CONFIG_PATH) as _f:
    _CONFIG = yaml.safe_load(_f)
NETWORK_CONFIG = TinyNetworkConfig.load_from_config(_CONFIG)


class ScalarPublisher(TinyNode):
    """Publisher that sends scalar sensor data."""

    def __init__(self) -> None:
        """Initialize the ScalarPublisher."""
        super().__init__(name="ScalarPublisher", network_config=NETWORK_CONFIG)
        _logger.info(f"ScalarPublisher: initialized on port {self.port}")

    def run(self) -> None:
        """Run the scalar publisher at 0.5 Hz."""
        rate = 0.5
        sleep_time = 1.0 / rate
        counter = 0
        try:
            while True:
                scalar_value = np.sin(counter * 0.1) * 100 + np.random.normal(0, 5)
                self.publish("scalar_data", float(scalar_value))
                _logger.info(
                    f"ScalarPublisher: published scalar = " f"{scalar_value:.2f}"
                )
                counter += 1
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            _logger.info("ScalarPublisher: shutting down")


class ImagePublisher(TinyNode):
    """Publisher that sends 16x16x2 image data."""

    def __init__(self) -> None:
        """Initialize the ImagePublisher."""
        super().__init__(name="ImagePublisher", network_config=NETWORK_CONFIG)
        _logger.info(f"ImagePublisher: initialized on port {self.port}")

    def run(self) -> None:
        """Run the image publisher at 1 Hz."""
        rate = 1.0
        sleep_time = 1.0 / rate
        counter = 0
        try:
            while True:
                image = np.random.randint(0, 255, (16, 16, 2), dtype=np.uint8)
                image[:, :, 0] = (np.sin(counter * 0.1) * 127 + 128).astype(np.uint8)
                image[:, :, 1] = (np.cos(counter * 0.1) * 127 + 128).astype(np.uint8)
                self.publish("image_data", image)
                _logger.info(
                    f"ImagePublisher: published image {image.shape}, "
                    f"sum = {np.sum(image)}"
                )
                counter += 1
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            _logger.info("ImagePublisher: shutting down")


class ControlProcessor(TinyNode):
    """Processor that collects data and publishes control commands."""

    def __init__(self) -> None:
        """Initialize the ControlProcessor."""
        super().__init__(name="ControlProcessor", network_config=NETWORK_CONFIG)
        self.latest_scalar = 0.0
        self.latest_image_sum = 0.0
        self.latest_feedback = 0.0
        _logger.info(f"ControlProcessor: initialized on port {self.port}")

    def on_scalar_data(self, data: float) -> None:
        """Callback for scalar data.

        Args:
            data: Received scalar value.
        """
        self.latest_scalar = data
        _logger.info(
            f"ControlProcessor: received scalar = " f"{self.latest_scalar:.2f}"
        )

    def on_image_data(self, data: np.ndarray) -> None:
        """Callback for image data.

        Args:
            data: Received image array.
        """
        image = np.array(data)
        self.latest_image_sum = float(np.sum(image))
        _logger.info(
            f"ControlProcessor: received image sum = " f"{self.latest_image_sum:.0f}"
        )

    def on_feedback_data(self, data: float) -> None:
        """Callback for feedback data.

        Args:
            data: Received feedback value.
        """
        self.latest_feedback = data
        _logger.info(
            f"ControlProcessor: received feedback = " f"{self.latest_feedback:.2f}"
        )

    def run(self) -> None:
        """Run the control processor at 0.5 Hz."""
        rate = 0.5
        sleep_time = 1.0 / rate
        try:
            while True:
                actuation_command = self.latest_scalar
                actuation_command += self.latest_image_sum * 0.001
                actuation_command += self.latest_feedback * 0.1
                self.publish("actuation_command", float(actuation_command))
                _logger.info(f"published actuation = {actuation_command:.2f}")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            _logger.info("ControlProcessor: shutting down")


class FeedbackProcessor(TinyNode):
    """Processor that simulates actuation feedback with noise."""

    def __init__(self) -> None:
        """Initialize the FeedbackProcessor."""
        super().__init__(name="FeedbackProcessor", network_config=NETWORK_CONFIG)
        _logger.info(f"FeedbackProcessor: initialized on port {self.port}")

    def on_actuation_command(self, data: float) -> None:
        """Callback for actuation commands.

        Args:
            data: Received actuation command.
        """
        actuation_value = float(data)
        feedback_value = actuation_value * 0.95 + np.random.normal(0, 2)
        self.publish("actuation_feedback", float(feedback_value))
        _logger.info(
            f"actuation = {actuation_value:.2f} -> " f"feedback = {feedback_value:.2f}"
        )

    def run(self) -> None:
        """Run the feedback processor."""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            _logger.info("FeedbackProcessor: shutting down")


def run_scalar_publisher() -> None:
    """Run scalar publisher process."""
    _logger.info("starting ScalarPublisher")
    ScalarPublisher().run()


def run_image_publisher() -> None:
    """Run image publisher process."""
    _logger.info("starting ImagePublisher")
    ImagePublisher().run()


def run_control_processor() -> None:
    """Run control processor process."""
    _logger.info("starting ControlProcessor")
    ControlProcessor().run()


def run_feedback_processor() -> None:
    """Run feedback processor process."""
    _logger.info("starting FeedbackProcessor")
    FeedbackProcessor().run()


def main() -> None:
    """Start all nodes in subprocesses and wait for Ctrl+C."""
    _logger.info("starting TinyROS multi-process example")
    procs: list[mp.Process] = []
    try:
        for target in (
            run_scalar_publisher,
            run_image_publisher,
            run_feedback_processor,
            run_control_processor,
        ):
            p = mp.Process(target=target, daemon=False)
            p.start()
            procs.append(p)

        _logger.info("all processes started; Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _logger.info("shutting down all processes")
    finally:
        # Ask nicely first: SIGTERM gives each child a chance to exit
        # cleanly (atexit runs only if the child installs a handler,
        # but Ctrl+C at the terminal also propagates SIGINT to the
        # whole process group -- most nodes exit gracefully by that
        # path alone).
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=3.0)
        # Escalate for any children that ignored SIGTERM.
        stragglers = [p for p in procs if p.is_alive()]
        for p in stragglers:
            _logger.warning(f"process {p.name} did not exit; killing")
            p.kill()
        for p in stragglers:
            p.join(timeout=2.0)
        _logger.info("all processes stopped")


if __name__ == "__main__":
    main()
