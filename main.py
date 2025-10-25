"""TinyROS Example Application.

This example demonstrates a multi-process robotics application using TinyROS with:
- Scalar publisher at 0.5 Hz
- Image publisher at 1 Hz
- Control processor that aggregates data and publishes actuation at 0.5 Hz
- Actuation feedback with noise

Each actor is implemented as a class inheriting from TinyNode.
"""

import logging
import os
import time

import numpy as np
import portal

from tinyros import TinyNetworkConfig, TinyNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load network configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'network_config.yaml')
NETWORK_CONFIG = TinyNetworkConfig.load_from_config(config_path)


class ScalarPublisher(TinyNode):
    """Publisher that sends scalar sensor data."""

    def __init__(self) -> None:
        """Initialize the ScalarPublisher."""
        super().__init__(
            name="ScalarPublisher",
            network_config=NETWORK_CONFIG
        )
        logger.info(f"ScalarPublisher: Initialized on port {self.port}")

    def run(self) -> None:
        """Run the scalar publisher at 0.5 Hz."""
        rate = 0.5  # Hz
        sleep_time = 1.0 / rate
        counter = 0

        try:
            while True:
                # Generate a simple scalar value (simulating sensor reading)
                scalar_value = (
                    np.sin(counter * 0.1) * 100 + np.random.normal(0, 5)
                )

                self.publish("scalar_data", float(scalar_value))
                logger.info(
                    f"ScalarPublisher: Published scalar = {scalar_value:.2f}")

                counter += 1
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("ScalarPublisher: Shutting down...")


class ImagePublisher(TinyNode):
    """Publisher that sends 16x16x2 image data."""

    def __init__(self) -> None:
        """Initialize the ImagePublisher."""
        super().__init__(
            name="ImagePublisher",
            network_config=NETWORK_CONFIG
        )
        logger.info(f"ImagePublisher: Initialized on port {self.port}")

    def run(self) -> None:
        """Run the image publisher at 1 Hz."""
        rate = 1.0  # Hz
        sleep_time = 1.0 / rate
        counter = 0

        try:
            while True:
                # Generate a 16x16x2 image
                image = np.random.randint(0, 255, (16, 16, 2), dtype=np.uint8)
                # Add some pattern
                image[:, :, 0] = (
                    np.sin(counter * 0.1) * 127 + 128).astype(np.uint8)
                image[:, :, 1] = (
                    np.cos(counter * 0.1) * 127 + 128).astype(np.uint8)

                self.publish("image_data", image)
                logger.info(
                    f"ImagePublisher: Published image {image.shape}, "
                    f"sum = {np.sum(image)}"
                )

                counter += 1
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("ImagePublisher: Shutting down...")


class ControlProcessor(TinyNode):
    """Processor that collects data and publishes control commands."""

    def __init__(self) -> None:
        """Initialize the ControlProcessor."""
        super().__init__(
            name="ControlProcessor",
            network_config=NETWORK_CONFIG
        )
        # Data storage
        self.latest_scalar = 0.0
        self.latest_image_sum = 0.0
        self.latest_feedback = 0.0

        logger.info(f"ControlProcessor: Initialized on port {self.port}")

    def on_scalar_data(self, data: float) -> None:
        """Callback for scalar data.

        Args:
            data (float): The received scalar data.
        """
        self.latest_scalar = data
        logger.info(
            f"ControlProcessor: Received scalar = {self.latest_scalar:.2f}")

    def on_image_data(self, data: np.ndarray) -> None:
        """Callback for image data.

        Args:
            data (np.ndarray): The received image data.
        """
        image = np.array(data)
        self.latest_image_sum = float(np.sum(image))
        logger.info(
            "ControlProcessor: Received image sum"
            f" = {self.latest_image_sum:.0f}"
        )

    def on_feedback_data(self, data: float) -> None:
        """Callback for feedback data.

        Args:
            data (float): The received feedback data.
        """
        self.latest_feedback = data
        logger.info(
            f"ControlProcessor: Received feedback = {self.latest_feedback:.2f}")

    def run(self) -> None:
        """Run the control processor at 0.5 Hz."""
        rate = 0.5  # Hz
        sleep_time = 1.0 / rate

        try:
            while True:
                # Process the collected data
                actuation_command = self.latest_scalar
                actuation_command += self.latest_image_sum * \
                    0.001  # Scale down image contribution
                actuation_command += self.latest_feedback * 0.1

                self.publish("actuation_command", float(actuation_command))
                logger.info(
                    f"Published actuation = {actuation_command:.2f}"
                )

                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("ControlProcessor: Shutting down...")


class FeedbackProcessor(TinyNode):
    """Processor that simulates actuation feedback with noise."""

    def __init__(self) -> None:
        """Initialize the FeedbackProcessor."""
        super().__init__(
            name="FeedbackProcessor",
            network_config=NETWORK_CONFIG
        )

        logger.info(f"FeedbackProcessor: Initialized on port {self.port}")

    def on_actuation_command(self, data: float) -> None:
        """Callback for actuation commands.

        Args:
            data (float): The received actuation command.
        """
        actuation_value = float(data)
        # Simulate actuator response with noise and some dynamics
        feedback_value = actuation_value * 0.95 + np.random.normal(0, 2)

        self.publish("actuation_feedback", float(feedback_value))
        logger.info(
            f"Actuation = {actuation_value:.2f} "
            f"-> Feedback = {feedback_value:.2f}"
        )

    def run(self) -> None:
        """Run the feedback processor."""
        try:
            # Keep the process alive
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("FeedbackProcessor: Shutting down...")


# Process runner functions
def run_scalar_publisher() -> None:
    """Run scalar publisher process."""
    logger.info("Starting ScalarPublisher...")
    publisher = ScalarPublisher()
    publisher.run()


def run_image_publisher() -> None:
    """Run image publisher process."""
    logger.info("Starting ImagePublisher...")
    publisher = ImagePublisher()
    publisher.run()


def run_control_processor() -> None:
    """Run control processor process."""
    logger.info("Starting ControlProcessor...")
    processor = ControlProcessor()
    processor.run()


def run_feedback_processor() -> None:
    """Run feedback processor process."""
    logger.info("Starting FeedbackProcessor...")
    processor = FeedbackProcessor()
    processor.run()


def main() -> None:
    """Main entry point - starts all processes."""
    logger.info("Starting TinyROS multi-process example...")

    processes = []

    try:
        # Start all processes using portal.Process
        scalar_proc = portal.Process(run_scalar_publisher, start=True)
        processes.append(scalar_proc)

        image_proc = portal.Process(run_image_publisher, start=True)
        processes.append(image_proc)

        feedback_proc = portal.Process(run_feedback_processor, start=True)
        processes.append(feedback_proc)

        control_proc = portal.Process(run_control_processor, start=True)
        processes.append(control_proc)

        print("All processes started. Press Ctrl+C to stop...")

        # Wait for user interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down all processes...")

    finally:
        # Clean shutdown of all processes
        for proc in processes:
            try:
                proc.kill()
                proc.join(timeout=5.0)
            except Exception as e:
                print(f"Error shutting down process: {e}")

        print("All processes stopped.")


if __name__ == "__main__":
    main()
