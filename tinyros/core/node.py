"""TinyROS node module."""

import os
import time
import abc
from typing import Dict, Tuple, List, Optional, Any
from multiprocessing import get_context

from tinyros.core import Client, Server
from tinyros.memory.buffer import Buffer
from tinyros.datatype import TinyROSMessageDefinition
from tinyros.utils import logger

_ctx = get_context(os.getenv("MULTIPROCESSING_CONTEXT", "spawn"))
_SpawnThread = _ctx.Process
_SpawnEvent = _ctx.Event


class Node(abc.ABC, _SpawnThread):
    """A ROS-like node that runs in its own process and spins at a fixed frequency.

    The node can create publishers and request subscriptions. Calling `request_stop()`
    signals the node to finish its current iteration and then exit cleanly.

    Subclasses must implement the `loop()` method to define per‐iteration work.
    """

    def __init__(
        self,
        name: str,
        subscriptions: List[str] = [],
        publishers: Dict[str, Tuple[TinyROSMessageDefinition, int]] = {},
        spin_frequency: float = 10.0,
    ):
        """Initialize a Node.

        Args:
            name (str): The name of the node.
            subscriptions (List[str]): A list of topic names to subscribe to.
            publishers (Dict[str, Tuple[TinyROSMessageDefinition, int]]):
                A dictionary mapping topic names to (msg_def, queue_capacity).
            spin_frequency (float): Frequency in Hz to call `loop()`.
            ctx (Optional[Context]): Context for multiprocessing.

        Raises:
            ValueError: If `spin_frequency` is not greater than zero.
        """
        abc.ABC.__init__(self)
        _SpawnThread.__init__(self, name=name)

        self._name = name
        self._stop_event = _SpawnEvent()
        self._publishers: Dict[str, Server] = {}
        self._subscribers: Dict[str, Client] = {}
        self._requested_subscriptions: List[str] = []

        if spin_frequency <= 0:
            raise ValueError("spin_frequency must be > 0")
        self._spin_frequency = spin_frequency
        self._spin_period = 1.0 / spin_frequency

        for topic in subscriptions:
            self.request_subscription(topic)
        for topic, (msg_def, capacity) in publishers.items():
            self.create_publisher(topic, msg_def, capacity)

    def create_publisher(
        self, topic: str, msg_def: TinyROSMessageDefinition, capacity: int
    ):
        """Create a publisher for the given topic.

        Args:
            topic (str): The topic to publish to.
            msg_def (TinyROSMessageDefinition): The message definition for the topic.
            capacity (int): The capacity of the publisher's buffer.

        Raises:
            ValueError: If already publishing to the topic.
        """
        if topic in self._publishers:
            raise ValueError(f"Already publishing to topic: {topic}")
        self._publishers[topic] = Server(msg_def=msg_def, capacity=capacity, ctx=_ctx)

    def request_subscription(self, topic: str):
        """Request a subscription to the given topic.

        Args:
            topic (str): The topic to subscribe to.

        Raises:
            ValueError: If subscription to the topic was already requested.
        """
        if topic in self._requested_subscriptions:
            raise ValueError(f"Already requested subscription to topic: {topic}")
        self._requested_subscriptions.append(topic)
        logger.info(f"{self._name} requested subscription to topic: {topic}")

    def publish(self, topic: str, **kwargs):
        """Publish a message to the given topic.

        Args:
            topic (str): The topic to publish to.
            **kwargs: The data to publish, as keyword arguments.

        Raises:
            ValueError: If not publishing to the topic.
        """
        if topic not in self._publishers:
            raise ValueError(f"Not publishing to topic: {topic}")
        self._publishers[topic].publish(**kwargs)

    def listen(
        self, topic: str, seq: int, timeout: float, latest: bool
    ) -> Optional[Dict[str, Any]]:
        """Get the latest message from the subscriber for the given topic.

        Args:
            topic (str): The topic to get the message from.
            seq (int): The sequence number to start listening from.
            timeout (float): Timeout in seconds to wait for a message.
            latest (bool): If True, get the latest message, else the next one.

        Returns:
            Optional[DataType]: The latest message received on the topic.

        Raises:
            ValueError: If not subscribed to the topic.
            TimeoutError: If no message is received within the timeout.
        """
        if topic not in self._subscribers:
            raise ValueError(f"Not subscribed to topic: {topic}")
        return self._subscribers[topic].try_get(seq=seq, timeout=timeout, latest=latest)

    def stop_subscriber(self, topic: str):
        """Stop subscribing to the given topic.

        Args:
            topic (str): The topic to stop subscribing to.

        Raises:
            ValueError: If not subscribed to the topic.
        """
        if topic not in self._subscribers:
            raise ValueError(f"Not subscribed to topic: {topic}")
        self._subscribers[topic].close()
        self._subscribers.pop(topic)
        self._requested_subscriptions.remove(topic)
        logger.info(f"{self._name} stopped subscription to topic: {topic}")

    def stop_publisher(self, topic: str):
        """Stop publishing to the given topic.

        Args:
            topic (str): The topic to stop publishing to.

        Raises:
            ValueError: If not publishing to the topic.
        """
        if topic not in self._publishers:
            raise ValueError(f"Not publishing to topic: {topic}")
        self._publishers[topic].close()
        self._publishers.pop(topic)
        logger.info(f"{self._name} stopped publishing to topic: {topic}")

    def close(self):
        """Close the node and clean up all publishers and subscribers."""
        for topic in list(self._subscribers.keys()):
            self.stop_subscriber(topic)
        for topic in list(self._publishers.keys()):
            self.stop_publisher(topic)
        logger.info(f"{self._name} closed successfully.")

    @property
    def publishing_at(self):
        """Return a view of topic names this node is currently publishing to."""
        return self._publishers.keys()

    @property
    def subscribing_at(self):
        """Return a view of topic names this node is currently subscribed to."""
        return self._subscribers.keys()

    def get_connection_to_publisher(self, topic: str) -> Buffer:
        """Get the buffer of the publisher for a given topic.

        Args:
            topic (str): The topic to get the publisher connection for.

        Returns:
            Buffer: The buffer of the publisher for the specified topic.

        Raises:
            ValueError: If not publishing to the topic.
        """
        if topic not in self._publishers:
            raise ValueError(f"Not publishing to topic: {topic}")
        server = self._publishers[topic]
        return server.buffer

    def connect_subscriber(self, topic: str, buffer: Buffer):
        """Connect a subscriber to a given topic using a previously requested callback.

        Args:
            topic (str): The topic to connect to.
            buffer (Buffer): The buffer to connect the subscriber to.

        Raises:
            ValueError: If already subscribed to the topic or if the subscription
                was not previously requested.
        """
        if topic in self._subscribers:
            raise ValueError(f"Already subscribed to topic: {topic}")
        if topic not in self._requested_subscriptions:
            raise ValueError(f"Subscription not previously requested to topic: {topic}")
        # The actual connection needs to be done in the new process context,
        # so we just store the buffer here for later use (at the start).
        self._subscribers[topic] = buffer
        logger.info(f"{self._name} prepared for connection to topic: {topic}")

    def request_stop(self):
        """Signal the node to stop after completing the current iteration."""
        self._stop_event.set()

    @abc.abstractmethod
    def loop(self):
        """Perform one iteration of work."""

    @abc.abstractmethod
    def cleanup(self):
        """Perform any necessary cleanup before the node stops."""

    def run(self):
        """Run the node process, spinning at the configured frequency until stopped."""
        # Connect all the requested subscriptions
        # Needs to be done in the new process context
        for topic in self._subscribers:
            self._subscribers[topic] = Client(
                self.name,
                buffer=self._subscribers[topic],
            )
        for topic in self._publishers:
            self._publishers[topic].buffer.open_in_buffer()

        logger.info(f"{self._name} starting at {self._spin_frequency} Hz")

        next_iteration = time.time()
        while not self._stop_event.is_set():
            try:
                self.loop()
            except Exception as e:
                logger.error(f"Error in node '{self._name}' during loop(): {e}")

            if self._stop_event.is_set():
                break

            next_iteration += self._spin_period
            sleep_duration = next_iteration - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_iteration = time.time()

        self.cleanup()
        self.close()
        logger.info(f"{self._name} has shut down.")
