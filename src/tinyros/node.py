"""TinyROS node implementation.

This module provides the core TinyROS functionality including:
- TinyNode: Base class for all ROS-like nodes
- TinySubscription: Data class for subscription configuration
- TinyNodeDescription: Data class for node network configuration
- TinyNetworkConfig: Configuration loader and manager for network topology

The module uses the portal library for inter-process communication and
supports dynamic topic-based publish/subscribe messaging.
"""

import atexit
import concurrent.futures
import logging
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Tuple

import portal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = getLogger(__name__)


@dataclass(frozen=True)
class TinySubscription:
    """Represents a subscription to a topic by an actor.

    Args:
        actor (str): The name of the subscribing actor/node
        cb_name (str): The callback method name to invoke on the actor
    """
    actor: str
    cb_name: str


@dataclass(frozen=True)
class TinyNodeDescription:
    """Describes network connection details for a TinyROS node.

    Args:
        port (int): The network port the node listens on
        host (str): The host address where the node is running
    """
    port: int
    host: str


@dataclass(frozen=True)
class TinyNetworkConfig:
    """Configuration for the TinyROS network topology.

    Manages the network configuration including node descriptions and
    publish/subscribe connections between nodes.

    Args:
        nodes (Dict[str, TinyNodeDescription]): Mapping of node names to their descriptions
        connections (Dict[str, Dict[str, List[TinySubscription]]]): Network topology mapping
            publisher_name -> topic_name -> list of subscriptions
    """
    nodes: Dict[str, TinyNodeDescription]
    connections: Dict[str, Dict[str, List[TinySubscription]]]

    def get_node_by_name(self, name: str) -> TinyNodeDescription:
        """Get node description by name."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in network config")
        return self.nodes[name]

    def get_publishers_for_node(
            self, node_name: str) -> Dict[str, List[TinySubscription]]:
        """Get all topics this node publishes and their subscribers."""
        return self.connections.get(node_name, {})

    def get_subscribers_for_node(self, node_name: str) -> Dict[str, str]:
        """Get all topics this node subscribes to and their callback names."""
        subscribers = {}
        for publisher_name, topics in self.connections.items():
            for topic_name, subscriptions in topics.items():
                for subscription in subscriptions:
                    if subscription.actor == node_name:
                        subscribers[topic_name] = subscription.cb_name
        return subscribers

    @classmethod
    def load_from_config(cls, config: dict) -> 'TinyNetworkConfig':
        """Load network configuration from a dictionary.

        Args:
            config (dict): Configuration dictionary
        """
        # Parse nodes
        nodes = {}
        for node_name, node_data in config['nodes'].items():
            nodes[node_name] = TinyNodeDescription(
                port=node_data['port'],
                host=node_data['host']
            )

        # Parse connections
        connections: Dict[str, Dict[str, List[TinySubscription]]] = {}
        for publisher_name, topics in config['connections'].items():
            connections[publisher_name] = {}
            for topic_name, subscribers in topics.items():
                connections[publisher_name][topic_name] = [
                    TinySubscription(
                        actor=sub['actor'],
                        cb_name=sub['cb_name']
                    )
                    for sub in subscribers
                ]

        return cls(nodes=nodes, connections=connections)


class TinyNode():
    """Base class for TinyROS nodes.

    Provides publish/subscribe functionality using the portal library for
    inter-process communication. Nodes can publish messages to topics and
    subscribe to topics with callback methods.

    The node automatically sets up connections based on the network configuration
    and provides methods for publishing messages and handling subscriptions.
    """

    def __init__(
        self,
        name: str,
        network_config: TinyNetworkConfig
    ):
        """Initialize a TinyROS node.

        Args:
            name (str): The name of this node in the network configuration
            network_config (TinyNetworkConfig): The network topology configuration

        Raises:
            ValueError: If the node name is not found in the network configuration
        """
        self.name = name
        self.network_config = network_config

        # Get port from network configuration
        node_description = self.network_config.get_node_by_name(name)
        self.port = node_description.port

        self.server = portal.Server(name=name + f"_{self.port}", port=self.port)

        # Two-storage approach for publishing
        # topic -> list((client_key, cb_name))
        self.topic_calls: Dict[str, List[Tuple[str, str]]] = {}
        # client_key -> client
        self.clients: Dict[str, portal.Client] = {}

        # Set up connections for topics this node publishes
        self._setup_publishing()

        # Set up callback bindings for topics this node subscribes to
        self._setup_subscriptions()

        # Ensure proper shutdown
        atexit.register(self.shutdown)

        # Start the server
        self.server.start(block=False)

    def _setup_publishing(self) -> None:
        """Set up publishing connections for topics this node publishes."""
        published_topics = self.network_config.get_publishers_for_node(
            self.name)

        for topic_name, subscriptions in published_topics.items():
            self.topic_calls[topic_name] = []

            for subscription in subscriptions:
                # Get subscriber node details
                subscriber_node = self.network_config.get_node_by_name(
                    subscription.actor)
                client_key = f"{subscriber_node.host}:{subscriber_node.port}"

                # Create client if it doesn't exist
                if client_key not in self.clients:
                    self.clients[client_key] = portal.Client(
                        client_key,
                        name=f"{self.name} -> {subscription.actor}"
                    )

                # Store the client_key and callback name for this topic
                self.topic_calls[topic_name].append(
                    (client_key, subscription.cb_name))

        logger.info(
            f"{self.name}: Set up publishing for topics:"
            f" {list(self.topic_calls.keys())}")
        logger.info(
            f"{self.name}: Created {len(self.clients)} client connections")

    def _setup_subscriptions(self) -> None:
        """Bind callback methods for topics this node subscribes to."""
        subscribed_topics = self.network_config.get_subscribers_for_node(
            self.name)

        for topic_name, callback_name in subscribed_topics.items():
            if hasattr(self, callback_name):
                self.server.bind(callback_name, getattr(self, callback_name))
                logger.info(
                    f"{self.name}: Bound callback '{callback_name}' "
                    f"for topic '{topic_name}'"
                )
            else:
                logger.error(
                    f"{self.name}: Callback method '{callback_name}' "
                    "not found!"
                )

    def publish(
            self, topic: str, message: Any) -> List[concurrent.futures.Future]:
        """Publish a message to all subscribers of a topic."""
        if topic not in self.topic_calls:
            logger.warning(f"{self.name}: No subscribers for topic '{topic}'")
            return []

        futures = []
        for client_key, cb_name in self.topic_calls[topic]:
            try:
                client = self.clients[client_key]
                futures.append(getattr(client, cb_name)(message))
            except Exception as e:
                logger.error(f"{self.name}: Failed to send message - {e}")

        return futures

    def shutdown(self) -> None:
        """Shutdown the node and close all connections."""
        logger.info(f"{self.name}: Shutting down...")
        self.server.close()

        # Close all clients
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                logger.warning(f"{self.name}: Error closing client: {e}")
