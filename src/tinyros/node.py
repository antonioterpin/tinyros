"""TinyROS node implementation.

Provides the user-facing pub/sub API:

- :class:`TinyNode`: base class for all ROS-like nodes. A node binds a
  server on its configured port, publishes to the servers of its
  subscribers, and invokes subscriber callbacks by name.
- :class:`TinySubscription`: descriptor for a single subscription.
- :class:`TinyNodeDescription`: network-level description of a node.
- :class:`TinyNetworkConfig`: immutable network topology.

The wire lives in :mod:`tinyros.transport`; nodes are unaware of the
underlying socket / shared-memory mechanics.
"""

from __future__ import annotations

import atexit
import concurrent.futures
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from ._logging import get_logger
from .transport import TinyClient, TinyServer

_logger = get_logger("tinyros.node", scope="tinyros.node")


@dataclass(frozen=True)
class TinySubscription:
    """A subscription from one node to another.

    Args:
        actor: Name of the subscribing node.
        cb_name: Callback method name to invoke on that node.
    """

    actor: str
    cb_name: str


@dataclass(frozen=True)
class TinyNodeDescription:
    """Network connection details for a TinyROS node.

    Args:
        port: TCP port the node listens on.
        host: Host address where the node is running.
    """

    port: int
    host: str


@dataclass(frozen=True)
class TinyNetworkConfig:
    """Immutable network topology.

    Args:
        nodes: Mapping of node name to its :class:`TinyNodeDescription`.
        connections: Mapping of ``publisher_name -> topic_name ->
            subscriptions``.
    """

    nodes: dict[str, TinyNodeDescription]
    connections: dict[str, dict[str, list[TinySubscription]]]

    def get_node_by_name(self, name: str) -> TinyNodeDescription:
        """Look up a node by name.

        Args:
            name: Node name to look up.

        Returns:
            The matching :class:`TinyNodeDescription`.

        Raises:
            ValueError: If ``name`` is not in the config.
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in network config")
        return self.nodes[name]

    def get_publishers_for_node(
        self, node_name: str
    ) -> dict[str, list[TinySubscription]]:
        """Get topics that ``node_name`` publishes and their subscribers.

        Args:
            node_name: Node whose outbound topics to return.

        Returns:
            Mapping of topic name to the list of subscriptions.
        """
        return self.connections.get(node_name, {})

    def get_subscribers_for_node(self, node_name: str) -> dict[str, str]:
        """Get topics that ``node_name`` subscribes to, keyed by topic.

        Args:
            node_name: Node whose inbound subscriptions to return.

        Returns:
            Mapping of topic name to callback name registered locally.
        """
        subscribers: dict[str, str] = {}
        for topics in self.connections.values():
            for topic_name, subscriptions in topics.items():
                for subscription in subscriptions:
                    if subscription.actor == node_name:
                        subscribers[topic_name] = subscription.cb_name
        return subscribers

    @classmethod
    def load_from_config(cls, config: dict) -> TinyNetworkConfig:
        """Parse a dictionary into a :class:`TinyNetworkConfig`.

        Args:
            config: Raw config dictionary (typically from YAML).

        Returns:
            The parsed immutable network config.
        """
        nodes = {
            node_name: TinyNodeDescription(
                port=node_data["port"], host=node_data["host"]
            )
            for node_name, node_data in config["nodes"].items()
        }
        connections: dict[str, dict[str, list[TinySubscription]]] = {}
        for publisher_name, topics in config["connections"].items():
            connections[publisher_name] = {
                topic_name: [
                    TinySubscription(actor=sub["actor"], cb_name=sub["cb_name"])
                    for sub in subscribers
                ]
                for topic_name, subscribers in topics.items()
            }
        return cls(nodes=nodes, connections=connections)


class TinyNode:
    """Base class for TinyROS nodes.

    A node:

    1. Reads its port/host from the network config.
    2. Starts a server bound to every callback method named in the config.
    3. Opens one client per distinct ``(host, port)`` it publishes to.
    4. Dispatches :meth:`publish` to every subscription for the topic.

    Long-running nodes should call :meth:`shutdown` explicitly or use
    the node as a context manager::

        with MyNode(name="pub", network_config=cfg) as node:
            node.publish("topic", payload)

    An ``atexit`` hook is registered as a best-effort safety net but
    is not a substitute for deterministic shutdown.
    """

    def __init__(
        self,
        name: str,
        network_config: TinyNetworkConfig,
        *,
        bind_host: str = "0.0.0.0",
    ) -> None:
        """Initialize the node.

        Args:
            name: Node name; must appear in ``network_config.nodes``.
            network_config: Immutable topology describing the network.
            bind_host: Local interface to bind the server on. Defaults to
                all interfaces so remote nodes can reach us.

        Raises:
            ValueError: If ``name`` is not present in the config.
        """
        self.name = name
        self.network_config = network_config
        node_description = self.network_config.get_node_by_name(name)
        self.port = node_description.port

        self.server = TinyServer(
            name=f"{name}_{self.port}",
            host=bind_host,
            port=self.port,
        )

        self.topic_calls: dict[str, list[tuple[str, str]]] = {}
        self.clients: dict[str, TinyClient] = {}

        # Order matters: bind the subscriber callbacks, then open the
        # listen socket so peers can connect to us, *then* dial out to
        # peers. With the reverse order a multi-process topology with
        # cyclic publishes (A -> B and B -> A) deadlocks: each node
        # blocks in TinyClient.__init__ waiting for its peer's listen
        # socket, which the peer can only open after its own outbound
        # dials succeed.
        self._setup_subscriptions()
        atexit.register(self.shutdown)
        self.server.start(block=False)
        self._setup_publishing()

    def _setup_publishing(self) -> None:
        """Open clients to each peer this node publishes to."""
        published_topics = self.network_config.get_publishers_for_node(self.name)
        for topic_name, subscriptions in published_topics.items():
            self.topic_calls[topic_name] = []
            for subscription in subscriptions:
                subscriber_node = self.network_config.get_node_by_name(
                    subscription.actor
                )
                client_key = f"{subscriber_node.host}:{subscriber_node.port}"
                if client_key not in self.clients:
                    self.clients[client_key] = TinyClient(
                        host=subscriber_node.host,
                        port=subscriber_node.port,
                        name=f"{self.name} -> {subscription.actor}",
                    )
                self.topic_calls[topic_name].append((client_key, subscription.cb_name))
        _logger.info(
            f"{self.name}: publishing topics "
            f"{list(self.topic_calls.keys())} via {len(self.clients)} "
            f"clients"
        )

    def _setup_subscriptions(self) -> None:
        """Bind server callbacks for topics this node subscribes to."""
        subscribed_topics = self.network_config.get_subscribers_for_node(self.name)
        for topic_name, callback_name in subscribed_topics.items():
            if hasattr(self, callback_name):
                self.server.bind(callback_name, getattr(self, callback_name))
                _logger.info(
                    f"{self.name}: bound '{callback_name}' " f"for topic '{topic_name}'"
                )
            else:
                _logger.error(
                    f"{self.name}: callback method " f"'{callback_name}' not found"
                )

    def publish(self, topic: str, message: Any) -> list[concurrent.futures.Future]:
        """Publish ``message`` to every subscriber of ``topic``.

        Args:
            topic: Topic name declared in the network config.
            message: Payload forwarded to each subscriber's callback.

        Returns:
            One future per subscriber, resolving with the callback's
            return value (typically ``None``). An empty list is
            returned when ``topic`` has no subscribers -- no error
            is raised so publishers can run before their consumers
            connect.
        """
        if topic not in self.topic_calls:
            _logger.warning(f"{self.name}: no subscribers for '{topic}'")
            return []
        futures: list[concurrent.futures.Future] = []
        for client_key, cb_name in self.topic_calls[topic]:
            try:
                client = self.clients[client_key]
                futures.append(client.call(cb_name, message))
            except (ConnectionError, OSError, RuntimeError) as exc:
                _logger.error(f"{self.name}: failed to send message - {exc}")
        return futures

    def shutdown(self) -> None:
        """Shut the server and every outbound client down."""
        _logger.info(f"{self.name}: shutting down")
        try:
            self.server.close()
        except (OSError, RuntimeError) as exc:
            _logger.warning(f"{self.name}: error closing server: {exc}")
        for client in self.clients.values():
            try:
                client.close()
            except (OSError, RuntimeError) as exc:
                _logger.warning(f"{self.name}: error closing client: {exc}")

    def __enter__(self) -> TinyNode:
        """Return self so ``with TinyNode(...) as node:`` works.

        Returns:
            The node itself, already started by :meth:`__init__`.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Tear the node down on context exit."""
        self.shutdown()
