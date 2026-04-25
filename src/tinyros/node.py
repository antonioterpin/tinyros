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
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any

from ._logging import get_logger
from .transport import TinyClient, TinyServer

_logger = get_logger("tinyros.node", scope="tinyros.node")

_LOOPBACK_HOST_ALIASES = frozenset({"localhost"})


def _is_loopback_host(host: str) -> bool:
    """Return True when ``host`` binds only to the loopback interface.

    Args:
        host: String passed to ``socket.bind`` in the current transport,
            which supports IPv4 literals and hostname aliases.
            ``0.0.0.0`` is treated as non-loopback (it binds every
            interface); any address in ``127.0.0.0/8`` and ``localhost``
            are loopback. IPv6 literals are not supported by the
            transport and would fail at bind regardless.

    Returns:
        ``True`` if ``host`` is known to be loopback, ``False``
        otherwise (including any custom hostname that cannot be
        verified statically).
    """
    if host in _LOOPBACK_HOST_ALIASES:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        # Current transport binds AF_INET only; IPv6 literals can't be
        # bound, so refuse to mark them loopback -- that would let a
        # user think ``::1`` is usable when it isn't.
        return False
    return addr.is_loopback


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

    ``nodes`` and the inner ``connections`` mappings are exposed as
    read-only ``MappingProxyType`` views, and subscription lists are
    stored as tuples. Mutation attempts raise ``TypeError`` on the
    mapping views and ``AttributeError`` on the tuple (e.g., calling
    ``.append`` on a subscription list), so the frozen-dataclass
    promise extends to the nested structures, not just attribute
    rebinding.

    Args:
        nodes: Mapping of node name to its :class:`TinyNodeDescription`.
        connections: Mapping of ``publisher_name -> topic_name ->
            subscriptions``.
    """

    nodes: Mapping[str, TinyNodeDescription]
    connections: Mapping[str, Mapping[str, tuple[TinySubscription, ...]]]

    def __post_init__(self) -> None:
        """Freeze nested mappings and subscription lists."""
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        frozen = {
            publisher: MappingProxyType(
                {topic: tuple(subs) for topic, subs in topics.items()}
            )
            for publisher, topics in self.connections.items()
        }
        object.__setattr__(self, "connections", MappingProxyType(frozen))

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
    ) -> Mapping[str, tuple[TinySubscription, ...]]:
        """Get topics that ``node_name`` publishes and their subscribers.

        Args:
            node_name: Node whose outbound topics to return.

        Returns:
            Mapping of topic name to the tuple of subscriptions.
        """
        return self.connections.get(node_name, MappingProxyType({}))

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
    def load_from_config(cls, config: dict[str, Any]) -> TinyNetworkConfig:
        """Parse a dictionary into a :class:`TinyNetworkConfig`.

        Validates that every publisher and every subscription actor is
        declared in ``nodes`` before returning, so a typo in the YAML
        raises a clear error here instead of blowing up later during
        :class:`TinyNode` setup.

        Args:
            config: Raw config dictionary (typically from YAML).

        Returns:
            The parsed immutable network config.

        Raises:
            ValueError: If a publisher or subscription actor references
                a node name that is not present in ``nodes``.
        """
        nodes = {
            node_name: TinyNodeDescription(
                port=node_data["port"], host=node_data["host"]
            )
            for node_name, node_data in config["nodes"].items()
        }
        connections: dict[str, dict[str, tuple[TinySubscription, ...]]] = {}
        for publisher_name, topics in config["connections"].items():
            if publisher_name not in nodes:
                raise ValueError(
                    f"network config: publisher {publisher_name!r} has "
                    f"connections but is not declared in 'nodes'"
                )
            connections[publisher_name] = {
                topic_name: tuple(
                    TinySubscription(actor=sub["actor"], cb_name=sub["cb_name"])
                    for sub in subscribers
                )
                for topic_name, subscribers in topics.items()
            }
            for topic_name, subs in connections[publisher_name].items():
                for sub in subs:
                    if sub.actor not in nodes:
                        raise ValueError(
                            f"network config: subscription in "
                            f"{publisher_name!r}/{topic_name!r} references "
                            f"actor {sub.actor!r} that is not in 'nodes'"
                        )
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
        bind_host: str = "127.0.0.1",
    ) -> None:
        """Initialize the node.

        Args:
            name: Node name; must appear in ``network_config.nodes``.
            network_config: Immutable topology describing the network.
            bind_host: Local interface to bind the server on. Defaults
                to the loopback interface (``127.0.0.1``) because the
                wire format deserializes with ``pickle.loads``, which
                is equivalent to arbitrary code execution for anyone
                who can open the port. Override with ``0.0.0.0`` or a
                specific interface address when a non-loopback bind is
                genuinely needed; a warning is logged in that case as
                a reminder that no authentication exists between peers.

        Raises:
            ValueError: If ``name`` is not present in the config.
        """
        self.name = name
        self.network_config = network_config
        node_description = self.network_config.get_node_by_name(name)
        self.port = node_description.port

        if not _is_loopback_host(bind_host):
            _logger.warning(
                f"{name}: binding to non-loopback host {bind_host!r}. "
                f"TinyROS deserializes the wire with pickle.loads, so "
                f"anyone who can connect to port {self.port} can "
                f"execute arbitrary code in this process. Use only on "
                f"a trusted network."
            )

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
        """Bind server callbacks for topics this node subscribes to.

        Raises:
            ValueError: If the config names a callback that is missing
                on the subclass or resolves to a non-callable attribute.
                Fails loudly at ``__init__`` time so a typo in
                ``network_config.yaml`` cannot silently leave a
                subscription unhandled at runtime.
        """
        subscribed_topics = self.network_config.get_subscribers_for_node(self.name)
        # Sentinel so we distinguish "attribute is missing" from
        # "attribute exists but the subclass shadowed it with None"
        # (e.g., ``some_cb = None`` as a placeholder). Using ``None``
        # as the default would conflate the two and blame the user for
        # a typo when the real issue is a non-callable shadow.
        _missing = object()
        for topic_name, callback_name in subscribed_topics.items():
            attr = getattr(self, callback_name, _missing)
            if attr is _missing:
                raise ValueError(
                    f"{self.name}: network config subscribes topic "
                    f"'{topic_name}' to callback '{callback_name}', "
                    f"but no such method is defined on "
                    f"{type(self).__name__}"
                )
            if not callable(attr):
                raise ValueError(
                    f"{self.name}: attribute '{callback_name}' "
                    f"(bound to topic '{topic_name}') is "
                    f"{type(attr).__name__}, not callable"
                )
            self.server.bind(callback_name, attr)
            _logger.info(
                f"{self.name}: bound '{callback_name}' " f"for topic '{topic_name}'"
            )

    def publish(self, topic: str, message: Any) -> list[concurrent.futures.Future]:
        """Publish ``message`` to every subscriber of ``topic``.

        Never raises synchronously. :meth:`TinyClient.call` always
        returns a future with any transport error already set on it, so
        callers must inspect each returned future to observe delivery
        or callback failures. When a subscriber's client is already
        torn down the future resolves immediately; the failure is
        logged here for visibility but the future is still returned so
        the caller's control flow is identical across sync and async
        failure paths.

        Args:
            topic: Topic name declared in the network config.
            message: Payload forwarded to each subscriber's callback.

        Returns:
            One future per subscriber, resolving with the callback's
            return value (typically ``None``) or with a transport /
            encoding / remote exception (``ConnectionError`` when the
            socket is down, ``pickle.PicklingError`` or similar when
            ``_encode_call`` fails, or whatever the subscriber's
            callback raised). An empty list is returned when ``topic``
            has no subscribers -- no error is raised so publishers can
            run before their consumers connect.
        """
        if topic not in self.topic_calls:
            _logger.warning(f"{self.name}: no subscribers for '{topic}'")
            return []
        futures: list[concurrent.futures.Future] = []
        for client_key, cb_name in self.topic_calls[topic]:
            fut = self.clients[client_key].call(cb_name, message)
            if fut.done():
                exc = fut.exception()
                if exc is not None:
                    _logger.warning(
                        f"{self.name}: immediate failure publishing "
                        f"'{topic}' -> {cb_name} on {client_key}: "
                        f"{exc}"
                    )
            futures.append(fut)
        return futures

    def shutdown(self) -> None:
        """Shut the server and every outbound client down.

        Unregisters the atexit hook up front so long-running processes
        that create and destroy nodes dynamically don't accumulate
        stale handlers in the atexit table.
        """
        atexit.unregister(self.shutdown)
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
