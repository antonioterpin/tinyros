"""TinyROS node implementation (iceoryx2 transport).

Provides the user-facing pub/sub API:

- :class:`TinyNode`: base class for all ROS-like nodes. Each node owns
  one iceoryx2 IPC node, opens a publisher per topic it produces, and
  spawns a subscriber dispatch thread per topic it consumes.
- :class:`TinySubscription`: descriptor for a single subscription.
- :class:`TinyNodeDescription`: network-level description of a node.
  ``host`` / ``port`` are kept for backward compatibility with
  pre-migration ``network_config.yaml`` files but are ignored by the
  iceoryx2 wire (services are addressed by name, not by socket).
- :class:`TinyNetworkConfig`: immutable network topology.

The wire lives in :mod:`tinyros.transport._iox`; nodes are unaware of
the underlying shared-memory mechanics.
"""

from __future__ import annotations

import atexit
import concurrent.futures
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any

from ._logging import get_logger
from .transport._iox import _Publisher, _Subscriber, make_node

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
    """Identity of a TinyROS node within a network.

    ``host`` and ``port`` are kept for backward compatibility with
    legacy ``network_config.yaml`` files written for the TCP transport.
    The iceoryx2 transport addresses peers by service name only and
    ignores both fields at runtime.

    Args:
        port: TCP port -- ignored, kept for legacy YAML compatibility.
        host: Host address -- ignored, kept for legacy YAML
            compatibility. Defaults to ``"localhost"``.
    """

    port: int = 0
    host: str = "localhost"


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

        Raises:
            ValueError: If ``name`` is not in the config.
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in network config")
        return self.nodes[name]

    def get_publishers_for_node(
        self, node_name: str
    ) -> Mapping[str, tuple[TinySubscription, ...]]:
        """Get topics that ``node_name`` publishes and their subscribers."""
        return self.connections.get(node_name, MappingProxyType({}))

    def get_subscribers_for_node(self, node_name: str) -> dict[str, tuple[str, str]]:
        """Get topics that ``node_name`` subscribes to.

        Returns:
            Mapping of ``topic_name -> (publisher_name, callback_name)``.
            Subscription names are unique per ``(publisher, topic)``
            pair, so the topic key alone would be ambiguous when the
            same topic name is used by multiple publishers.
        """
        subscribers: dict[str, tuple[str, str]] = {}
        for publisher, topics in self.connections.items():
            for topic_name, subscriptions in topics.items():
                for subscription in subscriptions:
                    if subscription.actor == node_name:
                        subscribers[topic_name] = (
                            publisher,
                            subscription.cb_name,
                        )
        return subscribers

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> TinyNetworkConfig:
        """Parse a dictionary into a :class:`TinyNetworkConfig`.

        Validates that every publisher and every subscription actor is
        declared in ``nodes`` before returning.

        Args:
            config: Raw config dictionary (typically from YAML).

        Raises:
            ValueError: If a publisher or subscription actor references
                a node name that is not present in ``nodes``.
        """
        nodes = {
            node_name: TinyNodeDescription(
                port=node_data.get("port", 0),
                host=node_data.get("host", "localhost"),
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
    """Base class for TinyROS nodes (iceoryx2 transport).

    Each node:

    1. Reads its identity from the network config.
    2. Opens an iceoryx2 publisher per topic it produces.
    3. Spawns a subscriber dispatch thread per topic it consumes,
       resolving the callback by attribute lookup on the subclass.

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
    ) -> None:
        """Initialize the node.

        Args:
            name: Node name; must appear in ``network_config.nodes``.
            network_config: Immutable topology describing the network.

        Raises:
            ValueError: If ``name`` is not present in the config or a
                subscribed callback is missing or non-callable.
        """
        self.name = name
        self.network_config = network_config
        # validate the name early so a typo fails before iceoryx2
        # creates any on-disk service files
        node_description = self.network_config.get_node_by_name(name)
        # ``port`` is retained as an attribute for backward compatibility
        # with example code that logs it; the iceoryx2 wire ignores it.
        self.port = node_description.port

        self._iox_node = make_node()
        self._publishers: dict[str, _Publisher] = {}
        self._subscribers: list[_Subscriber] = []

        self._setup_subscriptions()
        atexit.register(self.shutdown)
        try:
            self._setup_publishing()
        except BaseException:
            try:
                self.shutdown()
            except Exception as cleanup_exc:
                _logger.warning(
                    f"{self.name}: cleanup after failed __init__ "
                    f"raised: {cleanup_exc}"
                )
            raise

    def _setup_publishing(self) -> None:
        """Open one iceoryx2 publisher per outbound topic."""
        published_topics = self.network_config.get_publishers_for_node(self.name)
        for topic_name in published_topics:
            self._publishers[topic_name] = _Publisher(
                self._iox_node, self.name, topic_name
            )
        _logger.info(
            f"{self.name}: publishing topics " f"{list(self._publishers.keys())}"
        )

    def _setup_subscriptions(self) -> None:
        """Spawn a subscriber dispatch thread per inbound topic.

        Raises:
            ValueError: If the config names a callback that is missing
                on the subclass or resolves to a non-callable attribute.
        """
        _missing = object()
        subscribed = self.network_config.get_subscribers_for_node(self.name)
        for topic_name, (publisher_name, cb_name) in subscribed.items():
            attr = getattr(self, cb_name, _missing)
            if attr is _missing:
                raise ValueError(
                    f"{self.name}: network config subscribes topic "
                    f"'{topic_name}' to callback '{cb_name}', "
                    f"but no such method is defined on "
                    f"{type(self).__name__}"
                )
            if not callable(attr):
                raise ValueError(
                    f"{self.name}: attribute '{cb_name}' "
                    f"(bound to topic '{topic_name}') is "
                    f"{type(attr).__name__}, not callable"
                )
            self._subscribers.append(
                _Subscriber(
                    self._iox_node,
                    publisher_name,
                    topic_name,
                    attr,
                    on_error=lambda exc, t=topic_name, n=self.name: _logger.warning(  # noqa: E501
                        f"{n}: subscriber for '{t}' raised {exc!r}"
                    ),
                )
            )
            _logger.info(f"{self.name}: bound '{cb_name}' for topic '{topic_name}'")

    def publish(self, topic: str, message: Any) -> list[concurrent.futures.Future]:
        """Publish ``message`` on ``topic``.

        With the iceoryx2 wire, a single ``send`` reaches every
        subscriber of the service; the per-subscriber future-list of
        the legacy TCP transport is no longer meaningful. This method
        returns a list with **one already-resolved future** per
        configured subscriber so existing call sites that iterate and
        call ``.result()`` keep working.

        Args:
            topic: Topic name declared in the network config.
            message: Payload sent to subscribers' callbacks.

        Returns:
            A list of resolved futures (one per configured subscriber);
            empty when ``topic`` has no subscribers.
        """
        if topic not in self._publishers:
            _logger.warning(f"{self.name}: no subscribers for '{topic}'")
            return []
        try:
            self._publishers[topic].publish(message)
        except Exception as exc:  # noqa: BLE001
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_exception(exc)
            return [fut]
        n = len(self.network_config.get_publishers_for_node(self.name).get(topic, ()))
        out: list[concurrent.futures.Future] = []
        for _ in range(n):
            f: concurrent.futures.Future = concurrent.futures.Future()
            f.set_result(None)
            out.append(f)
        return out

    def shutdown(self) -> None:
        """Tear the iceoryx2 publishers and subscriber threads down."""
        atexit.unregister(self.shutdown)
        _logger.info(f"{self.name}: shutting down")
        for sub in self._subscribers:
            try:
                sub.close()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(f"{self.name}: subscriber close: {exc}")
        self._subscribers.clear()
        for pub in self._publishers.values():
            try:
                pub.close()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(f"{self.name}: publisher close: {exc}")
        self._publishers.clear()
        self._iox_node = None

    def __enter__(self) -> TinyNode:
        """Return self so ``with TinyNode(...) as node:`` works."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Tear the node down on context exit."""
        self.shutdown()
