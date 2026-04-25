"""Integration tests for TinyNode pub/sub.

These tests exercise the full stack -- config parsing, server binding,
client connection, publish fanout -- against the native transport.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tests.conftest import wait_port_free
from tinyros import (
    TinyNetworkConfig,
    TinyNode,
    TinyNodeDescription,
    TinySubscription,
)


def _make_config(ports: dict[str, int]) -> TinyNetworkConfig:
    """Build a three-node topology: ``pub`` fans out to ``sub_a`` / ``sub_b``.

    Args:
        ports: Mapping of node name to the ephemeral port to use.

    Returns:
        A :class:`TinyNetworkConfig` with the test topology.
    """
    return TinyNetworkConfig(
        nodes={
            name: TinyNodeDescription(port=port, host="127.0.0.1")
            for name, port in ports.items()
        },
        connections={
            "pub": {
                "topic": (
                    TinySubscription(actor="sub_a", cb_name="on_topic"),
                    TinySubscription(actor="sub_b", cb_name="on_topic"),
                ),
            },
        },
    )


class _Recorder(TinyNode):
    """TinyNode that appends every received message to an instance list."""

    def __init__(self, name: str, cfg: TinyNetworkConfig) -> None:
        """Initialize the recorder and its message buffer.

        Args:
            name: Node name registered in ``cfg``.
            cfg: Network configuration.
        """
        self.received: list = []
        self._received_event = threading.Event()
        super().__init__(name=name, network_config=cfg, bind_host="127.0.0.1")

    def on_topic(self, msg: object) -> str:
        """Record ``msg`` and signal that something arrived.

        Args:
            msg: Forwarded payload.

        Returns:
            A constant ack so tests can verify round-trip semantics.
        """
        self.received.append(msg)
        self._received_event.set()
        return "ack"


def _ports(free_port_factory: list[int]) -> dict[str, int]:
    """Materialize names -> ports for the three test nodes.

    Args:
        free_port_factory: List of three currently-free ports.

    Returns:
        Mapping usable by :func:`_make_config`.
    """
    return {
        "pub": free_port_factory[0],
        "sub_a": free_port_factory[1],
        "sub_b": free_port_factory[2],
    }


@pytest.fixture
def three_free_ports(free_port: int) -> list[int]:
    """Yield three distinct free ports.

    Args:
        free_port: The fixture from ``conftest.py``; used once to delegate
            binding logic for the first port.

    Returns:
        Three distinct ports currently unoccupied on the loopback interface.
    """
    # Reuse the free_port fixture once; compute two more independently.
    import socket  # noqa: PLC0415

    ports = [free_port]
    while len(ports) < 3:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        p = int(s.getsockname()[1])
        s.close()
        if p not in ports:
            ports.append(p)
    return ports


def test_publish_reaches_single_subscriber(
    three_free_ports: list[int],
) -> None:
    """A scalar message propagates from publisher to the bound callback."""
    cfg = _make_config(_ports(three_free_ports))
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg, bind_host="127.0.0.1")
    try:
        time.sleep(0.2)  # let connections establish
        futures = pub.publish("topic", 7)
        results = [f.result(timeout=2.0) for f in futures]
        assert results == [
            "ack",
            "ack",
        ], f"expected two acks from the fan-out, got {results}"
        assert sub_a.received == [7], sub_a.received
        assert sub_b.received == [7], sub_b.received
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()
        for p in three_free_ports:
            wait_port_free(p)


def test_publish_fans_out_ndarray_via_shm(
    three_free_ports: list[int],
) -> None:
    """ndarray payloads travel to every subscriber through the shm path."""
    cfg = _make_config(_ports(three_free_ports))
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg, bind_host="127.0.0.1")
    try:
        time.sleep(0.2)
        arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        futures = pub.publish("topic", arr)
        for fut in futures:
            fut.result(timeout=3.0)
        assert np.array_equal(
            sub_a.received[0], arr
        ), "sub_a should receive the ndarray byte-identically"
        assert np.array_equal(
            sub_b.received[0], arr
        ), "sub_b should receive the ndarray byte-identically"
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()
        for p in three_free_ports:
            wait_port_free(p)


def test_publish_unknown_topic_is_noop(three_free_ports: list[int]) -> None:
    """Publishing a topic with no subscribers returns an empty future list."""
    cfg = _make_config(_ports(three_free_ports))
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg, bind_host="127.0.0.1")
    try:
        time.sleep(0.1)
        futures = pub.publish("does-not-exist", 1)
        assert futures == [], (
            f"publishing an unknown topic should yield no futures; " f"got {futures}"
        )
        assert sub_a.received == [], "sub_a should not have received anything"
        assert sub_b.received == [], "sub_b should not have received anything"
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()
        for p in three_free_ports:
            wait_port_free(p)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),  # whole 127.0.0.0/8 range is loopback
        ("127.255.255.255", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.1", False),
        ("8.8.8.8", False),
        ("peer.internal", False),
        # Current transport is AF_INET only; IPv6 literals can't be
        # bound, so _is_loopback_host refuses to call them loopback.
        ("::1", False),
        ("::", False),
    ],
)
def test_is_loopback_host(host: str, expected: bool) -> None:
    """Loopback detection covers the full 127.0.0.0/8 range and ``localhost``.

    IPv4 loopback is the entire ``127.0.0.0/8`` block per RFC 6890, not
    just ``127.0.0.1``. The transport binds ``AF_INET`` only, so IPv6
    literals -- including ``::1`` -- are explicitly *not* treated as
    loopback: a user who types them expects a usable bind, and silently
    returning True would hide that the transport can't actually honor
    it.
    """
    from tinyros.node import _is_loopback_host  # noqa: PLC0415

    actual = _is_loopback_host(host)
    assert (
        actual is expected
    ), f"_is_loopback_host({host!r}) should be {expected}; got {actual}"


class _DefaultBindNode(TinyNode):
    """TinyNode that relies on the default ``bind_host`` for coverage tests."""

    def on_topic(self, msg: object) -> str:
        """Satisfy the subscription check; return value is not asserted on."""
        return "ok"


def test_default_bind_host_is_loopback(three_free_ports: list[int]) -> None:
    """A TinyNode constructed without ``bind_host`` binds loopback only.

    Pickle deserialization is arbitrary code execution for any peer
    that can open the port, so a safe default is essential. This
    guards against a forgetful constructor call exposing the node on
    every interface.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_a_port = three_free_ports[1]
    node = _DefaultBindNode(name="sub_a", network_config=cfg)
    try:
        assert (
            node.server.host == "127.0.0.1"
        ), f"default bind_host must be loopback; got {node.server.host!r}"
    finally:
        node.shutdown()
        wait_port_free(sub_a_port)


def test_init_rejects_unknown_node_name(
    three_free_ports: list[int],
) -> None:
    """Creating a node whose name is not in the config raises ValueError."""
    cfg = _make_config(_ports(three_free_ports))
    with pytest.raises(ValueError, match="ghost"):
        TinyNode("ghost", cfg, bind_host="127.0.0.1")


def test_init_rejects_missing_subscription_callback(
    three_free_ports: list[int],
) -> None:
    """A subscription naming a method that doesn't exist fails at init.

    Previously this was a silent log-and-continue; a typo in
    ``network_config.yaml`` left the subscriber bound to nothing and the
    publisher's messages vanished without trace. Loud failure lets the
    operator fix the config before anything starts running.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_port = three_free_ports[1]
    try:
        with pytest.raises(ValueError, match="on_topic"):
            # Bare TinyNode has no ``on_topic`` method, but the config
            # wires sub_a to invoke ``on_topic`` — init must reject this.
            TinyNode("sub_a", cfg, bind_host="127.0.0.1")
    finally:
        wait_port_free(sub_port)


class _ShadowedCallback(TinyNode):
    """Subclass that shadows the configured callback name with a value."""

    on_topic = "not a method"  # type: ignore[assignment]


class _NoneShadowedCallback(TinyNode):
    """Subclass that shadows the callback name with ``None``.

    Guards against conflating ``getattr(..., None)`` returning ``None``
    because the attribute is missing with ``None`` being the attribute
    itself. The two failure modes need different error messages.
    """

    on_topic = None  # type: ignore[assignment]


def test_init_rejects_non_callable_callback(
    three_free_ports: list[int],
) -> None:
    """If the callback name resolves to a non-callable attribute, raise."""
    cfg = _make_config(_ports(three_free_ports))
    sub_port = three_free_ports[1]
    try:
        with pytest.raises(ValueError, match="not callable"):
            _ShadowedCallback("sub_a", cfg, bind_host="127.0.0.1")
    finally:
        wait_port_free(sub_port)


def test_init_distinguishes_missing_from_none_shadow(
    three_free_ports: list[int],
) -> None:
    """A ``None``-valued attribute must report "not callable", not "missing".

    The original implementation used ``getattr(self, name, None)`` and
    branched on ``attr is None``, which conflates "no such attribute"
    (the common typo path) with "attribute exists but is ``None``" (a
    subclass placeholder or an accidental ``= None``). Use a sentinel
    default so the two surfaces tell the operator which one they hit.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_port = three_free_ports[1]
    try:
        with pytest.raises(ValueError, match="NoneType.*not callable"):
            _NoneShadowedCallback("sub_a", cfg, bind_host="127.0.0.1")
    finally:
        wait_port_free(sub_port)


class _Boomer(TinyNode):
    """Subscriber whose callback always raises."""

    def on_topic(self, msg: object) -> None:
        """Deliberately raise to test exception propagation.

        Args:
            msg: Unused.
        """
        raise ValueError(f"deliberate: {msg!r}")


def test_callback_exception_surfaces_on_publisher_future(
    three_free_ports: list[int],
) -> None:
    """A raise inside a subscriber's callback fails the publisher's future."""
    cfg = _make_config(_ports(three_free_ports))
    sub_a = _Boomer("sub_a", cfg, bind_host="127.0.0.1")
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg, bind_host="127.0.0.1")
    try:
        time.sleep(0.2)
        futures = pub.publish("topic", "ping")
        assert len(futures) == 2
        results: list[object] = []
        for fut in futures:
            try:
                results.append(fut.result(timeout=2.0))
            except ValueError as exc:
                results.append(exc)
        kinds = [type(r).__name__ if isinstance(r, Exception) else r for r in results]
        assert "ValueError" in kinds, (
            f"one future must surface the subscriber's ValueError; " f"got {kinds}"
        )
        assert "ack" in kinds, (
            f"the other subscriber must still ack normally; " f"got {kinds}"
        )
        assert sub_b.received == ["ping"]
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()
        for p in three_free_ports:
            wait_port_free(p)


def test_publish_returns_failed_future_for_closed_client(
    three_free_ports: list[int],
) -> None:
    """``publish()`` must not raise when a subscriber's client is torn
    down -- it returns an already-failed future instead, matching the
    async failure path.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg, bind_host="127.0.0.1")
    try:
        for client in pub.clients.values():
            client.close(timeout=1.0)
        futures = pub.publish("topic", "hi")
        assert len(futures) == 2, (
            f"publish should still return one future per subscriber "
            f"when clients are closed; got {len(futures)}"
        )
        for fut in futures:
            assert fut.done(), (
                "client.call() on a closed client must resolve the "
                "future synchronously so publish() never raises"
            )
            with pytest.raises(ConnectionError):
                fut.result(timeout=1.0)
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()
        for p in three_free_ports:
            wait_port_free(p)


def test_context_manager_shuts_down_on_exit(
    three_free_ports: list[int],
) -> None:
    """``with TinyNode(...) as n:`` releases the port on block exit.

    We use ``sub_a`` via :class:`_Recorder` because the topology wires
    it to the ``on_topic`` callback — the context-manager contract is
    independent of whether the node publishes, but ``__init__`` still
    verifies every declared subscription resolves to a real method.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_a_port = three_free_ports[1]
    with _Recorder("sub_a", cfg):
        pass
    wait_port_free(sub_a_port)
