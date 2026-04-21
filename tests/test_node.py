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
                "topic": [
                    TinySubscription(actor="sub_a", cb_name="on_topic"),
                    TinySubscription(actor="sub_b", cb_name="on_topic"),
                ],
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


def test_init_rejects_unknown_node_name(
    three_free_ports: list[int],
) -> None:
    """Creating a node whose name is not in the config raises ValueError."""
    cfg = _make_config(_ports(three_free_ports))
    with pytest.raises(ValueError, match="ghost"):
        TinyNode("ghost", cfg, bind_host="127.0.0.1")
