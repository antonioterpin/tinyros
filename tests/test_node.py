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


def test_cyclic_topology_starts_without_deadlock(
    three_free_ports: list[int],
) -> None:
    """Two nodes that publish to each other must come up concurrently.

    With the original startup order (open outbound clients, *then*
    bind + listen) two nodes that mutually publish would both block in
    ``TinyClient.__init__`` -- each waiting on a peer whose listen
    socket is gated by the same blocked init. The race usually shows
    up as a 10 s connect-timeout failure on both sides; here we cap
    the wait at 3 s so a regression fails loudly.

    The fix re-orders ``TinyNode.__init__`` so the listen socket comes
    up before any outbound dialing. Both nodes' ``__init__`` calls
    must complete inside a few hundred milliseconds.
    """
    port_x, port_y, _ = three_free_ports
    cfg = TinyNetworkConfig(
        nodes={
            "X": TinyNodeDescription(host="127.0.0.1", port=port_x),
            "Y": TinyNodeDescription(host="127.0.0.1", port=port_y),
        },
        connections={
            "X": {"x_to_y": [TinySubscription(actor="Y", cb_name="on_x")]},
            "Y": {"y_to_x": [TinySubscription(actor="X", cb_name="on_y")]},
        },
    )

    class _Pair(TinyNode):
        def on_x(self, _msg: object) -> None: ...
        def on_y(self, _msg: object) -> None: ...

    # Lock around writes from the two spawn threads. CPython's GIL
    # makes single-key dict assignment atomic, but adding a lock keeps
    # the test resilient if the scheduling pattern ever changes (and
    # makes the intent obvious).
    state_lock = threading.Lock()
    nodes: dict[str, _Pair] = {}
    errors: list[BaseException] = []

    def _spawn(name: str) -> None:
        try:
            built = _Pair(name, cfg)
        except BaseException as exc:  # noqa: BLE001
            with state_lock:
                errors.append(exc)
            return
        with state_lock:
            nodes[name] = built

    tx = threading.Thread(target=_spawn, args=("X",), name="spawn-X")
    ty = threading.Thread(target=_spawn, args=("Y",), name="spawn-Y")
    tx.start()
    ty.start()
    tx.join(timeout=3.0)
    ty.join(timeout=3.0)
    threads_finished = not tx.is_alive() and not ty.is_alive()

    try:
        assert threads_finished, (
            "TinyNode init deadlocked under a cyclic topology; "
            f"X alive={tx.is_alive()} Y alive={ty.is_alive()}"
        )
        with state_lock:
            assert not errors, f"unexpected init errors: {errors!r}"
            built_names = set(nodes)
        assert built_names == {
            "X",
            "Y",
        }, f"both nodes should be built; got {built_names}"
    finally:
        with state_lock:
            built = list(nodes.values())
        for n in built:
            n.shutdown()
        # Skip wait_port_free if the spawn threads are still alive: the
        # ports are still bound by the deadlocked init, and waiting for
        # them to free turns an assertion failure into a hang.
        if threads_finished:
            for p in (port_x, port_y):
                wait_port_free(p)


def test_context_manager_shuts_down_on_exit(
    three_free_ports: list[int],
) -> None:
    """``with TinyNode(...) as n:`` releases the port on block exit.

    We pick ``sub_a`` because it only binds a listener and has no
    outbound publishing in the test topology -- the context-manager
    contract is independent of whether the node publishes.
    """
    cfg = _make_config(_ports(three_free_ports))
    sub_a_port = three_free_ports[1]
    with TinyNode("sub_a", cfg, bind_host="127.0.0.1"):
        pass
    wait_port_free(sub_a_port)
