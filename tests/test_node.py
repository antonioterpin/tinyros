"""Integration tests for TinyNode pub/sub over the iceoryx2 transport.

These tests exercise the full stack -- config parsing, subscription
binding, publish fanout -- against the real iceoryx2 wire. They run
inside a single process; iceoryx2's ``ServiceType.Ipc`` still uses
shared memory, so the path is a faithful end-to-end exercise.
"""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import pytest

from tinyros import (
    TinyNetworkConfig,
    TinyNode,
    TinyNodeDescription,
    TinySubscription,
)


def _unique_topic() -> str:
    """Topic name unique per test run.

    Each test creates a fresh iceoryx2 service named after the topic;
    using a UUID per test prevents service-state bleed between runs.
    """
    return f"topic_{uuid.uuid4().hex[:12]}"


def _make_config(topic: str) -> TinyNetworkConfig:
    """Build a three-node topology where ``pub`` fans out to two subscribers."""
    return TinyNetworkConfig(
        nodes={
            "pub": TinyNodeDescription(),
            "sub_a": TinyNodeDescription(),
            "sub_b": TinyNodeDescription(),
        },
        connections={
            "pub": {
                topic: (
                    TinySubscription(actor="sub_a", cb_name="on_topic"),
                    TinySubscription(actor="sub_b", cb_name="on_topic"),
                ),
            },
        },
    )


class _Recorder(TinyNode):
    """TinyNode that appends every received message to an instance list."""

    def __init__(self, name: str, cfg: TinyNetworkConfig) -> None:
        """Initialize the recorder and its message buffer."""
        self.received: list = []
        self._received_event = threading.Event()
        super().__init__(name=name, network_config=cfg)

    def on_topic(self, msg: object) -> None:
        """Record ``msg`` and signal that something arrived."""
        self.received.append(msg)
        self._received_event.set()


def _wait(predicate, timeout: float = 3.0, interval: float = 1e-3) -> bool:
    """Spin until ``predicate()`` is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_publish_reaches_each_subscriber() -> None:
    """A scalar message propagates from publisher to every subscriber."""
    topic = _unique_topic()
    cfg = _make_config(topic)
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg)
    try:
        time.sleep(0.2)
        pub.publish(topic, 7)
        assert _wait(lambda: sub_a.received and sub_b.received)
        assert sub_a.received == [7]
        assert sub_b.received == [7]
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()


def test_publish_fans_out_ndarray() -> None:
    """ndarray payloads travel byte-identically to every subscriber."""
    topic = _unique_topic()
    cfg = _make_config(topic)
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg)
    try:
        time.sleep(0.2)
        arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        pub.publish(topic, arr)
        assert _wait(lambda: sub_a.received and sub_b.received)
        assert np.array_equal(sub_a.received[0], arr)
        assert np.array_equal(sub_b.received[0], arr)
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()


def test_publish_unknown_topic_is_noop() -> None:
    """Publishing a topic with no subscribers returns an empty future list."""
    cfg = _make_config(_unique_topic())
    sub_a = _Recorder("sub_a", cfg)
    sub_b = _Recorder("sub_b", cfg)
    pub = TinyNode("pub", cfg)
    try:
        time.sleep(0.1)
        futures = pub.publish("does-not-exist", 1)
        assert futures == []
        assert sub_a.received == []
        assert sub_b.received == []
    finally:
        pub.shutdown()
        sub_a.shutdown()
        sub_b.shutdown()


def test_init_rejects_unknown_node_name() -> None:
    """Creating a node whose name is not in the config raises ValueError."""
    cfg = _make_config(_unique_topic())
    with pytest.raises(ValueError, match="ghost"):
        TinyNode("ghost", cfg)


def test_init_rejects_missing_subscription_callback() -> None:
    """A subscription naming a method that doesn't exist fails at init."""
    cfg = _make_config(_unique_topic())
    with pytest.raises(ValueError, match="on_topic"):
        # Bare TinyNode has no ``on_topic`` method.
        TinyNode("sub_a", cfg)


class _ShadowedCallback(TinyNode):
    """Subclass that shadows the configured callback name with a value."""

    on_topic = "not a method"  # type: ignore[assignment]


def test_init_rejects_non_callable_callback() -> None:
    """If the callback name resolves to a non-callable attribute, raise."""
    cfg = _make_config(_unique_topic())
    with pytest.raises(ValueError, match="not callable"):
        _ShadowedCallback("sub_a", cfg)


def test_context_manager_shuts_down_on_exit() -> None:
    """``with TinyNode(...) as n:`` tears the node down on block exit."""
    cfg = _make_config(_unique_topic())
    with _Recorder("sub_a", cfg) as node:
        assert node._subscribers, "subscriber should be live inside the block"
    assert node._subscribers == [], "shutdown must drop the subscribers"
