"""Tests for TinyNetworkConfig parsing and lookups.

Verifies that the immutable topology correctly represents nodes and
connections and raises sensible errors on unknown nodes.
"""

from __future__ import annotations

import pytest

from tinyros import (
    TinyNetworkConfig,
    TinyNodeDescription,
    TinySubscription,
)

_RAW_CONFIG = {
    "nodes": {
        "A": {"port": 5001, "host": "localhost"},
        "B": {"port": 5002, "host": "localhost"},
        "C": {"port": 5003, "host": "localhost"},
        "D": {"port": 5004, "host": "localhost"},
    },
    "connections": {
        "A": {
            "t_a": [{"actor": "B", "cb_name": "on_a"}],
        },
        "B": {
            "t_b": [
                {"actor": "A", "cb_name": "on_b_for_a"},
                {"actor": "C", "cb_name": "on_b_for_c"},
            ],
        },
    },
}


def test_load_parses_nodes_into_descriptions() -> None:
    """load_from_config promotes each node entry to TinyNodeDescription."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    assert cfg.nodes == {
        "A": TinyNodeDescription(port=5001, host="localhost"),
        "B": TinyNodeDescription(port=5002, host="localhost"),
        "C": TinyNodeDescription(port=5003, host="localhost"),
        "D": TinyNodeDescription(port=5004, host="localhost"),
    }, f"expected four typed node descriptions, got {cfg.nodes}"


def test_load_parses_connections_into_subscriptions() -> None:
    """load_from_config produces TinySubscription lists, not raw dicts."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    assert cfg.connections["B"]["t_b"] == [
        TinySubscription(actor="A", cb_name="on_b_for_a"),
        TinySubscription(actor="C", cb_name="on_b_for_c"),
    ], "connections should preserve declaration order and subscription type"


def test_get_node_by_name_returns_description() -> None:
    """Known names resolve to the right description."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    assert cfg.get_node_by_name("A") == TinyNodeDescription(
        port=5001, host="localhost"
    ), "A should map to the (5001, localhost) description"


def test_get_node_by_name_raises_for_unknown() -> None:
    """Missing names raise ValueError so typos fail loudly."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    with pytest.raises(ValueError, match="ghost"):
        cfg.get_node_by_name("ghost")


def test_publishers_for_node_returns_own_topics() -> None:
    """get_publishers_for_node returns the publisher's outbound topic map."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    publishers = cfg.get_publishers_for_node("A")
    assert list(publishers.keys()) == [
        "t_a"
    ], f"A should publish only 't_a', got {list(publishers.keys())}"


def test_publishers_for_node_empty_for_pure_subscriber() -> None:
    """A node that only subscribes has no outbound topics."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    publishers = cfg.get_publishers_for_node("C")
    assert publishers == {}, "C publishes nothing; map should be empty"


def test_subscribers_for_node_walks_all_publishers() -> None:
    """get_subscribers_for_node aggregates across every publisher."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    subs = cfg.get_subscribers_for_node("A")
    assert subs == {
        "t_b": "on_b_for_a",
    }, f"A subscribes only to t_b via on_b_for_a, got {subs}"


def test_subscribers_for_node_empty_when_no_inbound() -> None:
    """A node declared with no subscriptions yields an empty dict."""
    cfg = TinyNetworkConfig.load_from_config(_RAW_CONFIG)
    subs = cfg.get_subscribers_for_node("D")
    assert subs == {}, "D has no subscriptions; subscription map should be empty"
