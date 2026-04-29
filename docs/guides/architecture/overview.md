# Architecture overview

TinyROS is a minimal pub/sub runtime for robotics: static network config, peer-to-peer RPC between nodes, and a single hand-written transport. This page gives the map; [`transport.md`](transport.md) goes deep on the wire protocol.

## Top-level layout

```
src/tinyros/
    __init__.py         Public surface (re-exports below)
    node.py             TinyNode, TinyNetworkConfig, descriptors
    transport/          The wire
        __init__.py     Re-exports TinyServer / TinyClient
        _common.py      Constants, env helpers, _PendingCall, _logger
        _framing.py     Wire helpers: _recvall, _frame, OOB pickle, CALL_LARGE shm bridge
        _server.py      TinyServer
        _client.py      TinyClient (+ _ClientMethod proxy)
tests/
    conftest.py         Marker-gated collection modifier
    test_*.py           Unit tests (mirror src/tinyros/*)
    benchmark/          -m run_explicitly latency / throughput benches
```

## Core concepts

### Static network config

Nodes, ports, and connections live in a YAML file that is parsed into an
immutable `TinyNetworkConfig`. Every publisher, every topic, and every
subscriber is declared upfront:

```yaml
nodes:
  SensorNode:       { port: 5001, host: localhost }
  ControlNode:      { port: 5002, host: localhost }

connections:
  SensorNode:
    obs:
      - { actor: ControlNode, cb_name: on_obs }
```

This is a design choice, not a limitation. See the README for rationale
(clarity, predictability, debuggability).

### Nodes

`TinyNode` is the user-facing base class. A node:

1. Reads its port from the network config.
2. Starts a server bound to every callback method named in the config.
3. Creates one client per distinct `(host, port)` it publishes to.
4. Dispatches `publish(topic, msg)` to all subscribers of that topic.

Callbacks are ordinary methods on the subclass; the framework binds them
by name.

### Transport

The `tinyros.transport` package provides the RPC plumbing under
`TinyNode`. Each node owns a `TinyServer` (accepts inbound calls) and
may own any number of `TinyClient`s (makes outbound calls to other
nodes).

See [`transport.md`](transport.md) for the wire protocol, framing, and
the shared-memory fast path for large numpy payloads.

## Public surface

The `tinyros` top-level package re-exports the minimum a user needs:

```python
from tinyros import (
    TinyNode,
    TinyNetworkConfig,
    TinyNodeDescription,
    TinySubscription,
)
```

Anything not listed in `__all__` is considered internal.

## Where to make changes

| Change | Module |
|---|---|
| Pub/sub semantics, callback binding, lifecycle | `src/tinyros/node.py` |
| Wire framing, OOB pickle, CALL_LARGE shm bridge | `src/tinyros/transport/_framing.py` |
| Server lifecycle, dispatch, backpressure | `src/tinyros/transport/_server.py` |
| Client lifecycle, send/recv loops, reconnect | `src/tinyros/transport/_client.py` |
| Constants and env-var defaults | `src/tinyros/transport/_common.py` |
| Network-config parsing | `src/tinyros/node.py` (`TinyNetworkConfig`) |
| Public re-exports | `src/tinyros/__init__.py`, `src/tinyros/transport/__init__.py` |

Test location mirrors module location (see [testing standards](../../standards/testing.md)).
