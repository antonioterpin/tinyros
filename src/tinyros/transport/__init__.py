"""TinyROS transport: minimal RPC wire between nodes.

Public surface:

- :class:`TinyServer`: binds a TCP port and dispatches inbound RPC
  calls to callbacks registered via :meth:`TinyServer.bind`.
- :class:`TinyClient`: connects to a :class:`TinyServer` and returns
  :class:`concurrent.futures.Future` objects from RPC calls.

The transport is single-host. Frames are length-prefixed; CALL bodies
use pickle protocol 5 with out-of-band buffers; large top-level ndarray
arguments take a shared-memory side-channel.

See:

- ``docs/guides/architecture/transport.md`` -- wire protocol, framing,
  shared-memory fast path, threading model.
- ``docs/guides/architecture/tiny-objects.md`` -- runtime behavior:
  state machines, backpressure, reconnect-on-send-failure, failure
  modes, cross-process startup choreography.
"""

from __future__ import annotations

from ._client import TinyClient

# Private symbols re-exported for tests and advanced users; they are
# not part of the public API and may change without notice.
from ._common import _MSG_CALL_LARGE  # noqa: F401
from ._framing import (  # noqa: F401
    _frame,
    _pack_call_large,
    _pack_oob,
    _recvall,
    _unpack_oob,
)
from ._server import TinyServer

__all__ = ["TinyClient", "TinyServer"]
