"""TinyROS transport: minimal RPC wire between nodes."""

from __future__ import annotations

from ._client import TinyClient

# Private symbols re-exported for tests and advanced users; they are
# not part of the public API and may change without notice.
from ._common import _MSG_CALL_LARGE  # noqa: F401
from ._errors import ConnectionLost, SerializationError, TransportError
from ._framing import (  # noqa: F401
    _frame,
    _pack_call_large,
    _pack_oob,
    _recvall,
    _unpack_oob,
)
from ._server import TinyServer

__all__ = [
    "ConnectionLost",
    "SerializationError",
    "TinyClient",
    "TinyServer",
    "TransportError",
]
