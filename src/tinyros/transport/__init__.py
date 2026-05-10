"""TinyROS transport: iceoryx2-backed pub/sub.

Public surface is the exception hierarchy in :mod:`._errors`. Node
plumbing (``_Publisher``, ``_Subscriber``, ``make_node``) lives in
:mod:`._iox` and is used directly by :mod:`tinyros.node` -- it is not
part of the public API.
"""

from __future__ import annotations

from ._errors import ConnectionLost, SerializationError, TransportError

__all__ = [
    "ConnectionLost",
    "SerializationError",
    "TransportError",
]
