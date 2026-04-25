"""Shared constants, env helpers, and small data carriers for the transport.

Imported by :mod:`tinyros.transport._framing`,
:mod:`tinyros.transport._server`, and :mod:`tinyros.transport._client`. No
public API: anything users need is re-exported from
:mod:`tinyros.transport`.
"""

from __future__ import annotations

import os
import socket
import struct
from typing import Any

from .._logging import get_logger

_logger = get_logger("tinyros.transport", scope="tinyros.transport")

# --- Wire kinds -----------------------------------------------------------

_MSG_CALL = 1
_MSG_CALL_LARGE = 2
_MSG_REPLY = 3
_MSG_BYE = 4

_HEADER_FMT = "!BI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# --- Defaults -------------------------------------------------------------

_DEFAULT_SHM_THRESHOLD = 65536
_DEFAULT_POOL_WORKERS = 32
_DEFAULT_MAX_FRAME_BYTES = 256 * 1024 * 1024
_ACCEPT_POLL_S = 0.1
_READ_POLL_S = 1.0
_CONNECT_TIMEOUT_S = 10.0
_RECONNECT_TIMEOUT_S = 2.0
# How long the reader thread waits for an in-flight slot before re-checking
# ``_running`` and looping. Short enough that close() is responsive; long
# enough that we don't burn CPU spinning on a saturated pool.
_INFLIGHT_ACQUIRE_POLL_S = 0.2
_LISTEN_BACKLOG = 64

_SENTINEL = object()


def _default_shm_threshold() -> int:
    """Shared-memory threshold in bytes (overridable via env).

    Returns:
        Minimum ndarray nbytes that triggers the shm side-channel.
    """
    raw = os.getenv("TINYROS_SHM_THRESHOLD")
    if raw is None:
        return _DEFAULT_SHM_THRESHOLD
    try:
        return max(0, int(raw))
    except ValueError:
        _logger.warning(
            f"TINYROS_SHM_THRESHOLD={raw!r} is not an integer; "
            f"falling back to default ({_DEFAULT_SHM_THRESHOLD})"
        )
        return _DEFAULT_SHM_THRESHOLD


def _default_max_frame_bytes() -> int:
    """Maximum inline frame body size in bytes (overridable via env).

    Returns:
        Upper bound the reader loops enforce on the ``length`` field of
        the wire header. Frames claiming more are rejected without
        buffering so a misbehaving peer cannot trigger an unbounded
        allocation. Only affects inline CALL / REPLY payloads -- ndarrays
        that take the shared-memory side-channel send only metadata on
        the socket and are unaffected.
    """
    raw = os.getenv("TINYROS_MAX_FRAME_BYTES")
    if raw is None:
        return _DEFAULT_MAX_FRAME_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        _logger.warning(
            f"TINYROS_MAX_FRAME_BYTES={raw!r} is not an integer; "
            f"falling back to default ({_DEFAULT_MAX_FRAME_BYTES})"
        )
        return _DEFAULT_MAX_FRAME_BYTES


class _PendingCall:
    """Container for an inbound call awaiting dispatch."""

    __slots__ = ("arg", "cb_name", "conn", "req_id")

    def __init__(
        self,
        conn: socket.socket,
        req_id: int,
        cb_name: str,
        arg: Any,
    ) -> None:
        """Capture the fields needed to execute and reply to a call.

        Args:
            conn: Peer socket to send the reply on.
            req_id: Monotonic request id.
            cb_name: Callback method name registered on the server.
            arg: Decoded argument to pass to the callback.
        """
        self.conn = conn
        self.req_id = req_id
        self.cb_name = cb_name
        self.arg = arg
