"""Wire-format helpers: framing, OOB pickle, CALL_LARGE shm bridge.

Pure functions used by both :mod:`tinyros.transport._server` and
:mod:`tinyros.transport._client`. No socket lifecycle or threading
primitives live here.
"""

from __future__ import annotations

import pickle
import socket
import struct
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ._common import _HEADER_FMT


def _recvall(
    sock: socket.socket, n: int, should_continue: Callable[[], bool]
) -> bytes | None:
    """Read exactly ``n`` bytes from ``sock``.

    The caller-provided ``should_continue`` predicate is consulted every
    time the underlying ``recv`` times out so that a blocked read can
    observe a shutdown request without relying on the peer closing the
    socket. The caller must have set a non-``None`` ``socket.settimeout``
    for the timeout path to be reachable.

    Args:
        sock: Connected stream socket.
        n: Number of bytes to read.
        should_continue: Predicate polled on every ``socket.timeout``;
            when it returns False, ``_recvall`` aborts and returns
            ``None``.

    Returns:
        The bytes read, ``None`` if the peer closed the connection or
        ``should_continue`` turned False mid-read.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except TimeoutError:
            if not should_continue():
                return None
            continue
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _frame(kind: int, body: bytes) -> bytes:
    """Prepend the fixed-size header to a frame body.

    Args:
        kind: Message kind (one of the ``_MSG_*`` constants).
        body: Payload bytes.

    Returns:
        Header + body, ready to send.
    """
    return struct.pack(_HEADER_FMT, kind, len(body)) + body


def _pack_oob(obj: Any) -> bytes:
    """Pickle ``obj`` with protocol-5 out-of-band buffers.

    The resulting byte stream is self-framing: callers can recover the
    main pickle and the out-of-band buffers with :func:`_unpack_oob`.

    Args:
        obj: Python object to serialize.

    Returns:
        Concatenated main pickle and out-of-band buffers.
    """
    buffers: list[pickle.PickleBuffer] = []
    main = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    out = bytearray()
    out += struct.pack("!I", len(main))
    out += main
    out += struct.pack("!I", len(buffers))
    for buf in buffers:
        mv = buf.raw()
        out += struct.pack("!I", len(mv))
        out += mv
    return bytes(out)


def _unpack_oob(payload: bytes) -> Any:
    """Reverse :func:`_pack_oob`.

    Args:
        payload: Bytes produced by :func:`_pack_oob`.

    Returns:
        The deserialized Python object.
    """
    offset = 0
    (main_len,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    main = payload[offset : offset + main_len]
    offset += main_len
    (num_bufs,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    buffers: list[bytes] = []
    for _ in range(num_bufs):
        (blen,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        buffers.append(payload[offset : offset + blen])
        offset += blen
    return pickle.loads(main, buffers=buffers)


def _pack_call_large(
    req_id: int,
    cb_name: str,
    arr: np.ndarray,
    shm_name: str,
) -> bytes:
    """Build a CALL_LARGE body referencing an ndarray in shared memory.

    Args:
        req_id: Monotonic request id used to correlate the reply.
        cb_name: Name of the callback to invoke on the server.
        arr: Source ndarray (its bytes have already been copied to shm).
        shm_name: Name of the shared-memory block.

    Returns:
        Pickled metadata ready to wrap in a frame header.
    """
    meta: dict[str, Any] = {
        "req_id": req_id,
        "cb_name": cb_name,
        "shm_name": shm_name,
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "nbytes": int(arr.nbytes),
    }
    return pickle.dumps(meta, protocol=5)


def _parse_call_large_meta(body: bytes) -> tuple[int, str, dict[str, Any]]:
    """Parse the metadata portion of a CALL_LARGE body.

    Split out from :func:`_unpack_call_large` so the worker can surface
    an ``ok=False`` reply when the shm block itself is missing or
    malformed: by the time materialization fails we already know the
    ``req_id`` and ``cb_name`` to address the reply to.

    Args:
        body: Pickled metadata produced by :func:`_pack_call_large`.

    Returns:
        Triple ``(req_id, cb_name, meta)``; ``meta`` is the full dict
        (including ``shm_name``/``dtype``/``shape``) so callers can
        continue into :func:`_materialize_call_large`.
    """
    meta = pickle.loads(body)
    return int(meta["req_id"]), str(meta["cb_name"]), meta


def _materialize_call_large(meta: dict[str, Any]) -> np.ndarray:
    """Map the shm block named in ``meta`` and copy its contents out.

    Args:
        meta: The dict returned by :func:`_parse_call_large_meta`.

    Returns:
        A freshly-copied ndarray of the declared ``shape``/``dtype``.
    """
    shm_name: str = meta["shm_name"]
    dtype = np.dtype(meta["dtype"])
    shape: tuple[int, ...] = tuple(meta["shape"])
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr = np.array(view, copy=True)
    finally:
        shm.close()
        _try_unlink_shm(shm_name)
    return arr


def _unpack_call_large(body: bytes) -> tuple[int, str, np.ndarray]:
    """Reconstruct a CALL_LARGE body and materialize the ndarray.

    Args:
        body: Pickled metadata produced by :func:`_pack_call_large`.

    Returns:
        Triple ``(req_id, cb_name, ndarray)``.
    """
    req_id, cb_name, meta = _parse_call_large_meta(body)
    return req_id, cb_name, _materialize_call_large(meta)


def _try_unlink_shm(name: str) -> None:
    """Best-effort unlink of a named shm block.

    Args:
        name: Name of the shared-memory block to remove.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
    except (FileNotFoundError, OSError):
        return
    try:
        shm.close()
    except OSError:
        pass
    try:
        shm.unlink()
    except (FileNotFoundError, OSError):
        pass
