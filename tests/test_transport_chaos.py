"""Chaos / fault-injection tests for the transport layer.

Exercises failure paths that pure happy-path tests cannot reach:
backpressure under a saturated worker pool, an abrupt mid-call client
disconnect, and the server logging contract when a frame handler
crashes with a non-:class:`SerializationError` exception.

Pairs with the typed exception hierarchy (#44) and the reader-loop
classification (#45). Each test uses a short timeout (<3 s) and a
fresh port so the suite stays bounded.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from tests.conftest import wait_port_free
from tinyros.transport import TinyClient, TinyServer


def test_slow_callback_does_not_starve_pool(free_port: int) -> None:
    """A slow callback must not block other concurrent calls indefinitely.

    Bind two callbacks to the same server: ``slow`` blocks for 0.5 s,
    ``fast`` returns immediately. Issue a burst of slow calls that
    saturates ``workers`` and then a single fast call. The fast call
    must complete before the last slow one -- otherwise the reader is
    serializing rather than running concurrently.
    """
    workers = 4
    server = TinyServer(
        name="t-chaos-slow",
        host="127.0.0.1",
        port=free_port,
        workers=workers,
    )
    server.bind("slow", lambda _x: (time.sleep(0.3), "ok")[1])
    server.bind("fast", lambda _x: "ok")
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-chaos-cli")
    try:
        slow_futs = [client.call("slow", i) for i in range(workers)]
        # Submit fast right after; with concurrent execution it should
        # land in workers' time. With serial execution it would have
        # to wait for all slow calls to drain.
        t0 = time.monotonic()
        fast_fut = client.call("fast", None)
        assert fast_fut.result(timeout=2.0) == "ok"
        fast_elapsed = time.monotonic() - t0
        # Each slow call is 0.3 s. With concurrent dispatch the fast
        # call should resolve in well under the 0.3 s slow window;
        # tolerate scheduling jitter.
        assert fast_elapsed < 0.6, (
            f"fast call took {fast_elapsed:.2f}s; pool appears to be "
            f"serializing slow callbacks"
        )
        # Drain slow ones so close() does not stall on the pool.
        for fut in slow_futs:
            assert fut.result(timeout=2.0) == "ok"
    finally:
        client.close(timeout=1.0)
        server.close(timeout=2.0)
        wait_port_free(free_port)


def test_partial_header_disconnect_does_not_crash_server(free_port: int) -> None:
    """A client that closes its TCP socket mid-header must not crash the server.

    Open a raw socket, send half a frame header, then orderly-close.
    The server's reader loop should drop the connection cleanly; a
    legitimate client must still be able to issue calls afterwards.
    """
    server = TinyServer(name="t-chaos-abrupt", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda x: x)
    server.start(block=False)
    try:
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.connect(("127.0.0.1", free_port))
        # 2 bytes of a 5-byte (u8 + u32) header.
        raw.sendall(b"\x01\x00")
        raw.close()

        # Give the server's reader loop a moment to observe the EOF
        # and drop the connection. _READ_POLL_S is 1.0, so 1.5 s is
        # the sound floor.
        time.sleep(1.5)

        # Legitimate client still works after the bad peer is dropped.
        good = TinyClient(host="127.0.0.1", port=free_port, name="t-chaos-good-cli")
        try:
            assert good.call("noop", 7).result(timeout=2.0) == 7
        finally:
            good.close(timeout=1.0)
    finally:
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_handler_crash_drops_conn_and_keeps_server_alive(free_port: int) -> None:
    """An unexpected handler crash must drop the conn but not the server.

    Patch ``_handle_frame`` to raise a non-:class:`SerializationError`.
    The reader loop should treat it as a server bug (error-level path
    in #45), drop the connection, and keep the server itself alive so
    a fresh client can still get served.
    """
    server = TinyServer(name="t-chaos-crash", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda x: x)
    server.start(block=False)

    original_handle = server._handle_frame  # noqa: SLF001
    crashed = threading.Event()

    def crashing_handle(*_args: object, **_kwargs: object) -> None:
        crashed.set()
        raise RuntimeError("synthetic handler bug")

    server._handle_frame = crashing_handle  # type: ignore[method-assign]  # noqa: SLF001

    client = TinyClient(host="127.0.0.1", port=free_port, name="t-chaos-crash-cli")
    try:
        fut = client.call("noop", 1)
        with pytest.raises(ConnectionError):
            fut.result(timeout=2.0)
        assert crashed.is_set(), "crashing_handle never ran"
        client.close(timeout=1.0)

        # Restore the real handler and verify the server still serves.
        server._handle_frame = original_handle  # type: ignore[method-assign]  # noqa: SLF001
        good = TinyClient(host="127.0.0.1", port=free_port, name="t-chaos-after-cli")
        try:
            assert good.call("noop", 9).result(timeout=2.0) == 9
        finally:
            good.close(timeout=1.0)
    finally:
        server._handle_frame = original_handle  # type: ignore[method-assign]  # noqa: SLF001
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_oversized_frame_drops_connection(free_port: int) -> None:
    """A frame whose declared length exceeds ``max_frame_bytes`` is rejected.

    Confirms that misbehaving peers cannot trigger an unbounded
    allocation: the server logs and drops the conn, then a fresh
    client must still be able to call.
    """
    cap = 1024  # 1 KiB cap; well below any real payload.
    server = TinyServer(
        name="t-chaos-oversized",
        host="127.0.0.1",
        port=free_port,
        max_frame_bytes=cap,
    )
    server.bind("noop", lambda x: x)
    server.start(block=False)
    try:
        # Hand-build a header claiming a 1 MiB body; never send the
        # body. Server should detect length>cap before any allocation.
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.connect(("127.0.0.1", free_port))
        # _HEADER_FMT = "!BI" (kind=u8, length=u32). kind=1 (CALL),
        # length=1<<20. The bogus length must be > cap.
        raw.sendall(struct.pack("!BI", 1, 1 << 20))
        raw.close()

        # Fresh client still works after the bad peer is dropped.
        good = TinyClient(host="127.0.0.1", port=free_port, name="t-chaos-after-cli")
        try:
            assert good.call("noop", 42).result(timeout=2.0) == 42
        finally:
            good.close(timeout=1.0)
    finally:
        server.close(timeout=1.0)
        wait_port_free(free_port)
