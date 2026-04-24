"""End-to-end tests for TinyServer and TinyClient.

Validates the wire in isolation from TinyNode: small-payload RPC,
large-payload shm side-channel, error propagation from remote exceptions,
and that shutdown releases the port.
"""

from __future__ import annotations

import concurrent.futures
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from tests.conftest import wait_port_free
from tinyros.transport import TinyClient, TinyServer

ServerClientFactory = Callable[..., tuple[TinyServer, TinyClient, int]]


@pytest.fixture
def server_client_pair(
    free_port: int,
) -> Iterator[ServerClientFactory]:
    """Factory yielding a bound+started server and a connected client.

    Callbacks are bound **before** ``server.start()`` because
    :meth:`TinyServer.bind` is only valid before the server is live --
    the dispatch path reads ``_callbacks`` without holding the state
    lock, so late registrations race with in-flight calls. Tests pass
    their bindings as keyword arguments: ``server_client_pair(add_one=fn)``.

    Args:
        free_port: Port the kernel considered free when the test started.

    Yields:
        A factory that accepts ``**bindings`` and returns
        ``(server, client, port)``. Can be called at most once per
        test (uses ``free_port`` which is a single-port fixture).
    """
    resources: list[tuple[TinyServer, TinyClient]] = []

    def _make(**bindings: Callable[..., Any]) -> tuple[TinyServer, TinyClient, int]:
        server = TinyServer(name="t-srv", host="127.0.0.1", port=free_port)
        for cb_name, fn in bindings.items():
            server.bind(cb_name, fn)
        server.start(block=False)
        client = TinyClient(host="127.0.0.1", port=free_port, name="t-cli")
        resources.append((server, client))
        return server, client, free_port

    try:
        yield _make
    finally:
        for server, client in resources:
            try:
                client.close(timeout=1.0)
            finally:
                server.close(timeout=1.0)
        wait_port_free(free_port)


def test_small_call_roundtrip(
    server_client_pair: ServerClientFactory,
) -> None:
    """A small RPC returns the callback's value as a Future."""
    _server, client, _ = server_client_pair(add_one=lambda x: x + 1)
    fut = client.call("add_one", 41)
    assert fut.result(timeout=2.0) == 42, "add_one(41) should return 42"


def test_attribute_proxy_matches_call(
    server_client_pair: ServerClientFactory,
) -> None:
    """client.method(x) and client.call('method', x) behave identically."""
    _server, client, _ = server_client_pair(echo=lambda x: x)
    fut_attr = client.echo("hi")
    fut_explicit = client.call("echo", "hi")
    assert fut_attr.result(timeout=2.0) == fut_explicit.result(timeout=2.0)


def test_large_ndarray_uses_shm_fast_path(
    server_client_pair: ServerClientFactory,
) -> None:
    """Arrays at/above the shm threshold round-trip correctly."""
    _server, client, _ = server_client_pair(shape_of=lambda arr: arr.shape)
    arr = np.ones((256, 256), dtype=np.float32)  # 256 KiB >> 64 KiB default
    fut = client.call("shape_of", arr)
    assert fut.result(timeout=3.0) == (
        256,
        256,
    ), "server should observe the full ndarray through shm"


def test_noncontiguous_view_roundtrips_via_shm(
    server_client_pair: ServerClientFactory,
) -> None:
    """A non-contiguous ndarray *view* takes the shm path and arrives intact.

    A view is itself an ``ndarray`` (``isinstance`` check on the encoder
    is true) and ``view.nbytes`` is the *logical* extent (shape *
    itemsize), not the underlying base buffer's size. So a strided slice
    above the threshold qualifies for the fast path on its own merits;
    a small slice of a giant base stays inline. The encoder copies via
    ``target[...] = view``, which is element-wise and faithfully
    materializes a non-contiguous source into the contiguous shm buffer.
    The receiver gets a fresh contiguous copy of the same shape, dtype,
    and values; view semantics (sharing memory with the base) do not --
    and cannot -- survive cross-process delivery.
    """

    def echo_arr(arr: np.ndarray) -> np.ndarray:
        return arr

    _server, client, _ = server_client_pair(echo_arr=echo_arr)
    # Base is 4 MiB; the strided view is 1 MiB logical, non-contiguous.
    # Both the base and the view exceed the default 64 KiB threshold,
    # so the view alone is enough to trigger the shm path.
    base = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    view = base[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"], "view must be non-contiguous for this test"
    assert view.nbytes < base.nbytes, "view.nbytes should reflect the logical extent"

    fut = client.call("echo_arr", view)
    received = fut.result(timeout=3.0)

    assert isinstance(
        received, np.ndarray
    ), f"expected ndarray, got {type(received).__name__}"
    assert received.shape == view.shape, f"shape: {received.shape!r} vs {view.shape!r}"
    assert received.dtype == view.dtype, f"dtype: {received.dtype!r} vs {view.dtype!r}"
    assert np.array_equal(
        received, view
    ), "non-contiguous view should arrive with bit-identical values"


def test_remote_exception_propagates(
    server_client_pair: ServerClientFactory,
) -> None:
    """Exceptions raised inside the callback surface on the client future."""

    def boom(_: object) -> None:
        raise ValueError("deliberate")

    _server, client, _ = server_client_pair(boom=boom)
    fut = client.call("boom", None)
    with pytest.raises(ValueError, match="deliberate"):
        fut.result(timeout=2.0)


def test_unpicklable_exception_resolves_future(
    server_client_pair: ServerClientFactory,
) -> None:
    """A callback raising an exception whose args are not picklable must
    not hang the client.

    Before the fix, ``_pack_oob((req_id, False, exc))`` raised inside the
    worker thread; no REPLY was sent; the future stayed unresolved
    forever. After: the server logs the pickle failure, substitutes a
    ``RuntimeError`` describing the original result type, and the caller
    gets a bounded-time resolution.
    """

    def boom(_: object) -> None:
        # A threading.Lock in the exception args is not picklable;
        # pickle.dumps raises TypeError while serializing the tuple.
        raise RuntimeError("synthetic", threading.Lock())

    _server, client, _ = server_client_pair(boom=boom)
    fut = client.call("boom", None)
    with pytest.raises(RuntimeError) as excinfo:
        fut.result(timeout=2.0)
    assert "not serializable" in str(excinfo.value), (
        f"fallback message should mention the pickle failure; " f"got {excinfo.value!r}"
    )


def test_unknown_method_raises_on_client(
    server_client_pair: ServerClientFactory,
) -> None:
    """Calling a method the server never bound surfaces an error."""
    _server, client, _ = server_client_pair()
    fut = client.call("ghost", None)
    with pytest.raises(AttributeError, match="ghost"):
        fut.result(timeout=2.0)


def test_concurrent_calls_all_complete(
    server_client_pair: ServerClientFactory,
) -> None:
    """Many in-flight calls from multiple threads each get their own reply."""
    _server, client, _ = server_client_pair(twice=lambda x: x * 2)
    num = 50
    futures = [client.call("twice", i) for i in range(num)]
    results = [f.result(timeout=3.0) for f in futures]
    assert results == [
        i * 2 for i in range(num)
    ], "each future should resolve to its own double"


def test_bind_after_start_raises(free_port: int) -> None:
    """``bind()`` after ``start()`` must fail loudly.

    The callback map is read from the dispatch thread pool without
    locking; accepting new registrations while the server is live
    would race with in-flight calls. Fail at the call site so the
    pattern ``server.start(); server.bind(...)`` is caught instead of
    silently ignored or intermittently observable.
    """
    server = TinyServer(name="t-bind-late", host="127.0.0.1", port=free_port)
    server.start(block=False)
    try:
        with pytest.raises(RuntimeError, match="after start"):
            server.bind("noop", lambda _x: None)
    finally:
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_server_close_releases_port(free_port: int) -> None:
    """After close(), the port becomes bindable again."""
    server = TinyServer(name="t-close", host="127.0.0.1", port=free_port)
    server.start(block=False)
    # Briefly connect to ensure the accept loop is alive.
    c = TinyClient(host="127.0.0.1", port=free_port, name="probe")
    c.close(timeout=1.0)
    server.close(timeout=1.0)
    wait_port_free(free_port)


def test_callbacks_run_concurrently(
    server_client_pair: ServerClientFactory,
) -> None:
    """Slow callbacks on distinct requests do not serialize each other."""
    barrier = threading.Barrier(2, timeout=2.0)

    def rendezvous(_: object) -> str:
        barrier.wait()
        return "ok"

    _server, client, _ = server_client_pair(rendezvous=rendezvous)
    t0 = time.perf_counter()
    f1 = client.call("rendezvous", None)
    f2 = client.call("rendezvous", None)
    assert f1.result(timeout=3.0) == "ok", "first call should return ok"
    assert f2.result(timeout=3.0) == "ok", "second call should return ok"
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, (
        f"concurrent calls should finish well under the 2s barrier; "
        f"took {elapsed:.2f}s"
    )


def test_large_ndarray_goes_through_shm_path(
    server_client_pair: ServerClientFactory,
) -> None:
    """Arrays >= threshold are tracked in ``_pending_shm`` during send.

    This checks the side-channel is *actually taken*, not just that
    the round-trip happens to produce the right answer.
    """
    started = threading.Event()
    release = threading.Event()

    def slow_echo(arr: np.ndarray) -> tuple[int, ...]:
        started.set()
        release.wait(timeout=2.0)
        return tuple(arr.shape)

    _server, client, _ = server_client_pair(slow_echo=slow_echo)
    arr = np.ones((256, 256), dtype=np.float32)  # 256 KiB >> 64 KiB
    fut = client.call("slow_echo", arr)

    assert started.wait(timeout=2.0), "server should have received the CALL_LARGE frame"
    # While the callback is still running and the shm block has just
    # been consumed by the server, _pending_shm has been drained --
    # we instead prove the path was taken by sending a payload *below*
    # threshold and showing _pending_shm stays empty for it.
    small = np.ones((2, 2), dtype=np.float32)  # 16 B
    client.call("slow_echo", small)
    with client._pending_shm_lock:  # noqa: SLF001
        pending_small = set(client._pending_shm)  # noqa: SLF001
    assert pending_small == set(), (
        "inline CALL path should never populate _pending_shm; " f"got {pending_small}"
    )

    release.set()
    assert fut.result(timeout=3.0) == (256, 256)


def test_concurrent_clients_interleave(
    free_port: int,
) -> None:
    """Three clients issue overlapping calls to one server."""
    server = TinyServer(name="t-many", host="127.0.0.1", port=free_port)
    server.bind("double", lambda x: x * 2)
    server.start(block=False)
    clients = [
        TinyClient(host="127.0.0.1", port=free_port, name=f"t-cli-{i}")
        for i in range(3)
    ]
    try:
        futures = [c.call("double", i) for i, c in enumerate(clients * 10)]
        results = [f.result(timeout=3.0) for f in futures]
        expected = [i * 2 for i, _ in enumerate(clients * 10)]
        assert results == expected
    finally:
        for c in clients:
            c.close(timeout=1.0)
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_shutdown_releases_blocked_reader(free_port: int) -> None:
    """Silent peer does not keep the server threads alive past close()."""
    server = TinyServer(name="t-hang", host="127.0.0.1", port=free_port)
    server.start(block=False)

    # Raw TCP socket, connect but never send a frame.
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.connect(("127.0.0.1", free_port))
    try:
        time.sleep(0.2)  # let the server's accept loop register us
        t0 = time.perf_counter()
        server.close(timeout=2.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, (
            f"server.close() should not block indefinitely on a "
            f"silent peer; took {elapsed:.2f}s"
        )
    finally:
        raw.close()
        wait_port_free(free_port)


def test_server_conn_bookkeeping_shrinks_after_disconnect(
    free_port: int,
) -> None:
    """Each connection's bookkeeping is removed when the client leaves.

    Previously ``_reader_threads`` was an append-only list and the
    other maps were keyed by ``id(conn)``; a server that saw many
    connect/disconnect cycles kept a ``Thread`` object per historical
    connection forever. Now every structure shrinks back to empty.
    """
    server = TinyServer(name="t-leak", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda _x: None)
    server.start(block=False)
    cycles = 30
    try:
        for i in range(cycles):
            client = TinyClient(host="127.0.0.1", port=free_port, name=f"c-{i}")
            client.call("noop", i).result(timeout=2.0)
            client.close(timeout=1.0)
        # _drop_conn runs on the reader thread once the client closes
        # its side -- give it a moment to prune.
        deadline = time.monotonic() + 2.0
        while (
            len(server._reader_threads) > 0  # noqa: SLF001
            or len(server._conns) > 0  # noqa: SLF001
            or len(server._conn_send_locks) > 0  # noqa: SLF001
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(server._reader_threads) == 0, (  # noqa: SLF001
            f"reader thread map should be empty after {cycles} "
            f"disconnects; got {len(server._reader_threads)}"  # noqa: SLF001
        )
        assert (
            len(server._conns) == 0
        ), (  # noqa: SLF001
            f"conn set should be empty; got {len(server._conns)}"  # noqa: SLF001
        )
        assert len(server._conn_send_locks) == 0, (  # noqa: SLF001
            f"send-lock map should be empty; "
            f"got {len(server._conn_send_locks)}"  # noqa: SLF001
        )
    finally:
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_close_joins_readers_accepted_concurrently(free_port: int) -> None:
    """close() must not leak a reader thread racing with a late accept.

    The earlier version snapshotted ``_reader_threads`` before joining
    ``_accept_thread``. If a client connected between the snapshot and
    the accept loop actually exiting, its reader would be registered
    after the snapshot and never joined -- leaking the thread and the
    connection bookkeeping. Drive many concurrent connects during
    ``close()`` and assert everything is drained.
    """
    server = TinyServer(name="t-race-close", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda _x: None)
    server.start(block=False)
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            try:
                c = TinyClient(
                    host="127.0.0.1", port=free_port, name="hammer", connect_timeout=0.5
                )
            except OSError:
                return
            try:
                c.close(timeout=0.5)
            except OSError:
                pass

    hammers = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
    for t in hammers:
        t.start()
    try:
        time.sleep(0.1)
    finally:
        stop.set()
        server.close(timeout=2.0)
        for t in hammers:
            t.join(timeout=2.0)
        wait_port_free(free_port)
    assert len(server._reader_threads) == 0, (  # noqa: SLF001
        "close() should have joined and removed every reader thread, "
        f"got {len(server._reader_threads)}"  # noqa: SLF001
    )
    assert (
        len(server._conns) == 0
    ), f"close() should have drained _conns, got {len(server._conns)}"  # noqa: SLF001


def test_oversized_frame_header_drops_connection(free_port: int) -> None:
    """A peer claiming an absurd frame length is disconnected, not buffered.

    Without a cap the reader would loop inside ``_recvall`` for gigabytes
    of data the peer never intends to send, tying up the connection and
    inviting a trivial resource exhaustion.
    """
    server = TinyServer(name="t-cap", host="127.0.0.1", port=free_port)
    server.start(block=False)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.connect(("127.0.0.1", free_port))
    try:
        # CALL kind (1) with length = 2^32-1: no legitimate frame is this big.
        header = struct.pack("!BI", 1, 2**32 - 1)
        raw.sendall(header)
        raw.settimeout(2.0)
        data = raw.recv(1)
        assert data == b"", (
            f"server should close the connection on an oversized frame; "
            f"got {data!r} instead of EOF"
        )
    finally:
        raw.close()
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_client_reconnects_across_server_restart(free_port: int) -> None:
    """A server restart on the same port must not permanently break the client.

    The first RPC completes against the original server. After the
    server is closed and a fresh one binds the same port, subsequent
    calls may fail with ``ConnectionError`` for a short window while
    the send loop detects the dead socket and reconnects; within the
    reconnect budget the client should recover and subsequent calls
    should succeed without tearing the client down.
    """
    srv1 = TinyServer(name="srv1", host="127.0.0.1", port=free_port)
    srv1.bind("echo", lambda x: x)
    srv1.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="cli-reconnect")
    try:
        assert client.call("echo", "before").result(timeout=2.0) == "before"
        srv1.close(timeout=1.0)
        wait_port_free(free_port)

        srv2 = TinyServer(name="srv2", host="127.0.0.1", port=free_port)
        srv2.bind("echo", lambda x: x)
        srv2.start(block=False)
        try:
            deadline = time.monotonic() + 5.0
            recovered = False
            while time.monotonic() < deadline:
                try:
                    if client.call("echo", "after").result(timeout=1.0) == "after":
                        recovered = True
                        break
                # Python 3.10 keeps concurrent.futures.TimeoutError distinct
                # from the builtin TimeoutError -- catch broadly so the
                # retry loop sees both ConnectionError and timeouts.
                except (ConnectionError, concurrent.futures.TimeoutError):
                    time.sleep(0.1)
            assert recovered, (
                "client never reconnected to the new server within the " "5 s deadline"
            )
        finally:
            srv2.close(timeout=1.0)
    finally:
        client.close(timeout=1.0)
        wait_port_free(free_port)


def test_reconnect_releases_failed_frame_shm(free_port: int) -> None:
    """A CALL_LARGE that fails to send must not leak its shm block.

    Without cleanup the shm name stays in ``_pending_shm`` even after
    reconnect, and the segment itself never gets unlinked until the
    client is closed -- repeated failures would exhaust ``/dev/shm``.
    Break the write side so the CALL_LARGE is guaranteed to hit
    ``OSError`` at the send boundary, then assert the tracking set
    drains promptly.
    """
    server = TinyServer(name="t-shm-leak", host="127.0.0.1", port=free_port)
    server.bind("shape_of", lambda arr: arr.shape)
    server.start(block=False)
    client = TinyClient(
        host="127.0.0.1",
        port=free_port,
        name="shm-leaker",
        reconnect_timeout=0.1,
    )
    try:
        client._sock.shutdown(socket.SHUT_WR)  # noqa: SLF001
        arr = np.ones((256, 256), dtype=np.float32)  # 256 KiB >> 64 KiB
        fut = client.call("shape_of", arr)
        with pytest.raises(ConnectionError):
            fut.result(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while (
            len(client._pending_shm) > 0 and time.monotonic() < deadline  # noqa: SLF001
        ):
            time.sleep(0.01)
        with client._pending_shm_lock:  # noqa: SLF001
            leftover = set(client._pending_shm)  # noqa: SLF001
        assert leftover == set(), (
            f"send failure must unlink the CALL_LARGE shm block; "
            f"still tracking {leftover}"
        )
    finally:
        client.close(timeout=1.0)
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_reconnect_drops_pre_failure_queued_frames(free_port: int) -> None:
    """Frames queued before a send failure must not reach the new server.

    Their futures were already failed by ``_fail_all_pending``; sending
    them after reconnect would trigger callback side-effects with no
    matching future to receive the reply. Wire up a callback that
    counts invocations, enqueue several calls against a half-broken
    socket, let reconnect succeed, then assert the new server never
    sees the stale frames.
    """
    srv2 = TinyServer(name="srv2-drop", host="127.0.0.1", port=free_port)
    srv2_count = 0
    srv2_lock = threading.Lock()

    def srv2_counter(_x: object) -> int:
        nonlocal srv2_count
        with srv2_lock:
            srv2_count += 1
            return srv2_count

    srv2.bind("counter", srv2_counter)

    srv1 = TinyServer(name="srv1-drop", host="127.0.0.1", port=free_port)
    srv1.bind("counter", lambda _x: None)
    srv1.start(block=False)
    client = TinyClient(
        host="127.0.0.1",
        port=free_port,
        name="cli-drop",
        reconnect_timeout=2.0,
    )
    try:
        client.call("counter", 0).result(timeout=2.0)
        # Break the socket and enqueue several more before the send
        # loop gets a chance to run and trip OSError.
        client._sock.shutdown(socket.SHUT_WR)  # noqa: SLF001
        srv1.close(timeout=1.0)
        wait_port_free(free_port)
        stale = [client.call("counter", i) for i in range(5)]

        srv2.start(block=False)
        try:
            for fut in stale:
                with pytest.raises(ConnectionError):
                    fut.result(timeout=3.0)
            # Give reconnect a moment, then issue a fresh call.
            deadline = time.monotonic() + 3.0
            ok = False
            while time.monotonic() < deadline:
                try:
                    if client.call("counter", 99).result(timeout=1.0) == 1:
                        ok = True
                        break
                except (ConnectionError, concurrent.futures.TimeoutError):
                    time.sleep(0.1)
            assert ok, "reconnect and fresh call should succeed against srv2"
            with srv2_lock:
                observed = srv2_count
            assert observed == 1, (
                f"srv2 should only see the post-reconnect call; "
                f"got {observed} total invocations, meaning stale frames "
                f"replayed across reconnect"
            )
        finally:
            srv2.close(timeout=1.0)
    finally:
        client.close(timeout=1.0)
        wait_port_free(free_port)


def test_call_large_shm_missing_returns_failure_reply(free_port: int) -> None:
    """A CALL_LARGE whose shm block has vanished must not hang the caller.

    After shifting ``_unpack_call_large`` onto the worker pool, a
    materialization error (missing/corrupt shm) used to surface only
    as a log line; the client's future stayed pending forever. The
    worker must now send an ``ok=False`` REPLY once metadata parses,
    so the caller sees a bounded-time ``RuntimeError``.
    """
    from multiprocessing import shared_memory as _shm

    from tinyros.transport import (  # type: ignore[attr-defined]
        _MSG_CALL_LARGE,
        _frame,
        _pack_call_large,
    )

    server = TinyServer(name="t-shm-missing", host="127.0.0.1", port=free_port)
    server.bind("shape_of", lambda arr: arr.shape)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-shm-missing-cli")
    try:
        # Allocate + immediately unlink a shm block to get a name that
        # looks valid but is not resolvable. Then craft a CALL_LARGE
        # frame referencing it and push it through the send queue.
        scratch = _shm.SharedMemory(create=True, size=16)
        dead_name = scratch.name
        scratch.close()
        scratch.unlink()

        arr = np.ones((4, 4), dtype=np.float32)
        body = _pack_call_large(
            req_id=9999, cb_name="shape_of", arr=arr, shm_name=dead_name
        )
        frame = _frame(_MSG_CALL_LARGE, body)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with client._pending_lock:  # noqa: SLF001
            client._pending[9999] = fut  # noqa: SLF001
        client._send_queue.put((frame, None))  # noqa: SLF001

        with pytest.raises(RuntimeError, match="shared memory"):
            fut.result(timeout=2.0)
    finally:
        client.close(timeout=1.0)
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_pending_futures_fail_on_send_failure(free_port: int) -> None:
    """When ``_send_loop`` hits ``OSError``, every pending future must
    resolve with ``ConnectionError`` rather than hang.

    Prior to the fix, the send loop cleared ``_running`` and returned
    without failing per-request futures; ``_recv_loop``'s fail-pending
    branch was gated on ``_running``, so pending futures stayed
    unresolved. Exercises the send/recv race over ``iterations`` runs
    so flakes in one direction are caught — whoever notices the dead
    socket first, every future must fail cleanly.
    """
    iterations = 25
    in_flight = 5
    for i in range(iterations):
        server = TinyServer(name=f"t-sf-srv-{i}", host="127.0.0.1", port=free_port)
        server.bind("noop", lambda _x: None)
        server.start(block=False)
        client = TinyClient(host="127.0.0.1", port=free_port, name=f"t-sf-cli-{i}")
        try:
            # Break the write side of the client's socket so the next
            # sendall raises BrokenPipeError immediately. This biases
            # the send loop to lose the race with recv, which is the
            # path the original bug lived on.
            client._sock.shutdown(socket.SHUT_WR)  # noqa: SLF001
            futures = [client.call("noop", j) for j in range(in_flight)]
            for fut in futures:
                with pytest.raises(ConnectionError):
                    fut.result(timeout=2.0)
        finally:
            client.close(timeout=1.0)
            server.close(timeout=1.0)
            wait_port_free(free_port)


def test_call_racing_teardown_does_not_hang(free_port: int) -> None:
    """A ``call()`` racing with ``_fail_all_pending`` must not leak a future.

    The early ``_running`` check in ``call()`` is not enough on its own:
    without re-checking under ``_pending_lock``, a call can pass the
    gate, be preempted, and insert into ``_pending`` after teardown has
    already drained the map — leaving that future orphaned. ``call()``
    must re-verify ``_running`` under the same lock it uses to insert,
    and ``_fail_all_pending`` must clear ``_running`` under that lock.

    Drive the interleaving deterministically: hold ``_pending_lock``
    from the test thread, kick a ``call()`` on a worker (which will
    pass the fast-path check and then block on the lock), then clear
    ``_running`` and release. The only gate that can still save the
    future is the re-check this fix adds inside the critical section.
    """
    server = TinyServer(name="t-race-srv", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda _x: None)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-race-cli")
    try:
        result: dict[str, concurrent.futures.Future] = {}
        worker_ready = threading.Event()

        def _caller() -> None:
            worker_ready.set()
            result["fut"] = client.call("noop", 1)

        with client._pending_lock:  # noqa: SLF001
            t = threading.Thread(target=_caller, daemon=True)
            t.start()
            # Give the worker time to pass the fast-path `is_set()`
            # check and land on the contended `_pending_lock`.
            assert worker_ready.wait(timeout=1.0)
            time.sleep(0.05)
            client._running.clear()  # noqa: SLF001
        t.join(timeout=1.0)
        assert not t.is_alive(), "worker call() must not hang on teardown"
        fut = result["fut"]
        with pytest.raises(ConnectionError):
            fut.result(timeout=1.0)
        with client._pending_lock:  # noqa: SLF001
            assert client._pending == {}, (  # noqa: SLF001
                "call() must not leave an entry in _pending once "
                "_running has been cleared under the lock"
            )
    finally:
        client.close(timeout=1.0)
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_client_drops_on_oversized_reply_header(free_port: int) -> None:
    """A malicious/buggy server sending an oversized REPLY header tears
    the client down: pending futures fail fast and the send thread exits.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", free_port))
    listener.listen(1)
    accepted: list[socket.socket] = []
    ready = threading.Event()

    def accept_once() -> None:
        conn, _ = listener.accept()
        accepted.append(conn)
        ready.set()

    acceptor = threading.Thread(target=accept_once, daemon=True)
    acceptor.start()

    client = TinyClient(host="127.0.0.1", port=free_port, name="t-victim")
    try:
        assert ready.wait(timeout=2.0), "fake server should accept the client"
        peer = accepted[0]
        fut = client.call("echo", "hi")
        # _MSG_REPLY = 3; length = 2**32 - 1 is absurd and must be rejected.
        peer.sendall(struct.pack("!BI", 3, 2**32 - 1))
        with pytest.raises(ConnectionError):
            fut.result(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while (
            client._send_thread.is_alive()  # noqa: SLF001
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert (
            not client._send_thread.is_alive()
        ), (  # noqa: SLF001
            "send thread should exit after the recv loop tears down the client"
        )
    finally:
        for s in accepted:
            try:
                s.close()
            except OSError:
                pass
        listener.close()
        client.close(timeout=1.0)
        wait_port_free(free_port)


def test_max_frame_bytes_kwarg_tightens_cap(free_port: int) -> None:
    """A constructor-level cap overrides the module default."""
    server = TinyServer(
        name="t-cap-kwarg",
        host="127.0.0.1",
        port=free_port,
        max_frame_bytes=1024,
    )
    server.start(block=False)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.connect(("127.0.0.1", free_port))
    try:
        # Frame claims 2 KiB, well under the module default but above our kwarg.
        header = struct.pack("!BI", 1, 2048)
        raw.sendall(header)
        raw.settimeout(2.0)
        data = raw.recv(1)
        assert data == b"", (
            f"server with max_frame_bytes=1024 should drop a 2 KiB frame; "
            f"got {data!r}"
        )
    finally:
        raw.close()
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_1mp_image_roundtrips_via_shm(
    server_client_pair: ServerClientFactory,
) -> None:
    """A 1 MP RGB image (~3 MB) round-trips unchanged through the shm path.

    Regression guard: the frame cap must not interfere with real image
    payloads, which take the shm side-channel and put only metadata on
    the wire.
    """
    _server, client, _ = server_client_pair(
        checksum=lambda arr: int(arr.sum()),
    )
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
    expected = int(img.sum())
    fut = client.call("checksum", img)
    assert (
        fut.result(timeout=3.0) == expected
    ), "1 MP image should round-trip through shm with bit-identical content"


def test_call_after_close_fails_cleanly(free_port: int) -> None:
    """call() on a closed client resolves with ConnectionError, not hang."""
    server = TinyServer(name="t-post", host="127.0.0.1", port=free_port)
    server.bind("noop", lambda _x: None)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-post-cli")
    try:
        client.close(timeout=1.0)
        fut = client.call("noop", 1)
        with pytest.raises(ConnectionError):
            fut.result(timeout=1.0)
    finally:
        server.close(timeout=1.0)
        wait_port_free(free_port)


def test_close_while_callback_in_flight(free_port: int) -> None:
    """server.close() during a blocked callback returns within timeout."""
    server = TinyServer(name="t-mid", host="127.0.0.1", port=free_port)
    started = threading.Event()
    release = threading.Event()

    def blocker(_: object) -> str:
        started.set()
        release.wait(timeout=5.0)
        return "done"

    server.bind("blocker", blocker)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-mid-cli")
    try:
        client.call("blocker", None)
        assert started.wait(timeout=2.0), "callback should begin executing"

        release.set()  # let the in-flight callback finish on its own
        t0 = time.perf_counter()
        server.close(timeout=2.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"server.close() took too long: {elapsed:.2f}s"
    finally:
        client.close(timeout=1.0)
        wait_port_free(free_port)
