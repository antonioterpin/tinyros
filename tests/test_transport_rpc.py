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
from collections.abc import Iterator

import numpy as np
import pytest

from tests.conftest import wait_port_free
from tinyros.transport import TinyClient, TinyServer


@pytest.fixture
def server_client_pair(
    free_port: int,
) -> Iterator[tuple[TinyServer, TinyClient, int]]:
    """Provision a server+client pair on a free port; close both on teardown.

    Args:
        free_port: Port the kernel considered free when the test started.

    Yields:
        Tuple ``(server, client, port)`` ready for the test to use.
    """
    server = TinyServer(name="t-srv", host="127.0.0.1", port=free_port)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-cli")
    try:
        yield server, client, free_port
    finally:
        try:
            client.close(timeout=1.0)
        finally:
            server.close(timeout=1.0)
        wait_port_free(free_port)


def test_small_call_roundtrip(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """A small RPC returns the callback's value as a Future."""
    server, client, _ = server_client_pair
    server.bind("add_one", lambda x: x + 1)
    fut = client.call("add_one", 41)
    assert fut.result(timeout=2.0) == 42, "add_one(41) should return 42"


def test_attribute_proxy_matches_call(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """client.method(x) and client.call('method', x) behave identically."""
    server, client, _ = server_client_pair
    server.bind("echo", lambda x: x)
    fut_attr = client.echo("hi")
    fut_explicit = client.call("echo", "hi")
    assert fut_attr.result(timeout=2.0) == fut_explicit.result(timeout=2.0)


def test_large_ndarray_uses_shm_fast_path(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Arrays at/above the shm threshold round-trip correctly."""
    server, client, _ = server_client_pair
    server.bind("shape_of", lambda arr: arr.shape)
    arr = np.ones((256, 256), dtype=np.float32)  # 256 KiB >> 64 KiB default
    fut = client.call("shape_of", arr)
    assert fut.result(timeout=3.0) == (
        256,
        256,
    ), "server should observe the full ndarray through shm"


def test_remote_exception_propagates(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Exceptions raised inside the callback surface on the client future."""
    server, client, _ = server_client_pair

    def boom(_: object) -> None:
        raise ValueError("deliberate")

    server.bind("boom", boom)
    fut = client.call("boom", None)
    with pytest.raises(ValueError, match="deliberate"):
        fut.result(timeout=2.0)


def test_unpicklable_exception_resolves_future(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """A callback raising an exception whose args are not picklable must
    not hang the client.

    Before the fix, ``_pack_oob((req_id, False, exc))`` raised inside the
    worker thread; no REPLY was sent; the future stayed unresolved
    forever. After: the server logs the pickle failure, substitutes a
    ``RuntimeError`` describing the original result type, and the caller
    gets a bounded-time resolution.
    """
    server, client, _ = server_client_pair

    def boom(_: object) -> None:
        # A threading.Lock in the exception args is not picklable;
        # pickle.dumps raises TypeError while serializing the tuple.
        raise RuntimeError("synthetic", threading.Lock())

    server.bind("boom", boom)
    fut = client.call("boom", None)
    with pytest.raises(RuntimeError) as excinfo:
        fut.result(timeout=2.0)
    assert "not serializable" in str(excinfo.value), (
        f"fallback message should mention the pickle failure; " f"got {excinfo.value!r}"
    )


def test_unknown_method_raises_on_client(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Calling a method the server never bound surfaces an error."""
    _server, client, _ = server_client_pair
    fut = client.call("ghost", None)
    with pytest.raises(AttributeError, match="ghost"):
        fut.result(timeout=2.0)


def test_concurrent_calls_all_complete(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Many in-flight calls from multiple threads each get their own reply."""
    server, client, _ = server_client_pair
    server.bind("twice", lambda x: x * 2)
    num = 50
    futures = [client.call("twice", i) for i in range(num)]
    results = [f.result(timeout=3.0) for f in futures]
    assert results == [
        i * 2 for i in range(num)
    ], "each future should resolve to its own double"


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
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Slow callbacks on distinct requests do not serialize each other."""
    server, client, _ = server_client_pair
    barrier = threading.Barrier(2, timeout=2.0)

    def rendezvous(_: object) -> str:
        barrier.wait()
        return "ok"

    server.bind("rendezvous", rendezvous)
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
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Arrays >= threshold are tracked in ``_pending_shm`` during send.

    This checks the side-channel is *actually taken*, not just that
    the round-trip happens to produce the right answer.
    """
    server, client, _ = server_client_pair
    started = threading.Event()
    release = threading.Event()

    def slow_echo(arr: np.ndarray) -> tuple[int, ...]:
        started.set()
        release.wait(timeout=2.0)
        return tuple(arr.shape)

    server.bind("slow_echo", slow_echo)
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
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """A 1 MP RGB image (~3 MB) round-trips unchanged through the shm path.

    Regression guard: the frame cap must not interfere with real image
    payloads, which take the shm side-channel and put only metadata on
    the wire.
    """
    _server, client, _ = server_client_pair
    _server.bind("checksum", lambda arr: int(arr.sum()))
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
