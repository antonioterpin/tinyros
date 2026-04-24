"""TinyROS transport: minimal RPC wire between nodes.

Two public classes:

- :class:`TinyServer`: binds a TCP port, dispatches inbound RPC calls to
  named callbacks registered via :meth:`TinyServer.bind`.
- :class:`TinyClient`: connects to a :class:`TinyServer` and makes RPC
  calls that return :class:`concurrent.futures.Future` objects.

Wire protocol (framed with a 1-byte kind + 4-byte length header):

- ``CALL``: inline pickle protocol 5 payload with out-of-band buffers.
- ``CALL_LARGE``: metadata frame whose single ndarray payload travels
  through :mod:`multiprocessing.shared_memory`, bypassing the socket.
- ``REPLY``: pickled ``(req_id, ok, result_or_exception)``.
- ``BYE``: graceful disconnect announcement.

The shared-memory side-channel activates when the call argument is an
ndarray whose ``nbytes`` is at least the configured threshold (default
64 KiB, set via the ``TINYROS_SHM_THRESHOLD`` env var or the
``shm_threshold`` constructor kwarg; ``0`` disables the fast path).
Only **top-level** ndarrays trigger the fast path -- arrays nested
inside a tuple, dict, or custom message class travel inline via
pickle protocol 5 out-of-band buffers. Pass a bare ndarray as the call argument
when you want cross-process delivery to skip the socket copy.

Inline frames (CALL without the shm path and every REPLY) are capped at
``TINYROS_MAX_FRAME_BYTES`` bytes (default 256 MiB) so a misbehaving
peer cannot force an unbounded allocation by claiming a huge length in
the header. Override per-instance via the ``max_frame_bytes`` kwarg on
:class:`TinyServer` / :class:`TinyClient`.

The whole transport assumes a single host.

See ``docs/guides/architecture/transport.md`` for the rationale behind
the design and the threading model.
"""

from __future__ import annotations

import concurrent.futures
import os
import pickle
import queue
import socket
import struct
import threading
import time
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ._logging import get_logger

_logger = get_logger("tinyros.transport", scope="tinyros.transport")

# --- Wire kinds -----------------------------------------------------------

_MSG_CALL = 1
_MSG_CALL_LARGE = 2
_MSG_REPLY = 3
_MSG_BYE = 4

_HEADER_FMT = "!BI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_DEFAULT_SHM_THRESHOLD = 65536
_DEFAULT_POOL_WORKERS = 32
_DEFAULT_MAX_FRAME_BYTES = 256 * 1024 * 1024
_ACCEPT_POLL_S = 0.1
_READ_POLL_S = 1.0
_CONNECT_TIMEOUT_S = 10.0
_RECONNECT_TIMEOUT_S = 2.0

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


# --- Packing helpers ------------------------------------------------------


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


def _unpack_call_large(body: bytes) -> tuple[int, str, np.ndarray]:
    """Reconstruct a CALL_LARGE body and materialize the ndarray.

    Args:
        body: Pickled metadata produced by :func:`_pack_call_large`.

    Returns:
        Triple ``(req_id, cb_name, ndarray)``.
    """
    meta = pickle.loads(body)
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
    return int(meta["req_id"]), str(meta["cb_name"]), arr


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


# --- Server ---------------------------------------------------------------


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


class TinyServer:
    """TCP RPC server.

    Binds a listening socket, accepts client connections, and dispatches
    incoming ``CALL`` frames to callbacks registered via :meth:`bind`.
    Callbacks run on a bounded thread pool so slow handlers do not block
    subsequent frames on the same connection.
    """

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        *,
        workers: int = _DEFAULT_POOL_WORKERS,
        max_frame_bytes: int | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            name: Human-readable label used in log messages.
            host: Interface address to bind on.
            port: TCP port to bind on.
            workers: Maximum concurrent callback executions.
            max_frame_bytes: Upper bound on inline frame body size. ``None``
                uses ``TINYROS_MAX_FRAME_BYTES`` or the 256 MiB default.
                Frames claiming more are rejected without buffering so a
                misbehaving peer cannot trigger an unbounded allocation.
        """
        self.name = name
        self.host = host
        self.port = port
        self._max_frame_bytes = (
            max_frame_bytes
            if max_frame_bytes is not None
            else _default_max_frame_bytes()
        )
        self._callbacks: dict[str, Callable[..., Any]] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"tinyros-server-{name}",
        )
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        # Per-connection bookkeeping is keyed by the connection socket
        # itself so drop_conn can prune each map and bounded memory
        # holds across many connect/disconnect cycles. id(conn) would
        # work while the object is live but is fragile if a future
        # refactor lets any map outlive the socket.
        self._reader_threads: dict[socket.socket, threading.Thread] = {}
        self._conns: set[socket.socket] = set()
        self._conn_send_locks: dict[socket.socket, threading.Lock] = {}
        self._conns_lock = threading.Lock()
        self._running = threading.Event()
        self._started = False
        self._state_lock = threading.Lock()

    def bind(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callback under ``name``.

        Must be called before :meth:`start`. The callback map is read
        from the dispatch thread pool without locking; adding entries
        while inbound calls are in flight would race with callers
        asking for the same name, and early calls could observe
        ``AttributeError`` transiently before the registration landed.

        Args:
            name: Method name clients will invoke.
            fn: Callable to execute when that method is called.

        Raises:
            RuntimeError: If called after :meth:`start` (the server is
                already accepting connections).
        """
        with self._state_lock:
            if self._started:
                raise RuntimeError(
                    f"{self.name}: bind({name!r}, ...) called after "
                    f"start(); register all callbacks before starting "
                    f"the server"
                )
            self._callbacks[name] = fn

    def start(self, *, block: bool = False) -> None:
        """Start accepting connections.

        Args:
            block: If True, block the caller thread; otherwise return
                immediately and serve in background threads.
        """
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._running.set()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(64)
        sock.settimeout(_ACCEPT_POLL_S)
        self._server_sock = sock

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name=f"tinyros-accept-{self.name}",
        )
        self._accept_thread.start()

        if block:
            self._accept_thread.join()

    def close(self, timeout: float | None = 2.0) -> None:
        """Stop the server and release resources.

        Args:
            timeout: Per-thread join timeout; ``None`` waits indefinitely.
        """
        with self._state_lock:
            if not self._running.is_set():
                return
            self._running.clear()

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        # Join the accept loop before snapshotting so a race-accepted
        # connection can't be registered after the snapshot and leak
        # its reader thread out of close().
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)

        with self._conns_lock:
            conns = list(self._conns)
            readers = list(self._reader_threads.values())
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

        for t in readers:
            t.join(timeout=timeout)

        self._pool.shutdown(wait=True, cancel_futures=True)

    def _accept_loop(self) -> None:
        """Accept inbound connections and spawn a reader per peer."""
        server_sock = self._server_sock
        if server_sock is None:
            raise RuntimeError(
                f"{self.name}: accept loop started without a bound socket"
            )
        while self._running.is_set():
            try:
                conn, _ = server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            # close() may have cleared _running while accept() was
            # blocking; drop the new conn instead of leaking a reader.
            if not self._running.is_set():
                try:
                    conn.close()
                except OSError:
                    pass
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(_READ_POLL_S)
            reader = threading.Thread(
                target=self._reader_loop,
                args=(conn,),
                daemon=True,
                name=f"tinyros-reader-{self.name}-{conn.fileno()}",
            )
            with self._conns_lock:
                self._conns.add(conn)
                self._conn_send_locks[conn] = threading.Lock()
                self._reader_threads[conn] = reader
                try:
                    reader.start()
                except RuntimeError:
                    # Starting the thread failed (e.g., interpreter
                    # shutdown). Unregister so close()'s reader snapshot
                    # doesn't try to join a thread that never started.
                    self._conns.discard(conn)
                    self._conn_send_locks.pop(conn, None)
                    self._reader_threads.pop(conn, None)
                    try:
                        conn.close()
                    except OSError:
                        pass
                    return

    def _reader_loop(self, conn: socket.socket) -> None:
        """Read framed messages from ``conn`` and dispatch callbacks.

        Args:
            conn: Connected peer socket.
        """
        try:
            while self._running.is_set():
                header = _recvall(conn, _HEADER_SIZE, self._running.is_set)
                if header is None:
                    return
                kind, length = struct.unpack(_HEADER_FMT, header)
                if length > self._max_frame_bytes:
                    _logger.warning(
                        f"{self.name}: oversized frame ({length} bytes, "
                        f"max {self._max_frame_bytes}); dropping connection"
                    )
                    return
                body = _recvall(conn, length, self._running.is_set) if length else b""
                if body is None:
                    return
                try:
                    self._handle_frame(conn, kind, body)
                except Exception as exc:
                    _logger.error(
                        f"{self.name}: frame handler failed "
                        f"(kind={kind}, peer_fd={conn.fileno()}): {exc}"
                    )
                    return
        finally:
            self._drop_conn(conn)

    def _handle_frame(self, conn: socket.socket, kind: int, body: bytes) -> None:
        """Dispatch a decoded frame to the right handler path.

        Args:
            conn: Peer socket the frame arrived on.
            kind: Message kind from the frame header.
            body: Frame body bytes.
        """
        if kind == _MSG_CALL:
            req_id, cb_name, arg = _unpack_oob(body)
            pending = _PendingCall(conn, int(req_id), str(cb_name), arg)
            self._pool.submit(self._execute_call, pending)
        elif kind == _MSG_CALL_LARGE:
            req_id, cb_name, arr = _unpack_call_large(body)
            pending = _PendingCall(conn, req_id, cb_name, arr)
            self._pool.submit(self._execute_call, pending)
        elif kind == _MSG_BYE:
            return
        else:
            _logger.warning(f"{self.name}: unknown frame kind {kind}")

    def _execute_call(self, call: _PendingCall) -> None:
        """Run the callback and send a REPLY frame.

        Args:
            call: Captured call metadata.
        """
        ok = True
        result: Any = None
        fn = self._callbacks.get(call.cb_name)
        if fn is None:
            ok = False
            result = AttributeError(f"{self.name}: no callback named {call.cb_name!r}")
        else:
            try:
                result = fn(call.arg)
            except Exception as exc:
                ok = False
                result = exc
        try:
            body = _pack_oob((call.req_id, ok, result))
        except Exception as pickle_exc:
            _logger.warning(
                f"{self.name}: reply for {call.cb_name!r} "
                f"(req_id={call.req_id}) is not picklable ({pickle_exc}); "
                f"substituting a RuntimeError so the caller does not hang"
            )
            body = _pack_oob(
                (
                    call.req_id,
                    False,
                    RuntimeError(
                        f"tinyros server {self.name!r}: reply for "
                        f"{call.cb_name!r} is not serializable "
                        f"({type(result).__name__}: {pickle_exc})"
                    ),
                )
            )
        frame = _frame(_MSG_REPLY, body)
        lock = self._conn_send_locks.get(call.conn)
        if lock is None:
            return
        with lock:
            try:
                call.conn.sendall(frame)
            except OSError:
                return

    def _drop_conn(self, conn: socket.socket) -> None:
        """Tear down a peer connection's bookkeeping.

        Args:
            conn: Peer socket being torn down.
        """
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass
        with self._conns_lock:
            self._conns.discard(conn)
            self._conn_send_locks.pop(conn, None)
            self._reader_threads.pop(conn, None)


# --- Client ---------------------------------------------------------------


class _ClientMethod:
    """Callable bound to a specific method name on a :class:`TinyClient`.

    Returned by :meth:`TinyClient.__getattr__` so code can write
    ``client.on_topic(msg)`` and get a :class:`concurrent.futures.Future`.
    """

    __slots__ = ("_client", "_name")

    def __init__(self, client: TinyClient, name: str) -> None:
        """Capture the client and the method name.

        Args:
            client: The owning :class:`TinyClient`.
            name: Callback method name on the peer server.
        """
        self._client = client
        self._name = name

    def __call__(self, arg: Any) -> concurrent.futures.Future:
        """Invoke the bound method asynchronously.

        Args:
            arg: Argument forwarded to the remote callback.

        Returns:
            A future that resolves with the callback's return value.
        """
        return self._client.call(self._name, arg)


class TinyClient:
    """TCP RPC client.

    Connects to a :class:`TinyServer` and sends CALL / CALL_LARGE frames;
    demultiplexes REPLY frames onto per-request futures. Attribute access
    returns a callable proxy so ``client.on_topic(message)`` dispatches
    an RPC to the remote callback of that name.
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        *,
        shm_threshold: int | None = None,
        max_frame_bytes: int | None = None,
        connect_timeout: float = _CONNECT_TIMEOUT_S,
        reconnect_timeout: float = _RECONNECT_TIMEOUT_S,
    ) -> None:
        """Initialize the client and connect to the server.

        Args:
            host: Server host to connect to.
            port: Server TCP port.
            name: Human-readable label used in log messages.
            shm_threshold: Payload size (bytes) at or above which ndarray
                arguments travel through shared memory. ``None`` uses
                ``TINYROS_SHM_THRESHOLD`` or the 64 KiB default. ``0``
                disables the shm fast path.
            max_frame_bytes: Upper bound on inline reply body size. ``None``
                uses ``TINYROS_MAX_FRAME_BYTES`` or the 256 MiB default.
                Reply frames claiming more tear the client down instead of
                buffering.
            connect_timeout: Max seconds to wait for the server to accept
                the initial connection.
            reconnect_timeout: Max seconds to retry reconnecting after
                a send failure before giving up and tearing the client
                down. The send loop attempts one reconnect per detected
                socket failure; callers who want long-running resilience
                can retry across client lifetimes.
        """
        self.host = host
        self.port = port
        self.name = name
        self._shm_threshold = (
            shm_threshold if shm_threshold is not None else _default_shm_threshold()
        )
        self._max_frame_bytes = (
            max_frame_bytes
            if max_frame_bytes is not None
            else _default_max_frame_bytes()
        )
        self._reconnect_timeout = reconnect_timeout
        self._sock = self._connect(connect_timeout)
        self._sock.settimeout(_READ_POLL_S)
        self._send_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._pending_lock = threading.Lock()
        self._next_req_id = 0
        self._req_id_lock = threading.Lock()
        self._pending_shm: set[str] = set()
        self._pending_shm_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._shutdown_called = False
        self._state_lock = threading.Lock()
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name=f"tinyros-client-send-{name}",
        )
        self._send_thread.start()
        self._recv_thread = self._start_recv_thread(self._sock)

    def _connect(self, timeout: float) -> socket.socket:
        """Connect to the server, retrying until ``timeout`` elapses.

        Args:
            timeout: Max seconds to keep retrying.

        Returns:
            The connected stream socket.

        Raises:
            ConnectionError: If no connection could be established.
        """
        deadline = time.monotonic() + timeout
        last_exc: OSError | None = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            try:
                sock.connect((self.host, self.port))
            except OSError as exc:
                last_exc = exc
                sock.close()
                time.sleep(0.05)
                continue
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        raise ConnectionError(
            f"tinyros client {self.name!r} could not connect to "
            f"{self.host}:{self.port} within {timeout:.1f}s: {last_exc}"
        )

    def __getattr__(self, name: str) -> _ClientMethod:
        """Return a method proxy for attribute access.

        Args:
            name: Remote callback name.

        Returns:
            Callable that forwards its argument as an RPC call.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return _ClientMethod(self, name)

    def call(self, method: str, arg: Any) -> concurrent.futures.Future:
        """Send an RPC call and return a future for the reply.

        When ``arg`` is a bare ndarray whose ``nbytes`` meets the shm
        threshold, the array travels through shared memory. Nested
        ndarrays (inside a tuple, dict, or custom message class) always
        travel inline via pickle protocol 5 out-of-band buffers. For
        cross-process delivery, that path still copies the bytes
        through the socket. Pass the array at the top level when the
        shm fast path matters.

        Args:
            method: Callback name registered on the server.
            arg: Argument forwarded to the callback.

        Returns:
            A future that resolves with the callback's return value.
        """
        fut: concurrent.futures.Future = concurrent.futures.Future()
        if not self._running.is_set():
            fut.set_exception(
                ConnectionError(f"tinyros client {self.name!r} is no longer running")
            )
            return fut
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
        with self._pending_lock:
            if not self._running.is_set():
                fut.set_exception(
                    ConnectionError(
                        f"tinyros client {self.name!r} is no longer running"
                    )
                )
                return fut
            self._pending[req_id] = fut
        try:
            frame, shm_name = self._encode_call(req_id, method, arg)
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            fut.set_exception(exc)
            return fut
        self._send_queue.put((frame, shm_name))
        return fut

    def _encode_call(
        self, req_id: int, method: str, arg: Any
    ) -> tuple[bytes, str | None]:
        """Encode a single call as a framed byte string.

        Args:
            req_id: Monotonic request id.
            method: Remote callback name.
            arg: Argument to send.

        Returns:
            ``(frame, shm_name)``. ``shm_name`` is set only when the
            large-payload fast path was used.
        """
        if (
            self._shm_threshold > 0
            and isinstance(arg, np.ndarray)
            and arg.nbytes >= self._shm_threshold
        ):
            shm = shared_memory.SharedMemory(create=True, size=max(1, arg.nbytes))
            shm_name = shm.name
            with self._pending_shm_lock:
                self._pending_shm.add(shm_name)
            try:
                view = np.ndarray(arg.shape, dtype=arg.dtype, buffer=shm.buf)
                view[...] = arg
                body = _pack_call_large(req_id, method, arg, shm_name)
            except BaseException:
                try:
                    shm.close()
                finally:
                    _try_unlink_shm(shm_name)
                with self._pending_shm_lock:
                    self._pending_shm.discard(shm_name)
                raise
            finally:
                shm.close()
            return _frame(_MSG_CALL_LARGE, body), shm_name
        body = _pack_oob((req_id, method, arg))
        return _frame(_MSG_CALL, body), None

    def _send_loop(self) -> None:
        """Drain the outbound queue and push frames to the socket.

        On ``OSError`` the loop fails every pending future (the current
        frame included -- there is no safe way to know whether the
        server started processing it), discards any frames that were
        queued before the failure (their futures are already failed, so
        transmitting them on a reconnected socket would be a stale
        side-effect), and attempts a single best-effort reconnect. If
        the reconnect succeeds the loop resumes on the new socket;
        callers who need the failed frame re-delivered must re-issue
        it. If the reconnect fails the client is torn down permanently.
        """
        while True:
            item = self._send_queue.get()
            if item is _SENTINEL:
                return
            frame, shm_name = item
            try:
                self._sock.sendall(frame)
            except OSError as exc:
                _logger.warning(f"{self.name}: send failed ({exc})")
                exc_for_pending = ConnectionError(
                    f"tinyros client {self.name!r}: send failed ({exc})"
                )
                self._fail_all_pending(exc_for_pending)
                # Server never consumed this frame, so unlink its shm
                # block ourselves before draining further queued frames.
                self._unlink_orphan_shm(shm_name)
                self._drain_queued_sends()
                if self._shutdown_called or not self._try_reconnect():
                    self._stop_running(exc_for_pending)
                    self._shutdown_io()
                    return
                continue
            else:
                if shm_name is not None:
                    # Server is responsible for unlinking on success;
                    # just release our tracking entry.
                    with self._pending_shm_lock:
                        self._pending_shm.discard(shm_name)

    def _unlink_orphan_shm(self, shm_name: str | None) -> None:
        """Release a shm block the server will never consume.

        Used on the send-failure path where the receiving server never
        got the CALL_LARGE frame (or the frame was dropped before being
        delivered), so the unlink responsibility falls back on the
        client. No-op for inline frames.

        Args:
            shm_name: Name of the shm block to release; ``None`` for
                inline frames.
        """
        if shm_name is None:
            return
        with self._pending_shm_lock:
            if shm_name not in self._pending_shm:
                return
            self._pending_shm.discard(shm_name)
        _try_unlink_shm(shm_name)

    def _drain_queued_sends(self) -> None:
        """Drop every frame still queued behind a failed send.

        Their futures have already been failed by ``_fail_all_pending``;
        re-sending them on a reconnected socket would only produce
        server-side side effects and orphan REPLY frames. Sentinels are
        put back so ``close()`` can still terminate the loop.
        """
        requeue_sentinel = False
        while True:
            try:
                item = self._send_queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                requeue_sentinel = True
                continue
            _frame, shm_name = item
            self._unlink_orphan_shm(shm_name)
        if requeue_sentinel:
            self._send_queue.put(_SENTINEL)

    def _recv_loop(self, sock: socket.socket) -> None:
        """Read REPLY frames from ``sock`` and complete matching futures.

        Bound to a specific socket so a reconnect can spawn a fresh
        recv thread on the new socket without racing the old one.
        Does not clear ``_running`` -- the client stays reusable for
        the next send, which may trigger a reconnect attempt.

        Args:
            sock: The connected socket this loop reads from.
        """
        while True:
            header = _recvall(sock, _HEADER_SIZE, self._running.is_set)
            if header is None:
                if self._running.is_set():
                    self._fail_all_pending(
                        ConnectionError(
                            f"tinyros client {self.name!r}: server "
                            f"closed the connection"
                        )
                    )
                return
            kind, length = struct.unpack(_HEADER_FMT, header)
            if length > self._max_frame_bytes:
                _logger.warning(
                    f"{self.name}: oversized reply frame ({length} bytes, "
                    f"max {self._max_frame_bytes}); tearing down"
                )
                self._stop_running(
                    ConnectionError(
                        f"tinyros client {self.name!r}: oversized reply "
                        f"({length} bytes)"
                    )
                )
                self._shutdown_io()
                return
            body = _recvall(sock, length, self._running.is_set) if length else b""
            if body is None:
                if self._running.is_set():
                    self._fail_all_pending(
                        ConnectionError(f"tinyros client {self.name!r}: short read")
                    )
                return
            if kind != _MSG_REPLY:
                continue
            try:
                req_id, ok, result = _unpack_oob(body)
            except Exception as exc:
                _logger.error(f"{self.name}: failed to decode reply: {exc}")
                continue
            with self._pending_lock:
                fut = self._pending.pop(int(req_id), None)
            if fut is None:
                continue
            if ok:
                fut.set_result(result)
            else:
                fut.set_exception(
                    result
                    if isinstance(result, BaseException)
                    else RuntimeError(str(result))
                )

    def _fail_all_pending(self, exc: BaseException) -> None:
        """Resolve every outstanding future with an exception.

        Does not change the client's running state; callers intending
        a full teardown must clear ``_running`` via :meth:`_stop_running`
        so a ``call()`` racing with teardown sees the cleared flag
        under ``_pending_lock`` instead of orphaning a future. Leaving
        ``_running`` alone means a transient socket failure can still
        be recovered by :meth:`_try_reconnect`.

        Args:
            exc: Exception to set on each pending future.
        """
        with self._pending_lock:
            pending = dict(self._pending)
            self._pending.clear()
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    def _stop_running(self, exc: BaseException) -> None:
        """Clear ``_running`` atomically with draining ``_pending``.

        Used on permanent-teardown paths (close, reconnect-failed,
        oversized reply). ``call()`` re-checks ``_running`` under
        ``_pending_lock`` before inserting, so clearing the flag here
        -- while the lock is held around the final drain -- is what
        keeps a racing caller from leaving an orphan future behind.

        Args:
            exc: Exception to set on each pending future drained by
                this call.
        """
        with self._pending_lock:
            self._running.clear()
            pending = dict(self._pending)
            self._pending.clear()
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    def _try_reconnect(self) -> bool:
        """Replace the dead socket with a fresh connection.

        Closes the old socket (waking the old recv thread), waits
        briefly for the old recv to exit, then calls :meth:`_connect`
        with the reconnect timeout. On success swaps in the new
        socket and spawns a fresh recv thread bound to it.

        Returns:
            ``True`` if the new socket is live and a new recv thread
            is running; ``False`` if the reconnect budget expired.
        """
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        try:
            new_sock = self._connect(self._reconnect_timeout)
        except ConnectionError as exc:
            _logger.warning(f"{self.name}: reconnect failed: {exc}")
            return False
        new_sock.settimeout(_READ_POLL_S)
        self._sock = new_sock
        self._recv_thread = self._start_recv_thread(new_sock)
        _logger.info(f"{self.name}: reconnected to {self.host}:{self.port}")
        return True

    def _start_recv_thread(self, sock: socket.socket) -> threading.Thread:
        """Spawn a recv thread bound to ``sock``. Returns the started thread."""
        t = threading.Thread(
            target=self._recv_loop,
            args=(sock,),
            daemon=True,
            name=f"tinyros-client-recv-{self.name}",
        )
        t.start()
        return t

    def _shutdown_io(self) -> None:
        """Release the socket and wake the send loop without joining threads.

        Safe to call from a worker thread -- notably :meth:`_recv_loop`,
        where joining sibling threads would self-deadlock. Idempotent,
        so :meth:`close` can still run its full shutdown sequence
        afterwards. Unlinks any shm blocks the send loop never got to.
        """
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._send_queue.put(_SENTINEL)
        except (RuntimeError, ValueError):
            pass
        with self._pending_shm_lock:
            stragglers = list(self._pending_shm)
            self._pending_shm.clear()
        for name in stragglers:
            _try_unlink_shm(name)

    def close(self, timeout: float | None = 2.0) -> None:
        """Close the client and release resources.

        Args:
            timeout: Per-thread join timeout; ``None`` waits indefinitely.
        """
        with self._state_lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        # Atomically block new call()s and drain anything already in
        # flight so a racing caller can't slip a future in after the
        # recv loop has been joined.
        self._stop_running(ConnectionError(f"tinyros client {self.name!r} was closed"))

        try:
            self._send_queue.put((_frame(_MSG_BYE, b""), None))
        except (RuntimeError, ValueError):
            pass
        self._send_queue.put(_SENTINEL)
        if self._send_thread.is_alive():
            self._send_thread.join(timeout=timeout)

        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=timeout)

        with self._pending_shm_lock:
            stragglers = list(self._pending_shm)
            self._pending_shm.clear()
        for n in stragglers:
            _try_unlink_shm(n)
