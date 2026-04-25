"""TinyROS server: accept connections, dispatch CALL frames to callbacks."""

from __future__ import annotations

import concurrent.futures
import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

from ._common import (
    _ACCEPT_POLL_S,
    _DEFAULT_POOL_WORKERS,
    _HEADER_FMT,
    _HEADER_SIZE,
    _INFLIGHT_ACQUIRE_POLL_S,
    _LISTEN_BACKLOG,
    _MSG_BYE,
    _MSG_CALL,
    _MSG_CALL_LARGE,
    _MSG_REPLY,
    _READ_POLL_S,
    _default_max_frame_bytes,
    _logger,
    _PendingCall,
)
from ._errors import SerializationError
from ._framing import (
    _frame,
    _materialize_call_large,
    _pack_oob,
    _parse_call_large_meta,
    _recvall,
    _unpack_oob,
)


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
        max_in_flight: int | None = None,
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
            max_in_flight: Maximum number of calls in flight (running +
                queued) before the reader thread blocks on new frames.
                ``None`` defaults to ``workers * 3``. See
                ``docs/guides/architecture/tiny-objects.md`` ("Worker
                pool and in-flight cap" / "Submit during shutdown") for
                the backpressure model and the during-shutdown drop
                semantics.

        Raises:
            ValueError: If ``workers`` is not at least 1, or if the
                resolved ``max_in_flight`` is not at least 1.
        """
        self.name = name
        self.host = host
        self.port = port
        if workers < 1:
            raise ValueError(f"workers must be at least 1; got {workers}")
        self._max_frame_bytes = (
            max_frame_bytes
            if max_frame_bytes is not None
            else _default_max_frame_bytes()
        )
        effective_in_flight = (
            max_in_flight if max_in_flight is not None else workers * 3
        )
        if effective_in_flight < 1:
            raise ValueError(
                "max_in_flight must resolve to at least 1; "
                f"got {effective_in_flight}"
            )
        self._pool_semaphore = threading.BoundedSemaphore(effective_in_flight)
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
        sock.listen(_LISTEN_BACKLOG)
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

        ``timeout`` gates the joins for the accept and reader threads.
        The worker pool is drained with ``wait=True`` afterwards, so a
        callback already running can extend ``close()`` past the
        configured timeout; queued-but-not-yet-running callbacks are
        cancelled via ``cancel_futures=True``. If you need a hard upper
        bound, detach the server in a supervisor and reap the process
        externally.

        Args:
            timeout: Per-thread join timeout for the accept and reader
                threads; ``None`` waits indefinitely.
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
                # _recvall returns b"" for n=0 without touching the socket,
                # so the length==0 path needs no special casing here.
                body = _recvall(conn, length, self._running.is_set)
                if body is None:
                    return
                try:
                    self._handle_frame(conn, kind, body)
                except SerializationError as exc:
                    # Peer sent garbage we cannot decode. Drop the
                    # connection but log at warning -- this is bad
                    # input, not a server bug.
                    _logger.warning(
                        f"{self.name}: dropping peer (kind={kind}, "
                        f"peer_fd={conn.fileno()}): {exc}"
                    )
                    return
                except Exception:
                    # Anything else is unexpected -- include the
                    # traceback so the operator can find the bug.
                    _logger.exception(
                        f"{self.name}: frame handler crashed "
                        f"(kind={kind}, peer_fd={conn.fileno()})"
                    )
                    return
        finally:
            self._drop_conn(conn)

    def _handle_frame(self, conn: socket.socket, kind: int, body: bytes) -> None:
        """Dispatch a decoded frame to the right handler path.

        ``CALL_LARGE`` materialization (shm open, memcpy, unlink) is
        pushed onto the worker pool so a large payload does not stall
        the per-connection reader while it copies hundreds of MiB out
        of shared memory -- the next frame on the same connection can
        begin decoding immediately.

        Args:
            conn: Peer socket the frame arrived on.
            kind: Message kind from the frame header.
            body: Frame body bytes.
        """
        if kind == _MSG_CALL:
            try:
                req_id, cb_name, arg = _unpack_oob(body)
            except Exception as exc:
                raise SerializationError(
                    f"tinyros server {self.name!r}: failed to decode CALL "
                    f"frame: {type(exc).__name__}: {exc}"
                ) from exc
            pending = _PendingCall(conn, int(req_id), str(cb_name), arg)
            self._submit_call(self._execute_call, pending)
        elif kind == _MSG_CALL_LARGE:
            self._submit_call(self._execute_large_call, conn, body)
        elif kind == _MSG_BYE:
            return
        else:
            _logger.warning(f"{self.name}: unknown frame kind {kind}")

    def _submit_call(self, fn: Callable[..., Any], *args: Any) -> None:
        """Submit a call to the worker pool with in-flight backpressure.

        See ``docs/guides/architecture/tiny-objects.md`` ("Worker pool
        and in-flight cap" / "Submit during shutdown") for the model.

        Args:
            fn: Worker method (:meth:`_execute_call` or
                :meth:`_execute_large_call`).
            *args: Arguments forwarded to ``fn``.
        """
        while self._running.is_set():
            if self._pool_semaphore.acquire(timeout=_INFLIGHT_ACQUIRE_POLL_S):
                break
        else:
            return
        try:
            fut = self._pool.submit(fn, *args)
        except RuntimeError:
            # ThreadPoolExecutor.submit() raises RuntimeError exactly
            # when the pool is shut down. Expected during close():
            # drop the slot and the call; the reader loop will notice
            # _running fall and exit on its own without logging a
            # spurious error.
            self._pool_semaphore.release()
            return
        except Exception:
            # Anything else from submit() is unexpected. Release the
            # slot we just acquired and surface the traceback so an
            # operator can find the bug; previously this re-raised
            # silently and got conflated with deserialization failures
            # at the reader loop's catch-site.
            self._pool_semaphore.release()
            _logger.exception(f"{self.name}: unexpected error submitting work to pool")
            raise
        fut.add_done_callback(lambda _f: self._pool_semaphore.release())

    def _execute_large_call(self, conn: socket.socket, body: bytes) -> None:
        """Unpack a ``CALL_LARGE`` body and run the callback.

        Runs on the worker pool, not the reader thread, so the shm
        memcpy does not stall the connection. See
        ``docs/guides/architecture/tiny-objects.md``
        ("CALL_LARGE failure handling") for the parse-fail vs
        materialize-fail behavior.

        Args:
            conn: Peer socket the frame arrived on.
            body: Pickled metadata produced by :func:`_pack_call_large`.
        """
        try:
            req_id, cb_name, meta = _parse_call_large_meta(body)
        except Exception as exc:
            _logger.error(
                f"{self.name}: corrupt CALL_LARGE metadata; "
                f"dropping connection: {exc}"
            )
            self._drop_conn(conn)
            return
        try:
            arr = _materialize_call_large(meta)
        except Exception as exc:
            _logger.error(
                f"{self.name}: failed to materialize CALL_LARGE "
                f"(req_id={req_id}, cb={cb_name!r}): {exc}"
            )
            self._reply_failure(
                conn,
                req_id,
                cb_name,
                RuntimeError(
                    f"tinyros server {self.name!r}: CALL_LARGE payload for "
                    f"{cb_name!r} could not be read from shared memory: {exc}"
                ),
            )
            return
        self._execute_call(_PendingCall(conn, req_id, cb_name, arr))

    def _reply_failure(
        self,
        conn: socket.socket,
        req_id: int,
        cb_name: str,
        exc: BaseException,
    ) -> None:
        """Send a synthesized ``ok=False`` REPLY without running a callback.

        Used when dispatch itself fails before the callback has a
        chance to run (e.g., CALL_LARGE shm materialization errors).

        Args:
            conn: Peer socket the original frame arrived on.
            req_id: Request id to address the reply to.
            cb_name: Callback name -- only used for the log line.
            exc: Exception to ship back as the failure cause.
        """
        try:
            body = _pack_oob((req_id, False, exc))
        except Exception as pickle_exc:
            _logger.warning(
                f"{self.name}: failure reply for {cb_name!r} is not "
                f"picklable ({pickle_exc}); substituting RuntimeError"
            )
            body = _pack_oob(
                (
                    req_id,
                    False,
                    RuntimeError(
                        f"tinyros server {self.name!r}: dispatch error "
                        f"for {cb_name!r} is not serializable "
                        f"({type(exc).__name__}: {pickle_exc})"
                    ),
                )
            )
        frame = _frame(_MSG_REPLY, body)
        lock = self._conn_send_locks.get(conn)
        if lock is None:
            return
        with lock:
            try:
                conn.sendall(frame)
            except OSError:
                return

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
