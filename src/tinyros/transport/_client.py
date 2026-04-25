"""TinyROS client: connect to a server, send CALL frames, demux REPLY frames."""

from __future__ import annotations

import concurrent.futures
import queue
import socket
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ._common import (
    _CONNECT_TIMEOUT_S,
    _HEADER_FMT,
    _HEADER_SIZE,
    _MSG_BYE,
    _MSG_CALL,
    _MSG_CALL_LARGE,
    _MSG_REPLY,
    _READ_POLL_S,
    _RECONNECT_TIMEOUT_S,
    _SENTINEL,
    _default_max_frame_bytes,
    _default_shm_threshold,
    _logger,
)
from ._errors import ConnectionLost, SerializationError
from ._framing import (
    _frame,
    _pack_call_large,
    _pack_oob,
    _recvall,
    _try_unlink_shm,
    _unpack_oob,
)


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
                a send failure before giving up. One reconnect attempt
                per detected socket failure. See
                ``docs/guides/architecture/tiny-objects.md``
                ("Reconnect on send failure") for the full
                choreography.
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
            ConnectionLost: If no connection could be established.
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
        raise ConnectionLost(
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
                ConnectionLost(f"tinyros client {self.name!r} is no longer running")
            )
            return fut
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
        try:
            frame, shm_name = self._encode_call(req_id, method, arg)
        except Exception as exc:
            fut.set_exception(
                SerializationError(
                    f"tinyros client {self.name!r}: failed to encode call to "
                    f"{method!r}: {exc}"
                )
            )
            return fut
        # Insert into _pending and put on _send_queue under one lock so
        # the failure-handling block in _send_loop (which holds the same
        # lock around fail-pending + drain-queue) cannot interleave and
        # silently drop a frame whose future is still pending.
        with self._pending_lock:
            if not self._running.is_set():
                fut.set_exception(
                    ConnectionLost(f"tinyros client {self.name!r} is no longer running")
                )
                if shm_name is not None:
                    with self._pending_shm_lock:
                        self._pending_shm.discard(shm_name)
                    _try_unlink_shm(shm_name)
                return fut
            self._pending[req_id] = fut
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

        On ``OSError`` the loop runs the reconnect-on-send-failure
        choreography. See
        ``docs/guides/architecture/tiny-objects.md``
        ("Reconnect on send failure") for the step-by-step. ``_running``
        is also consulted before reconnecting so a teardown initiated
        by another path (e.g., ``_recv_loop``'s oversized-reply branch)
        cannot be inadvertently revived by an OSError on the next
        ``sendall``.
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
                exc_for_pending = ConnectionLost(
                    f"tinyros client {self.name!r}: send failed ({exc})"
                )
                # Hold _pending_lock across the snapshot + clear + drain
                # so a concurrent ``call()`` (which takes the same lock
                # around insert + put) cannot slip a frame past us:
                # either its insert+put completes before us and is
                # included in the failure, or it waits for the lock,
                # observes the cleared state (post-_stop_running) or
                # the new socket, and behaves correctly.
                with self._pending_lock:
                    pending_snapshot = dict(self._pending)
                    self._pending.clear()
                    # Server never consumed the failing frame, so unlink
                    # its shm block before draining further queued frames.
                    self._unlink_orphan_shm(shm_name)
                    self._drain_queued_sends()
                for fut in pending_snapshot.values():
                    if not fut.done():
                        fut.set_exception(exc_for_pending)
                if (
                    self._shutdown_called
                    or not self._running.is_set()
                    or not self._try_reconnect()
                ):
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
            _frame_bytes, shm_name = item
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
                        ConnectionLost(
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
                    ConnectionLost(
                        f"tinyros client {self.name!r}: oversized reply "
                        f"({length} bytes)"
                    )
                )
                self._shutdown_io()
                return
            body = _recvall(sock, length, self._running.is_set)
            if body is None:
                if self._running.is_set():
                    self._fail_all_pending(
                        ConnectionLost(f"tinyros client {self.name!r}: short read")
                    )
                return
            if kind != _MSG_REPLY:
                _logger.warning(
                    f"{self.name}: unexpected non-reply frame kind "
                    f"{kind} on the client socket; ignoring"
                )
                continue
            try:
                req_id, ok, result = _unpack_oob(body)
            except Exception as exc:
                _logger.error(
                    f"{self.name}: failed to decode reply: "
                    f"{type(exc).__name__}: {exc}",
                )
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

        See ``docs/guides/architecture/tiny-objects.md``
        ("Reconnect on send failure") for the choreography.

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
        except ConnectionLost as exc:
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
        self._stop_running(ConnectionLost(f"tinyros client {self.name!r} was closed"))

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
