"""iceoryx2-backed transport for TinyROS.

Each ``(publisher_node, topic)`` declared in the network config maps
to a single iceoryx2 publish-subscribe service named
``tinyros/<publisher>/<topic>``. Publishers loan a dynamic-length
slice in shared memory, write an 8-byte length prefix followed by the
pickled message body, and send. Subscribers attach to the same
service, run a daemon dispatch thread, and invoke the configured
callback on each received sample.

This module replaces the previous TCP + pickle proto-5 OOB +
hand-rolled SHM threshold transport (~2k lines across
``_client.py`` / ``_server.py`` / ``_framing.py`` / ``_common.py``).
iceoryx2 owns all of:

- the wire (true cross-process zero-copy SHM),
- backpressure and lifetime management,
- the cross-OS endpoint split (``ipc://`` vs TCP loopback),
- service discovery within a host.
"""

from __future__ import annotations

import ctypes
import pickle
import threading
import time
from collections.abc import Callable
from typing import Any

import iceoryx2 as iox2
from iceoryx2 import Slice

from ._errors import SerializationError, TransportError

__all__ = ["TransportError", "SerializationError"]

_HEADER_BYTES = 8  # u64 little-endian length prefix


def _service_name(publisher: str, topic: str) -> Any:
    return iox2.ServiceName.new(f"tinyros/{publisher}/{topic}")


def _open_service(node: Any, name: Any) -> Any:
    return (
        node.service_builder(name)
        .publish_subscribe(Slice[ctypes.c_uint8])
        .open_or_create()
    )


class _Publisher:
    """One iceoryx2 publisher tied to a single ``(node, topic)`` pair.

    The slice capacity is grown lazily: the first publish that doesn't
    fit recreates the underlying publisher with a doubled
    ``initial_max_slice_len``. Cost amortizes to zero for steady-state
    workloads.
    """

    def __init__(
        self,
        node: Any,
        publisher_name: str,
        topic: str,
        *,
        initial_capacity: int = 4096,
    ) -> None:
        """Initialize the publisher backed by an iceoryx2 service."""
        self._service = _open_service(
            node, _service_name(publisher_name, topic)
        )
        self._capacity = max(initial_capacity, _HEADER_BYTES)
        self._publisher = self._build(self._capacity)

    def _build(self, capacity: int) -> Any:
        return (
            self._service.publisher_builder()
            .initial_max_slice_len(capacity)
            .create()
        )

    def publish(self, message: Any) -> None:
        """Serialize ``message`` once and send it on the wire.

        Raises:
            SerializationError: pickling the message failed.
            TransportError: the iceoryx2 send call failed.
        """
        try:
            body = pickle.dumps(message, protocol=5)
        except Exception as exc:  # noqa: BLE001
            raise SerializationError(
                f"failed to pickle message of type "
                f"{type(message).__name__}: {exc}"
            ) from exc
        total = _HEADER_BYTES + len(body)
        if total > self._capacity:
            self._capacity = max(total, self._capacity * 2)
            self._publisher = self._build(self._capacity)
        try:
            sample = self._publisher.loan_slice_uninit(total)
            base = sample.payload_ptr
            ctypes.memmove(
                base, len(body).to_bytes(_HEADER_BYTES, "little"), _HEADER_BYTES
            )
            if body:
                ctypes.memmove(base + _HEADER_BYTES, body, len(body))
            sample.assume_init().send()
        except Exception as exc:  # noqa: BLE001
            raise TransportError(f"iceoryx2 publish failed: {exc}") from exc

    def close(self) -> None:
        """Drop the iceoryx2 publisher handle."""
        self._publisher = None


class _Subscriber:
    """One iceoryx2 subscriber + a daemon dispatch thread.

    The thread polls ``subscriber.receive()`` non-blockingly. Each
    received sample is decoded and fed to ``callback``; callback
    failures are logged but never kill the dispatcher.
    """

    def __init__(
        self,
        node: Any,
        publisher_name: str,
        topic: str,
        callback: Callable[[Any], Any],
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Initialize subscriber and start its dispatch thread."""
        self._service = _open_service(
            node, _service_name(publisher_name, topic)
        )
        self._subscriber = self._service.subscriber_builder().create()
        self._callback = callback
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"tinyros-sub:{publisher_name}/{topic}",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._subscriber.receive()
            if sample is None:
                time.sleep(1e-5)
                continue
            sl = sample.payload()
            n = sl.number_of_elements
            buf = (ctypes.c_uint8 * n).from_address(sl.data_ptr)
            length = int.from_bytes(bytes(buf[:_HEADER_BYTES]), "little")
            try:
                message = pickle.loads(
                    bytes(buf[_HEADER_BYTES : _HEADER_BYTES + length])
                )
            except Exception as exc:  # noqa: BLE001
                if self._on_error is not None:
                    self._on_error(SerializationError(str(exc)))
                continue
            try:
                self._callback(message)
            except BaseException as exc:  # noqa: BLE001
                if self._on_error is not None:
                    self._on_error(exc)

    def close(self) -> None:
        """Stop the dispatch thread and drop the iceoryx2 handles."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._subscriber = None


def make_node() -> Any:
    """Create a fresh iceoryx2 IPC node.

    Each ``TinyNode`` owns one of these for the lifetime of the
    process; iceoryx2 services are shared across nodes by name so
    multiple ``TinyNode`` instances (in the same or different
    processes) can attach to the same service.
    """
    return iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
