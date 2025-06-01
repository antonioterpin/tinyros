"""Module for a process-safe ring buffer using shared memory."""

import contextlib
import ctypes
import os
import struct
import time
import cupy as cp
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional, Tuple, Dict, Any, List

from tinyros.datatype import TinyROSMessageDefinition, CupySharray
from tinyros.utils import logger


class BufferClosedError(Exception):
    """Custom exception for broken pipe errors in the buffer."""

    def __init__(self, message: str):
        """Initialize the BufferClosedError with a message.

        Args:
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.message = message


class BufferAlreadyOpenError(Exception):
    """Custom exception for attempting to open a buffer that is already open."""

    def __init__(self, message: str):
        """Initialize the BufferAlreadyOpenError with a message.

        Args:
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.message = message


class Buffer:
    """A process-safe ring buffer using multiprocessing.SharedMemory.

    Stores up to `capacity` fixed-size slots tagged with
    a monotonic 32-bit sequence number.

    Semantics:
      - put(data) writes data into the next free slot,
        bumps a global 32-bit sequence counter, and signals all waiting readers.
      - get_newest(t, timeout) returns the newest (seq, data) with seq>t, or None
        on timeout.
      - get_next_after(t, timeout) scans from oldest-held items up to the most
        recent, returning the first (seq, data) with seq>t, or None on timeout.
    """

    def __init__(self, capacity: int, msg_def: TinyROSMessageDefinition, ctx=None):
        """Initialize the buffer.

        Args:
            capacity (int): Maximum number of items the buffer can hold.
            msg_def (TinyROSMessageDefinition):
                Definition of the message that the buffer will hold.
            ctx: Optional context for multiprocessing (default is None).

        Raises:
            ValueError: If capacity or slot_size is not a positive integer.
        """
        self._out_buffers: Dict[str, List[Any]] = {}
        self._in_buffers: Dict[str, List[Any]] = {}
        self._in_buffers_ptrs: Dict[str, List[Any]] = {}
        self._ctx = ctx or get_context("fork")

        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self._capacity = capacity
        self._owner_pid = os.getpid()
        self._data = {field.name: field.dtype for field in msg_def}

        # ---------------------------------------------------------
        # Create shared memory for the raw payloads and the sequence numbers.
        # We allocate N + 1 separate SharedMemory (see _allocate_buffer) regions:
        #   1) _shm_items:   Dict[str, SharedMemory]  (each item is a fixed-size slot)
        #   2) _shm_seqs:    capacity * 4 bytes   (each seq is a signed 32-bit int)
        #
        # These are never resized.  We index into them as:
        #   seq_array[idx * 4 : idx*4 + 4]   for the 32-bit sequence
        # ---------------------------------------------------------
        self._allocate_buffer(msg_def)
        self._shm_seqs = SharedMemory(create=True, size=capacity * 4)

        # ---------------------------------------------------------
        # Three RawValue counters with no internal locks:
        #   _head    = next index to write (0 <= head < capacity)
        #   _count   = how many slots are currently "occupied" (<= capacity)
        #   _counter = global 32-bit sequence counter (monotonic, starts at 0)
        # We do not use a lock on these RawValues, because either their access
        # does not change the result (i.e., we can perform fast checks), or
        # we hold a readers-writer lock while reading/writing them.
        #
        # We also have a POSIX semaphore that readers block on when
        # there's no new data.  Writers do sem.release() once per put().
        # ---------------------------------------------------------
        self._head = self._ctx.RawValue(ctypes.c_int32, 0)
        self._count = self._ctx.RawValue(ctypes.c_int32, 0)
        self._counter = self._ctx.RawValue(ctypes.c_int32, 0)

        # ReadersWriterLock to ensure safe concurrent access
        self._rwlock = ReadersWriterLock(ctx=self._ctx)
        # A lock for the data available condition variable
        self._data_available_lock = self._ctx.Lock()
        self._data_available_cond = self._ctx.Condition(self._data_available_lock)

    def _allocate_buffer(self, msg_def: TinyROSMessageDefinition) -> None:
        """Allocate shared memory for the buffer items.

        Args:
            msg_def (TinyROSMessageFieldDefinition):
                Definition of the message that the buffer will hold.

        Raises:
            ValueError: If slot_size is not a positive integer.
        """
        self._in_buffers = {}
        self._in_buffers_ptrs = {}
        for field in msg_def:
            if field.dtype == cp.ndarray:
                if "shape" not in field.kwargs or "dtype" not in field.kwargs:
                    raise ValueError(
                        "cp.ndarray fields must have 'shape' and 'dtype' in kwargs"
                    )
                # For each Sharray type, we need to allocate the buffers.
                self._in_buffers[field.name] = [
                    cp.empty(field.kwargs["shape"], dtype=field.kwargs["dtype"])
                    for _ in range(self._capacity)
                ]
                self._in_buffers_ptrs[field.name] = [
                    CupySharray.from_array(self._in_buffers[field.name][i])
                    for i in range(self._capacity)
                ]
            else:
                ValueError(
                    f"Unsupported field type: {field.dtype}."
                    " Convert your data to CupySharray."
                )

    def is_closed(self) -> bool:
        """Check if the buffer has been closed or unlinked."""
        return self._in_buffers is None or self._shm_seqs.buf is None

    @property
    def capacity(self) -> int:
        """Get the maximum number of items the buffer can hold."""
        return self._capacity

    def put(self, data: Dict[str, Any]) -> None:
        """Write raw bytes (must be exactly slot_size) with a new sequence.

        Raises:
            KeyError: If any required key is missing in the data.
            TypeError: If data is not matching.
            BufferClosedError: If the buffer has been closed or unlinked.
        """
        if self.is_closed():
            raise BufferClosedError("Buffer has been closed or unlinked")

        # Acquire the write lock so that no readers are interleaved with this write:
        t0 = time.perf_counter()
        with self._rwlock.for_write():
            # 1) Grab the next index to write:
            idx = self._head.value

            # 2) Bump the global sequence by 1:
            seq = self._counter.value + 1

            # 3) Write the sequence number into _shm_seqs at offset:
            struct.pack_into("i", self._shm_seqs.buf, idx * 4, seq)

            # 4) Copy the payload into _in_buffers
            for key in self._data:
                if key not in data:
                    raise KeyError(f"Missing required key: {key}")
                if not isinstance(data[key], self._data[key]):
                    raise TypeError(
                        f"Expected type {self._data[key]} for key '{key}', "
                        f"but got {type(data[key])}"
                    )
                # TODO: Check how much of a pain it is for the process using the Server
                # to directly work on this. We would halve the number of copies.
                # On the other hand, these copies are relatively fast.
                CupySharray.copy_to(data[key], self._in_buffers[key][idx])
                # Synchronize the CUDA stream to ensure the copy is complete
                cp.cuda.Stream.null.synchronize()

            # 5) Advance head, adjust count (capped at capacity), and update counter:
            self._head.value = (idx + 1) % self._capacity
            if self._count.value < self._capacity:
                self._count.value += 1
            self._counter.value = seq

            # 6) Notify all waiting readers that new data is available:
            with self._data_available_cond:
                self._data_available_cond.notify_all()
            logger.debug(
                (
                    f"Put data with seq {seq} at index {idx} "
                    + f"in {time.perf_counter() - t0:.4f} s"
                )
            )
            # 7) Release the write lock here (implicit on exiting context).

    def _wait_and_find(
        self,
        t: int,
        timeout: float,
        finder: Callable[[int], Tuple[int, bytes]],
    ) -> Optional[Tuple[int, bytes]]:
        """Blocks until a finder returns or times out.

        Args:
            t (int): The sequence threshold. We want seq > t.
            timeout (float): Max time to wait in seconds.
            finder (callable): A function that takes `t` and returns (seq, data).

        Returns:
            Optional[Tuple[int, bytes]]:
                The first (seq, data) matching seq > t, or None on timeout.

        Raises:
            BufferClosedError: If the buffer has been closed or unlinked.
        """
        if self.is_closed():
            raise BufferClosedError("Buffer has been closed or unlinked")

        if not isinstance(t, int) or t < 0:
            raise ValueError("t must be a non-negative integer")

        t0 = time.perf_counter()
        with self._data_available_cond:
            while self._counter.value <= t:
                # compute how much time remains
                elapsed = time.perf_counter() - t0
                remaining = timeout - elapsed
                if remaining <= 0:
                    logger.debug(
                        f"Timeout after {time.perf_counter() - t0:.4f}s "
                        f"(requested was {timeout}) while waiting for seq > {t}."
                    )
                    return None

                # This call atomically releases data_available_lock and sleeps.
                # If a writer notifies after bumping counter, we wake immediately.
                self._data_available_cond.wait(remaining)

        with self._rwlock.for_read(timeout - time.perf_counter() + t0):
            if self._counter.value > t:
                res = finder(t)
                logger.debug(
                    (
                        f"Found data with seq {res[0]} after "
                        + f"waiting {time.perf_counter() - t0:.4f}s"
                    )
                )
                logger.debug(
                    f"Transfer time: {time.time() - res[1]['data'][0]} seconds"
                )
                return res
        logger.debug(
            (
                f"Timeout after {time.perf_counter() - t0:.4f}s "
                + f"(requested is {timeout}) while waiting for read lock access"
            )
        )
        return None

    def get_newest(
        self, name: str, t: int, timeout: float
    ) -> Optional[Tuple[int, bytes]]:
        """Return the newest (seq, data) with seq > t, or None on timeout.

        Args:
            name (str): The name of the client that opened the out buffer.
            t (int): The sequence threshold. We want seq > t.
            timeout (float): Max time to wait in seconds.

        Returns:
            Optional[Tuple[int, bytes]]:
                The newest (seq, data) with seq > t, or None on timeout.
        """

        def _finder(_: int) -> Tuple[int, bytes]:
            # The newest-written slot is at index (head - 1) mod capacity:
            idx = (self._head.value - 1 + self._capacity) % self._capacity
            seq, payload = self._unpack(name, idx)
            return seq, payload

        return self._wait_and_find(t, timeout, _finder)

    def get_next_after(
        self, name: str, t: int, timeout: float
    ) -> Optional[Tuple[int, bytes]]:
        """Return the first (seq, data) with seq > t, or None on timeout.

        Args:
            name (str): The name of the client that opened the out buffer.
            t (int): The sequence threshold. We want seq > t.
            timeout (float): Max time to wait in seconds.

        Returns:
            Optional[Tuple[int, bytes]]:
                The first (seq, data) with seq > t, or None on timeout.
        """

        def _finder(threshold: int) -> Tuple[int, bytes]:
            n = self._count.value
            head = self._head.value
            # The oldest “valid” slot is at index (head - n) mod capacity:
            start_index = (head - n + self._capacity) % self._capacity

            # Scan up to n slots in order of increasing sequence:
            for i in range(n):
                idx = (start_index + i + self._capacity) % self._capacity
                seq = struct.unpack_from("i", self._shm_seqs.buf, idx * 4)[0]
                if seq > threshold:
                    _, payload = self._unpack(name, idx)
                    return seq, payload
            # This should never happen...
            return None

        return self._wait_and_find(t, timeout, _finder)

    def _unpack(self, name: str, idx: int) -> Dict[str, Any]:
        """Read back the (seq, data) from slot `idx`. Must be called under a read-lock."""
        if name not in self._out_buffers:
            raise BufferClosedError("Did you open the out buffers?")
        seq = struct.unpack_from("i", self._shm_seqs.buf, idx * 4)[0]
        data = {
            key: self._out_buffers[name][key][
                idx
            ]  # this is not a copy, but a reference
            for key in self._in_buffers_ptrs
        }
        return seq, data

    def close(self):
        """Close shared-memory handles in this process."""
        try:
            self._in_buffers = None
            self._in_buffers_ptrs = None
            self._shm_seqs.close()
        except Exception:
            pass

    def open_in_buffer(self):
        """Open the shared-memory segments for writing in this process.

        This registers the current PID as the owner of the segments.
        """
        if self.is_closed():
            raise BufferClosedError("Buffer has been closed or unlinked")

        # With the current logic, the in buffer is first open in the master process
        # so we need to open it here as well.
        self._in_buffers = {
            key: [ptr.open() for ptr in self._in_buffers_ptrs[key]]
            for key in self._in_buffers_ptrs
        }

    def open_out_buffer(self, client_name: str):
        """Open the shared-memory segments for reading in this process.

        This registers the current PID as the owner of the segments.
        """
        if self.is_closed():
            raise BufferClosedError("Buffer has been closed or unlinked")

        if client_name in self._out_buffers:
            raise BufferAlreadyOpenError(
                f"Buffer already open for client {client_name}"
            )

        self._out_buffers[client_name] = {
            key: [ptr.open() for ptr in self._in_buffers_ptrs[key]]
            for key in self._in_buffers_ptrs
        }

    def unlink(self):
        """Unlink (destroy) the shared-memory segments (only once, by the creator)."""
        if os.getpid() == self._owner_pid:
            try:
                self._shm_seqs.unlink()
            except Exception:
                pass

    def cleanup(self):
        """Cleanup shared-memory segments (close + unlink)."""
        self.close()
        self.unlink()


class ReadersWriterLock:
    """Multiprocessing-compatible readers/writer lock.

    See https://en.wikipedia.org/wiki/Readers-writers_problem

    The implementation is so that no reader or writer can starve.
    """

    def __init__(self, ctx=None):
        """Initialize the readers-writer lock.

        Args:
            ctx: Optional context for multiprocessing (default is None).
        """
        self._ctx = ctx or get_context("fork")
        self._mutex = self._ctx.Lock()
        self._readers_cond = self._ctx.Condition(self._mutex)
        self._writer_cond = self._ctx.Condition(self._mutex)
        self._readers = self._ctx.RawValue(ctypes.c_uint, 0)
        self._writer = self._ctx.RawValue(ctypes.c_bool, False)

    def _acquire_reader_lock(self, timeout: Optional[float] = None):
        """Acquire the lock for reading (multiple readers allowed).

        The mutex is held only briefly while checking the writer status.
        Then, the reader count is incremented atomically and the mutex is released.
        """
        with self._mutex:
            # Wait until no writer is active:
            while self._writer.value:
                self._readers_cond.wait(timeout)
            # Register ourselves as a reader:
            self._readers.value += 1

    def _release_reader_lock(self):
        """Release the lock for reading.

        This decrements the reader count and notifies any waiting writer.
        """
        with self._mutex:
            self._readers.value -= 1
            # If we were the last reader, notify any waiting writer:
            if self._readers.value == 0:
                self._writer_cond.notify()

    @contextlib.contextmanager
    def for_read(self, timeout: Optional[float] = None):
        """Acquire the lock for reading (multiple readers allowed).

        Args:
            timeout (float, optional): Maximum time to wait for the lock in seconds.
                If None, wait indefinitely.
        """
        self._acquire_reader_lock(timeout)
        try:
            yield
        finally:
            self._release_reader_lock()

    def _acquire_writer_lock(self, timeout: Optional[float] = None):
        """Acquire the lock for writing (exclusive, no readers allowed).

        The mutex is held while waiting for no readers and no other writer.

        Args:
            timeout (float, optional): Maximum time to wait for the lock in seconds.
                If None, wait indefinitely.
        """
        with self._mutex:
            # Wait until no readers and no other writer:
            while self._writer.value or self._readers.value > 0:
                self._writer_cond.wait(timeout)
            self._writer.value = True

    def _release_writer_lock(self):
        """Release the lock for writing.

        This resets the writer status and notifies all waiting readers and one writer.
        """
        with self._mutex:
            self._writer.value = False
            # Wake all waiting readers, and wake one waiting writer:
            self._readers_cond.notify_all()
            self._writer_cond.notify()

    @contextlib.contextmanager
    def for_write(self, timeout: Optional[float] = None):
        """Acquire the lock for writing (exclusive)."""
        self._acquire_writer_lock(timeout)
        try:
            yield
        finally:
            self._release_writer_lock()
