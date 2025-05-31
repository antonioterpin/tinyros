"""Module for a process-safe ring buffer using shared memory."""

import contextlib
import ctypes
import os
import struct
import time
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional, Tuple

class BufferClosedError(Exception):
    """Custom exception for broken pipe errors in the buffer."""
    pass


class Buffer:
    """A process-safe ring buffer using multiprocessing.SharedMemory.

    Stores up to `capacity` fixed-size slots (each of size `slot_size`),
    each tagged with a monotonic 32-bit sequence number.

    Semantics:
      - put(data) always writes exactly `slot_size` bytes into the next free slot,
        bumps a global 32-bit sequence counter, and signals all waiting readers.
      - get_newest(t, timeout) returns the newest (seq, data) with seq>t, or None
        on timeout.
      - get_next_after(t, timeout) scans from oldest-held items up to the most
        recent, returning the first (seq, data) with seq>t, or None on timeout.

    Internals:
      - A ReadersWriterLock ensures that reads and writes to the data+sequence
        arrays never race.  Writers hold a short “for_write()” lock while
        updating head/count/counter and the shared-memory arrays.  Readers
        briefly grab a “for_read()” lock to fetch sequence+payload atomically.
      - A POSIX semaphore is used to wake up waiting readers.  Readers spin,
        checking counter > t, and if no new data is available, block on
        `Semaphore.acquire(timeout=...)`.  Writers do `Semaphore.release()`
        once per `put()` to wake any blocked reader.
    """

    def __init__(self, capacity: int, slot_size: int, ctx=None):
        """Initialize the buffer with given capacity and slot size.

        Args:
            capacity (int): Maximum number of items the buffer can hold.
            slot_size (int): Size of each item in bytes.
            ctx: Optional context for multiprocessing (default is None).

        Raises:
            ValueError: If capacity or slot_size is not a positive integer.
        """
        # By default, we explicitly use "spawn" on POSIX or Windows to avoid
        # unexpected state‐copying semantics that arise with "fork".
        # (Note: On Linux, the default is "fork".)
        self._ctx = ctx or get_context("spawn")

        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(slot_size, int) or slot_size <= 0:
            raise ValueError("slot_size must be a positive integer")

        self._capacity = capacity
        self._slot_size = slot_size
        self._owner_pid = os.getpid()

        # ---------------------------------------------------------
        # Create shared memory for the raw payloads and the sequence numbers.
        # We allocate two separate SharedMemory regions:
        #   1) _shm_items:   capacity * slot_size bytes
        #   2) _shm_seqs:    capacity * 4 bytes   (each seq is a signed 32-bit int)
        #
        # These are never resized.  We index into them as:
        #   seq_array[idx * 4 : idx*4 + 4]   for the 32-bit sequence
        #   items_array[idx * slot_size : idx*slot_size + slot_size]  for the payload
        # ---------------------------------------------------------
        self._shm_items = SharedMemory(create=True, size=capacity * slot_size)
        self._shm_seqs = SharedMemory(create=True, size=capacity * 4)

        # ---------------------------------------------------------
        # Three RawValue counters with no internal locks:
        #   _head    = next index to write (0 <= head < capacity)
        #   _count   = how many slots are currently “occupied” (<= capacity)
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

        # POSIX semaphore to signal “new item available”
        self._sem = self._ctx.Semaphore(0)
        # ReadersWriterLock to ensure safe concurrent access
        self._rwlock = ReadersWriterLock(ctx=self._ctx)

    @property
    def capacity(self) -> int:
        """Get the maximum number of items the buffer can hold."""
        return self._capacity

    @property
    def slot_size(self) -> int:
        """Get the size of each item in bytes."""
        return self._slot_size

    def put(self, data: bytes) -> None:
        """Write raw bytes (must be exactly slot_size) with a new sequence.

        Raises:
            TypeError: If data is not bytes/bytearray.
            ValueError: If len(data) != slot_size.
        """
        if self._shm_items.buf is None or self._shm_seqs.buf is None:
            raise BufferClosedError("Buffer has been closed or unlinked")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("put() expects bytes or bytearray")
        length = len(data)
        if length != self._slot_size:
            raise ValueError(
                f"Data length {length} must equal slot_size {self._slot_size}"
            )

        # Acquire the write lock so that no readers are interleaved with this write:
        with self._rwlock.for_write():
            # 1) Grab the next index to write:
            idx = self._head.value

            # 2) Bump the global sequence by 1:
            seq = self._counter.value + 1

            # 3) Write the sequence number into _shm_seqs at offset:
            struct.pack_into("i", self._shm_seqs.buf, idx * 4, seq)

            # 4) Copy the payload into _shm_items at offset (idx * slot_size):
            start = idx * self._slot_size
            self._shm_items.buf[start : start + self._slot_size] = data

            # 5) Advance head, adjust count (capped at capacity), and update counter:
            self._head.value = (idx + 1) % self._capacity
            if self._count.value < self._capacity:
                self._count.value += 1
            self._counter.value = seq

            # 6) Release the write lock here (implicit on exiting context).

        # 7) Signal one or more waiting readers that “a new item is available”:
        #    We only need one release() per put(); if multiple readers are blocked,
        #    they’ll all wake up in turn (each get() does its own sem.acquire()).
        self._sem.release()

    def _wait_and_find(
        self,
        t: int,
        timeout: float,
        finder: Callable[[int], Tuple[int, bytes]],
    ) -> Optional[Tuple[int, bytes]]:
        """Generic wait loop: blocks until a finder returns or times out.

        Args:
            t (int): The sequence threshold. We want seq > t.
            timeout (float): Max time to wait in seconds.
            finder (callable): A function that takes `t` and returns (seq, data).

        Returns:
            Optional[Tuple[int, bytes]]: 
                The first (seq, data) matching seq > t, or None on timeout.
        """
        if self._shm_seqs.buf is None or self._shm_items.buf is None:
            raise BufferClosedError("Buffer has been closed or unlinked")

        if not isinstance(t, int) or t < 0:
            raise ValueError("t must be a non-negative integer")

        deadline = time.time() + timeout

        while True:
            # Fast check: is there any sequence > t?
            # No need for a lock here, since the counter is monotonic.
            # Thus, if its value changes, the condition will still hold.
            if self._counter.value > t:
                with self._rwlock.for_read():
                    return finder(t)

            # No candidate found yet; check timeout then block on semaphore
            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            got_it = self._sem.acquire(timeout=remaining)
            if not got_it:
                return None
            # Once sem.acquire() returns True, loop back and re‐check.

    def get_newest(self, t: int, timeout: float) -> Optional[Tuple[int, bytes]]:
        """Return the newest (seq, data) with seq > t, or None on timeout.

        Args:
            t (int): The sequence threshold. We want seq > t.
            timeout (float): Max time to wait in seconds.

        Returns:
            Optional[Tuple[int, bytes]]:
                The newest (seq, data) with seq > t, or None on timeout.
        """

        def _finder(_: int) -> Tuple[int, bytes]:
            # The newest-written slot is at index (head - 1) mod capacity:
            idx = (self._head.value - 1 + self._capacity) % self._capacity
            seq, payload = self._unpack(idx)
            return seq, payload

        return self._wait_and_find(t, timeout, _finder)

    def get_next_after(self, t: int, timeout: float) -> Optional[Tuple[int, bytes]]:
        """Return the first (seq, data) with seq > t, or None on timeout.

        Args:
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
                    _, payload = self._unpack(idx)
                    return seq, payload
            # This should never happen...
            return None

        return self._wait_and_find(t, timeout, _finder)

    def _unpack(self, idx: int) -> Tuple[int, bytes]:
        """Read back the (seq, data) from slot `idx`. Must be called under a read-lock."""
        seq = struct.unpack_from("i", self._shm_seqs.buf, idx * 4)[0]
        start = idx * self._slot_size
        raw = bytes(self._shm_items.buf[start : start + self._slot_size])
        return seq, raw

    def close(self):
        """Close shared-memory handles in this process."""
        try:
            self._shm_items.close()
            self._shm_seqs.close()
        except Exception:
            pass

    def unlink(self):
        """Unlink (destroy) the shared-memory segments (only once, by the creator)."""
        if os.getpid() == self._owner_pid:
            try:
                self._shm_items.unlink()
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
    """

    def __init__(self, ctx=None):
        """Initialize the readers-writer lock.

        Args:
            ctx: Optional context for multiprocessing (default is None).
        """
        # By default, we explicitly use "spawn" on POSIX or Windows to avoid
        # unexpected state‐copying semantics that arise with "fork".
        # (Note: On Linux, the default is "fork".)
        self._ctx = ctx or get_context("spawn")
        self._mutex = self._ctx.Lock()
        self._readers_cond = self._ctx.Condition(self._mutex)
        self._writer_cond = self._ctx.Condition(self._mutex)
        self._readers = self._ctx.RawValue(ctypes.c_uint, 0)
        self._writer = self._ctx.RawValue(ctypes.c_bool, False)

    def _acquire_reader_lock(self):
        """Acquire the lock for reading (multiple readers allowed).

        The mutex is held only briefly while checking the writer status.
        Then, the reader count is incremented atomically and the mutex is released.
        """
        with self._mutex:
            # Wait until no writer is active:
            while self._writer.value:
                self._readers_cond.wait()
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
    def for_read(self):
        """Acquire the lock for reading (multiple readers allowed)."""
        self._acquire_reader_lock()
        try:
            yield
        finally:
            self._release_reader_lock()

    def _acquire_writer_lock(self):
        """Acquire the lock for writing (exclusive, no readers allowed).

        The mutex is held while waiting for no readers and no other writer.
        """
        with self._mutex:
            # Wait until no readers and no other writer:
            while self._writer.value or self._readers.value > 0:
                self._writer_cond.wait()
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
    def for_write(self):
        """Acquire the lock for writing (exclusive)."""
        self._acquire_writer_lock()
        try:
            yield
        finally:
            self._release_writer_lock()
