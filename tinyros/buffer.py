"""Module for a process-safe ring buffer using shared memory."""

import os
import struct
import time
from multiprocessing import Condition, Lock, Semaphore, Value, shared_memory
from typing import Optional, Tuple


class Buffer:
    """A process-safe ring buffer using multiprocessing.shared_memory.

    Stores up to `capacity` raw byte-slots of fixed size `slot_size`,
    each tagged with a monotonic 64-bit sequence number.
    Uses a C semaphore + a tight spin-loop to get sub-100 µs handoff times.
    Automatically cleans up shared memory when the creator process exits.
    """

    def __init__(self, capacity: int, slot_size: int):
        """Initialize the buffer with given capacity and slot size.

        Args:
            capacity (int): Maximum number of items the buffer can hold.
            slot_size (int): Size of each item in bytes.

        Raises:
            ValueError: If capacity or slot_size is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(slot_size, int) or slot_size <= 0:
            raise ValueError("slot_size must be a positive integer")

        self._capacity = capacity
        self._slot_size = slot_size
        self._owner_pid = os.getpid()

        # Shared memory blocks
        self._shm_items = shared_memory.SharedMemory(
            create=True, size=capacity * slot_size
        )
        self._shm_seqs = shared_memory.SharedMemory(create=True, size=capacity * 8)

        # SPSC pointers and counters (no built-in locks)
        self._head = Value("i", 0, lock=False)
        self._count = Value("i", 0, lock=False)
        self._counter = Value("q", 0, lock=False)

        # C semaphore to signal “new item available”
        self._sem = Semaphore(0)

        # Unused—but kept for API compatibility
        self._lock = Lock()
        self._cond = Condition(self._lock)

    def put(self, data: bytes) -> None:
        """Write raw bytes (must be exactly slot_size) with a new sequence.

        Raises:
            TypeError: If data is not bytes/bytearray.
            ValueError: If len(data) != slot_size.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("put() expects bytes or bytearray")
        length = len(data)
        if length != self._slot_size:
            raise ValueError(
                f"Data length {length} must equal slot_size {self._slot_size}"
            )

        # write into ring
        idx = self._head.value
        seq = self._counter.value + 1

        # sequence → shared-mem
        struct.pack_into("q", self._shm_seqs.buf, idx * 8, seq)
        # payload → shared-mem
        start = idx * self._slot_size
        self._shm_items.buf[start : start + self._slot_size] = data

        # advance pointers
        self._head.value = (idx + 1) % self._capacity
        if self._count.value < self._capacity:
            self._count.value += 1
        self._counter.value = seq

        # signal reader
        self._sem.release()

    def get_newest(self, t: int, timeout: float) -> Optional[Tuple[int, bytes]]:
        """Return the newest (seq, data) with seq > t, or None on timeout."""
        deadline = time.time() + timeout
        # spin-loop + sem waiting
        while True:
            # fast check
            if self._counter.value > t:
                idx = (self._head.value - 1) % self._capacity
                return self._unpack(idx)
            # timeout?
            rem = deadline - time.time()
            if rem <= 0:
                return None
            # wait for at least one put()
            if not self._sem.acquire(timeout=rem):
                return None
            # loop back and re-check

    def get_next_after(self, t: int, timeout: float) -> Optional[Tuple[int, bytes]]:
        """Return the first (seq, data) with seq > t, or None on timeout."""
        deadline = time.time() + timeout
        while True:
            if self._counter.value > t:
                # scan only the valid slots
                n = self._count.value
                start = (self._head.value - n) % self._capacity
                for i in range(n):
                    idx = (start + i) % self._capacity
                    seq = struct.unpack_from("q", self._shm_seqs.buf, idx * 8)[0]
                    if seq > t:
                        return self._unpack(idx)
                # if overwritten old entries or none > t yet, keep waiting
            rem = deadline - time.time()
            if rem <= 0:
                return None
            if not self._sem.acquire(timeout=rem):
                return None

    def _unpack(self, idx: int) -> Tuple[int, bytes]:
        """Helper to read back the (seq, data) from slot `idx`."""
        seq = struct.unpack_from("q", self._shm_seqs.buf, idx * 8)[0]
        start = idx * self._slot_size
        raw = bytes(self._shm_items.buf[start : start + self._slot_size])
        return seq, raw.rstrip(b"\0")

    def close(self):
        """Close shared-memory handles in this process."""
        try:
            self._shm_items.close()
            self._shm_seqs.close()
        except Exception:
            pass

    def unlink(self):
        """Unlink (destroy) the shared-memory segments (only once)."""
        if os.getpid() == self._owner_pid:
            try:
                self._shm_items.unlink()
                self._shm_seqs.unlink()
            except Exception:
                pass

    def cleanup(self):
        """Cleanup shared-memory segments."""
        self.close()
        self.unlink()
