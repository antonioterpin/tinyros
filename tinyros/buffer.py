import time
from multiprocessing import Manager, Value, Lock
from typing import Any, Optional, Tuple, List


class Buffer:
    """A process-safe ring buffer that stores items with a monotonic sequence counter.

    This buffer uses multiprocessing primitives to allow safe concurrent access
    across processes. New items overwrite the oldest when capacity is reached.

    Attributes:
        _data (List[Optional[Tuple[int, Any]]]): Shared list holding (seq, item) pairs.
        _capacity (int): Maximum number of items the buffer can hold.
        _head (Value): Index in `_data` for the next write.
        _count (Value): Current number of valid items in the buffer.
        _counter (Value): Monotonic sequence counter for items.
        _lock (Lock): Mutex protecting all buffer operations.
    """

    def __init__(self, capacity: int):
        """Initialize a ring buffer with the given capacity.

        Args:
            capacity (int): Maximum number of items the buffer can hold.

        Raises:
            ValueError: If `capacity` is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        self._data = Manager().list([None] * capacity)
        self._capacity = capacity

        self._head = Value('i', 0)
        self._count = Value('i', 0)
        self._counter = Value('i', 0)
        self._lock = Lock()

    def put(self, item: Any) -> None:
        """Insert an item with an auto-incremented sequence number.

        If the buffer is full, the oldest entry is dropped.

        Args:
            item (Any): The item to insert into the buffer.
        """
        with self._lock:
            # Increment global sequence counter
            self._counter.value += 1
            seq = self._counter.value

            # Place new entry at head position
            idx = self._head.value
            self._data[idx] = (seq, item)

            # Advance head pointer, wrapping around capacity
            self._head.value = (idx + 1) % self._capacity

            # Increase count up to capacity
            if self._count.value < self._capacity:
                self._count.value += 1

    def get_oldest(self) -> Optional[Tuple[int, Any]]:
        """Return the oldest entry in the buffer.

        Returns:
            Optional[Tuple[int, Any]]: The oldest (sequence, item) pair,
            or `None` if the buffer is empty.
        """
        with self._lock:
            if self._count.value == 0:
                return None
            start = (self._head.value - self._count.value) % self._capacity
            return self._data[start]

    def get_newest(self) -> Optional[Tuple[int, Any]]:
        """Return the newest entry in the buffer.

        Returns:
            Optional[Tuple[int, Any]]: 
                The newest (sequence, item) pair, or `None` if the buffer is empty.
        """
        with self._lock:
            if self._count.value == 0:
                return None
            newest_idx = (self._head.value - 1) % self._capacity
            return self._data[newest_idx]

    def get_next_after(self, seq: int) -> Optional[Tuple[int, Any]]:
        """Find the first entry with a sequence number greater than `seq`.

        Args:
            seq (int): Sequence number to compare against.

        Returns:
            Optional[Tuple[int, Any]]: The first (sequence, item) pair
            whose sequence is > `seq`, or `None` if no such entry exists.
        """
        with self._lock:
            n = self._count.value
            start = (self._head.value - n) % self._capacity
            for i in range(n):
                entry = self._data[(start + i) % self._capacity]
                if entry is not None and entry[0] > seq:
                    return entry
            return None

    def get_all(self) -> List[Tuple[int, Any]]:
        """Return a snapshot of all entries from oldest to newest.

        Returns:
            List[Tuple[int, Any]]: List of all (sequence, item) pairs
            in order from oldest to newest.
        """
        with self._lock:
            n = self._count.value
            start = (self._head.value - n) % self._capacity
            return [
                self._data[(start + i) % self._capacity]
                for i in range(n)
            ]
