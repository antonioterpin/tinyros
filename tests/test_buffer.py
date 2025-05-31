import pickle
import time
from multiprocessing import Manager, Process

import pytest

from tinyros.buffer import Buffer


def serialize(item):
    """A simple serialization function for testing."""
    return pickle.dumps(item)


def deserialize(serialized_item):
    """A simple deserialization function for testing."""
    return pickle.loads(serialized_item)


@pytest.mark.parametrize("capacity", [0, -1, 2.5])
def test_init_invalid(capacity):
    with pytest.raises(ValueError):
        Buffer(capacity, 1)


@pytest.mark.parametrize("size", [0, -1, 2.5])
def test_init_invalid_size(size):
    with pytest.raises(ValueError):
        Buffer(1, size)


@pytest.mark.parametrize(
    "capacity, items",
    [
        (3, ["a", "b", "c"]),
        (5, [1, 2, 3, 4, 5]),
        (2, [{"arr": [1, 2, 3]}, {"arr": [4, 5, 6]}]),
    ],
)
def test_put_and_get_single_thread(capacity, items):
    size = len(serialize(items[0]))
    buf = Buffer(capacity, size)
    seqs = []

    # insert items and collect their sequence numbers
    last_t = 0
    for item in items:
        buf.put(serialize(item))
        seq, _ = buf.get_newest(t=last_t, timeout=1)
        seqs.append(seq)
        last_t = seq

    # strictly increasing
    assert seqs == sorted(seqs)

    # newest is the last we inserted
    _, newest_item = buf.get_newest(t=0, timeout=1)
    newest_item = deserialize(newest_item)
    assert newest_item == items[-1]

    # oldest is the first
    _, oldest = buf.get_next_after(t=0, timeout=1)
    oldest = deserialize(oldest)
    assert oldest == items[0]

    # reconstruct “all” in order using get_next_after
    all_items = []
    t = 0
    while True:
        nxt = buf.get_next_after(t, timeout=1)
        if nxt is None:
            break
        seq, itm = nxt
        all_items.append(deserialize(itm))
        t = seq
    assert all_items == items

    # get_next_after each intermediate seq
    for i, seq in enumerate(seqs[:-1]):
        seq, nxt = buf.get_next_after(seq, timeout=1)
        nxt = deserialize(nxt)
        assert nxt == items[i + 1]

    # beyond last seq, there’s no next
    assert buf.get_next_after(seqs[-1], timeout=1) is None


def test_overwrite_behavior():
    size = len(serialize(0))
    buf = Buffer(3, size)
    for i in range(1, 7):
        buf.put(serialize(i))

    # We expect the buffer to hold only [4,5,6]
    # Reconstruct via get_next_after
    collected = []
    t = 0
    while True:
        nxt = buf.get_next_after(t, timeout=1)
        if nxt is None:
            break
        collected.append(deserialize(nxt[1]))
        t = nxt[0]

    assert collected == [4, 5, 6]

    # oldest seq is 4, newest is 6
    assert collected[0] == 4
    last_seq, _ = buf.get_newest(0, timeout=1)
    assert last_seq == 6

    # next after 4 is 5, after 6 is None
    assert deserialize(buf.get_next_after(4, timeout=1)[1]) == 5
    assert buf.get_next_after(6, timeout=1) is None


def test_concurrent_read_during_write():
    capacity = 10
    total = 50
    size = len(serialize(0))
    buf = Buffer(capacity, size)
    mgr = Manager()
    read_counts = mgr.list()

    def writer():
        for i in range(total):
            buf.put(serialize(i))
            time.sleep(0.002)

    def reader():
        t = 0
        for _ in range(total * 5):
            nxt = buf.get_next_after(t, timeout=1)
            if nxt is None:
                break
            t = nxt[0]
            read_counts.append(t)
            time.sleep(0.001)

    pw = Process(target=writer)
    pr = Process(target=reader)
    pw.start()
    pr.start()
    pw.join()
    pr.join()

    assert [i for i in read_counts] == [i + 1 for i in range(total)]


@pytest.mark.parametrize("frequency", [10, 50, 100, 1000])
def test_transfer_latency(frequency):
    size = len(serialize(time.perf_counter()))
    buf = Buffer(10, size)
    mgr = Manager()
    latencies = mgr.list()
    N = 100

    def writer():
        # stamp the current wall‐clock time into the item
        for i in range(N):
            buf.put(serialize(time.perf_counter()))
            time.sleep(1 / frequency)

    def reader():
        seq = 0
        for _ in range(N):
            # wait for an item to be available
            entry = buf.get_newest(t=seq, timeout=2)
            if entry is None:
                raise RuntimeError("No item available in time")
            t1 = time.perf_counter()
            (seq, t0) = entry
            latencies.append((t1 - deserialize(t0)) * 1000)

    pw = Process(target=writer)
    pr = Process(target=reader)
    pw.start()
    pr.start()
    pw.join()
    pr.join()

    # calculate the average latency
    latency = sum(latencies) / len(latencies)

    # calculate the expected latency of the serialization
    expected_serialization_latency = 0
    for _ in range(N):
        t0 = time.perf_counter()
        serialize(time.perf_counter())
        expected_serialization_latency += (time.perf_counter() - t0) * 1000
    expected_serialization_latency /= N
    latency -= expected_serialization_latency

    assert latency < 0.12, f"Data transfer took too long: {latency:.3f}ms"
