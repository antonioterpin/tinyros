import pytest
import time
from multiprocessing import Process, Manager
from tinyros.buffer import Buffer


@pytest.mark.parametrize("capacity", [0, -1, 2.5])
def test_init_invalid(capacity):
    with pytest.raises(ValueError):
        Buffer(capacity)


@pytest.mark.parametrize(
    "capacity, items",
    [
        (3, ["a", "b", "c"]),
        (5, [1, 2, 3, 4, 5]),
        (4, [None, True, False, "x"]),
    ],
)
def test_put_and_get_single_thread(capacity, items):
    buf = Buffer(capacity)
    seqs = []
    # put items
    for item in items:
        buf.put(item)
        seqs.append(buf.get_newest()[0])

    # sequences strictly increasing
    assert seqs == sorted(seqs)
    # newest is last item
    assert buf.get_newest()[1] == items[-1]
    # oldest is first item (no wrap yet)
    assert buf.get_oldest()[1] == items[0]
    # get_all returns all in order
    assert [it for _, it in buf.get_all()] == items
    # get_next_after for each
    for i, seq in enumerate(seqs[:-1]):
        nxt = buf.get_next_after(seq)
        assert nxt[1] == items[i+1]
    # after last seq, no next
    assert buf.get_next_after(seqs[-1]) is None


def test_overwrite_behavior():
    buf = Buffer(3)
    for i in range(1, 7):
        buf.put(i)
    # we've written 1..6 into capacity=3 buffer => should hold [4,5,6]
    all_items = buf.get_all()
    assert [it for _, it in all_items] == [4, 5, 6]
    # oldest seq should be 4, newest 6
    assert buf.get_oldest()[0] == all_items[0][0] == 4
    assert buf.get_newest()[0] == all_items[-1][0] == 6
    # next after 4 is 5, after 6 is None
    assert buf.get_next_after(4)[1] == 5
    assert buf.get_next_after(6) is None


def test_concurrent_read_during_write():
    capacity = 10
    total = 50
    buf = Buffer(capacity)
    mgr = Manager()
    read_lengths = mgr.list()

    def writer():
        for i in range(total):
            buf.put(i)
            time.sleep(0.002)

    def reader():
        # sample the buffer frequently
        for _ in range(total * 2):
            snapshot = buf.get_all()
            read_lengths.append(len(snapshot))
            time.sleep(0.001)

    p_writer = Process(target=writer)
    p_reader = Process(target=reader)
    p_writer.start()
    p_reader.start()
    p_writer.join()
    p_reader.join()

    # every observed length must be between 0 and capacity
    assert all(0 <= ln <= capacity for ln in read_lengths)
    # eventually should see a full buffer
    assert capacity in read_lengths
