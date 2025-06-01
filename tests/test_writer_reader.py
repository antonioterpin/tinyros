import time
from multiprocessing import Manager, get_context

import pytest

from tinyros.memory.buffer import ReadersWriterLock

_ctx = get_context("fork")


def reader_task(
    lock: ReadersWriterLock, shared_list, idx: int, n_iters: int, sleep_inside: float
):
    """A reader process that:
     1. Loops n_iters times
     2. Acquires lock.for_read()
     3. Records ("reader-start", idx, timestamp)
     4. Sleeps for sleep_inside seconds (to force overlap)
     5. Records ("reader-end", idx, timestamp)
     6. Releases lock automatically via context manager
    Each process shares the same `shared_list` via a multiprocessing.Manager.list().
    """
    for _ in range(n_iters):
        with lock.for_read():
            shared_list.append(("reader-start", idx, time.time()))
            time.sleep(sleep_inside)
            shared_list.append(("reader-end", idx, time.time()))
        # Small sleep between iterations so that writers/readers can interleave
        time.sleep(0.01)


def writer_task(
    lock: ReadersWriterLock, shared_list, idx: int, n_iters: int, sleep_inside: float
):
    """A writer process that:
    1. Loops n_iters times
    2. Acquires lock.for_write()
    3. Records ("writer-start", idx, timestamp)
    4. Sleeps for sleep_inside seconds
    5. Records ("writer-end", idx, timestamp)
    6. Releases lock automatically via context manager
    """
    for i in range(n_iters):
        with lock.for_write():
            shared_list.append(("writer-start", idx, time.time()))
            time.sleep(sleep_inside)
            shared_list.append(("writer-end", idx, time.time()))
        time.sleep(0.01)


@pytest.mark.parametrize(
    "num_readers, read_iterations, inside_sleep, min_concurrent",
    [
        (2, 10, 0.05, 2),
        (5, 10, 0.02, 2),
        (5, 10, 0.01, 2),
        (100, 10, 0.01, 2),
        (100, 10, 0.001, 2),
    ],
)
def test_multiple_readers_overlap(
    num_readers, read_iterations, inside_sleep, min_concurrent
):
    """Test that multiple readers can overlap in their critical sections.

    Spawn `num_readers` processes, each doing `read_iterations` read-critical sections.
    Each reader sleeps `inside_sleep` while holding the read lock.
    Because read locks allow concurrent readers, we assert that at least at one
    point min_concurrent readers are inside the critical region ("reader-start" recorded)
    before any "reader-end" occurs.
    """
    mgr = Manager()
    shared_list = mgr.list()
    lock = ReadersWriterLock(ctx=_ctx)

    procs = []
    for idx in range(num_readers):
        p = _ctx.Process(
            target=reader_task,
            args=(lock, shared_list, idx, read_iterations, inside_sleep),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=10)
        assert not p.is_alive(), "A reader process hung unexpectedly"

    # Now inspect shared_list for overlap
    # We want to find at least one window where all readers have appended "reader-start"
    # before any single "reader-end".
    events = list(shared_list)
    # events is a list of tuples: ("reader-start"/"reader-end", idx, timestamp)
    # For each iteration (0..read_iterations-1), check that
    # all reader-start for that iteration happen before any reader-end for that iteration.
    # Because each reader writes its own events in sequence, but they may interleave,
    # we only need to see that for at least one iteration, N “reader-start” events appear
    # before ANY “reader-end” event in that same iteration.
    # This is of course not guaranteed to happen in every run, but with enough readers
    # and iterations, it should be statistically likely.

    # Group events by iteration count:
    # since we can’t explicitly mark iteration, we infer by chronological ordering.
    # Instead, we check: at time t*, count how many readers have started
    # without any end. If that hits num_readers at any point, success.
    current_readers = set()
    peak_overlap = 0

    # Sort by timestamp to be sure
    events.sort(key=lambda x: x[2])
    for tag, idx, ts in events:
        if tag == "reader-start":
            current_readers.add(idx)
        elif tag == "reader-end" and idx in current_readers:
            current_readers.remove(idx)
        peak_overlap = max(peak_overlap, len(current_readers))

    assert peak_overlap >= min_concurrent, (
        f"Expected up to {num_readers} readers concurrently, "
        f"but saw at most {peak_overlap}"
    )


@pytest.mark.parametrize(
    "num_readers, num_writers, read_iterations, write_iterations, inside_sleep",
    [
        (3, 1, 10, 10, 0.03),
        (2, 2, 10, 10, 0.05),
        (10, 5, 10, 10, 0.001),
    ],
)
def test_readers_block_writer_until_done(
    num_readers, num_writers, read_iterations, write_iterations, inside_sleep
):
    """Test writes are exclusive and block incoming readers.

    Spawn `num_readers` reader processes and `num_writers` writer processes.
    Each process does its respective critical section for `*_iterations` times,
    sleeping `inside_sleep`.
    We check that:
      1) No "writer-start" timestamp ever occurs while any reader is inside (i.e., between
         a "reader-start" and "reader-end" for that reader).
      2) Similarly, no "reader-start" occurs while a writer is inside (i.e., between
         a "writer-start" and "writer-end").
    In other words, reads can overlap amongst themselves, writes are exclusive, and
    writes block incoming readers until the writer is done.
    """
    mgr = Manager()
    shared_list = mgr.list()
    lock = ReadersWriterLock()

    procs = []
    # Launch readers
    for ridx in range(num_readers):
        p = _ctx.Process(
            target=reader_task,
            args=(lock, shared_list, f"R{ridx}", read_iterations, inside_sleep),
        )
        p.start()
        procs.append(p)

    # Launch writers
    for widx in range(num_writers):
        p = _ctx.Process(
            target=writer_task,
            args=(lock, shared_list, f"W{widx}", write_iterations, inside_sleep),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=10)
        assert not p.is_alive(), "A process hung unexpectedly"

    # Sort events by timestamp
    events = sorted(list(shared_list), key=lambda x: x[2])

    # We will sweep through the timeline, tracking which readers or writers are "active".
    active_readers = set()
    active_writers = set()

    for tag, idx, ts in events:
        if tag == "reader-start":
            # A reader is trying to start; ensure no writer currently active
            assert (
                len(active_writers) == 0
            ), f"{idx} started reading while writer was active"
            active_readers.add(idx)
        elif tag == "reader-end":
            if idx in active_readers:
                active_readers.remove(idx)
        elif tag == "writer-start":
            # A writer is trying to start; ensure no reader or writer currently active
            assert (
                len(active_readers) == 0
            ), f"{idx} started writing while readers were active"
            assert (
                len(active_writers) == 0
            ), f"{idx} started writing while another writer was active"
            active_writers.add(idx)
        elif tag == "writer-end":
            if idx in active_writers:
                active_writers.remove(idx)
