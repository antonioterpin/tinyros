import pytest
import cupy as cp
import time
import multiprocessing as mp

from tinyros.datatype.sharray import CupySharray

N_ITERS = 1000


def producer_cp(
    num_elements, dtype, barrier, update_event, ack_event, sharr_queue, time_queue
):
    # Pre-warm the CUDA context in this process
    arr = cp.arange(num_elements, dtype=dtype)
    arr.fill(0)

    # Send Sharray to consumer
    sharr_queue.put(CupySharray.from_array(arr))

    ack_event.clear()
    update_event.clear()
    barrier.wait()

    start = time.perf_counter()
    for i in range(N_ITERS):
        arr.fill(i)
        cp.cuda.Stream.null.synchronize()
        # Signal the consumer that new data is ready
        update_event.set()

        # Wait until the consumer reads this iteration
        ack_event.wait()
        ack_event.clear()
    end = time.perf_counter()

    time_queue.put(end - start)


def consumer_cp(barrier, update_event, ack_event, sharr_queue, sums_queue, time_queue):
    # Receive Sharray
    sharr = sharr_queue.get()
    arr = sharr.open()
    barrier.wait()

    total = 0
    t0 = time.perf_counter()
    for i in range(N_ITERS):
        update_event.wait()
        update_event.clear()
        total += int(arr[0])
        ack_event.set()
    t1 = time.perf_counter()

    sums_queue.put(total)
    elapsed = t1 - t0
    time_queue.put(elapsed)


@pytest.mark.parametrize(
    "num_elements,dtype,threshold_ms",
    [
        (100, cp.int32, 0.1),
        (100_000, cp.int32, 0.1),
        (100_000, cp.float32, 0.1),
        (100_000, cp.float64, 0.1),
        (1_000_000, cp.int32, 0.11),
        (1_000_000, cp.float32, 0.11),
        (1_000_000, cp.float64, 0.11),
        (10_000_000, cp.int32, 0.18),
        (10_000_000, cp.float32, 0.18),
        (10_000_000, cp.float64, 0.26),
        (100_000_000, cp.int32, 0.95),
        (100_000_000, cp.float32, 0.95),
        (100_000_000, cp.float64, 2.0),  # TODO: Can also this be improved?
    ],
)
@pytest.mark.parametrize("spawn_type", ["fork", "spawn"])
def test_cupy_sharray_ipc_performance_and_correctness(
    num_elements, dtype, threshold_ms, spawn_type
):
    """IPC test for CupySharray via two processes."""
    ctx = mp.get_context(spawn_type)
    barrier = ctx.Barrier(2)
    update_event = ctx.Event()
    ack_event = ctx.Event()
    sharr_queue = ctx.Queue()
    time_producer_queue = ctx.Queue()
    time_consumer_queue = ctx.Queue()
    sums_queue = ctx.Queue()

    cons = ctx.Process(
        target=consumer_cp,
        args=(
            barrier,
            update_event,
            ack_event,
            sharr_queue,
            sums_queue,
            time_consumer_queue,
        ),
    )
    prod = ctx.Process(
        target=producer_cp,
        args=(
            num_elements,
            dtype,
            barrier,
            update_event,
            ack_event,
            sharr_queue,
            time_producer_queue,
        ),
    )

    cons.start()
    prod.start()

    elapsed_1 = time_producer_queue.get()
    elapsed_2 = time_consumer_queue.get()
    elapsed = elapsed_1 + elapsed_2
    total_sum = sums_queue.get()

    prod.join(timeout=10)
    cons.join(timeout=10)
    assert not prod.is_alive(), "Cupy producer did not finish"
    assert not cons.is_alive(), "Cupy consumer did not finish"

    expected_total = N_ITERS * (N_ITERS - 1) // 2
    assert total_sum == expected_total

    avg_ms = (elapsed / N_ITERS) * 1e3
    assert (
        avg_ms < threshold_ms
    ), f"Average Cupy round-trip {avg_ms:.3f}ms exceeds {threshold_ms}ms"


def test_cupy_copy_to_correctness():
    """Test that CuPySharray.copy_to correctly copies data between CuPy arrays."""
    a = cp.arange(100, dtype=cp.int32)
    b = cp.zeros_like(a)
    CupySharray.copy_to(a, b)
    assert cp.array_equal(a, b)

    a2 = cp.arange(64, dtype=cp.float32).reshape(8, 8)
    b2 = cp.zeros_like(a2)
    CupySharray.copy_to(a2, b2)
    assert cp.array_equal(a2, b2)


@pytest.mark.parametrize(
    "dim,threshold",
    [(100_000, 0.008), (1_000_000, 0.011), (10_000_000, 0.08), (100_000_000, 0.9)],
)
def test_cupy_copy_to_speed(dim, threshold):
    a = cp.arange(dim, dtype=cp.float32)
    b = cp.zeros_like(a)
    start = time.perf_counter()
    for _ in range(1000):
        CupySharray.copy_to(a, b)
        cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()
    elapsed = end - start  # ms average
    assert elapsed < threshold, (
        f"CupySharray.copy_to took too long: {elapsed:.6f} ms on average."
        f"Threshold is {threshold} ms."
    )
