import time
from multiprocessing import Manager, get_context

import cupy as cp
import numpy as np
import pytest

from tinyros.buffer import Buffer
from tinyros.datatype import DataType

ctx = get_context("spawn")


def writer(b, frequency, N):
    new_obj = DataType()
    p = cp.arange(100_000_000, dtype=cp.float32)
    setattr(new_obj, "arr", p)
    for _ in range(N):
        t0 = time.perf_counter()
        setattr(new_obj, "t0", t0)
        b.put(new_obj.serialize())
        time.sleep(1 / frequency)


def reader(b, latencies, N):
    seq = 0
    _latencies = [0.0] * N
    for i in range(N):
        el = b.get_newest(seq, timeout=2)
        if el is None:
            break
        seq, data = el
        data = DataType().deserialize(data)
        t1 = time.perf_counter()
        t0 = getattr(data, "t0")
        _latencies[i] = (t1 - t0) * 1000
    for latency in _latencies:
        if latency is not None:
            latencies.append(latency)


@pytest.mark.parametrize("frequency", [100, 1000])
def test_buffer_roundtrip_cross_process(frequency):
    obj = DataType()
    p = cp.arange(100_000_000, dtype=cp.float32)
    setattr(obj, "arr", p)
    setattr(obj, "t0", time.perf_counter())

    serialized_obj = obj.serialize()
    buf = Buffer(capacity=10, slot_size=len(serialized_obj), ctx=ctx)
    latencies = Manager().list()

    N = 1000

    pr = ctx.Process(target=reader, args=(buf, latencies, N))
    pw = ctx.Process(target=writer, args=(buf, frequency, N))

    pr.start()
    pw.start()

    pw.join(timeout=N / frequency * 5)
    pr.join(timeout=N / frequency * 5)

    latency = np.mean(latencies)
    threshold = 0.5
    assert latency < threshold, (
        f"Average latency {latency:.3f} ms exceeds threshold {threshold:.3f} "
        f"for frequency {frequency} Hz"
    )
    assert len(latencies) == N, f"Expected {N} items read, but got {len(latencies)}"
