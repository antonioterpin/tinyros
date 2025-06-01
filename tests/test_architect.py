import pytest
import cupy as cp
import time
import os
from multiprocessing import get_context
import ctypes

from tinyros.datatype import TinyROSMessageFieldDefinition
from tinyros.core import Node, Architect

TOPIC = "data"


class SimplePublisherNode(Node):
    def __init__(self, frequency, dim, dtype):
        super().__init__(
            name="PublisherNode",
            subscriptions=[],
            publishers={
                TOPIC: (
                    [
                        TinyROSMessageFieldDefinition(
                            name=TOPIC, field_type=cp.ndarray, shape=(dim,), dtype=dtype
                        )
                    ],
                    1,
                )
            },
            spin_frequency=frequency,
        )
        self.arr = cp.zeros(dim, dtype=dtype)
        _ctx = get_context(os.getenv("MULTIPROCESSING_CONTEXT", "spawn"))
        self.i = _ctx.RawValue(ctypes.c_int32, -1)

    def loop(self):
        self.arr.fill(time.time())
        self.i.value += 1
        self.publish(topic=TOPIC, data=self.arr)

    def cleanup(self):
        pass


class SimpleConsumerNode(Node):
    def __init__(self, frequency):
        super().__init__(
            name="ConsumerNode",
            subscriptions=[TOPIC],
            publishers={},
            spin_frequency=10 * frequency,
        )
        self.last_seq = 0
        _ctx = get_context(os.getenv("MULTIPROCESSING_CONTEXT", "spawn"))
        self.total_time = _ctx.RawValue(ctypes.c_float, 0)
        self.n = _ctx.RawValue(ctypes.c_int32, 0)
        self.first = True

    def loop(self):
        ret = self.listen(TOPIC, seq=self.last_seq, timeout=2, latest=True)
        t1 = time.time()
        if ret is None:
            return
        if self.first:
            self.first = False
            self.last_seq = ret[0]
            return
        self.last_seq, data = ret
        delta = t1 - data["data"][0]
        self.total_time.value += delta
        self.n.value += 1

    def cleanup(self):
        return


@pytest.mark.parametrize("frequency", [50, 100])
@pytest.mark.parametrize(
    "dim,threshold",
    [(100_000, 0.45), (1_000_000, 0.45), (10_000_000, 0.7), (100_000_000, 3.0)],
)
def test_client_server(frequency, dim, threshold):
    publisher_node = SimplePublisherNode(frequency, dim, cp.float64)
    consumer_node = SimpleConsumerNode(frequency)

    # Use Architect to connect and run both nodes
    with Architect(consumer_node, publisher_node):
        # Let them run for a bit
        T = min(10, 1000 / frequency)
        time.sleep(T)

    # At this point, __exit__ has already requested both nodes to stop

    n = consumer_node.n.value
    i = publisher_node.i.value
    total_time = consumer_node.total_time.value

    # Perform the same assertions as before
    assert (
        n >= (i - 1) * 0.9 and i >= frequency * T * 0.9
    ), f"Consumer received {n} messages, but publisher sent {i} messages."

    average_time_ms = total_time / n * 1e3
    assert (
        average_time_ms < threshold
    ), f"Average time {average_time_ms:.3f} exceeds threshold {threshold:.3f}."
