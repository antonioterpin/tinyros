import pytest
import cupy as cp
import time
import os
from multiprocessing import get_context
import ctypes

from tinyros.datatype import TinyROSMessageFieldDefinition
from tinyros.core import Node
from tinyros.utils import logger

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
        # we discard the first message for warmup
        # self.last_time = time.time()

    def loop(self):
        # t = time.time()
        # print(f"Expected frequency: {self._spin_frequency} Hz, "
        #       f"Actual frequency: {1 / (t - self.last_time):.2f} Hz")
        # self.last_time = t

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


# TODO: To improve the 100_000_000 performances, we need to skip the copies completely.
@pytest.mark.parametrize("frequency", [50, 100])
@pytest.mark.parametrize(
    "dim,threshold",
    [(100_000, 0.45), (1_000_000, 0.45), (10_000_000, 0.7), (100_000_000, 3.0)],
)
def test_client_server(frequency, dim, threshold):
    # Create nodes
    publisher_node = SimplePublisherNode(frequency, dim, cp.float64)
    consumer_node = SimpleConsumerNode(frequency)

    # Connect nodes
    for topic in publisher_node._publishers:
        if topic in consumer_node._requested_subscriptions:
            logger.info(f"Connecting {topic} from publisher to consumer")
            consumer_node.connect_subscriber(
                topic, publisher_node.get_connection_to_publisher(topic)
            )
    for topic in consumer_node._publishers:
        if topic in publisher_node._requested_subscriptions:
            logger.info(f"Connecting {topic} from consumer to publisher")
            publisher_node.connect_subscriber(
                topic, consumer_node._publishers[topic].buffer
            )

    # # Start nodes
    consumer_node.start()
    publisher_node.start()

    # Run for a while
    T = min(10, 1000 / frequency)
    time.sleep(T)

    # Stop nodes
    publisher_node.request_stop()
    consumer_node.request_stop()

    n = consumer_node.n.value
    i = publisher_node.i.value
    total_time = consumer_node.total_time.value

    del publisher_node
    del consumer_node

    assert n >= (i - 1) * 0.95 and i >= (
        frequency * T * 0.95
    ), f"Consumer received {n} messages, but publisher sent {i} messages."

    average_time_ms = total_time / n * 1e3
    assert (
        average_time_ms < threshold
    ), f"Average time {average_time_ms:.3f} exceeds threshold {threshold:.3f}."
