import pytest
from multiprocessing import get_context

from tinyros.core import Client, Server
from tinyros.datatype import DataType
from tinyros.buffer import Buffer, BufferClosedError

ctx = get_context("spawn")

def test_server_make_and_publish_puts_data_into_buffer():
    """Verify Server.make and publish put serialized objects into the buffer."""
    dt0 = DataType()
    dt0.x = 0

    server = Server.make(3, dt0)
    buf = server.buffer
    assert isinstance(buf, Buffer)

    for i in range(3):
        dti = DataType()
        dti.x = i + 100
        server.publish(dti)

    seq = 0
    received = []
    for _ in range(3):
        el = buf.get_next_after(seq, timeout=0.1)
        assert el is not None, "Data expeceted in buffer"
        seq, raw = el
        obj = DataType().deserialize(raw)
        received.append(obj.x)

    assert received == [100, 101, 102]


def test_server_publish_then_close_and_buffer_no_longer_accepts():
    """Ensure that after close, buffer stops returning new data."""
    dt0 = DataType()
    dt0.x = 0
    server = Server.make(2, dt0)
    buf = server.buffer

    d1 = DataType()
    d1.x = 42
    server.publish(d1)

    el = buf.get_newest(0, timeout=1)

    if el is not None:
        _, data = el
        obj = DataType().deserialize(data)
        assert obj.x == 42

    server.publish(d1)
    server.close()

    with pytest.raises(BufferClosedError):
        _ = buf.get_newest(0, timeout=0.05)

    with pytest.raises(BufferClosedError):
        server.publish(d1)

def test_client_make_and_try_get_returns_deserialized_data():
    """Verify Client.make and try_get returns deserialized data."""
    dt0 = DataType()
    dt0.x = 0

    server = Server.make(3, dt0)
    buf = server.buffer
    client = Client.make(buf, dt0)

    for i in range(3):
        dti = DataType()
        dti.x = i + 100
        server.publish(dti)

    seq = 0
    received = []
    for _ in range(3):
        ret = client.try_get(seq, timeout=0.1, latest=False)
        assert ret is not None, "Data expected from client"
        seq, obj = ret
        received.append(obj.x)

    assert received == [100, 101, 102]

    obj = client.try_get(0, timeout=0.1, latest=True)
    assert obj is not None, "Data expected from client"
    obj = client.try_get(obj[0], timeout=0.1, latest=True)
    assert obj is None, "No more data expected from client"

    received = []
    for _ in range(3):
        ret = client.try_get(0, timeout=0.1, latest=True)
        assert ret is not None, "Data expected from client"
        seq, obj = ret
        received.append(obj.x)
    assert received == [102, 102, 102], "Expected latest data to be the same"

def test_client_try_get_after_server_close_raises_error():
    """Ensure that after server close, client cannot get data."""
    dt0 = DataType()
    dt0.x = 0

    server = Server.make(3, dt0)
    buf = server.buffer
    client = Client.make(buf, dt0)

    for i in range(3):
        dti = DataType()
        dti.x = i + 100
        server.publish(dti)

    server.close()

    with pytest.raises(BufferClosedError):
        _ = client.try_get(0, timeout=0.1, latest=False)

    with pytest.raises(BufferClosedError):
        _ = client.try_get(0, timeout=0.1, latest=True)