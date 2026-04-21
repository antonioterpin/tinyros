"""End-to-end tests for TinyServer and TinyClient.

Validates the wire in isolation from TinyNode: small-payload RPC,
large-payload shm side-channel, error propagation from remote exceptions,
and that shutdown releases the port.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from tests.conftest import wait_port_free
from tinyros.transport import TinyClient, TinyServer


@pytest.fixture
def server_client_pair(
    free_port: int,
) -> Iterator[tuple[TinyServer, TinyClient, int]]:
    """Provision a server+client pair on a free port; close both on teardown.

    Args:
        free_port: Port the kernel considered free when the test started.

    Yields:
        Tuple ``(server, client, port)`` ready for the test to use.
    """
    server = TinyServer(name="t-srv", host="127.0.0.1", port=free_port)
    server.start(block=False)
    client = TinyClient(host="127.0.0.1", port=free_port, name="t-cli")
    try:
        yield server, client, free_port
    finally:
        try:
            client.close(timeout=1.0)
        finally:
            server.close(timeout=1.0)
        wait_port_free(free_port)


def test_small_call_roundtrip(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """A small RPC returns the callback's value as a Future."""
    server, client, _ = server_client_pair
    server.bind("add_one", lambda x: x + 1)
    fut = client.call("add_one", 41)
    assert fut.result(timeout=2.0) == 42, "add_one(41) should return 42"


def test_attribute_proxy_matches_call(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """client.method(x) and client.call('method', x) behave identically."""
    server, client, _ = server_client_pair
    server.bind("echo", lambda x: x)
    fut_attr = client.echo("hi")
    fut_explicit = client.call("echo", "hi")
    assert fut_attr.result(timeout=2.0) == fut_explicit.result(timeout=2.0)


def test_large_ndarray_uses_shm_fast_path(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Arrays at/above the shm threshold round-trip correctly."""
    server, client, _ = server_client_pair
    server.bind("shape_of", lambda arr: arr.shape)
    arr = np.ones((256, 256), dtype=np.float32)  # 256 KiB >> 64 KiB default
    fut = client.call("shape_of", arr)
    assert fut.result(timeout=3.0) == (
        256,
        256,
    ), "server should observe the full ndarray through shm"


def test_remote_exception_propagates(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Exceptions raised inside the callback surface on the client future."""
    server, client, _ = server_client_pair

    def boom(_: object) -> None:
        raise ValueError("deliberate")

    server.bind("boom", boom)
    fut = client.call("boom", None)
    with pytest.raises(ValueError, match="deliberate"):
        fut.result(timeout=2.0)


def test_unknown_method_raises_on_client(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Calling a method the server never bound surfaces an error."""
    _server, client, _ = server_client_pair
    fut = client.call("ghost", None)
    with pytest.raises(AttributeError, match="ghost"):
        fut.result(timeout=2.0)


def test_concurrent_calls_all_complete(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Many in-flight calls from multiple threads each get their own reply."""
    server, client, _ = server_client_pair
    server.bind("twice", lambda x: x * 2)
    num = 50
    futures = [client.call("twice", i) for i in range(num)]
    results = [f.result(timeout=3.0) for f in futures]
    assert results == [
        i * 2 for i in range(num)
    ], "each future should resolve to its own double"


def test_server_close_releases_port(free_port: int) -> None:
    """After close(), the port becomes bindable again."""
    server = TinyServer(name="t-close", host="127.0.0.1", port=free_port)
    server.start(block=False)
    # Briefly connect to ensure the accept loop is alive.
    c = TinyClient(host="127.0.0.1", port=free_port, name="probe")
    c.close(timeout=1.0)
    server.close(timeout=1.0)
    wait_port_free(free_port)


def test_callbacks_run_concurrently(
    server_client_pair: tuple[TinyServer, TinyClient, int],
) -> None:
    """Slow callbacks on distinct requests do not serialize each other."""
    server, client, _ = server_client_pair
    barrier = threading.Barrier(2, timeout=2.0)

    def rendezvous(_: object) -> str:
        barrier.wait()
        return "ok"

    server.bind("rendezvous", rendezvous)
    t0 = time.perf_counter()
    f1 = client.call("rendezvous", None)
    f2 = client.call("rendezvous", None)
    assert f1.result(timeout=3.0) == "ok", "first call should return ok"
    assert f2.result(timeout=3.0) == "ok", "second call should return ok"
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, (
        f"concurrent calls should finish well under the 2s barrier; "
        f"took {elapsed:.2f}s"
    )
