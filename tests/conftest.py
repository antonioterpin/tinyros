"""Shared pytest fixtures and collection configuration for tinyros tests."""

from __future__ import annotations

import socket
import time

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``run_explicitly`` tests unless selected with ``-m run_explicitly``.

    Args:
        config: Active pytest configuration.
        items: Collected test items (mutated in place).
    """
    if config.getoption("-m") and "run_explicitly" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(
        reason="Skipped unless explicitly selected with -m run_explicitly"
    )
    for item in items:
        if "run_explicitly" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def free_port() -> int:
    """Bind and release a loopback port to pick one the kernel considers free.

    Returns:
        A port number that was free at the moment the fixture was evaluated.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether ``port`` currently accepts connections.

    Args:
        port: TCP port to probe.
        host: Interface to probe on.

    Returns:
        True if connecting to ``(host, port)`` fails (port is free).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.05)
        return sock.connect_ex((host, port)) != 0


def wait_port_free(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> None:
    """Block until ``(host, port)`` stops accepting connections.

    Args:
        port: TCP port to probe.
        host: Interface to probe on.
        timeout: Max seconds to wait before raising.

    Raises:
        AssertionError: If the port is still open past ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_free(port, host):
            return
        time.sleep(0.01)
    raise AssertionError(f"port {port} on {host} still open after {timeout:.1f}s")
