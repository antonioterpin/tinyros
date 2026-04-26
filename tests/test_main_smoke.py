"""End-to-end smoke test for the example application in ``main.py``.

The example is the most user-visible artifact in the repo. This test
launches it as a subprocess, verifies that all four nodes start and
that a cross-node round-trip occurs (FeedbackProcessor reacts to a
ControlProcessor publish), then sends SIGTERM and asserts a clean
shutdown.

Skipped on Windows (POSIX-style signal semantics assumed) and when
the configured demo ports are already in use.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PY = _REPO_ROOT / "main.py"
_CONFIG_YAML = _REPO_ROOT / "network_config.yaml"

# How long we let the example run before sending SIGTERM. The cycle
# requires ~4--6s to observe a full round-trip (publishers tick at
# 0.5--1 Hz; the second tick of the control loop is what proves
# bidirectional flow).
_RUN_SECONDS = 8.0
# Grace window for the parent + four children to exit after SIGTERM.
_SHUTDOWN_TIMEOUT = 6.0


def _ports_from_config() -> list[int]:
    """Return the list of TCP ports the example will try to bind."""
    with _CONFIG_YAML.open() as fh:
        cfg = yaml.safe_load(fh)
    return [int(node["port"]) for node in cfg["nodes"].values()]


def _ports_free(ports: list[int]) -> bool:
    """Return True if every port is currently free on loopback."""
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.05)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return False
    return True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX signal semantics assumed",
)
def test_main_runs_and_shuts_down_cleanly() -> None:
    """Run ``main.py``; assert round-trip, SIGTERM, clean exit."""
    ports = _ports_from_config()
    if not _ports_free(ports):
        pytest.skip(f"demo ports {ports} not free on this host")

    env = os.environ.copy()
    # Force unbuffered stdio so the parent sees logs in real time.
    env["PYTHONUNBUFFERED"] = "1"
    # Bypass goggles for this subprocess: its async event bus does not
    # route output through stdout/stderr in a way subprocess.PIPE can
    # reliably capture, especially across mp-spawn children. The
    # stdlib fallback writes synchronously and shows up in the pipe.
    env["TINYROS_LOG_FORCE_STDLIB"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(_MAIN_PY)],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # New process group so SIGTERM reaches all children at once.
        start_new_session=True,
    )

    try:
        time.sleep(_RUN_SECONDS)
        # Send SIGTERM to the whole group so the four mp children get it
        # at the same time as the parent.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, _ = proc.communicate(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, _ = proc.communicate(timeout=2.0)
            pytest.fail(
                "main.py did not exit within "
                f"{_SHUTDOWN_TIMEOUT}s after SIGTERM:\n{stdout}"
            )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2.0)

    # All four nodes must have started.
    for marker in (
        "starting ScalarPublisher",
        "starting ImagePublisher",
        "starting ControlProcessor",
        "starting FeedbackProcessor",
    ):
        assert (
            marker in stdout
        ), f"missing startup marker {marker!r} in main.py output:\n{stdout}"

    # Cross-node round-trip: FeedbackProcessor's callback fires only
    # after ControlProcessor has published an actuation command, which
    # itself depends on subscriptions being wired and the first tick
    # having landed. Seeing this line is the strongest single signal
    # that the example "works".
    assert "feedback =" in stdout, (
        "no FeedbackProcessor reaction observed -- the round-trip "
        f"between ControlProcessor and FeedbackProcessor did not "
        f"complete within {_RUN_SECONDS}s.\n{stdout}"
    )
