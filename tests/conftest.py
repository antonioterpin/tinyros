"""Conftest.py for tests."""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Collection modifier
# ──────────────────────────────────────────────────────────────────────────────
def pytest_collection_modifyitems(
        config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests unless explicitly selected with -m run_explicitly."""
    if config.getoption("-m") and "run_explicitly" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(
        reason="Skipped unless explicitly selected with -m run_explicitly"
    )
    for item in items:
        if "run_explicitly" in item.keywords:
            item.add_marker(skip)
