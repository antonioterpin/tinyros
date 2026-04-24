"""Internal logging helper.

Uses :mod:`goggles` when available (keeps parity with the broader ecosystem
scopes / metrics), otherwise falls back to the standard library so that
``pip install tinyros`` does not pull a transitive dependency chain.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import goggles as _gg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in minimal installs
    _gg = None


def get_logger(name: str, *, scope: str | None = None) -> Any:
    """Return a logger bound to ``name`` (scoped if goggles is present).

    Args:
        name: Logger name, used for both goggles and stdlib logging.
        scope: Goggles scope. Ignored when goggles is not installed.

    Returns:
        A logger that exposes ``.info`` / ``.warning`` / ``.error`` /
        ``.debug`` methods.
    """
    if _gg is not None:
        return _gg.get_logger(name, scope=scope or name)
    return logging.getLogger(name)
