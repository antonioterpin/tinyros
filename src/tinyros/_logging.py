"""Internal logging helper.

Uses :mod:`goggles` when available (keeps parity with the broader ecosystem
scopes / metrics), otherwise falls back to the standard library so that
``pip install tinyros`` does not pull a transitive dependency chain.

Set ``TINYROS_LOG_FORCE_STDLIB=1`` in the environment to bypass goggles even
when it is installed and use the stdlib logger directly. This exists for
tests that capture subprocess stdout/stderr -- goggles emits through an
async event bus whose output does not always survive ``subprocess.PIPE``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import goggles as _gg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in minimal installs
    _gg = None


def _stdlib_forced() -> bool:
    """Honor ``TINYROS_LOG_FORCE_STDLIB`` to bypass goggles when set."""
    return os.environ.get("TINYROS_LOG_FORCE_STDLIB") in ("1", "true", "True")


def get_logger(name: str, *, scope: str | None = None) -> Any:
    """Return a logger bound to ``name`` (scoped if goggles is present).

    Args:
        name: Logger name, used for both goggles and stdlib logging.
        scope: Goggles scope. Ignored when goggles is not installed.

    Returns:
        A logger that exposes ``.info`` / ``.warning`` / ``.error`` /
        ``.debug`` methods.
    """
    if _gg is not None and not _stdlib_forced():
        return _gg.get_logger(name, scope=scope or name)
    return logging.getLogger(name)


def setup_console_logging(level: int = logging.INFO) -> None:
    """Route tinyros log records to the console with sane defaults.

    Goggles drops messages whose scope has no attached handler, so a
    fresh process needs an explicit attach to see anything; the stdlib
    fallback needs ``basicConfig`` for the same reason. This helper
    handles both cases so example code does not need a try/import dance.

    Safe to call multiple times -- goggles deduplicates handlers and
    ``basicConfig`` is a no-op when the root logger already has a
    handler.

    Args:
        level: Minimum severity to emit. Defaults to INFO.
    """
    if _gg is not None and not _stdlib_forced():
        _gg.attach(_gg.ConsoleHandler(level=level), scopes=["tinyros"])
        return
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    )
