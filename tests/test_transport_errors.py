"""Tests for the public transport exception hierarchy.

Confirms the inheritance contract callers rely on -- in particular
that pre-hierarchy code using ``except ConnectionError`` still
catches the new typed :class:`ConnectionLost`.
"""

from __future__ import annotations

import pytest

from tinyros import (
    ConnectionLost,
    SerializationError,
    TransportError,
)
from tinyros.transport._errors import (
    ConnectionLost as _ConnectionLost,
)
from tinyros.transport._errors import (
    SerializationError as _SerializationError,
)
from tinyros.transport._errors import (
    TransportError as _TransportError,
)


def test_top_level_reexports_match_module_classes() -> None:
    """Public re-exports are the same objects as the module-level classes."""
    assert TransportError is _TransportError
    assert ConnectionLost is _ConnectionLost
    assert SerializationError is _SerializationError


def test_connection_lost_is_subclass_of_transport_error() -> None:
    """``except TransportError`` covers every drop the hierarchy emits."""
    assert issubclass(ConnectionLost, TransportError)


def test_serialization_error_is_subclass_of_transport_error() -> None:
    """SerializationError participates in the unified hierarchy."""
    assert issubclass(SerializationError, TransportError)


def test_connection_lost_is_subclass_of_builtin_connection_error() -> None:
    """Pre-hierarchy ``except ConnectionError`` callers keep working."""
    assert issubclass(ConnectionLost, ConnectionError)


def test_connection_lost_is_caught_by_except_connection_error() -> None:
    """Demonstrate the backward-compat behavior end to end."""
    with pytest.raises(ConnectionError):
        raise ConnectionLost("peer gone")


def test_serialization_error_is_not_a_connection_error() -> None:
    """The two failure modes are disjoint at the catch-site."""
    assert not issubclass(SerializationError, ConnectionError)


def test_transport_error_is_an_exception() -> None:
    """TransportError sits under Exception, not BaseException."""
    assert issubclass(TransportError, Exception)
