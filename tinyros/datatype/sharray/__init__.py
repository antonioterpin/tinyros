"""Shared array module for TinyROS."""

from .base import Sharray
from .cupy import CupySharray

__all__ = [
    "Sharray",
    "CupySharray",
]

# Try to import JaxSharray if jax is installed
try:
    from .jax import JaxSharray  # noqa: F401

    __all__.append("JaxSharray")
except ImportError:
    pass
