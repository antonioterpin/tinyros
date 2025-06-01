"""Shared array module for TinyROS."""

from tinyros.datatype.sharray.base import Sharray
from tinyros.datatype.sharray.cupy import CupySharray
from tinyros.datatype.sharray.string import StringSharray
from tinyros.datatype.sharray.numpy import NumpySharray

__all__ = [
    "Sharray",
    "CupySharray",
    "StringSharray",
    "NumpySharray",
]

# Try to import JaxSharray if jax is installed
try:
    from tinyros.datatype.sharray.jax import JaxSharray
    __all__.append("JaxSharray")
except ImportError:
    pass
