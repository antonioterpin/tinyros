"""Data types for TinyROS."""

from tinyros.datatype.base import DataType, SerializedCuPyArray

__all__ = [
    "DataType",
    "SerializedCuPyArray",
]

try:
    from tinyros.datatype.jax import JaxDataType, SerializedJaxArray  # noqa: F401

    __all__.append("JaxDataType")
    __all__.append("SerializedJaxArray")
except ImportError:
    pass
