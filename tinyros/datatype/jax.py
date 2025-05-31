"""JAX data type for TinyROS."""

from dataclasses import dataclass
from typing import Any

import cupy as cp
import jax.numpy as jnp
from jax import dlpack

from tinyros.datatype.base import DataType, SerializedCuPyArray


@dataclass
class SerializedJaxArray:
    """Serialized representation of a JAX array."""

    cupy_serialized: SerializedCuPyArray


class JaxDataType(DataType):
    """Data type for JAX arrays in TinyROS."""

    def can_serialize(self, data: Any) -> bool:
        """Check if the JAX data type can be serialized.

        TODO: Extend to pytree support.

        Args:
            data: The data to check for serialization capability.

        Returns:
            bool: True if the data can be serialized, False otherwise.
        """
        return isinstance(data, jnp.ndarray)

    def serialize_data(self, data: jnp.ndarray) -> bytes:
        """Serialize a JAX array into bytes.

        Args:
            data (jnp.ndarray): The JAX array to serialize.

        Returns:
            bytes: Serialized data as bytes.
        """
        return SerializedJaxArray(
            cupy_serialized=self.serialize_cupy(self.jax_to_cupy(data))
        )

    def can_deserialize(self, data: Any) -> bool:
        """Check if the JAX data type can be deserialized.

        Args:
            data: The data to check for deserialization capability.

        Returns:
            bool: True if the data can be deserialized, False otherwise.
        """
        return isinstance(data, SerializedJaxArray)

    def deserialize_data(self, data: SerializedJaxArray) -> jnp.ndarray:
        """Deserialize a JAX array from serialized data.

        Args:
            data (SerializedJaxArray): The serialized JAX array.

        Returns:
            jnp.ndarray: The deserialized JAX array.
        """
        return self.cupy_to_jax(self.deserialize_cupy(data.cupy_serialized))

    def cupy_to_jax(self, cp_arr: cp.ndarray) -> jnp.ndarray:
        """Convert a CuPy array to a JAX array.

        This is at the moment the bottleneck, needs to be optimized.
        """
        return jnp.from_dlpack(cp_arr, copy=False)

    def jax_to_cupy(self, jax_arr: jnp.ndarray) -> cp.ndarray:
        """Convert a JAX array to a CuPy array with zero-copy.

        Args:
            jax_arr (jnp.ndarray): The JAX array to convert.

        Returns:
            cp.ndarray: The converted CuPy array.
        """
        return cp.from_dlpack(dlpack.to_dlpack(jax_arr))
