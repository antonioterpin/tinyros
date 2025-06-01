"""JAX shared array implementation."""

import jax.numpy as jnp
import cupy as cp
from jax import dlpack
from dataclasses import dataclass

from tinyros.datatype.sharray.cupy import CupySharray


def cupy_to_jax(cp_arr: cp.ndarray) -> jnp.ndarray:
    """Convert a CuPy array to a JAX array.

    TODO: This is at the moment a bottleneck, needs to be optimized.
    """
    return jnp.from_dlpack(cp_arr, copy=False)


def jax_to_cupy(jax_arr: jnp.ndarray) -> cp.ndarray:
    """Convert a JAX array to a CuPy array with zero-copy.

    Args:
        jax_arr (jnp.ndarray): The JAX array to convert.

    Returns:
        cp.ndarray: The converted CuPy array.
    """
    return cp.from_dlpack(dlpack.to_dlpack(jax_arr))


@dataclass
class JaxSharray(CupySharray):
    """Serialized representation of a JAX array."""

    @classmethod
    def from_array(cls, arr_jax: jnp.ndarray) -> "JaxSharray":
        """Create a JaxSharray from a JAX array.

        Args:
            arr_jax (jnp.ndarray): The JAX array to serialize.

        Returns:
            JaxSharray: An instance containing serialized data.
        """
        # Convert JAX array to CuPy for serialization
        arr_cp = jax_to_cupy(arr_jax)
        return CupySharray.from_array(arr_cp)

    def open(self) -> jnp.ndarray:
        """Open the shared JAX array and return the underlying data.

        This method reconstructs a JAX array from the serialized data,
        including the IPC handle, shape, dtype, and strides.

        Returns:
            jnp.ndarray: The reconstructed JAX ndarray.
        """
        # Open the CuPy array first
        arr_cp = super().open()
        # Convert CuPy array to JAX array
        return cupy_to_jax(arr_cp)

    @classmethod
    def copy_to(cls, a: jnp.ndarray, b: jnp.ndarray) -> None:
        """Copy a JAX array to another JAX array (possibly efficiently).

        Args:
            a (jnp.ndarray): The source JAX array.
            b (jnp.ndarray): The destination JAX array.
        """
        # Convert JAX arrays to CuPy for copying (JAX arrays are immutable)
        a_cp = jax_to_cupy(a)
        b_cp = jax_to_cupy(b)
        # Use the parent class method to perform the copy
        CupySharray.copy_to(a_cp, b_cp)
