"""Base class for shared arrays in TinyROS."""

from abc import abstractmethod
from typing import Any


class Sharray:
    """Base class for shared arrays in TinyROS.

    This class is intended to be subclassed for specific shared array types.
    It provides a common interface for serialization and deserialization.
    """

    @classmethod
    @abstractmethod
    def from_array(cls, arr: Any) -> "Sharray":
        """Create a Sharray from a given array.

        Args:
            arr (Any): The array to serialize.

        Returns:
            Sharray: An instance containing serialized data.
        """

    @abstractmethod
    def open(self) -> Any:
        """Open the shared array and return the underlying data.

        This method should be implemented by subclasses to define how the
        shared array is opened and accessed.

        Returns:
            Any: The underlying data of the shared array.
        """

    @classmethod
    @abstractmethod
    def copy_to(self, a: Any, b: Any) -> None:
        """Copy a to b (possibly efficiently).

        Args:
            a (Any): The source object.
            b (Any): The destination object.
        """
