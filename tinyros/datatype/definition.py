"""TinyROS message definition module."""

from typing import List
import cupy as cp


class TinyROSMessageFieldDefinition:
    """A class to describe a field of a TinyROS message."""

    name: str
    dtype: type
    kwargs: dict

    def __init__(self, name: str, field_type: type, **kwargs):
        """Initialize a TinyROS message field definition.

        Args:
            name (str): The name of the field.
            field_type (type): The type of the field, e.g., `cp.ndarray`.
            **kwargs: Additional keyword arguments for the field.

        Raises:
            TypeError: If the name is not a string or is empty.
            ValueError: If the field_type is not a valid type.
        """
        if not isinstance(name, str) or name == "":
            raise TypeError("Field name must be a string.")
        if (
            field_type
            not in [cp.ndarray]
            # # TODO: probably there is a better way to handle these other types
            # and not isinstance(dtype, int)
            # and not isinstance(dtype, float)
            # and not isinstance(dtype, str)
            # and not isinstance(dtype, bool)
        ):
            raise ValueError(f"Type '{field_type}' is not a valid type.")

        self.name = name
        self.dtype = field_type
        self.kwargs = kwargs


TinyROSMessageDefinition = List[TinyROSMessageFieldDefinition]
