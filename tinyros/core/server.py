"""Server class for managing a message buffer in a TinyROS system."""

from tinyros.memory import Buffer
from tinyros.datatype import TinyROSMessageDefinition


class Server:
    """A Server class to manage a message buffer."""

    def __init__(
        self, msg_def: TinyROSMessageDefinition, capacity: int, ctx=None
    ) -> "Server":
        """Initialize a Server instance.

        Args:
            msg_def (TinyROSMessageDefinition):
                The definition of the message that the server will handle.
            capacity (int): The capacity of the server's buffer.
            ctx (Optional[Context]):
                The context for multiprocessing. Defaults to None, which uses 'fork'.
        """
        self._buffer = Buffer(capacity=capacity, msg_def=msg_def, ctx=ctx)
        self._data_description = {f.name: f.dtype for f in msg_def}

    def publish(self, **kwargs) -> None:
        """Publish data to the server's buffer.

        Args:
            **kwargs: The data to be published, as keyword arguments.
        """
        self._check_data_description(kwargs)
        self._buffer.put({k: kwargs[k] for k in self._data_description.keys()})

    def _check_data_description(self, kwargs: dict) -> None:
        """Check if the provided kwargs match the server's data description.

        Args:
            kwargs (dict): The keyword arguments to check against the data description.

        Raises:
            ValueError: If any required keys are missing from kwargs.
            TypeError:
                If the types of the values in kwargs do not match the data description.
        """
        missing_keys = [k for k in self._data_description if k not in kwargs]
        if missing_keys:
            raise ValueError(
                f"The following required keys are missing from kwargs: {missing_keys}"
            )

        type_mismatches = {
            k: (type(kwargs[k]), self._data_description[k])
            for k in kwargs
            if not isinstance(kwargs[k], self._data_description[k])
        }
        if type_mismatches:
            mismatch_details = ", ".join(
                f"{k}: expected {expected}, got {actual}"
                for k, (actual, expected) in type_mismatches.items()
            )
            raise TypeError(
                f"Data types in kwargs do not match the data description. "
                f"Mismatches: {mismatch_details}"
            )

    @property
    def buffer(self) -> Buffer:
        """Get the server's buffer."""
        return self._buffer

    def close(self):
        """Close the server's buffer."""
        self.buffer.close()
        del self._buffer
