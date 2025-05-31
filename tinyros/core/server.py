from multiprocessing import get_context

from tinyros.buffer import Buffer
from tinyros.datatype import DataType

class Server:
    def __init__(
        self,
        buffer: Buffer,
    ):
        """Initialize a Server instance.
        
        Args:
            topic (str): The topic to publish to.
            mock_data (DataType): The mock data to initialize the buffer.
            buffer (Buffer): The buffer for storing messages.
        """
        self._buffer = buffer

    def publish(self, data: DataType):
        """Publish data to the server's buffer.
        
        Args:
            data (DataType): The data to publish.
        """
        self._buffer.put(data.serialize())

    @property
    def buffer(self) -> Buffer:
        """Get the server's buffer."""
        return self._buffer

    def close(self):
        """Close the server's buffer."""
        self.buffer.close()

    @classmethod
    def make(
        cls,
        capacity: int,
        mock_data: DataType,
    ) -> "Server":
        """Create a server instance.
        
        Args:
            capacity (int): The capacity of the buffer.
            mock_data (DataType): The mock data to initialize the buffer.
            synchronizer (Optional[Synchronizer]): 
                An optional synchronizer for the server.

        Returns:
            Server: An instance of the Server class.
        """
        # Explicitly use 'spawn' context for multiprocessing
        # Else, in ubuntu, it will use 'fork' by default
        ctx = get_context("spawn")
        buffer = Buffer(
            capacity=capacity, slot_size=len(mock_data.serialize()), ctx=ctx)

        return cls(buffer)