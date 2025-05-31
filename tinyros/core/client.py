from tinyros.buffer import Buffer
from tinyros.datatype import DataType

class Client:
    def __init__(self, buffer: Buffer, mock_data: DataType):
        """Initialize a Client instance.
        
        Args:
            buffer (Buffer): The buffer where to read messages from.
            mock_data (DataType): The mock data to initialize the buffer.
        """
        self.buffer = buffer
        self.data = mock_data

    def try_get(self, seq: int, timeout: float = None, latest: bool=True) -> DataType:
        """Try to get a message from the buffer.
        
        Args:
            seq (int): The sequence number of the message to retrieve.
            timeout (float, optional): The timeout for getting the message.
            latest (bool, optional): If True, get the latest message, else the next one.
        
        Returns:
            DataType: The deserialized data if found, else None.
        """
        ret = None
        if latest:
            ret = self.buffer.get_newest(seq, timeout=timeout)
        else:
            ret = self.buffer.get_next_after(seq, timeout=timeout)
        if ret is not None:
            t, raw_data = ret
            return t, self.data.deserialize(raw_data)
        return None

    @classmethod
    def make(
        cls,
        buffer: Buffer,
        mock_data: DataType,
    ) -> "Client":
        """
        Create a Client instance.

        Args:
            buffer (Buffer): The buffer where to read messages from.
            mock_data (DataType): The mock data to initialize the buffer.

        Returns:
            Client: An instance of the Client class.
        """
        return cls(buffer=buffer, mock_data=mock_data)
