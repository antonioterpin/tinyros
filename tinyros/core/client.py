"""Client class to read messages from a Buffer."""

from typing import Optional, Dict, Any

from tinyros.memory.buffer import Buffer


class Client:
    """A Client class to read messages from a Buffer."""

    def __init__(self, name: str, buffer: Buffer):
        """Initialize a Client instance.

        Args:
            name (str): The name of the client.
            buffer (Buffer): The buffer where to read messages from.
        """
        self._buffer = buffer
        self.name = name
        self._buffer.open_out_buffer(name)

    def try_get(
        self, seq: int, timeout: float = None, latest: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Try to get a message from the buffer.

        Args:
            seq (int): The sequence number of the message to retrieve.
            timeout (float, optional): The timeout for getting the message.
            latest (bool, optional): If True, get the latest message, else the next one.

        Returns:
            Optional[Dict[str, Any]]: The message data if available, otherwise None.
        """
        if self._buffer.is_closed():
            raise RuntimeError("Buffer is closed, cannot get messages.")
        ret = None
        if latest:
            ret = self._buffer.get_newest(name=self.name, t=seq, timeout=timeout)
        else:
            ret = self._buffer.get_next_after(name=self.name, t=seq, timeout=timeout)
        return ret

    def close(self):
        """Close the client's buffer."""
        self._buffer.close()
        del self._buffer
