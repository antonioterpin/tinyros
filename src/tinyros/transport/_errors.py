"""Public exception hierarchy for the transport layer.

Callers can ``except`` on these to distinguish expected failure modes
from bugs:

- :class:`TransportError` -- base; catches everything raised
  deliberately by this package.
- :class:`ConnectionLost` -- the peer socket is gone (graceful close,
  reset, reconnect deadline elapsed). Recovery is reconnect-and-retry.
- :class:`SerializationError` -- the wire payload could not be packed
  or unpacked (pickle, OOB metadata). Recovery is fix-the-payload --
  retrying the same input will fail the same way.
"""

from __future__ import annotations


class TransportError(Exception):
    """Base class for all transport-layer exceptions raised by tinyros."""


class ConnectionLost(TransportError, ConnectionError):
    """The peer socket is gone or never became reachable.

    Multi-inherits from :class:`ConnectionError` so that callers
    written before the typed hierarchy (``except ConnectionError``)
    keep catching it.
    """


class SerializationError(TransportError):
    """A wire payload could not be packed or unpacked."""
