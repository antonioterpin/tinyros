"""TinyROS public API.

Nothing outside of this module's ``__all__`` is considered public.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._logging import get_logger, setup_console_logging
from .node import (
    TinyNetworkConfig,
    TinyNode,
    TinyNodeDescription,
    TinySubscription,
)
from .transport import (
    ConnectionLost,
    SerializationError,
    TinyClient,
    TinyServer,
    TransportError,
)

try:
    __version__ = _version("tinyros")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__all__ = [
    "ConnectionLost",
    "SerializationError",
    "TinyClient",
    "TinyNetworkConfig",
    "TinyNode",
    "TinyNodeDescription",
    "TinyServer",
    "TinySubscription",
    "TransportError",
    "get_logger",
    "setup_console_logging",
]
