"""
Serena Memory Bridge

Public API surface. Import `Memory` to interact with the persistence
and semantic-search bridge.
"""

# Use FTS5-enabled sqlite build if available – fall back silently
import sys
from importlib import metadata as _metadata

# Re-export core API objects
from .settings import settings  # noqa: F401

try:
    import pysqlite3 as _pysqlite3  # noqa: WPS433

    sys.modules["sqlite3"] = _pysqlite3  # noqa: WPS437 monkey-patch
except ImportError:  # system sqlite will be used (may lack FTS5)
    pass

# Package metadata
# Public symbols
__all__ = ["settings"]

try:
    __version__ = _metadata.version("serena")
except _metadata.PackageNotFoundError:  # local dev / editable install
    __version__ = "0.1.0"

__author__ = "SmartWalletFX Team"
