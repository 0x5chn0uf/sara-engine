from __future__ import annotations

"""Sara CLI package.

This lightweight wrapper re-exports the ``main`` function from
:pyfile:`serena.cli.main` so external callers can simply do::

    from cli import main
    main()

The heavy logic will be progressively split into specialised modules
inside :pymod:`serena.cli.*`.
"""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING
from typing import Any as _Any


def _lazy_import(name: str) -> _ModuleType:  # noqa: D401
    """Import *name* on first access to defer heavy imports."""

    module = _import_module(name)
    globals()[name.rsplit(".", 1)[-1]] = module  # cache in module globals
    return module


if TYPE_CHECKING:  # pragma: no cover – mypy/IDE support only
    from .main import main  # noqa: F401
else:

    def __getattr__(attr: str) -> _Any:  # noqa: D401
        if attr == "main":
            return _lazy_import("sara.cli.main").main  # type: ignore[attr-defined]
        raise AttributeError(attr)


__all__ = ["main"]
