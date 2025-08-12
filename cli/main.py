from __future__ import annotations

"""Sara CLI entry-point (new modular implementation)."""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable

# Ensure `sara` package is importable when CLI executed as module script
if __package__ is None or __package__ == "":  # pragma: no cover – direct script
    spec = importlib.util.spec_from_file_location(
        "sara", Path(__file__).resolve().parent.parent / "__init__.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        sys.modules["sara"] = module

from cli import common  # noqa: E402  import after path fix

# ---------------------------------------------------------------------------
# Utility for dynamic command registrations
# ---------------------------------------------------------------------------

_COMMAND_MODULES = [
    "init_cmd",
    "index_cmd",
    "embed_cmd",
    "search_cmd",
    "serve_cmd",
    "get_cmd",
    "latest_cmd",
    "delete_cmd",
    "pool_cmd",
    "maintenance_cmd",
    "context_cmd",
    "watch_cmd",
]


def _register_commands(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[name-defined]
    for mod_name in _COMMAND_MODULES:
        mod = importlib.import_module(f"cli.{mod_name}")
        register: Callable[[Any], None] = getattr(mod, "register")
        register(subparsers)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        "sara", description="Sara memory CLI (modular)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _register_commands(sub)

    args = parser.parse_args(argv)
    # Many sub-commands reuse common.setup_logging flag
    if getattr(args, "verbose", False):
        common.setup_logging(True)

    # Dispatch
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
