from __future__ import annotations

"""`serena init` command."""

import sys
from typing import Any

from sara.infrastructure.database import init_database


def cmd_init(args) -> None:
    """Initialize Sara database and configuration."""
    try:
        # Initialize the database
        init_database()
        # TODO: Add configuration file creation if needed
        # TODO: Add auto-detection of project structure

        print("✅ Sara initialized successfully!")

    except KeyboardInterrupt:
        print("\n🛑 Initialization cancelled by user")
        sys.exit(1)
    except PermissionError as e:
        print(f"❌ Permission denied: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)


def register(sub: Any) -> None:  # sub is argparse subparser
    """Register the init command."""
    parser = sub.add_parser("init", help="Initialize Sara (DB, config, auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=cmd_init)
