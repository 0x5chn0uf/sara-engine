"""Latest command for showing recent archives."""

import logging
from typing import Any

from cli.common import RemoteMemory


def cmd_latest(args) -> None:
    """Show latest archives."""
    remote_memory = None

    try:
        # Remote-only mode - fail if server not available
        remote_memory = RemoteMemory()
        if not remote_memory.is_server_available():
            print("❌ Server not available - remote operations only")
            return

        # Use server API
        response = remote_memory._make_request(
            "GET", "/archives", params={"limit": args.limit, "offset": 0}
        )
        archives = response.get("archives", [])
        memory_type = "server"

        if not archives:
            print("📭 No archives found")
            return

        # Display header
        if getattr(args, "verbose", False):
            print(f"🔍 Using {memory_type} memory")

        print(f"📚 Latest {len(archives)} archives:")
        print()

        # Display archives (server API format only)
        for i, archive in enumerate(archives, 1):
            task_id = archive.get("task_id")
            title = archive.get("title")
            kind = archive.get("kind")
            status = archive.get("status")
            completed_at = archive.get("completed_at")
            created_at = archive.get("created_at")
            filepath = archive.get("filepath", "N/A")

            print(f"{i:2d}. 📄 {task_id}")
            print(f"    📝 {title}")

            if kind:
                print(f"    🏷️  {kind}", end="")
                if status:
                    print(f" | 📊 {status}", end="")
                print()

            if completed_at:
                print(f"    ✅ Completed: {completed_at}")
            elif created_at:
                print(f"    📅 Created: {created_at}")

            if getattr(args, "show_path", False):
                print(f"    📁 {filepath}")

            print()

        # Show total count if using server
        if memory_type == "server" and "response" in locals():
            total = response.get("total", len(archives))
            if total > len(archives):
                print(f"📊 Showing {len(archives)} of {total} total archives")

    except Exception as e:
        print(f"❌ Failed to get latest archives: {e}")

    finally:
        # Cleanup connections to prevent hanging
        if remote_memory:
            try:
                # Wait for server completion and close connections
                remote_memory.wait_for_server_completion(timeout=5.0)
                remote_memory.close()
            except Exception as cleanup_e:
                print(f"Cleanup warning: {cleanup_e}")


def register(sub: Any) -> None:
    """Register the latest command."""
    p = sub.add_parser("latest", help="Show latest archives")
    p.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="Number of recent archives to show (default: 10)",
    )
    p.add_argument("--show-path", action="store_true", help="Show file paths")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p.set_defaults(func=cmd_latest)
