from __future__ import annotations

"""Background maintenance operations for the Sara memory system."""

import logging
import sys
import time
from typing import Any, Dict, List, Optional

from cli.common import RemoteMemory

# All maintenance operations have been moved to server-side only
# Local database maintenance is no longer supported


def cmd_maintenance(args) -> None:
    """Perform maintenance operations via server API."""
    print("Starting maintenance operation")

    try:
        # Check if server is available for all operations
        remote_memory = RemoteMemory()
        if not remote_memory.is_server_available():
            print("❌ Server not available")
            print("   💡 Start the server with: sara serve")
            sys.exit(1)

        if args.operation == "cleanup":
            print("❌ Local maintenance is no longer supported.")
            print("   All maintenance operations must be performed through the server.")
            print("   Use the server API endpoints or wait for automatic maintenance.")
            sys.exit(1)

        elif args.operation == "status":
            print("📊 Server Status:")
            try:
                # Get server health status
                health_response = remote_memory._make_request("GET", "/health")
                print(f"   Status: {health_response.get('status', 'unknown')}")

                # Get maintenance status if available
                try:
                    maintenance_response = remote_memory._make_request(
                        "GET", "/maintenance/status"
                    )
                    if "health" in maintenance_response:
                        health_info = maintenance_response["health"]
                        print(
                            f"   Database size: {health_info.get('database_size', 0) / (1024*1024):.1f} MB"
                        )
                        print(
                            f"   Archive count: {health_info.get('archive_count', 0)}"
                        )
                        print(
                            f"   Embedding count: {health_info.get('embedding_count', 0)}"
                        )
                except Exception:
                    print("   Maintenance info not available")

            except Exception as e:
                print(f"   ❌ Failed to get server status: {e}")

        elif args.operation == "checkpoint":
            print("🔧 Running database checkpoint...")
            try:
                response = remote_memory._make_request(
                    "POST", "/maintenance/run/checkpoint"
                )
                # _make_request returns the data portion for successful responses
                if response and "message" in response:
                    print("✅ Database checkpoint completed successfully")
                    if args.verbose:
                        print(f"   Details: {response['message']}")
                else:
                    print(f"❌ Checkpoint failed: {response}")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Failed to run checkpoint: {e}")
                sys.exit(1)

        elif args.operation == "vacuum":
            print("🧹 Running database vacuum...")
            try:
                response = remote_memory._make_request(
                    "POST", "/maintenance/run/vacuum"
                )
                # _make_request returns the data portion for successful responses
                if response and "message" in response:
                    print("✅ Database vacuum completed successfully")
                    if args.verbose:
                        print(f"   Details: {response['message']}")
                else:
                    print(f"❌ Vacuum failed: {response}")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Failed to run vacuum: {e}")
                sys.exit(1)

        else:
            print(f"❌ Unknown maintenance operation: {args.operation}")
            print("   Available operations: status, checkpoint, vacuum")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Maintenance operation cancelled by user")
        sys.exit(1)
    except Exception:
        print("❌ Maintenance failed: unexpected error. See logs for details.")
        sys.exit(1)


def register(sub: Any) -> None:
    """Register the maintenance command."""
    p = sub.add_parser("maintenance", help="Perform maintenance operations")
    p.add_argument(
        "operation",
        choices=["status", "checkpoint", "vacuum"],
        help="Maintenance operation to perform",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p.set_defaults(func=cmd_maintenance)
