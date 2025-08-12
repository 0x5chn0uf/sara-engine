"""CLI command for code embedding functionality."""

import argparse
from pathlib import Path

from cli.common import setup_logging, validate_configuration, RemoteMemory
from settings import settings


def cmd_embed(args: argparse.Namespace) -> int:
    """Command handler for code embedding operations."""
    setup_logging(args.verbose)

    try:
        validate_configuration()
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1

    # Use server-only mode - check if server is available
    remote_memory = RemoteMemory()
    if not remote_memory.is_server_available():
        print("❌ Sara server is not running")
        print("   💡 Start the server with: sara serve")
        print("   Embed operations require the server to use pre-loaded embedding models")
        return 1

    if args.action == "index":
        return _cmd_embed_index_server(remote_memory, args)
    elif args.action == "search":
        return _cmd_embed_search_server(remote_memory, args)
    elif args.action == "stats":
        return _cmd_embed_stats_server(remote_memory, args)
    else:
        print(f"❌ Unknown embed action: {args.action}")
        return 1


def _cmd_embed_index_server(remote_memory: RemoteMemory, args: argparse.Namespace) -> int:
    """Handle code indexing command via server API."""
    try:
        print("🚀 Starting codebase embedding via server...")
        print("   ⚡ Using server API (preloaded embedding model)")

        # Prepare the API request
        payload = {
            "force_reindex": args.force,
        }
        
        if args.files:
            payload["files"] = args.files
            print(f"📁 Indexing {len(args.files)} specific files...")
        else:
            print("📁 Indexing entire codebase...")

        # Make the server API call
        response = remote_memory._make_request("POST", "/embed/index", json=payload)
        
        if response and response.get("success"):
            stats = response.get("stats", {})
            print(f"\n📊 Embedding Complete:")
            print(f"   📁 Files found: {stats.get('files_found', 0)}")
            print(f"   ✅ Files processed: {stats.get('files_processed', 0)}")
            print(f"   ⏭️  Files skipped: {stats.get('files_skipped', 0)}")
            print(f"   🧩 Chunks created: {stats.get('chunks_created', 0)}")
            print(f"   🎯 Embeddings generated: {stats.get('embeddings_generated', 0)}")
            print(f"   ❌ Errors: {stats.get('errors', 0)}")
            
            # Wait for server completion
            print("⏳ Waiting for server to complete processing...")
            remote_memory.wait_for_server_completion(timeout=30.0)
            
            return 0 if stats.get("errors", 0) == 0 else 1
        else:
            error_msg = response.get("error", "Unknown server error") if response else "No response from server"
            print(f"❌ Server embedding failed: {error_msg}")
            return 1

    except Exception as e:
        print(f"❌ Error during server embedding: {e}")
        return 1


def _cmd_embed_search_server(remote_memory: RemoteMemory, args: argparse.Namespace) -> int:
    """Handle code search command via server API."""
    try:
        query = args.query
        limit = args.limit or 10

        print(f"🔍 Searching codebase for: '{query}'")
        print("   ⚡ Using server API (preloaded embedding model)")

        # Make the server API call
        response = remote_memory._make_request("POST", "/embed/search", json={
            "query": query,
            "limit": limit
        })

        if response and response.get("success"):
            results = response.get("results", [])
            
            if not results:
                print("No results found.")
                return 0

            print(f"\n📊 Found {len(results)} results:")
            print("-" * 80)

            for i, result in enumerate(results, 1):
                print(
                    f"\n{i}. {result['filepath']}:{result['start_line']}-{result['end_line']}"
                )
                print(f"   📊 Similarity: {result['similarity']:.3f}")
                print(f"   💻 Preview: {result['preview']}")

            return 0
        else:
            error_msg = response.get("error", "Unknown server error") if response else "No response from server"
            print(f"❌ Server search failed: {error_msg}")
            return 1

    except Exception as e:
        print(f"❌ Error during server search: {e}")
        return 1


def _cmd_embed_stats_server(remote_memory: RemoteMemory, args: argparse.Namespace) -> int:
    """Handle stats command via server API."""
    try:
        print("📊 Getting code embedding statistics from server...")
        print("   ⚡ Using server API")

        # Make the server API call
        response = remote_memory._make_request("GET", "/embed/stats")

        if response and response.get("success"):
            stats = response.get("stats", {})
            
            print("📊 Code Embedding Statistics:")
            print(f"   📁 Files indexed: {stats.get('files_indexed', 0)}")
            print(f"   🎯 Embeddings generated: {stats.get('embeddings_generated', 0)}")
            print(f"   📝 Files tracked in indexed_files: {stats.get('indexed_files_tracked', 0)}")
            print(f"   📈 Average chunks per file: {stats.get('average_chunks_per_file', 0):.1f}")

            return 0
        else:
            error_msg = response.get("error", "Unknown server error") if response else "No response from server"
            print(f"❌ Server stats failed: {error_msg}")
            return 1

    except Exception as e:
        print(f"❌ Error getting server stats: {e}")
        return 1


def register(subparsers) -> None:
    """Register the embed command with the argument parser."""
    parser = subparsers.add_parser(
        "embed",
        help="Code embedding operations for semantic code search",
        description="Embed code files (*.py, *.ts, *.tsx, *.js, *.jsx) into semantic search index. "
                   "For documentation indexing, use 'sara index' instead.",
    )

    # Add verbose flag
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    # Add project root option
    parser.add_argument(
        "--project-root",
        type=str,
        help="Project root directory (default: current directory)",
    )

    # Add subcommands
    subparsers_embed = parser.add_subparsers(
        dest="action", help="Embed actions", required=True
    )

    # Index subcommand
    parser_index = subparsers_embed.add_parser(
        "index", help="Index codebase or specific files for embedding"
    )
    parser_index.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing of all files (ignore SHA-256 checksums)",
    )
    parser_index.add_argument(
        "--files",
        nargs="*",
        help="Specific files to index (if not provided, indexes entire codebase)",
    )

    # Search subcommand
    parser_search = subparsers_embed.add_parser(
        "search", help="Search embedded code using semantic similarity"
    )
    parser_search.add_argument("query", type=str, help="Search query")
    parser_search.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results (default: 10)"
    )

    # Stats subcommand
    subparsers_embed.add_parser("stats", help="Show embedding statistics")

    parser.set_defaults(func=cmd_embed)
