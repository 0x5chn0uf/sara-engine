from __future__ import annotations

"""`sara index` command."""

import time
from typing import Any

from sara.cli.common import detect_taskmaster_directories
from sara.infrastructure.indexer import MemoryIndexer


def cmd_index(args) -> None:
    """Index memories from directories or individual files."""
    try:
        # Determine what to scan - directories, files, or both
        directories = []
        files = []

        if args.directories:
            directories = [d.strip() for d in args.directories.split(",")]
        if args.files:
            files = [f.strip() for f in args.files.split(",")]

        # If nothing specified, use auto-detected directories
        if not directories and not files:
            directories = detect_taskmaster_directories()
            if not directories:
                print(
                    "⚠️ No directories or files found to index. Use --directories or --files to specify manually."
                )
                return

        # Create indexer
        indexer = MemoryIndexer(max_workers=args.workers)

        # Show mode
        if directories and files:
            print(
                f"🔍 Indexing {len(directories)} directories and {len(files)} files using server memory"
            )
        elif directories:
            print(
                f"🔍 Indexing directories using server memory: {', '.join(directories)}"
            )
        else:
            print(f"🔍 Indexing {len(files)} individual files using server memory")

        # Start overall timing
        start_time = time.time()

        # Process directories and files
        if directories:
            stats = indexer.scan_directories(
                directories=directories, force_reindex=args.force, show_progress=True
            )
        else:
            stats = {
                "files_found": 0,
                "files_indexed": 0,
                "files_skipped": 0,
                "files_failed": 0,
                "directories_scanned": 0,
                "scan_time_seconds": 0.0,
                "indexing_time_seconds": 0.0,
                "total_time_seconds": 0.0,
            }

        # Process individual files if specified
        if files:
            file_stats = indexer.scan_files(
                files=files, force_reindex=args.force, show_progress=True
            )
            # Merge stats
            stats["files_found"] += file_stats["files_found"]
            stats["files_indexed"] += file_stats["files_indexed"]
            stats["files_skipped"] += file_stats["files_skipped"]
            stats["files_failed"] += file_stats["files_failed"]
            stats["indexing_time_seconds"] += file_stats["indexing_time_seconds"]

        total_command_time = time.time() - start_time

        # Display comprehensive results with performance metrics
        print("\n✅ Indexing complete!")
        print(f"📁 Directories scanned: {stats['directories_scanned']}")
        print(f"📄 Files found: {stats.get('files_found', 0)}")
        print(f"📄 Files indexed: {stats['files_indexed']}")
        print(f"⏭️ Files skipped: {stats['files_skipped']}")
        if stats["files_failed"] > 0:
            print(f"❌ Files failed: {stats['files_failed']}")

        # Performance metrics
        print("\n⏱️ Performance Summary:")
        scan_time = stats.get("scan_time_seconds", 0)
        indexing_time = stats.get("indexing_time_seconds", 0)
        print(f"   - Directory scan: {scan_time:.2f}s")
        print(f"   - File processing: {indexing_time:.2f}s")
        print(f"   - Total time: {total_command_time:.2f}s")

        # Throughput calculations
        if indexing_time > 0 and stats["files_indexed"] > 0:
            files_per_second = stats["files_indexed"] / indexing_time
            print(f"   - Throughput: {files_per_second:.1f} files/second")

        if stats["files_indexed"] > 0:
            avg_time_per_file = (
                indexing_time / stats["files_indexed"] * 1000
            )  # Convert to ms
            print(f"   - Average per file: {avg_time_per_file:.1f}ms")

        # Efficiency metrics
        total_files = stats.get("files_found", 0)
        if total_files > 0:
            efficiency = (stats["files_indexed"] / total_files) * 100
            print(f"   - Processing efficiency: {efficiency:.1f}%")

        # Ensure proper cleanup of background resources
        _cleanup_indexing_resources(indexer)

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        raise


def _cleanup_indexing_resources(indexer) -> None:
    """Clean up indexing resources to prevent hanging."""
    try:
        print("⏳ Waiting for server to complete processing...")
        if hasattr(indexer.memory, "wait_for_server_completion"):
            completion_success = indexer.memory.wait_for_server_completion(timeout=10.0)
            if completion_success:
                print("✅ Server processing completed")
            else:
                print(
                    "⚠️ Timeout waiting for server - some operations may still be processing"
                )

        # Close HTTP connections
        if hasattr(indexer.memory, "close"):
            indexer.memory.close()
            print("✅ Remote connections closed")

        # No cleanup needed for RemoteMemory - it just makes HTTP requests

    except Exception as e:
        # Don't let cleanup errors prevent command completion
        print(f"⚠️ Cleanup warning: {e}")
        pass


def register(sub: Any) -> None:
    """Register the index command."""
    p = sub.add_parser(
        "index",
        help="Index documentation and content files (*.md, *.txt, *.json)",
        description="Index documentation, TaskMaster archives, and content files for semantic search. "
        "For code embedding, use 'sara embed index' instead.",
    )
    p.add_argument(
        "--directories",
        help="Comma-separated directories to scan (default: auto-detect TaskMaster dirs)",
    )
    p.add_argument("--files", help="Comma-separated individual files to index")
    p.add_argument("--force", action="store_true", help="Force reindex of all files")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p.set_defaults(func=cmd_index)
