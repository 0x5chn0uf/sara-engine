"""Watch command for continuous file monitoring and indexing."""

import logging
import signal
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def cmd_watch(args) -> None:
    """Start file watcher for continuous monitoring and indexing."""
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    from sara.cli.common import RemoteMemory
    from sara.infrastructure.indexer.indexer_core import MemoryIndexer
    
    try:
        # Initialize remote memory connection
        memory = RemoteMemory()
        if not memory.is_server_available():
            print("❌ Sara server is not running. Start it with: sara serve")
            sys.exit(1)
        
        print("✅ Connected to Sara server")
        
        # Create indexer
        indexer = MemoryIndexer(memory=memory, max_workers=args.workers)
        
        # Setup directories to watch
        directories = args.directories if args.directories else indexer.scan_dirs
        print(f"📁 Watching directories: {', '.join(directories)}")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down watcher...")
            indexer.stop_watching()
            print("✅ File watcher stopped")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Start watching
        print("🚀 Starting file watcher...")
        indexer.watch_directories(directories)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
    except Exception as exc:
        logger.error(f"Watch command failed: {exc}")
        print(f"❌ Watch command failed: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def register(sub: Any) -> None:
    """Register the watch command."""
    p = sub.add_parser(
        "watch", 
        help="Start file watcher for continuous monitoring and indexing"
    )
    p.add_argument(
        "directories",
        nargs="*",
        help="Directories to watch (default: auto-detect TaskMaster and project dirs)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for indexing (default: 4)"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    p.set_defaults(func=cmd_watch)