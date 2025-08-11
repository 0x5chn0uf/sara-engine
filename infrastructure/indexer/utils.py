"""Indexer utility functions."""

import asyncio
from typing import Dict, List, Optional

from .indexer_core import MemoryIndexer


async def index_memories_async(
    directories: Optional[List[str]] = None,
    force_reindex: bool = False,
    max_workers: int = 4,
) -> Dict[str, int]:
    """Async wrapper for memory indexing."""
    indexer = MemoryIndexer(max_workers=max_workers)

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, indexer.scan_directories, directories, force_reindex, True
    )


def index_memories(
    directories: Optional[List[str]] = None,
    force_reindex: bool = False,
    max_workers: int = 4,
    show_progress: bool = True,
) -> Dict[str, int]:
    """Synchronous function to index memories."""
    indexer = MemoryIndexer(max_workers=max_workers)
    return indexer.scan_directories(directories, force_reindex, show_progress)
