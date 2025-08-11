"""Search utility functions and convenience wrappers."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sara.core.models import SearchResult, TaskKind, TaskStatus
from sara.database.session import get_db_session as get_session
from sara.infrastructure.search.search_core import SearchEngine


def search_memories(
    query: str,
    k: int = 10,
    kind_filter: Optional[List[str]] = None,
    status_filter: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> List[SearchResult]:
    """
    Convenience function for searching memories.

    Args:
        query: Search query text
        k: Number of results to return
        kind_filter: Optional filter by task kinds (string values)
        status_filter: Optional filter by task statuses (string values)
        db_path: Optional database path

    Returns:
        List[SearchResult]: Search results
    """
    engine = SearchEngine(db_path)

    # Convert string filters to enums
    kind_enum_filter = None
    if kind_filter:
        kind_enum_filter = [TaskKind(k) for k in kind_filter]

    status_enum_filter = None
    if status_filter:
        status_enum_filter = [TaskStatus(s) for s in status_filter]

    return engine.search(
        query=query, k=k, kind_filter=kind_enum_filter, status_filter=status_enum_filter
    )


def get_latest_tasks(
    n: int = 10, kind_filter: Optional[List[str]] = None, db_path: Optional[str] = None
) -> List[SearchResult]:
    """
    Get the most recently completed tasks.

    Args:
        n: Number of tasks to return
        kind_filter: Optional filter by task kinds
        db_path: Optional database path

    Returns:
        List[SearchResult]: Latest tasks
    """
    with get_session(db_path) as session:
        from sqlalchemy import func

        from sara.core.models import Archive

        query = session.query(Archive)

        if kind_filter:
            query = query.filter(Archive.kind.in_(kind_filter))

        # Sort by the most recent completion date, using updated_at as fallback for NULL completed_at
        # NULLS LAST ensures NULL completed_at values don't interfere with the sorting
        archives = (
            query.order_by(
                Archive.completed_at.desc().nulls_last(), Archive.updated_at.desc()
            )
            .limit(n)
            .all()
        )

        results = []
        for archive in archives:
            result = SearchResult(
                task_id=archive.task_id,
                title=archive.title,
                score=1.0,  # Not applicable for latest query
                excerpt="",  # Not needed for latest query
                kind=TaskKind(archive.kind),
                status=TaskStatus(archive.status) if archive.status else None,
                completed_at=archive.completed_at,
                filepath=archive.filepath,
            )
            results.append(result)

        return results
