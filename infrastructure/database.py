"""Database connection and schema management for Serena memory bridge.

All functions use SQLAlchemy ORM with proper migrations and context managers.
"""

from __future__ import annotations

from typing import Optional

from settings import settings


def get_database_path() -> str:
    """Return the configured SQLite database path."""
    return settings.memory_db


def init_database(db_path: Optional[str] = None) -> str:
    """Initialize database using SQLAlchemy ORM with migrations.

    Args:
        db_path: Path to database file

    Returns:
        str: Path to initialized database
    """
    from database.migrations import auto_migrate
    from database.session import get_db_manager

    if db_path is None:
        db_path = get_database_path()

    # Auto-migrate database to latest schema
    migrated = auto_migrate(db_path)

    # Verify database manager works
    db_manager = get_db_manager(db_path)
    if not db_manager.health_check():
        raise RuntimeError("Database health check failed")

    if migrated:
        print(f"✅ Database initialized with migrations at {db_path}")
    else:
        print(f"Database already up to date at {db_path}")

    return db_path


def get_session(db_path: Optional[str] = None):
    """Get SQLAlchemy session context manager.

    Args:
        db_path: Path to database file

    Returns:
        Context manager for SQLAlchemy session

    Example:
        with get_session() as session:
            archives = session.query(Archive).all()
    """
    from database.session import get_db_session

    return get_db_session(db_path)


def checkpoint_database(db_path: Optional[str] = None) -> None:
    """Checkpoint the database WAL file."""
    from database.session import get_db_manager

    db_manager = get_db_manager(db_path)
    db_manager.checkpoint()


def vacuum_database(db_path: Optional[str] = None) -> None:
    """Run VACUUM operation to reclaim space."""
    from database.session import get_db_manager

    db_manager = get_db_manager(db_path)
    db_manager.vacuum()
