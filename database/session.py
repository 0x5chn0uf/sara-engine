"""Database session management with SQLAlchemy ORM."""

import os
import threading
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from sara.core.models import Base
from sara.settings import database_config, settings


class DatabaseManager:
    """Manages SQLAlchemy database sessions and connections."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Use centralized database configuration
        if db_path is not None:
            self.db_config = settings.get_database_config(db_path)
        else:
            self.db_config = database_config

        # Validate configuration early
        try:
            self.db_config.validate_configuration()
        except Exception as e:
            print("Database configuration validation failed: %s", e)
            raise

        self.db_path = self.db_config.db_path
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Ensure database directory exists using centralized method
        self.db_config.ensure_database_directory()

        # Create tables if they don't exist
        self._create_tables()

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with environment-optimized settings."""
        # Use centralized database URL
        db_url = self.db_config.db_url

        # Environment-specific engine configuration
        if settings.is_production:
            # Production: Optimize for reliability and performance
            engine_kwargs = {
                "poolclass": QueuePool,
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 3600,  # Recycle connections every hour
                "pool_pre_ping": True,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 30,  # Longer timeout in production
                },
                "echo": False,
            }
        else:
            # Development: Optimize for debugging and development
            engine_kwargs = {
                "poolclass": StaticPool,
                "pool_pre_ping": True,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20,
                },
                "echo": settings.log_level.upper() == "DEBUG",
            }

        engine = create_engine(db_url, **engine_kwargs)

        print(
            f"Database engine created for {settings.environment} environment with {engine_kwargs['poolclass'].__name__} pooling"
        )

        # Configure SQLite pragmas with environment-specific settings
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()

            try:
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")

                # Environment-specific optimizations
                if settings.is_production:
                    # Production: Prioritize data safety and performance
                    cursor.execute("PRAGMA synchronous=FULL")  # Maximum safety
                    cursor.execute("PRAGMA cache_size=-32000")  # 32MB cache
                    cursor.execute("PRAGMA mmap_size=536870912")  # 512MB memory map
                    cursor.execute(
                        "PRAGMA wal_autocheckpoint=1000"
                    )  # Conservative checkpointing
                    cursor.execute("PRAGMA busy_timeout=60000")  # 60s busy timeout
                else:
                    # Development: Balance safety and speed
                    cursor.execute(
                        "PRAGMA synchronous=NORMAL"
                    )  # Balance safety and speed
                    cursor.execute("PRAGMA cache_size=-16000")  # 16MB cache
                    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
                    cursor.execute(
                        "PRAGMA wal_autocheckpoint=500"
                    )  # More frequent checkpoints
                    cursor.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout

                # Common settings for all environments
                cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
                cursor.execute(
                    "PRAGMA foreign_keys=ON"
                )  # Enable foreign key constraints
                cursor.execute("PRAGMA optimize")  # Run optimization on connect
                cursor.execute("PRAGMA query_only=OFF")  # Allow writes
                cursor.execute(
                    "PRAGMA recursive_triggers=ON"
                )  # Enable recursive triggers

            except Exception as e:
                print(f"Failed to set SQLite pragma: {e}")
            finally:
                cursor.close()

        return engine

    def _create_tables(self) -> None:
        """Create and initialize database tables with proper error handling."""
        try:
            # Create tables
            Base.metadata.create_all(bind=self.engine)

            # Verify table creation and create indexes
            with self.get_session() as session:
                # Verify core tables exist
                tables_to_check = ["archives", "embeddings"]
                for table_name in tables_to_check:
                    result = session.execute(
                        text(
                            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                        )
                    ).fetchone()
                    if not result:
                        raise Exception(f"Table '{table_name}' was not created")

                # Create performance indexes if they don't exist
                session.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_archives_task_id ON archives(task_id)"
                    )
                )
                session.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_archives_completed_at ON archives(completed_at DESC)"
                    )
                )
                session.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_embeddings_task_id ON embeddings(task_id)"
                    )
                )

                # Create FTS table for text search
                session.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS archives_fts USING fts5(task_id, title, summary)"
                    )
                )

                session.commit()

            print(f"Database tables and indexes initialized at {self.db_path}")

        except Exception as exc:
            print(f"❌ Failed to initialize database tables: {exc}")
            raise

    def initialize_for_deployment(self) -> dict:
        """Initialize database for deployment with comprehensive validation."""
        init_info = {"status": "success", "steps": [], "warnings": [], "errors": []}

        try:
            # Step 1: Ensure directory exists
            init_info["steps"].append("Creating database directory")
            self.db_config.ensure_database_directory()

            # Step 2: Test database connectivity
            init_info["steps"].append("Testing database connectivity")
            health = self.health_check()
            if health["status"] == "unhealthy":
                init_info["errors"].extend(
                    ["Database connectivity test failed"] + health.get("warnings", [])
                )
                init_info["status"] = "failed"
                return init_info

            # Step 3: Initialize schema
            init_info["steps"].append("Initializing database schema")
            try:
                self._create_tables()
            except Exception as e:
                init_info["errors"].append(f"Schema initialization failed: {e}")
                init_info["status"] = "failed"
                return init_info

            # Step 4: Final validation
            init_info["steps"].append("Validating deployment readiness")
            final_health = self.health_check()
            if final_health["warnings"]:
                init_info["warnings"].extend(final_health["warnings"])

            if final_health["status"] == "unhealthy":
                init_info["status"] = "degraded"
                init_info["errors"].append("Post-initialization health check failed")

            init_info["steps"].append("Database initialization completed successfully")

        except Exception as e:
            init_info["errors"].append(f"Initialization failed: {e}")
            init_info["status"] = "failed"

        return init_info

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup.

        Yields:
            Session: SQLAlchemy session

        Example:
            with db_manager.get_session() as session:
                archive = session.query(Archive).filter_by(task_id="123").first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def get_transaction(self) -> Generator[Session, None, None]:
        """Get database session with explicit transaction control.

        Yields:
            Session: SQLAlchemy session (auto-commit disabled)

        Example:
            with db_manager.get_transaction() as session:
                # Make multiple changes
                session.add(archive1)
                session.add(archive2)
                # Commit happens automatically on success
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> dict:
        """Comprehensive database health check with detailed metrics.

        Returns:
            dict: Health check results with status and metrics
        """
        import time
        from pathlib import Path

        start_time = time.time()
        health_info = {"status": "healthy", "checks": {}, "metrics": {}, "warnings": []}

        try:
            # Test basic connectivity
            with self.get_session() as session:
                # Basic connectivity test
                connect_start = time.time()
                session.execute(text("SELECT 1")).fetchone()
                connect_time = (time.time() - connect_start) * 1000

                health_info["checks"]["connectivity"] = {
                    "status": "ok",
                    "response_time_ms": round(connect_time, 2),
                }

                # Check database file stats
                db_path = Path(self.db_path)
                if db_path.exists():
                    db_stat = db_path.stat()
                    health_info["metrics"]["file"] = {
                        "size_mb": round(db_stat.st_size / (1024 * 1024), 2),
                        "path": str(db_path),
                        "writable": os.access(db_path, os.W_OK),
                    }

                # Test table existence and get counts
                try:
                    archive_count = session.execute(
                        text("SELECT COUNT(*) FROM archives")
                    ).scalar()
                    embedding_count = session.execute(
                        text("SELECT COUNT(*) FROM embeddings")
                    ).scalar()

                    health_info["metrics"]["tables"] = {
                        "archives": archive_count,
                        "embeddings": embedding_count,
                    }

                    health_info["checks"]["tables"] = {"status": "ok"}

                except Exception as e:
                    health_info["checks"]["tables"] = {
                        "status": "error",
                        "message": f"Table access failed: {e}",
                    }
                    health_info["warnings"].append(
                        "Database tables may need initialization"
                    )

                # Check database settings
                try:
                    wal_mode = session.execute(text("PRAGMA journal_mode")).scalar()
                    foreign_keys = session.execute(text("PRAGMA foreign_keys")).scalar()
                    cache_size = session.execute(text("PRAGMA cache_size")).scalar()

                    health_info["metrics"]["settings"] = {
                        "journal_mode": wal_mode,
                        "foreign_keys_enabled": bool(foreign_keys),
                        "cache_size": cache_size,
                    }

                    if wal_mode != "wal":
                        health_info["warnings"].append(
                            "Database not using WAL mode - performance may be suboptimal"
                        )

                except Exception as e:
                    health_info["warnings"].append(
                        f"Could not check database settings: {e}"
                    )

                # Check connection pool status
                pool_status = self.get_pool_status()
                if pool_status:
                    health_info["metrics"]["pool"] = pool_status

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["checks"]["connectivity"] = {"status": "error", "error": str(e)}
            print("Database health check failed: %s", e)

        # Overall health determination
        total_time = (time.time() - start_time) * 1000
        health_info["response_time_ms"] = round(total_time, 2)

        # Check for conditions that should affect status
        if health_info["warnings"] and health_info["status"] == "healthy":
            health_info["status"] = "degraded"

        return health_info

    def vacuum(self) -> None:
        """Run VACUUM operation to reclaim space."""
        try:
            with self.get_session() as session:
                session.execute(text("VACUUM"))
                session.commit()
            print("Database VACUUM completed")
        except Exception as exc:
            print(f"Database VACUUM failed: {exc}")
            raise

    def checkpoint(self) -> None:
        """Checkpoint the WAL file."""
        try:
            # Use raw connection for PRAGMA commands instead of ORM session
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                conn.commit()
            print("WAL checkpoint completed")
        except Exception as exc:
            print("WAL checkpoint failed: %s", exc)
            raise

    def get_pool_status(self) -> dict:
        """Get connection pool status information."""
        try:
            pool = self.engine.pool
            return {
                "pool_size": getattr(pool, "size", lambda: 0)(),
                "pool_checked_in": getattr(pool, "checkedin", lambda: 0)(),
                "pool_checked_out": getattr(pool, "checkedout", lambda: 0)(),
                "pool_overflow": getattr(pool, "overflow", lambda: 0)(),
                "pool_invalid": getattr(pool, "invalidated", lambda: 0)(),
            }
        except Exception as exc:
            print("Failed to get pool status: %s", exc)
            return {}

    def close(self) -> None:
        """Close the database engine and cleanup connections."""
        if hasattr(self, "engine"):
            try:
                # Log final pool status
                status = self.get_pool_status()
                print(f"Closing database engine. Final pool status: {status}")

                # Dispose of all connections in pool
                self.engine.dispose()
                print("Database engine closed successfully")
            except Exception as exc:
                print("Error closing database engine: %s", exc)


# Global database manager instance with thread safety
_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager(db_path: Optional[str] = None) -> DatabaseManager:
    """Get global database manager instance (thread-safe).

    Args:
        db_path: Path to database file (only used on first call)

    Returns:
        DatabaseManager: Global database manager
    """
    global _db_manager
    if _db_manager is None:
        with _db_manager_lock:
            # Double-check after acquiring lock
            if _db_manager is None:
                _db_manager = DatabaseManager(db_path)
    return _db_manager


@contextmanager
def get_db_session(db_path: Optional[str] = None) -> Generator[Session, None, None]:
    """Convenience function to get a database session.

    Args:
        db_path: Path to database file

    Yields:
        Session: SQLAlchemy session with connection pool monitoring

    Example:
        with get_db_session() as session:
            archives = session.query(Archive).limit(10).all()
    """
    db_manager = get_db_manager(db_path)

    # Log pool status on high usage (for monitoring)
    try:
        status = db_manager.get_pool_status()
        checked_out = status.get("pool_checked_out", 0)
        pool_size = status.get("pool_size", 0)

        if checked_out > pool_size * 0.8:  # Log when 80% of pool is used
            print(
                "High database connection usage: %d/%d connections in use",
                checked_out,
                pool_size,
            )
    except Exception:
        pass  # Don't fail session creation due to monitoring

    with db_manager.get_session() as session:
        yield session


def get_pool_metrics(db_path: Optional[str] = None) -> dict:
    """Get detailed connection pool metrics for monitoring.

    Args:
        db_path: Path to database file

    Returns:
        dict: Pool metrics
    """
    db_manager = get_db_manager(db_path)
    return db_manager.get_pool_status()
