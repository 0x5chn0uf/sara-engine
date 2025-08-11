"""Add performance indexes for query optimization

Revision ID: 002_add_performance_indexes
Revises: 001_initial_schema_with_fts
Create Date: 2025-07-28 12:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "002_add_performance_indexes"
down_revision = "001_initial_schema_with_fts"
branch_labels = None
depends_on = None


def upgrade():
    """Add performance indexes for common query patterns."""

    # Index for task_id lookups (most common operation)
    op.execute("CREATE INDEX IF NOT EXISTS idx_archives_task_id ON archives(task_id);")

    # Index for kind-based filtering
    op.execute("CREATE INDEX IF NOT EXISTS idx_archives_kind ON archives(kind);")

    # Index for completion date ordering (used in latest queries)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_archives_completed_at ON archives(completed_at DESC);"
    )

    # Index for status filtering
    op.execute("CREATE INDEX IF NOT EXISTS idx_archives_status ON archives(status);")

    # Composite index for kind + completion date queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_archives_kind_completed ON archives(kind, completed_at DESC);"
    )

    # Index for embeddings task_id foreign key lookups
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_task_id ON embeddings(task_id);"
    )

    # Index for embeddings chunk queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_task_chunk ON embeddings(task_id, chunk_id);"
    )

    # Index for SHA256 hash uniqueness checks (faster duplicate detection)
    op.execute("CREATE INDEX IF NOT EXISTS idx_archives_sha256 ON archives(sha256);")

    # Index for filepath uniqueness checks
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_archives_filepath ON archives(filepath);"
    )

    # Index for updated_at timestamp (for sync/change tracking)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_archives_updated_at ON archives(updated_at DESC);"
    )


def downgrade():
    """Remove performance indexes."""
    op.execute("DROP INDEX IF EXISTS idx_archives_updated_at;")
    op.execute("DROP INDEX IF EXISTS idx_archives_filepath;")
    op.execute("DROP INDEX IF EXISTS idx_archives_sha256;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_task_chunk;")
    op.execute("DROP INDEX IF EXISTS idx_embeddings_task_id;")
    op.execute("DROP INDEX IF EXISTS idx_archives_kind_completed;")
    op.execute("DROP INDEX IF EXISTS idx_archives_status;")
    op.execute("DROP INDEX IF EXISTS idx_archives_completed_at;")
    op.execute("DROP INDEX IF EXISTS idx_archives_kind;")
    op.execute("DROP INDEX IF EXISTS idx_archives_task_id;")
