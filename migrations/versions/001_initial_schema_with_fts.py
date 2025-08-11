"""Initial schema with FTS5 virtual table

Revision ID: 001_initial_schema_with_fts
Revises: 
Create Date: 2025-07-28 03:30:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "001_initial_schema_with_fts"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial schema including FTS5 virtual table."""
    # Basic tables are already created by SQLAlchemy create_all()
    # We just need to add the FTS5 virtual table

    # Create FTS5 virtual table for full-text search
    op.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS archives_fts USING fts5(
            task_id,
            title,
            summary,
            content='archives',
            content_rowid='rowid'
        );
    """
    )

    # Create triggers to keep FTS table in sync with archives table
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS archives_fts_insert AFTER INSERT ON archives BEGIN
            INSERT INTO archives_fts(task_id, title, summary) 
            VALUES (new.task_id, new.title, new.summary);
        END;
    """
    )

    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS archives_fts_delete AFTER DELETE ON archives BEGIN
            INSERT INTO archives_fts(archives_fts, task_id, title, summary) 
            VALUES('delete', old.task_id, old.title, old.summary);
        END;
    """
    )

    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS archives_fts_update AFTER UPDATE ON archives BEGIN
            INSERT INTO archives_fts(archives_fts, task_id, title, summary) 
            VALUES('delete', old.task_id, old.title, old.summary);
            INSERT INTO archives_fts(task_id, title, summary) 
            VALUES (new.task_id, new.title, new.summary);
        END;
    """
    )

    # Populate FTS table with existing data
    op.execute(
        """
        INSERT INTO archives_fts(task_id, title, summary)
        SELECT task_id, title, summary FROM archives;
    """
    )


def downgrade():
    """Remove FTS5 virtual table and triggers."""
    op.execute("DROP TRIGGER IF EXISTS archives_fts_update;")
    op.execute("DROP TRIGGER IF EXISTS archives_fts_delete;")
    op.execute("DROP TRIGGER IF EXISTS archives_fts_insert;")
    op.execute("DROP TABLE IF EXISTS archives_fts;")
