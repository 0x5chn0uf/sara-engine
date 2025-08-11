"""add_indexed_files_table

Revision ID: 003_add_indexed_files_table
Revises: 002_add_performance_indexes
Create Date: 2025-01-30 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_indexed_files_table'
down_revision: Union[str, None] = '002_add_performance_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add indexed_files table for tracking indexed files."""
    op.create_table(
        'indexed_files',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('filepath', sa.Text(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('kind', sa.String(length=20), nullable=False),
        sa.Column('task_id', sa.String(length=255), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=False, default=0),
        sa.Column('last_modified', sa.DateTime(), nullable=False),
        sa.Column('indexed_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('filepath'),
        sa.CheckConstraint(
            "kind IN ('archive', 'reflection', 'doc', 'rule', 'code')",
            name='ck_indexed_files_kind'
        )
    )
    
    # Create index on filepath for faster lookups
    op.create_index('ix_indexed_files_filepath', 'indexed_files', ['filepath'])
    
    # Create index on kind for filtering
    op.create_index('ix_indexed_files_kind', 'indexed_files', ['kind'])
    
    # Create index on last_modified for change detection
    op.create_index('ix_indexed_files_last_modified', 'indexed_files', ['last_modified'])


def downgrade() -> None:
    """Remove indexed_files table."""
    op.drop_index('ix_indexed_files_last_modified', table_name='indexed_files')
    op.drop_index('ix_indexed_files_kind', table_name='indexed_files')
    op.drop_index('ix_indexed_files_filepath', table_name='indexed_files')
    op.drop_table('indexed_files')