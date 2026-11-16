"""cascade_thread_project_and_index_messages

Revision ID: 74415a46bb0c
Revises: beda827dcffe
Create Date: 2026-11-16 14:03:18.550870

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74415a46bb0c'
down_revision: Union[str, Sequence[str], None] = 'beda827dcffe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index('ix_messages_thread_sequence', 'messages', ['thread_id', 'sequence'], unique=False)
    op.drop_constraint('threads_project_id_fkey', 'threads', type_='foreignkey')
    op.create_foreign_key('threads_project_id_fkey', 'threads', 'projects', ['project_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('threads_project_id_fkey', 'threads', type_='foreignkey')
    op.create_foreign_key('threads_project_id_fkey', 'threads', 'projects', ['project_id'], ['id'], ondelete='SET NULL')
    op.drop_index('ix_messages_thread_sequence', table_name='messages')
