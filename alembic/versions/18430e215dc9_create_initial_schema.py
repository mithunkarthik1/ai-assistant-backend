"""create_initial_schema

Revision ID: 18430e215dc9
Revises: 
Create Date: 2026-10-06 09:55:43.801628

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '18430e215dc9'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema idempotently."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = set(inspector.get_table_names())

    # 1. Documents Table
    if 'documents' not in existing_tables:
        op.create_table(
            'documents',
            sa.Column('document_id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('file_name', sa.String(255), nullable=False),
            sa.Column('file_hash', sa.String(64), nullable=False),
            sa.Column('file_type', sa.String(50), nullable=False, server_default='pdf'),
            sa.Column('status', sa.String(50), nullable=False, server_default='UPLOADED'),
            sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index('ix_documents_document_id', 'documents', ['document_id'])
        op.create_index('ix_documents_file_name', 'documents', ['file_name'])
        op.create_index('ix_documents_file_hash', 'documents', ['file_hash'])
    else:
        doc_cols = {c['name'] for c in inspector.get_columns('documents')}
        if 'id' in doc_cols and 'document_id' not in doc_cols:
            op.alter_column('documents', 'id', new_column_name='document_id')
        if 'filename' in doc_cols and 'file_name' not in doc_cols:
            op.alter_column('documents', 'filename', new_column_name='file_name')
        if 'file_hash' not in doc_cols:
            op.add_column('documents', sa.Column('file_hash', sa.String(64), nullable=False, server_default=''))
        if 'chunk_count' not in doc_cols:
            op.add_column('documents', sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'))
        if 'updated_at' not in doc_cols:
            op.add_column('documents', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()))
        if 'file_size' in doc_cols:
            op.drop_column('documents', 'file_size')

    # 2. Document Chunks Table
    if 'document_chunks' not in existing_tables:
        op.create_table(
            'document_chunks',
            sa.Column('chunk_id', sa.String(255), primary_key=True),
            sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.document_id', ondelete='CASCADE'), nullable=False),
            sa.Column('section', sa.String(255), nullable=True),
            sa.Column('topic', sa.String(255), nullable=True),
            sa.Column('chunk_index', sa.Integer(), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('content_hash', sa.String(64), nullable=False),
            sa.Column('page_number', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index('ix_document_chunks_chunk_id', 'document_chunks', ['chunk_id'])
        op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])
        op.create_index('ix_document_chunks_content_hash', 'document_chunks', ['content_hash'])

    # 3. Chat Sessions Table
    if 'chat_sessions' not in existing_tables:
        op.create_table(
            'chat_sessions',
            sa.Column('id', sa.String(255), primary_key=True),
            sa.Column('title', sa.String(255), nullable=False, server_default='New Chat'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index('ix_chat_sessions_id', 'chat_sessions', ['id'])

    # 4. Chat Messages Table
    if 'chat_messages' not in existing_tables:
        op.create_table(
            'chat_messages',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.document_id', ondelete='SET NULL'), nullable=True),
            sa.Column('session_id', sa.String(255), nullable=True),
            sa.Column('role', sa.String(50), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index('ix_chat_messages_session_id', 'chat_messages', ['session_id'])
    else:
        msg_cols = {c['name'] for c in inspector.get_columns('chat_messages')}
        if 'session_id' not in msg_cols:
            op.add_column('chat_messages', sa.Column('session_id', sa.String(255), nullable=True))
            op.create_index('ix_chat_messages_session_id', 'chat_messages', ['session_id'])

    # Drop legacy langchain tables if present
    if 'langchain_pg_embedding' in existing_tables:
        op.drop_table('langchain_pg_embedding')
    if 'langchain_pg_collection' in existing_tables:
        op.drop_table('langchain_pg_collection')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    op.drop_table('document_chunks')
    op.drop_table('documents')
