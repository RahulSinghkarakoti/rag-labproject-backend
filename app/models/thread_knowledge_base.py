# app/models/thread_knowledge_base.py

from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


thread_knowledge_bases = Table(
    "thread_knowledge_bases",
    Base.metadata,

    Column(
        "thread_id",
        UUID(as_uuid=True),
        ForeignKey("threads.id", ondelete="CASCADE"),
        primary_key=True,
    ),

    Column(
        "knowledge_base_id",
        UUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
)