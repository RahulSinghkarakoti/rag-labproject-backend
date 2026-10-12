# app/models/thread.py

import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    ForeignKey,
    DateTime,
    Index,Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class Thread(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "threads"

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
    )

    title: Mapped[str | None] = mapped_column(String(500))

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    pipeline_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "pipelines.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    pipeline_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "pipeline_versions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    project = relationship(
        "Project",
        back_populates="threads",
    )

    messages = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Message.sequence",
    )

    runs = relationship(
        "Run",
        back_populates="thread",
        cascade="all, delete-orphan",
    )

    summaries = relationship(
        "ThreadSummary",
        back_populates="thread",
        cascade="all, delete-orphan",
    )

    knowledge_bases = relationship(
        "KnowledgeBase",
        secondary="thread_knowledge_bases",
        back_populates="threads",
    )

    __table_args__ = (
        Index(
            "ix_threads_user_updated",
            "user_id",
            "updated_at",
        ),
    )

    
class ThreadSummary(Base, UUIDMixin):
    __tablename__ = "thread_summaries"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("threads.id", ondelete="CASCADE"),
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    covered_until_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True)
    )

    token_count: Mapped[int | None]

    thread = relationship(
        "Thread",
        back_populates="summaries",
    )