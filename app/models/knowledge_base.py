# app/models/knowledge_base.py

import uuid

from sqlalchemy import String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class KnowledgeBase(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "knowledge_bases"

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(Text)

    embedding_config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    chunking_config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    indexing_config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    project = relationship(
        "Project",
        back_populates="knowledge_bases",
    )

    documents = relationship(
        "Document",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )

    threads = relationship(
        "Thread",
        secondary="thread_knowledge_bases",
        back_populates="knowledge_bases",
    )