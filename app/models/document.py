# app/models/document.py

import uuid

from sqlalchemy import (
    String,
    Text,
    BigInteger,
    ForeignKey
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(100)
    )

    size_bytes: Mapped[int | None] = mapped_column(
        BigInteger
    )

    storage_path: Mapped[str | None] = mapped_column(
        Text
    )

    checksum: Mapped[str | None] = mapped_column(
        String(128),
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="processing",
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    knowledge_base = relationship(
        "KnowledgeBase",
        back_populates="documents",
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base, UUIDMixin):
    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int | None]

    page_number: Mapped[int | None]

    section: Mapped[str | None] = mapped_column(
        String(500)
    )

    embedding: Mapped[list[float]] = mapped_column(
    Vector(),
    nullable=False,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(255)
    )
    embedding_dimension: Mapped[int | None]
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    document = relationship(
        "Document",
        back_populates="chunks",
    )