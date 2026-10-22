# app/models/run.py

import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    ForeignKey,
    DateTime,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin


class Run(Base, UUIDMixin):
    __tablename__ = "runs"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    input_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
    )

    output_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
    )

    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
    )

    run_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="running",
        nullable=False,
    )

    pipeline_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "pipeline_versions.id",
            ondelete="SET NULL",
        ),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    error: Mapped[dict | None] = mapped_column(JSONB)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    thread = relationship(
        "Thread",
        back_populates="runs",
    )

    input_message = relationship(
        "Message",
        foreign_keys=[input_message_id],
        back_populates="input_runs",
    )

    output_message = relationship(
        "Message",
        foreign_keys=[output_message_id],
        back_populates="output_runs",
    )

    steps = relationship(
        "RunStep",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    model_calls = relationship(
        "ModelCall",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    tool_calls = relationship(
        "ToolCall",
        back_populates="run",
        cascade="all, delete-orphan",
    )

class RunStep(Base, UUIDMixin):
    __tablename__ = "run_steps"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    parent_step_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("run_steps.id", ondelete="SET NULL"),
    )

    step_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    sequence: Mapped[int] = mapped_column(
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
        nullable=False,
    )

    input: Mapped[dict | None] = mapped_column(JSONB)

    output: Mapped[dict | None] = mapped_column(JSONB)

    config: Mapped[dict | None] = mapped_column(JSONB)

    error: Mapped[dict | None] = mapped_column(JSONB)

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    run = relationship(
        "Run",
        back_populates="steps",
    )

class ModelCall(Base, UUIDMixin):
    __tablename__ = "model_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    request: Mapped[dict | None] = mapped_column(JSONB)

    response: Mapped[dict | None] = mapped_column(JSONB)

    input_tokens: Mapped[int | None]

    output_tokens: Mapped[int | None]

    total_tokens: Mapped[int | None]

    latency_ms: Mapped[int | None]

    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
    )

    error: Mapped[dict | None] = mapped_column(JSONB)

    run = relationship(
        "Run",
        back_populates="model_calls",
    )

class ToolCall(Base, UUIDMixin):
    __tablename__ = "tool_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
    )

    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    call_id: Mapped[str | None] = mapped_column(
        String(255),
        index=True,
    )

    arguments: Mapped[dict | None] = mapped_column(JSONB)

    result: Mapped[dict | None] = mapped_column(JSONB)

    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
    )

    error: Mapped[dict | None] = mapped_column(JSONB)

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    run = relationship(
        "Run",
        back_populates="tool_calls",
    )

class RetrievalRun(Base, UUIDMixin):
    __tablename__ = "retrieval_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    retriever_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    top_k: Mapped[int | None]

    results = relationship(
        "RetrievalResult",
        back_populates="retrieval_run",
        cascade="all, delete-orphan",
    )


class RetrievalResult(Base, UUIDMixin):
    __tablename__ = "retrieval_results"

    retrieval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_chunks.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    rank: Mapped[int | None]

    score: Mapped[float | None]

    rerank_score: Mapped[float | None]

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    retrieval_run = relationship(
        "RetrievalRun",
        back_populates="results",
    )
 