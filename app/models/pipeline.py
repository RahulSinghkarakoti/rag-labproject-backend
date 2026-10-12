# app/models/pipeline.py

import uuid

from sqlalchemy import String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin, TimestampMixin


class Pipeline(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "pipelines"

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(Text)

    project = relationship(
        "Project",
        back_populates="pipelines",
    )

    versions = relationship(
        "PipelineVersion",
        back_populates="pipeline",
        cascade="all, delete-orphan",
    )


class PipelineVersion(Base, UUIDMixin):
    __tablename__ = "pipeline_versions"

    pipeline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "pipelines.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        nullable=False
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    pipeline = relationship(
        "Pipeline",
        back_populates="versions",
    )

    __table_args__ = (
        UniqueConstraint(
            "pipeline_id",
            "version",
            name="uq_pipeline_version",
        ),
    )