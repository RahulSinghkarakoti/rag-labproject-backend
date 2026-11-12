# app/models/project.py


from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDMixin


class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    description: Mapped[str | None] = mapped_column(Text)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    threads = relationship(
        "Thread",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    knowledge_bases = relationship(
        "KnowledgeBase",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    pipelines = relationship(
        "Pipeline",
        back_populates="project",
        cascade="all, delete-orphan",
    )
