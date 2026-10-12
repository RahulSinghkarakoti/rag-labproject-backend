# app/models/message.py

import uuid

from sqlalchemy import (
    String,
    Text,
    ForeignKey,
    BigInteger,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDMixin


class Message(Base, UUIDMixin):
    __tablename__ = "messages"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    parent_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    role: Mapped[str] = mapped_column(
        String(30), # human assistant system developer tool
        nullable=False,
    )

    content: Mapped[str | None] = mapped_column(Text)

    content_json: Mapped[dict | list | None] = mapped_column(
        JSONB
    )

    sequence: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    thread = relationship(
        "Thread",
        back_populates="messages",
    )

    parent = relationship(
        "Message",
        remote_side="Message.id",
    )

    input_runs = relationship(
        "Run",
        foreign_keys="Run.input_message_id",
        back_populates="input_message",
    )

    output_runs = relationship(
        "Run",
        foreign_keys="Run.output_message_id",
        back_populates="output_message",
    )