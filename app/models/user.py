from datetime import datetime
from typing import List

from sqlalchemy import Boolean, DateTime, String, func ,Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,)

    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        index=True,
        nullable=False,
    )

    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    avatar_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    oauth_accounts: Mapped[List["OAuthAccount"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    auth_logs: Mapped[List["AuthLog"]] = relationship(
        back_populates="user",
    )