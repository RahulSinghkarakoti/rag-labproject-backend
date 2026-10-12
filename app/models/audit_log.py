from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, JSON, String, func,Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base  import Base

if TYPE_CHECKING:
    from app.models.user import User


class AuthLog(Base):
    __tablename__ = "auth_logs"

    id: Mapped[int] = mapped_column(primary_key=True)

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    event: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    provider: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    log_metadata: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    user: Mapped["User"] = relationship(
        back_populates="auth_logs",
    )