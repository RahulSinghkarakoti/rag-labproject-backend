"""
Conversation memory management for the RAG query pipeline.

Loads prior thread messages from the database so the pipeline
can use chat history as context during query transformation and generation.
"""
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Loads and manages conversation history for a RAG query pipeline run.

    Reads prior messages from the database in chronological order so that
    query transformers (e.g. QueryRewriter) and the generation prompt can
    resolve pronouns and maintain conversational continuity.
    """

    def __init__(self, db: AsyncSession, max_turns: int = 50) -> None:
        """
        Args:
            db:         Async SQLAlchemy session.
            max_turns:  Maximum number of prior messages to return.
                        Guards against unbounded context growth.
        """
        self.db = db
        self.max_turns = max_turns

    async def load(
        self,
        thread_id: Any,
        before_sequence: int | None,
    ) -> list[dict[str, Any]]:
        """
        Load prior conversation messages for *thread_id* that appear before
        *before_sequence* (the current user message's sequence number).

        Returns a list of ``{"role": ..., "content": ...}`` dicts in
        chronological order, or an empty list when memory cannot be loaded.
        """
        if not thread_id:
            return []

        try:
            stmt = (
                select(Message)
                .where(
                    Message.thread_id == thread_id,
                    Message.sequence < (before_sequence or 1_000_000),
                )
                .order_by(Message.sequence.asc())
                .limit(self.max_turns)
            )
            result = await self.db.scalars(stmt)
            messages = result.all()
            return [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.content
            ]
        except Exception:
            logger.warning(
                "Failed to load conversation memory for thread %s",
                thread_id,
                exc_info=True,
            )
            return []
