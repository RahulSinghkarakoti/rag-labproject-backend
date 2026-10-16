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

    def to_langchain_messages(self, turns: list[dict[str, Any]]) -> list[Any]:
        """Convert role/content turns into LangChain BaseMessage objects."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            if role in {"user", "human"}:
                lc_messages.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
        return lc_messages

    async def load_messages(
        self,
        thread_id: Any,
        before_sequence: int | None,
    ) -> list[Any]:
        """Load prior conversation history directly as LangChain BaseMessage objects."""
        turns = await self.load(thread_id=thread_id, before_sequence=before_sequence)
        return self.to_langchain_messages(turns)

