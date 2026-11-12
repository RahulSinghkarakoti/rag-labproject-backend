"""
Conversation memory management for the RAG query pipeline.

Loads prior thread messages from the database so the pipeline
can use chat history as context during query transformation and generation.
"""
import logging
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.message import Message
from app.models.thread import Thread, ThreadSummary
from app.rag.generation.base import BaseGenerationProvider
from app.rag.memory.context import estimate_context_tokens
from app.rag.memory.summarizer import ThreadSummarizer, resolve_summarizer_provider

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
        before_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load the most recent prior conversation messages for *thread_id* that
        appear before *before_sequence*, returned in chronological order.
        """
        if not thread_id:
            return []

        fetch_limit = limit or self.max_turns
        try:
            stmt = (
                select(Message)
                .where(
                    Message.thread_id == thread_id,
                    Message.sequence < (before_sequence or 1_000_000),
                )
                .order_by(Message.sequence.desc())
                .limit(fetch_limit)
            )
            result = await self.db.scalars(stmt)
            messages = list(result.all())
            messages.reverse()  # Restore chronological order for recent turns
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

    async def get_conversation_context(
        self,
        thread: Thread | None,
        before_sequence: int | None = None,
        provider: BaseGenerationProvider | None = None,
        recent_window_turns: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        """
        Load conversation context applying progressive compaction.

        Keeps the latest `recent_window_turns` (default from settings) verbatim,
        and compacts older dialogue turns into a rolling summary stored in
        `ThreadSummary` records.

        Returns:
            (recent_memory_turns, conversation_summary, is_summarized)
        """
        if not thread:
            return [], None, False

        window_turns = recent_window_turns or settings.THREAD_SUMMARY_RECENT_TURNS
        trigger_threshold = settings.THREAD_SUMMARY_TRIGGER_THRESHOLD

        try:
            # 1. Fetch all messages up to before_sequence
            stmt = (
                select(Message)
                .where(
                    Message.thread_id == thread.id,
                    Message.sequence < (before_sequence or 1_000_000),
                )
                .order_by(Message.sequence.asc())
            )
            res = await self.db.scalars(stmt)
            all_messages = list(res.all())

            msg_seq_map = {
                m.id: (m.sequence or 0)
                for m in all_messages
                if getattr(m, "id", None) is not None
            }

            # 2. Fetch latest ThreadSummary record for this thread
            existing_summary: str = ""
            summarized_up_to: int = 0

            summary_stmt = (
                select(ThreadSummary, Message.sequence)
                .outerjoin(Message, ThreadSummary.covered_until_message_id == Message.id)
                .where(ThreadSummary.thread_id == thread.id)
                .order_by(Message.sequence.desc().nullslast(), ThreadSummary.id.desc())
                .limit(1)
            )
            try:
                exec_res = await self.db.execute(summary_stmt)
                first_row = exec_res.first()
                if first_row and isinstance(first_row[0], ThreadSummary):
                    latest_summary = first_row[0]
                    existing_summary = latest_summary.summary or ""
                    covered_id = latest_summary.covered_until_message_id
                    summarized_up_to = (
                        msg_seq_map.get(covered_id)
                        if covered_id and covered_id in msg_seq_map
                        else (first_row[1] or 0)
                    )
            except Exception:
                pass

            # Fallback check for legacy metadata during transition
            if not existing_summary and thread.metadata_ and isinstance(thread.metadata_, dict):
                existing_summary = thread.metadata_.get("conversation_summary", "")
                summarized_up_to = thread.metadata_.get("summarized_up_to_sequence", 0)

            # If total messages fit within sliding window, return verbatim
            if len(all_messages) <= window_turns:
                recent_turns = [
                    {"role": m.role, "content": m.content}
                    for m in all_messages
                    if m.content
                ]
                return (
                    recent_turns,
                    existing_summary if existing_summary else None,
                    bool(existing_summary),
                )

            # Sliding window: keep the latest `window_turns` verbatim
            recent_messages = all_messages[-window_turns:]
            older_messages = all_messages[:-window_turns]

            # Find unsummarized older messages
            unsummarized_older = [
                m
                for m in older_messages
                if (m.sequence or 0) > summarized_up_to and m.content
            ]

            # Check if compaction should trigger
            active_provider = resolve_summarizer_provider(provider)
            if len(unsummarized_older) >= trigger_threshold and active_provider:
                logger.info(
                    "[MemoryManager] Compacting %d older turns for thread %s",
                    len(unsummarized_older),
                    thread.id,
                )
                summarizer = ThreadSummarizer(active_provider)
                turns_to_compact = [
                    {"role": m.role, "content": m.content}
                    for m in unsummarized_older
                ]
                updated_summary = await summarizer.summarize_turns(
                    existing_summary, turns_to_compact
                )

                if updated_summary:
                    last_covered = unsummarized_older[-1]
                    last_covered_id = getattr(last_covered, "id", None)
                    if not last_covered_id:
                        last_covered_id = uuid4()
                        last_covered.id = last_covered_id

                    summary_record = ThreadSummary(
                        thread_id=thread.id,
                        summary=updated_summary,
                        covered_until_message_id=last_covered_id,
                        token_count=estimate_context_tokens(updated_summary),
                    )
                    self.db.add(summary_record)
                    try:
                        await self.db.commit()
                        existing_summary = updated_summary
                    except Exception:
                        logger.warning(
                            "[MemoryManager] Failed to persist ThreadSummary for thread %s",
                            thread.id,
                            exc_info=True,
                        )
                        if getattr(self.db, "is_active", False):
                            await self.db.rollback()

            recent_turns = [
                {"role": m.role, "content": m.content}
                for m in recent_messages
                if m.content
            ]
            return (
                recent_turns,
                existing_summary if existing_summary else None,
                bool(existing_summary),
            )
        except Exception:
            logger.warning(
                "Failed to get conversation context with summary for thread %s, falling back to load()",
                thread.id,
                exc_info=True,
            )
            fallback_turns = await self.load(thread.id, before_sequence, limit=window_turns)
            return fallback_turns, None, False

