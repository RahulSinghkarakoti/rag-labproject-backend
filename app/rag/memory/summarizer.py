"""
Thread conversation summarizer for rolling memory compaction.

Compacts older conversation turns into a progressive summary using
the generation model selected by the user in thread configs (or .env override).
"""
import logging
from typing import Any

from app.core.config import settings
from app.rag.generation.base import BaseGenerationProvider

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT_TEMPLATE = """You are an expert conversation memory summarizer for an AI assistant.
Your task is to update the running summary of an ongoing conversation using the newly completed dialogue turns.

Guidelines:
1. Synthesize user goals, questions asked, and key constraints.
2. Note important facts, data points, or decisions established in the dialogue.
3. Incorporate new dialogue turns cleanly into the existing summary.
4. Keep the summary dense, factual, objective, and under {max_tokens} words.
5. Do NOT include pleasantries, greetings, or conversational filler.

{existing_summary_section}

New dialogue turns to incorporate:
{dialogue_turns}

Updated Conversation Summary:"""


def resolve_summarizer_provider(
    default_provider: BaseGenerationProvider | None = None,
) -> BaseGenerationProvider | None:
    """
    Resolve the provider to use for summarization.
    Prefers .env overrides (THREAD_SUMMARY_PROVIDER, THREAD_SUMMARY_MODEL) if set,
    otherwise uses the thread's selected generation provider.
    """
    if settings.THREAD_SUMMARY_PROVIDER and settings.THREAD_SUMMARY_MODEL:
        try:
            from app.rag.generation.factory import provider_model_factory
            from app.schemas.thread import GenerationConfig

            gen_config = GenerationConfig(
                provider=settings.THREAD_SUMMARY_PROVIDER,
                llm_model=settings.THREAD_SUMMARY_MODEL,
            )
            return provider_model_factory.registry.get(
                settings.THREAD_SUMMARY_PROVIDER, gen_config
            )
        except Exception as exc:
            logger.warning(
                "[ThreadSummarizer] Failed to create .env summary provider override: %s",
                exc,
            )
    return default_provider


class ThreadSummarizer:
    """Progressive conversation summarizer for rolling memory compaction."""

    def __init__(self, provider: BaseGenerationProvider) -> None:
        self.provider = provider
        self.max_tokens = settings.THREAD_SUMMARY_MAX_TOKENS

    async def summarize_turns(
        self,
        existing_summary: str | None,
        turns: list[dict[str, Any]],
    ) -> str:
        """
        Generate an updated running summary by combining any existing summary
        with the specified older turns.
        """
        if not turns:
            return existing_summary or ""

        dialogue_text = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in turns
            if turn.get("content")
        )
        if not dialogue_text.strip():
            return existing_summary or ""

        existing_section = (
            f"Existing conversation summary:\n{existing_summary.strip()}"
            if existing_summary and existing_summary.strip()
            else "Existing conversation summary: None (start of conversation)."
        )

        prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
            max_tokens=self.max_tokens,
            existing_summary_section=existing_section,
            dialogue_turns=dialogue_text,
        )

        try:
            summary = await self.provider.ainvoke(prompt)
            clean_summary = summary.strip()
            logger.info(
                "[ThreadSummarizer] Successfully generated rolling summary (%d chars)",
                len(clean_summary),
            )
            return clean_summary
        except Exception:
            logger.warning(
                "[ThreadSummarizer] Failed to summarize dialogue turns, keeping existing summary",
                exc_info=True,
            )
            return existing_summary or ""
