import re
from abc import ABC, abstractmethod
from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.schemas.thread import QueryTransformationConfig


def clean_queries(raw_text: str, max_count: int = 5, fallback_query: str = "") -> List[str]:
    """
    Cleans raw LLM response text into a list of clean, distinct search queries.
    Strips numbering, bullets, quotes, and conversational preamble.
    """
    if not raw_text or not raw_text.strip():
        return [fallback_query.strip()] if fallback_query.strip() else []

    lines = raw_text.strip().splitlines()
    cleaned: list[str] = []

    # Patterns to ignore conversational boilerplate lines
    ignore_prefixes = (
        "here are",
        "here is",
        "sure,",
        "certainly",
        "rewritten query:",
        "alternative queries:",
        "hypothetical passage:",
        "hypothetical document:",
        "step-back query:",
        "sub-queries:",
    )

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        lower = line_str.lower()
        if any(lower.startswith(prefix) for prefix in ignore_prefixes) and (
            line_str.endswith(":") or len(line_str.split()) < 6
        ):
            continue

        # Strip list markers like "1. ", "1) ", "- ", "* ", "(1) "
        stripped = re.sub(r"^(\(?\d+[\.\)]|\*|-|\+)\s*", "", line_str)
        # Strip surrounding markdown quotes or backticks
        stripped = stripped.strip(" `\"'")

        if stripped and len(stripped) > 2:
            cleaned.append(stripped)

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(cleaned))

    if not deduped and fallback_query.strip():
        deduped = [fallback_query.strip()]

    return deduped[:max_count]


class BaseQueryTransformer(ABC):
    """Abstract base class for all query transformation techniques."""

    @abstractmethod
    async def transform(
        self,
        query: str,
        provider: BaseGenerationProvider,
        config: QueryTransformationConfig,
        memory: Any = None,
    ) -> List[str]:
        """
        Transforms the input query into one or more retrieval query strings.
        """
        raise NotImplementedError
