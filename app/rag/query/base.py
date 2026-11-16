import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from app.rag.generation.base import BaseGenerationProvider
from app.schemas.thread import QueryTransformationConfig

logger = logging.getLogger(__name__)


def clean_queries(raw_text: str, max_count: int = 5, fallback_query: str = "") -> list[str]:
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
        run_manager: Any = None,
    ) -> list[str]:
        """
        Transforms the input query into one or more retrieval query strings.
        """
        raise NotImplementedError


async def execute_prompt_chain(
    prompt: Any,
    inputs: dict[str, Any],
    provider: BaseGenerationProvider,
    run_manager: Any = None,
    stage_info: dict[str, Any] | None = None,
) -> str:
    """
    Executes a prompt using a LangChain LCEL chain (prompt | model | StrOutputParser)
    when the provider exposes a LangChain BaseChatModel, or falls back to provider.ainvoke.
    Records model telemetry to run_manager if provided.
    """
    import time
    from app.rag.memory.context import estimate_context_tokens

    formatted = prompt.format(**inputs) if hasattr(prompt, "format") else str(prompt)
    start = time.monotonic()
    raw_response = None
    error = None

    try:
        model = getattr(provider, "model", None)
        if model is not None and hasattr(model, "ainvoke"):
            try:
                from langchain_core.output_parsers import StrOutputParser

                chain = prompt | model | StrOutputParser()
                raw_response = await chain.ainvoke(inputs)
            except Exception:
                raw_response = await provider.ainvoke_safe(formatted)
        else:
            raw_response = await provider.ainvoke(formatted)
        return raw_response
    except Exception as exc:
        error = exc
        raise
    finally:
        if run_manager and hasattr(run_manager, "record_model_call"):
            elapsed_ms = int((time.monotonic() - start) * 1000)
            in_tokens = estimate_context_tokens(formatted)
            out_tokens = estimate_context_tokens(raw_response or "") if raw_response else 0
            provider_name = getattr(provider, "provider_name", "unknown")
            model_name = getattr(getattr(provider, "config", None), "model", "unknown")
            req_data = {
                "stage": "query_transformation",
                "prompt": formatted,
                "inputs": inputs,
                **(stage_info or {}),
            }
            try:
                if error:
                    await run_manager.record_model_call(
                        provider=provider_name,
                        model=model_name,
                        request=req_data,
                        response={"content": raw_response} if raw_response else None,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=in_tokens + out_tokens,
                        latency_ms=elapsed_ms,
                        status="failed",
                        error={"message": str(error)},
                    )
                else:
                    await run_manager.record_model_call(
                        provider=provider_name,
                        model=model_name,
                        request=req_data,
                        response={"content": raw_response},
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=in_tokens + out_tokens,
                        latency_ms=elapsed_ms,
                        status="completed",
                    )
            except Exception as rec_exc:
                logger.warning("[execute_prompt_chain] Failed to record model call: %s", rec_exc)

