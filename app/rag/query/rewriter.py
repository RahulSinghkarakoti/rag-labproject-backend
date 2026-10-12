from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer, clean_queries
from app.schemas.thread import QueryTransformationConfig


class QueryRewriter(BaseQueryTransformer):
    """
    Rewrites the user's query into an optimized, self-contained retrieval query.
    Resolves conversational context/pronouns if chat memory is available.
    """

    DEFAULT_PROMPT = (
        "You are an expert search query optimizer. Given the following user query, "
        "rewrite it into a concise, specific, and self-contained search query for a vector "
        "and keyword search engine.\n"
        "Do NOT answer the question. Do NOT include preambles or explanations.\n"
        "Return ONLY the rewritten query on a single line:\n\n"
        "Query: {query}"
    )

    MEMORY_PROMPT = (
        "You are an expert search query optimizer. Given the chat conversation context and "
        "the latest user query, resolve all pronouns and context to produce a single, self-contained, "
        "and specific search query suitable for document retrieval.\n"
        "Do NOT answer the question. Do NOT include preambles or explanations.\n"
        "Return ONLY the rewritten search query on a single line:\n\n"
        "Conversation History:\n{history}\n\n"
        "Latest Query: {query}"
    )

    async def transform(
        self,
        query: str,
        provider: BaseGenerationProvider,
        config: QueryTransformationConfig,
        memory: Any = None,
    ) -> List[str]:
        stripped = query.strip()
        if not stripped:
            return []

        # Build prompt
        if config.prompt_template:
            prompt = config.prompt_template.format(query=stripped)
        elif memory and isinstance(memory, list) and len(memory) > 0:
            history_str = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in memory
                if isinstance(m, dict) and m.get("content")
            )
            if history_str:
                prompt = self.MEMORY_PROMPT.format(history=history_str, query=stripped)
            else:
                prompt = self.DEFAULT_PROMPT.format(query=stripped)
        else:
            prompt = self.DEFAULT_PROMPT.format(query=stripped)

        raw_response = await provider.ainvoke(prompt)
        rewritten_list = clean_queries(raw_response, max_count=1, fallback_query=stripped)
        rewritten = rewritten_list[0] if rewritten_list else stripped

        if config.include_original and stripped.lower() != rewritten.lower():
            return [rewritten, stripped]
        return [rewritten]
