from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer, clean_queries
from app.schemas.thread import QueryTransformationConfig


class MultiQueryExpander(BaseQueryTransformer):
    """
    Multi-Query / Query Expansion.
    Generates multiple diverse search queries from different perspectives
    to overcome retrieval vocabulary mismatch and improve recall.
    """

    DEFAULT_PROMPT = (
        "You are an AI assistant designed to optimize information retrieval. "
        "Given the user question, generate {count} diverse search queries from different perspectives "
        "and using alternative phrasing/terminology to retrieve relevant documents.\n"
        "Return one search query per line without any numbers, bullets, or explanations:\n\n"
        "Question: {query}"
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

        count = max(1, min(config.query_count, 10))

        if config.prompt_template:
            prompt = config.prompt_template.format(query=stripped, count=count)
        else:
            prompt = self.DEFAULT_PROMPT.format(query=stripped, count=count)

        raw_response = await provider.ainvoke(prompt)
        expanded = clean_queries(raw_response, max_count=count, fallback_query=stripped)

        if config.include_original:
            all_queries = [stripped] + [q for q in expanded if q.lower() != stripped.lower()]
            return list(dict.fromkeys(all_queries))[: count + 1]

        return expanded or [stripped]
