from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer, clean_queries
from app.schemas.thread import QueryTransformationConfig


class HyDETransformer(BaseQueryTransformer):
    """
    Hypothetical Document Embeddings (HyDE).
    Generates hypothetical document excerpt(s) answering the query to retrieve
    semantically aligned passages from vector embeddings.
    """

    DEFAULT_SINGLE_PROMPT = (
        "Write one concise hypothetical document excerpt that directly answers "
        "the following user question. Write authoritatively as if this text appeared "
        "inside a reference document.\n"
        "Do NOT mention that this is hypothetical. Do NOT include preambles, greetings, or explanations.\n"
        "Return ONLY the hypothetical passage:\n\n"
        "Question: {query}"
    )

    DEFAULT_MULTI_PROMPT = (
        "Write {count} distinct concise hypothetical answers or document passages that would answer "
        "the following user question from different angles. Return each hypothetical passage separated by a blank line.\n"
        "Do NOT mention that this is hypothetical. Do NOT include numbering, bullet points, or commentary:\n\n"
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

        count = max(1, min(config.query_count, 5))

        if config.prompt_template:
            prompt = config.prompt_template.format(query=stripped, count=count)
        elif count > 1:
            prompt = self.DEFAULT_MULTI_PROMPT.format(query=stripped, count=count)
        else:
            prompt = self.DEFAULT_SINGLE_PROMPT.format(query=stripped)

        raw_response = await provider.ainvoke(prompt)
        hypotheses = clean_queries(raw_response, max_count=count, fallback_query=stripped)

        results = hypotheses if hypotheses else [stripped]

        if config.include_original:
            results = list(dict.fromkeys(results + [stripped]))

        return results
