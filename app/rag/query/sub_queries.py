from typing import Any, List

from langchain_core.prompts import PromptTemplate

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer, clean_queries, execute_prompt_chain
from app.schemas.thread import QueryTransformationConfig


class SubQueryDecomposer(BaseQueryTransformer):
    """
    Sub-Query Decomposition.
    Decomposes a complex, multi-faceted, or comparative question into distinct,
    simpler sub-questions that can each be searched independently.
    """

    DEFAULT_PROMPT = PromptTemplate.from_template(
        "Decompose the following complex user question into {count} distinct, simple, and independent "
        "sub-questions needed to comprehensively answer the original question.\n"
        "Return one sub-question per line without any numbers, bullet points, or explanations:\n\n"
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
            prompt = PromptTemplate.from_template(config.prompt_template)
        else:
            prompt = self.DEFAULT_PROMPT

        inputs = {"query": stripped, "count": count}
        raw_response = await execute_prompt_chain(prompt, inputs, provider)
        sub_questions = clean_queries(raw_response, max_count=count, fallback_query=stripped)

        if config.include_original:
            all_queries = [stripped] + [q for q in sub_questions if q.lower() != stripped.lower()]
            return list(dict.fromkeys(all_queries))[: count + 1]

        return sub_questions or [stripped]

