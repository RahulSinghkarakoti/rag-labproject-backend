from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer, clean_queries
from app.schemas.thread import QueryTransformationConfig


class StepBackTransformer(BaseQueryTransformer):
    """
    Step-Back Prompting (Zheng et al., 2023).
    Generates a broader, higher-level conceptual question to retrieve foundational
    principles, concepts, or background context.
    """

    DEFAULT_PROMPT = (
        "You are an expert at step-back prompting. Given the following specific user question, "
        "generate a broader, higher-level conceptual question that steps back to retrieve "
        "the foundational principles, high-level background concepts, or definitions needed to answer it.\n"
        "Do NOT answer the question. Do NOT include preambles or explanations.\n"
        "Return ONLY the single step-back question on one line:\n\n"
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

        if config.prompt_template:
            prompt = config.prompt_template.format(query=stripped)
        else:
            prompt = self.DEFAULT_PROMPT.format(query=stripped)

        raw_response = await provider.ainvoke(prompt)
        step_back_list = clean_queries(raw_response, max_count=1, fallback_query=stripped)
        step_back_query = step_back_list[0] if step_back_list else stripped

        if config.include_original and stripped.lower() != step_back_query.lower():
            return [step_back_query, stripped]

        return [step_back_query]
