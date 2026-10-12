from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.processor import query_transformation_processor
from app.schemas.thread import QueryTransformationConfig


async def transform_query(
    query: str,
    provider: BaseGenerationProvider,
    config: QueryTransformationConfig | dict | None = None,
    memory: Any = None,
) -> List[str]:
    """
    Transforms the query into one or more retrieval queries using the configured technique.
    Delegates to the modular QueryTransformationProcessor.
    """
    return await query_transformation_processor.process(
        query=query,
        provider=provider,
        config=config,
        memory=memory,
    )
