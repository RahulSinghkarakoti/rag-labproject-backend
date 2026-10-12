from typing import Any, List

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer
from app.schemas.thread import QueryTransformationConfig


class PassThroughTransformer(BaseQueryTransformer):
    """
    Direct pass-through: performs no transformation and returns the original query.
    """

    async def transform(
        self,
        query: str,
        provider: BaseGenerationProvider,
        config: QueryTransformationConfig,
        memory: Any = None,
    ) -> List[str]:
        stripped = query.strip()
        return [stripped] if stripped else []
