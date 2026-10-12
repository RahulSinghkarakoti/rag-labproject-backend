from typing import Any, Dict, List, Type

from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import BaseQueryTransformer
from app.rag.query.expander import MultiQueryExpander
from app.rag.query.hyde import HyDETransformer
from app.rag.query.passthrough import PassThroughTransformer
from app.rag.query.rewriter import QueryRewriter
from app.rag.query.step_back import StepBackTransformer
from app.rag.query.sub_queries import SubQueryDecomposer
from app.schemas.thread import QueryTransformationConfig


class QueryTransformationProcessor:
    """
    Central coordinator for query transformations.
    Resolves the appropriate strategy based on the QueryTransformationConfig mode.
    """

    def __init__(self):
        self._transformers: Dict[str, BaseQueryTransformer] = {
            "none": PassThroughTransformer(),
            "rewrite": QueryRewriter(),
            "hyde": HyDETransformer(),
            "multi_query": MultiQueryExpander(),
            "step_back": StepBackTransformer(),
            "sub_queries": SubQueryDecomposer(),
        }

    def register(self, mode: str, transformer: BaseQueryTransformer) -> None:
        """Register a custom or plugin query transformer."""
        self._transformers[mode.lower()] = transformer

    def get_transformer(self, mode: str) -> BaseQueryTransformer:
        """Retrieve the transformer instance for the specified mode."""
        transformer = self._transformers.get(mode.lower())
        if not transformer:
            valid_modes = ", ".join(self._transformers.keys())
            raise ValueError(f"Unsupported query transformation mode: '{mode}'. Valid modes: {valid_modes}")
        return transformer

    async def process(
        self,
        query: str,
        provider: BaseGenerationProvider,
        config: QueryTransformationConfig | dict | None = None,
        memory: Any = None,
    ) -> List[str]:
        """
        Executes query transformation using the configured technique.
        Returns a list of search query strings.
        """
        stripped_query = query.strip()
        if not stripped_query:
            return []

        # Normalize config to QueryTransformationConfig
        if config is None:
            norm_config = QueryTransformationConfig()
        elif isinstance(config, dict):
            norm_config = QueryTransformationConfig.model_validate(config)
        elif isinstance(config, QueryTransformationConfig):
            norm_config = config
        else:
            norm_config = QueryTransformationConfig()

        mode = norm_config.mode.lower()
        transformer = self.get_transformer(mode)

        transformed = await transformer.transform(
            query=stripped_query,
            provider=provider,
            config=norm_config,
            memory=memory,
        )

        return transformed or [stripped_query]


# Default singleton instance
query_transformation_processor = QueryTransformationProcessor()
