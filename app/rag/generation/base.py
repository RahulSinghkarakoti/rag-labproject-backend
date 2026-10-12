from abc import ABC, abstractmethod
from typing import Any


class BaseGenerationProvider(ABC):
    """Shared provider interface used by every RAG pipeline stage."""

    provider_name: str = "base"

    def __init__(self, config: Any):
        self.config = config
        self.model = self._build_model(config)

    @abstractmethod
    def _build_model(self, config: Any):
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        raise NotImplementedError

    async def astream(self, prompt: str):
        """Yield text chunks without changing existing provider interfaces."""
        try:
            if not hasattr(self.model, "astream"):
                yield await self.ainvoke(prompt)
                return
            async for chunk in self.model.astream(prompt):
                text = str(getattr(chunk, "content", chunk))
                if text.strip():
                    yield text
        except Exception as exc:
            from app.core.exceptions import AppException, sanitize_model_error
            if isinstance(exc, AppException):
                raise
            raise sanitize_model_error(
                exc,
                provider=self.provider_name,
                model=getattr(self.config, "model", "unknown"),
            ) from exc

    @staticmethod
    def content(response: Any) -> str:
        return str(getattr(response, "content", response)).strip()

    async def ainvoke_safe(self, prompt: str) -> str:
        try:
            return await self.ainvoke(prompt)
        except Exception as exc:
            from app.core.exceptions import AppException, sanitize_model_error
            if isinstance(exc, AppException):
                raise
            provider_name = getattr(self, "provider_name", "unknown")
            model_name = getattr(self.config, "model", "unknown")
            raise sanitize_model_error(exc, provider=provider_name, model=model_name) from exc


