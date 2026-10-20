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

    @classmethod
    def _extract_text_content(cls, content: Any) -> str:
        """Extract clean text string from strings, lists, or structured dict blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_val = item.get("text")
                    if text_val and isinstance(text_val, str):
                        parts.append(text_val)
                elif hasattr(item, "text"):
                    text_val = getattr(item, "text", "")
                    if text_val and isinstance(text_val, str):
                        parts.append(text_val)
            return "".join(parts)
        if isinstance(content, dict):
            text_val = content.get("text")
            if text_val and isinstance(text_val, str):
                return text_val
        return str(content)

    async def astream(self, prompt: str):
        """Yield text chunks without changing existing provider interfaces."""
        try:
            if not hasattr(self.model, "astream"):
                yield await self.ainvoke(prompt)
                return
            async for chunk in self.model.astream(prompt):
                raw_content = getattr(chunk, "content", chunk)
                text = self._extract_text_content(raw_content)
                if text:
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

    @classmethod
    def content(cls, response: Any) -> str:
        raw_content = getattr(response, "content", response)
        return cls._extract_text_content(raw_content).strip()

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


