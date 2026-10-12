from app.rag.generation.base import BaseGenerationProvider 

class GroqProvider(BaseGenerationProvider):
    provider_name = "groq"

    def _build_model(self, config):
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "Groq provider requires the langchain-groq package"
            ) from exc

        return ChatGroq(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    async def ainvoke(self, prompt: str) -> str:
        try:
            return self.content(await self.model.ainvoke(prompt))
        except Exception as exc:
            from app.core.exceptions import AppException, sanitize_model_error
            if isinstance(exc, AppException):
                raise
            raise sanitize_model_error(exc, provider=self.provider_name, model=self.config.model) from exc

