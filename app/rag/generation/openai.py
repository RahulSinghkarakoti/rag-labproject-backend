from app.rag.generation.base import BaseGenerationProvider


class OpenAIProvider(BaseGenerationProvider):
    provider_name = "openai"

    def _build_model(self, config):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI provider requires the langchain-openai package"
            ) from exc

        return ChatOpenAI(
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

