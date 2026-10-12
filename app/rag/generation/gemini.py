from app.rag.generation.base import BaseGenerationProvider


class GeminiProvider(BaseGenerationProvider):
    provider_name = "gemini"

    def _build_model(self, config):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Gemini provider requires the langchain-google-genai package"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    async def ainvoke(self, prompt: str) -> str:
        try:
            return self.content(await self.model.ainvoke(prompt))
        except Exception as exc:
            from app.core.exceptions import AppException, sanitize_model_error
            if isinstance(exc, AppException):
                raise
            raise sanitize_model_error(exc, provider=self.provider_name, model=self.config.model) from exc

