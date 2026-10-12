from collections.abc import Callable
from typing import Any

from app.schemas.thread import GenerationConfig
from app.rag.generation.base import BaseGenerationProvider
from app.rag.generation.gemini import GeminiProvider
from app.rag.generation.groq import GroqProvider
from app.rag.generation.openai import OpenAIProvider


ProviderClass = Callable[[GenerationConfig], BaseGenerationProvider]


class ProviderRegistry:
    """Registry of available generation providers, matching LoaderRegistry."""

    def __init__(self):
        self._providers: dict[str, ProviderClass] = {
            "groq": GroqProvider,
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
        }

    def register(self, name: str, provider_class: ProviderClass) -> None:
        self._providers[name.lower()] = provider_class

    def get(self, name: str, config: GenerationConfig) -> BaseGenerationProvider:
        provider_class = self._providers.get(name.lower())
        if not provider_class:
            from app.core.exceptions import ModelProviderError
            raise ModelProviderError(
                message=f"No generation provider registered for: '{name}'. Supported providers: {list(self._providers.keys())}",
                provider=name,
                model=getattr(config, "model", "unknown"),
                error_type="invalid_provider",
                status_code=400,
            )
        return provider_class(config)


class ProviderModelFactory:
    """Read generation config once and create the shared provider for a run."""

    def __init__(self, registry: ProviderRegistry | None = None):
        self.registry = registry or ProviderRegistry()

    def create(self, config: dict[str, Any]) -> BaseGenerationProvider:
        gen_data = config.get("generation", {}) if isinstance(config, dict) else {}
        generation = GenerationConfig.model_validate(gen_data)
        return self.registry.get(generation.provider, generation)



provider_model_factory = ProviderModelFactory()
