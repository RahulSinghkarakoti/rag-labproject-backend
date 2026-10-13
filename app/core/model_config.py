"""Global allowlists for models and generation providers."""

from sqlalchemy import label


SUPPORTED_EMBEDDING_MODELS = frozenset({
    "BAAI/bge-large-en-v1.5",
    "all-MiniLM-L6-v2",
})

SUPPORTED_RERANKING_MODELS = frozenset({
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
})

SUPPORTED_GENERATION_MODELS = {
    # "openai": frozenset({"gpt-4o"}),
    "groq": frozenset({
        "gpt-oss-120b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "whisper-large-v3",
    }),
    "gemini": frozenset({"gemini-2.5-pro","gemini-2.5-flash","gemini-3.8-flash" ,  "gemini-3.7-flash" ,
      "gemini-3.5-flash" , "gemini-3.5-flash-lite"}),
}


def validate_embedding_model(model: str) -> str:
    if model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding model '{model}'. "
            f"Supported models: {sorted(SUPPORTED_EMBEDDING_MODELS)}"
        )
    return model


def validate_reranking_model(model: str) -> str:
    if model not in SUPPORTED_RERANKING_MODELS:
        raise ValueError(
            f"Unsupported reranking model '{model}'. "
            f"Supported models: {sorted(SUPPORTED_RERANKING_MODELS)}"
        )
    return model


def validate_generation_model(provider: str, model: str) -> str:
    supported = SUPPORTED_GENERATION_MODELS.get(provider, frozenset())
    if model not in supported:
        raise ValueError(
            f"Unsupported model '{model}' for provider '{provider}'. "
            f"Supported models: {sorted(supported)}"
        )
    return model
