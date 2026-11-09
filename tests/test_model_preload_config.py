import pytest

from app.core.config import Settings
from app.core.lifespan import resolve_preload_models
from app.core.model_config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANKING_MODELS


def test_resolve_preload_models_preload_all():
    # When preload_all is True, all supported models are returned sorted
    result = resolve_preload_models(
        SUPPORTED_EMBEDDING_MODELS,
        configured_value="none",
        preload_all=True,
    )
    assert result == sorted(SUPPORTED_EMBEDDING_MODELS)


def test_resolve_preload_models_configured_all():
    # When configured_value is "all", all supported models are returned sorted
    result = resolve_preload_models(
        SUPPORTED_EMBEDDING_MODELS,
        configured_value="all",
        preload_all=False,
    )
    assert result == sorted(SUPPORTED_EMBEDDING_MODELS)


def test_resolve_preload_models_none_or_empty():
    # When configured_value is "none", "", or None and preload_all is False, returns []
    assert resolve_preload_models(SUPPORTED_EMBEDDING_MODELS, "none") == []
    assert resolve_preload_models(SUPPORTED_EMBEDDING_MODELS, "None") == []
    assert resolve_preload_models(SUPPORTED_EMBEDDING_MODELS, "") == []
    assert resolve_preload_models(SUPPORTED_EMBEDDING_MODELS, None) == []


def test_resolve_preload_models_specific_valid():
    # Specific valid model
    result = resolve_preload_models(
        SUPPORTED_EMBEDDING_MODELS,
        configured_value="all-MiniLM-L6-v2",
        preload_all=False,
    )
    assert result == ["all-MiniLM-L6-v2"]

    # Multiple comma-separated valid models
    result = resolve_preload_models(
        SUPPORTED_RERANKING_MODELS,
        configured_value="BAAI/bge-reranker-base, BAAI/bge-reranker-large",
        preload_all=False,
    )
    assert set(result) == set(SUPPORTED_RERANKING_MODELS)


def test_resolve_preload_models_invalid_raises():
    # Invalid model name raises ValueError
    with pytest.raises(ValueError) as exc_info:
        resolve_preload_models(
            SUPPORTED_EMBEDDING_MODELS,
            configured_value="nonexistent-model",
            preload_all=False,
        )
    assert "Unknown model(s) configured for preload" in str(exc_info.value)
    assert "nonexistent-model" in str(exc_info.value)


def test_settings_model_preload_defaults():
    # Default settings should have preloading disabled
    test_settings = Settings(
        DATABASE_URL="postgresql://user:pass@localhost:5432/db",
        NEXTAUTH_INTERNAL_SECRET="secret",
        JWT_SECRET_KEY="secret",
        JWT_ALGORITHM="HS256",
        ACCESS_TOKEN_EXPIRE_MINUTES=15,
        REFRESH_TOKEN_EXPIRE_DAYS=30,
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        HF_TOKEN="hf_token",
    )
    assert test_settings.PRELOAD_ALL_MODELS is False
    assert test_settings.PRELOAD_EMBEDDING_MODELS == "none"
    assert test_settings.PRELOAD_RERANKING_MODELS == "none"
