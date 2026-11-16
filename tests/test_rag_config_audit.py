import pytest
from unittest.mock import MagicMock, AsyncMock

from app.schemas.thread import (
    RerankingConfig,
    RetrievalConfig,
    ThreadConfig,
    GenerationConfig,
)
from app.core.model_config import (
    SUPPORTED_GENERATION_MODELS,
    validate_generation_model,
)
from app.rag.reranking.reranker import rerank_documents
from app.rag.memory.context import (
    build_generation_prompt,
    FOLLOWUP_SUGGESTIONS_PROMPT,
    CONVERSATIONAL_SYSTEM_PROMPT,
)
from app.rag.query.sub_queries import SubQueryDecomposer
from app.rag.query.hyde import HyDETransformer


def test_reranking_config_top_n_sync():
    """Verify top_n and top_k sync properly."""
    cfg = RerankingConfig(top_n=15)
    assert cfg.top_n == 15
    assert cfg.top_k == 15

    cfg2 = RerankingConfig(top_k=8)
    assert cfg2.top_k == 8
    assert cfg2.top_n == 8

    # Default is 10 for both
    default_cfg = RerankingConfig()
    assert default_cfg.top_n == 10
    assert default_cfg.top_k == 10


def test_rerank_documents_respects_top_n(monkeypatch):
    """Verify rerank_documents uses config.top_n to return more than 5 documents."""
    class DummyDoc:
        def __init__(self, doc_id, content):
            self.id = doc_id
            self.content = content
            self.rerank_score = 0.0

    # 12 documents
    docs = [DummyDoc(f"doc_{i}", f"content {i}") for i in range(12)]
    # Mock model that returns descending scores
    mock_model = MagicMock()
    mock_model.predict.return_value = [float(12 - i) for i in range(12)]

    monkeypatch.setattr("app.rag.reranking.reranker._load_model", lambda name: mock_model)

    cfg = RerankingConfig(top_n=10, score_threshold=None, return_documents=True)
    results = rerank_documents(["test query"], docs, cfg)

    assert len(results) == 10
    assert results[0].id == "doc_0"
    assert results[-1].id == "doc_9"


def test_whisper_not_in_generation_allowlist():
    """Ensure audio transcription model whisper-large-v3 cannot be used as an LLM."""
    assert "whisper-large-v3" not in SUPPORTED_GENERATION_MODELS.get("groq", set())

    with pytest.raises(ValueError, match="Unsupported model 'whisper-large-v3'"):
        validate_generation_model("groq", "whisper-large-v3")


def test_build_generation_prompt_followups_with_system_prompt():
    """Verify suggest_followups works even with custom system_prompt."""
    prompt = build_generation_prompt(
        query="Explain quantum computing",
        context="Retrieved info",
        system_prompt="You are a quantum physicist.",
        suggest_followups=True,
        chat_mode="conversational",
    )
    assert "You are a quantum physicist." in prompt
    assert FOLLOWUP_SUGGESTIONS_PROMPT.strip() in prompt


def test_build_generation_prompt_strict_mode_omits_followups():
    """Strict mode should omit follow-ups even if suggest_followups is True."""
    prompt = build_generation_prompt(
        query="Explain quantum computing",
        context="Retrieved info",
        system_prompt="You are a quantum physicist.",
        suggest_followups=True,
        chat_mode="strict",
    )
    assert "You are a quantum physicist." in prompt
    assert FOLLOWUP_SUGGESTIONS_PROMPT.strip() not in prompt


@pytest.mark.anyio
async def test_sub_queries_transformer_query_count_cap():
    """Verify sub_queries respects query_count > 5 up to 10."""
    transformer = SubQueryDecomposer()
    mock_provider = AsyncMock()
    mock_provider.model = None
    mock_provider.ainvoke.return_value = "\n".join([f"sub query {i}" for i in range(8)])
    mock_provider.ainvoke_safe.return_value = "\n".join([f"sub query {i}" for i in range(8)])

    config = ThreadConfig().query_transformation
    config = config.model_copy(update={"mode": "sub_queries", "query_count": 8, "include_original": False})

    queries = await transformer.transform("complex query", mock_provider, config)
    assert len(queries) == 8


@pytest.mark.anyio
async def test_hyde_transformer_query_count_cap():
    """Verify hyde respects query_count > 5 up to 10."""
    transformer = HyDETransformer()
    mock_provider = AsyncMock()
    mock_provider.model = None
    mock_provider.ainvoke.return_value = "\n\n".join([f"passage {i}" for i in range(7)])
    mock_provider.ainvoke_safe.return_value = "\n\n".join([f"passage {i}" for i in range(7)])

    config = ThreadConfig().query_transformation
    config = config.model_copy(update={"mode": "hyde", "query_count": 7, "include_original": False})

    queries = await transformer.transform("complex query", mock_provider, config)
    assert len(queries) == 7
