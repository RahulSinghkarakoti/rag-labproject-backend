import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import ModelProviderError, sanitize_model_error
from app.rag.generation.base import BaseGenerationProvider
from app.rag.query.base import clean_queries
from app.rag.query.expander import MultiQueryExpander
from app.rag.query.hyde import HyDETransformer
from app.rag.query.passthrough import PassThroughTransformer
from app.rag.query.processor import (
    QueryTransformationProcessor,
    query_transformation_processor,
)
from app.rag.query.rewriter import QueryRewriter
from app.rag.query.step_back import StepBackTransformer
from app.rag.query.sub_queries import SubQueryDecomposer
from app.rag.query.transformer import transform_query
from app.schemas.thread import (
    GenerationConfig,
    QueryTransformationConfig,
    RerankingConfig,
    RetrievalConfig,
    ThreadConfig,
)


class DummyProvider(BaseGenerationProvider):
    provider_name = "test_provider"

    def __init__(self, canned_response: str = ""):
        self.canned_response = canned_response
        self.last_prompt = None

    def _build_model(self, config):
        return None

    async def ainvoke(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.canned_response


# =========================================================================
# 1. ThreadConfig & Sub-configs: Validation & Non-ambiguity Tests
# =========================================================================

def test_thread_config_defaults():
    config = ThreadConfig()
    assert config.query_transformation.mode == "none"
    assert config.query_transformation.query_count == 3
    assert config.query_transformation.include_original is True
    assert config.retrieval.top_k == 8
    assert config.retrieval.alpha == 0.5
    assert config.reranking.enabled is True
    assert config.reranking.model == "BAAI/bge-reranker-base"
    assert config.reranking.top_n == 10
    assert config.reranking.top_k == 5
    assert config.reranking.batch_size == 16
    assert config.generation.provider == "groq"
    assert config.generation.llm_model == "gpt-oss-120b"
    assert config.generation.model == "gpt-oss-120b"


def test_thread_config_all_transformation_modes():
    modes = ["none", "rewrite", "hyde", "multi_query", "step_back", "sub_queries"]
    for mode in modes:
        cfg = ThreadConfig.model_validate({
            "query_transformation": {"mode": mode, "query_count": 4, "include_original": False},
        })
        assert cfg.query_transformation.mode == mode
        assert cfg.query_transformation.query_count == 4
        assert cfg.query_transformation.include_original is False


def test_thread_config_legacy_normalization_hyde():
    # hyde_enabled: True should normalize to mode="hyde"
    cfg = ThreadConfig.model_validate({
        "query_transformation": {"hyde_enabled": True},
    })
    assert cfg.query_transformation.mode == "hyde"


def test_thread_config_legacy_normalization_multi_query():
    # multi_query_count: 5 should normalize to mode="multi_query" and query_count=5
    cfg = ThreadConfig.model_validate({
        "query_transformation": {"multi_query_count": 5},
    })
    assert cfg.query_transformation.mode == "multi_query"
    assert cfg.query_transformation.query_count == 5


def test_thread_config_extra_forbidden():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ThreadConfig.model_validate({
            "query_transformation": {"mode": "rewrite", "invalid_param": 123},
        })

    with pytest.raises(ValidationError):
        ThreadConfig.model_validate({
            "unknown_section": {},
        })


def test_generation_config_provider_auto_inference():
    # Inferred from model name
    cfg_gemini = GenerationConfig.model_validate({"model": "gemini-2.5-pro"})
    assert cfg_gemini.provider == "gemini"
    assert cfg_gemini.model == "gemini-2.5-pro"

    cfg_groq = GenerationConfig.model_validate({"model": "gpt-oss-120b"})
    assert cfg_groq.provider == "groq"

    cfg_groq = GenerationConfig.model_validate({"llm_model": "gpt-oss-120b"})
    assert cfg_groq.provider == "groq"
    assert cfg_groq.model == "gpt-oss-120b"


# =========================================================================
# 2. Text Cleaner Helper Tests
# =========================================================================

def test_clean_queries_handles_numbering_and_boilerplate():
    raw_response = (
        "Here are the queries for search:\n"
        "1. What is quantum computing?\n"
        "2) Applications of quantum computing\n"
        "- Future challenges in quantum computing\n"
        "* Quantum hardware architectures\n"
    )
    cleaned = clean_queries(raw_response, max_count=5)
    assert cleaned == [
        "What is quantum computing?",
        "Applications of quantum computing",
        "Future challenges in quantum computing",
        "Quantum hardware architectures",
    ]


def test_clean_queries_deduplication():
    raw_response = "1. AI safety\n2. AI safety\n3. AI ethics"
    cleaned = clean_queries(raw_response, max_count=5)
    assert cleaned == ["AI safety", "AI ethics"]


# =========================================================================
# 3. Individual Query Transformer Technique Tests
# =========================================================================

@pytest.mark.anyio
async def test_passthrough_transformer():
    transformer = PassThroughTransformer()
    provider = DummyProvider()
    config = QueryTransformationConfig(mode="none")

    result = await transformer.transform("  how to fine-tune an LLM?  ", provider, config)
    assert result == ["how to fine-tune an LLM?"]

    # Empty query
    assert await transformer.transform("", provider, config) == []


@pytest.mark.anyio
async def test_query_rewriter_without_memory():
    transformer = QueryRewriter()
    provider = DummyProvider(canned_response="optimizing transformer inference latency")
    config = QueryTransformationConfig(mode="rewrite", include_original=False)

    result = await transformer.transform("how make transformer fast?", provider, config)
    assert result == ["optimizing transformer inference latency"]
    assert "how make transformer fast?" in provider.last_prompt


@pytest.mark.anyio
async def test_query_rewriter_with_memory():
    transformer = QueryRewriter()
    provider = DummyProvider(canned_response="Apple Inc quarterly earnings Q3 2024")
    config = QueryTransformationConfig(mode="rewrite", include_original=True)
    memory = [
        {"role": "user", "content": "Tell me about Apple revenue"},
        {"role": "assistant", "content": "Apple reported strong performance."},
    ]

    result = await transformer.transform("what were their quarterly earnings?", provider, config, memory=memory)
    assert "Conversation History:" in provider.last_prompt
    assert result == [
        "Apple Inc quarterly earnings Q3 2024",
        "what were their quarterly earnings?",
    ]


@pytest.mark.anyio
async def test_hyde_transformer():
    transformer = HyDETransformer()
    hypothetical_passage = "Supervised fine-tuning involves training a pre-trained model on domain-specific pairs."
    provider = DummyProvider(canned_response=hypothetical_passage)
    config = QueryTransformationConfig(mode="hyde", query_count=1, include_original=True)

    result = await transformer.transform("what is fine tuning?", provider, config)
    assert hypothetical_passage in result
    assert "what is fine tuning?" in result


@pytest.mark.anyio
async def test_multi_query_expander():
    transformer = MultiQueryExpander()
    canned = (
        "1. Postgres pgvector similarity search performance\n"
        "2. Vector indexing techniques in PostgreSQL\n"
        "3. HNSW vs IVFFlat benchmarks pgvector\n"
    )
    provider = DummyProvider(canned_response=canned)
    config = QueryTransformationConfig(mode="multi_query", query_count=3, include_original=True)

    result = await transformer.transform("pgvector speed", provider, config)
    assert "pgvector speed" in result
    assert "Postgres pgvector similarity search performance" in result
    assert len(result) == 4  # original + 3 generated


@pytest.mark.anyio
async def test_step_back_transformer():
    transformer = StepBackTransformer()
    provider = DummyProvider(canned_response="What are the fundamental principles of neural network regularization?")
    config = QueryTransformationConfig(mode="step_back", include_original=True)

    result = await transformer.transform("How does dropout prevent overfitting in ResNet-50?", provider, config)
    assert len(result) == 2
    assert result[0] == "What are the fundamental principles of neural network regularization?"
    assert result[1] == "How does dropout prevent overfitting in ResNet-50?"


@pytest.mark.anyio
async def test_sub_query_decomposer():
    transformer = SubQueryDecomposer()
    canned = (
        "1. What are the advantages of FastAPI?\n"
        "2. What are the advantages of Django?\n"
    )
    provider = DummyProvider(canned_response=canned)
    config = QueryTransformationConfig(mode="sub_queries", query_count=2, include_original=False)

    result = await transformer.transform("Compare FastAPI and Django", provider, config)
    assert result == [
        "What are the advantages of FastAPI?",
        "What are the advantages of Django?",
    ]


# =========================================================================
# 4. Processor & Coordinator Dispatch Tests
# =========================================================================

@pytest.mark.anyio
async def test_processor_dispatch_all_modes():
    provider = DummyProvider(canned_response="transformed query output")

    for mode in ["none", "rewrite", "hyde", "multi_query", "step_back", "sub_queries"]:
        res = await transform_query(
            query="test query",
            provider=provider,
            config={"mode": mode},
        )
        assert isinstance(res, list)
        assert len(res) >= 1


@pytest.mark.anyio
async def test_processor_unsupported_mode():
    provider = DummyProvider()
    with pytest.raises(Exception):
        await transform_query(
            query="test",
            provider=provider,
            config={"mode": "non_existent_mode"},
        )


# =========================================================================
# 5. Safe Error Sanitization Tests
# =========================================================================

def test_sanitize_model_error_redacts_api_keys():
    raw_error = Exception("Invalid API Key: sk-proj-1234567890abcdef1234567890 provided for org.")
    sanitized = sanitize_model_error(raw_error, provider="groq", model="gpt-oss-120b")

    assert isinstance(sanitized, ModelProviderError)
    assert "sk-proj" not in str(sanitized.message)
    assert "sk-proj" not in str(sanitized.details)
    assert sanitized.details["error_type"] == "authentication_failed"
    assert "Authentication failed for groq" in sanitized.message


def test_sanitize_model_error_rate_limit():
    raw_error = Exception("429 Too Many Requests: Rate limit reached for groq gpt-oss-120b")
    sanitized = sanitize_model_error(raw_error, provider="groq", model="gpt-oss-120b")

    assert sanitized.status_code == 429
    assert sanitized.details["error_type"] == "rate_limit_exceeded"
    assert "gpt-oss-120b" in sanitized.message


def test_sanitize_model_error_model_not_found():
    raw_error = Exception("404 NotFound: The model `gemini-unknown-model` was not found.")
    sanitized = sanitize_model_error(raw_error, provider="gemini", model="gemini-unknown-model")

    assert sanitized.status_code == 400
    assert sanitized.details["error_type"] == "model_not_found"


# =========================================================================
# 6. Pipeline Execution Tests
# =========================================================================

@pytest.mark.anyio
async def test_query_pipeline_execution():
    from unittest.mock import MagicMock
    from uuid import uuid4
    from app.rag.pipeline.query import QueryPipeline

    provider = DummyProvider(canned_response="Rewritten: What is retrieval augmented generation?")
    thread_mock = MagicMock()
    thread_mock.id = uuid4()
    message_mock = MagicMock()
    message_mock.content = "what is RAG?"
    message_mock.sequence = 1
    run_mock = MagicMock()
    run_mock.id = uuid4()
    db_mock = AsyncMock()
    db_mock.scalars.return_value = MagicMock()
    db_mock.scalars.return_value.all.return_value = []

    pipeline = QueryPipeline(
        db=db_mock,
        config={
            "query_transformation": {"mode": "rewrite", "include_original": True},
            "generation": {"model": "gpt-oss-120b"},
        },
        run=run_mock,
        thread=thread_mock,
        message=message_mock,
        provider=provider,
    )

    result = await pipeline.execute()

    assert result["original_query"] == "what is RAG?"
    assert result["transformation_mode"] == "rewrite"
    assert "what is RAG?" in result["transformed_queries"]
    assert "Rewritten: What is retrieval augmented generation?" in result["transformed_queries"]
    assert "Processed Query" in result
    assert result["response"] == "Rewritten: What is retrieval augmented generation?"

