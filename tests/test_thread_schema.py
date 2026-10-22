import pytest
from pydantic import ValidationError

from app.schemas.thread import ThreadConfig


def test_thread_config_validates_frontend_payload_and_rejects_unknown_values():
    config = ThreadConfig.model_validate({
        "query_transformation": {"hyde_enabled": True, "multi_query_count": 3},
        "retrieval": {
            "alpha": 0.7,
            "top_k": 20,
            "sparse_model": "bm25",
            "dense_model": "all-MiniLM-L6-v2",
        },
        "reranking": {
            "enabled": True,
            "model": "BAAI/bge-reranker-base",
            "top_n": 5,
            "top_k": 3,
            "batch_size": 8,
            "score_threshold": 0.2,
            "return_documents": True,
        },
        "generation": {
            "llm_model": "gpt-oss-120b",
            "temperature": 0.2,
            "citation_enforced": True,
        },
    })

    assert config.retrieval.top_k == 20
    assert config.model_dump()["generation"]["llm_model"] == "gpt-oss-120b"

    with pytest.raises(ValidationError):
        ThreadConfig.model_validate({
            "retrieval": {"alpha": 1.1, "unknown": True},
            "generation": {"llm_model": "gpt-oss-120b"},
        })
