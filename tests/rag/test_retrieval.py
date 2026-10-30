from types import SimpleNamespace

from app.rag.retrieval.fusion import weighted_fusion
from app.rag.retrieval.mmr import maximal_marginal_relevance
from app.rag.retrieval.sparse import bm25


def test_bm25_ranks_matching_documents():
    documents = [
        SimpleNamespace(id="one", content="retrieval augmented generation"),
        SimpleNamespace(id="two", content="database administration"),
    ]

    results = bm25("retrieval generation", documents)

    assert results[0][0].id == "one"
    assert results[0][1] > 0


def test_weighted_fusion_uses_alpha_and_deduplicates():
    first = SimpleNamespace(id="one")
    second = SimpleNamespace(id="two")

    results = weighted_fusion(
        [(first, 1.0), (second, 0.2)],
        [(second, 1.0)],
        alpha=1.0,
        top_k=2,
    )

    assert [document.id for document in results] == ["one", "two"]


def test_weighted_fusion_preserves_order_with_negative_scores():
    better = SimpleNamespace(id="better")
    worse = SimpleNamespace(id="worse")

    results = weighted_fusion(
        [(better, -0.1), (worse, -0.8)],
        [],
        alpha=1.0,
        top_k=2,
    )

    assert [document.id for document in results] == ["better", "worse"]
    assert results[0].score > results[1].score


def test_mmr_keeps_relevant_results_with_diversity():
    first = SimpleNamespace(id="one", embedding=[1.0, 0.0])
    duplicate = SimpleNamespace(id="duplicate", embedding=[1.0, 0.0])
    diverse = SimpleNamespace(id="diverse", embedding=[0.0, 1.0])

    results = maximal_marginal_relevance(
        [1.0, 0.0],
        [(first, 1.0), (duplicate, 0.95), (diverse, 0.8)],
        top_k=2,
        lambda_mult=0.5,
    )

    assert [document.id for document in results] == ["one", "diverse"]
