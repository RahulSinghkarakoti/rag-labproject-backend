import math


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def maximal_marginal_relevance(
    query_embedding: list[float],
    results: list[tuple[object, float]],
    *,
    top_k: int,
    lambda_mult: float,
):
    candidates = list(results)
    selected = []
    while candidates and len(selected) < top_k:
        def score(item):
            document, relevance = item
            diversity = max(
                (_cosine(document.embedding, chosen.embedding) for chosen, _ in selected),
                default=0.0,
            )
            return lambda_mult * relevance - (1 - lambda_mult) * diversity

        best = max(candidates, key=score)
        candidates.remove(best)
        selected.append(best)
    return [document for document, _ in selected]
