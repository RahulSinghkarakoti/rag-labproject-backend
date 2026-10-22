def _normalize(results: list[tuple[object, float]]) -> dict[str, tuple[object, float]]:
    if not results:
        return {}
    maximum = max(score for _, score in results) or 1.0
    return {str(document.id): (document, score / maximum) for document, score in results}


def weighted_fusion(
    dense_results: list[tuple[object, float]],
    sparse_results: list[tuple[object, float]],
    *,
    alpha: float,
    top_k: int,
):
    dense = _normalize(dense_results)
    sparse = _normalize(sparse_results)
    fused = {}
    for chunk_id in dense.keys() | sparse.keys():
        document = (dense.get(chunk_id) or sparse[chunk_id])[0]
        score = alpha * dense.get(chunk_id, (document, 0.0))[1]
        score += (1 - alpha) * sparse.get(chunk_id, (document, 0.0))[1]
        fused[chunk_id] = (document, score)
    ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:top_k]
    for document, score in ranked:
        document.score = float(score)
    return [document for document, _ in ranked]
