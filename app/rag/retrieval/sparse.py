import math
import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25(query: str, documents: list, *, k1: float = 1.2, b: float = 0.75):
    query_terms = set(_tokens(query))
    tokenized = [_tokens(document.content) for document in documents]
    if not query_terms or not documents:
        return []

    average_length = sum(map(len, tokenized)) / len(tokenized) or 1
    document_frequency = Counter(
        term for terms in tokenized for term in query_terms if term in terms
    )
    scored = []
    for document, terms in zip(documents, tokenized):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            if not counts[term]:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            tf = counts[term]
            score += idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * len(terms) / average_length)
            )
        if score > 0:
            scored.append((document, score))
    return scored


async def retrieve_sparse(vector_store, query: str, *, model: str, top_k: int, filters: dict):
    if model.lower() != "bm25":
        raise ValueError(f"Unsupported sparse retrieval model: {model}")
    documents = await vector_store.keyword_search(query, filters=filters)
    return sorted(bm25(query, documents), key=lambda item: item[1], reverse=True)[:top_k]
