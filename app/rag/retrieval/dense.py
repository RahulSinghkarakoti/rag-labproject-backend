async def retrieve_dense(vector_store, embedder, query: str, *, top_k: int, filters: dict):
    return await vector_store.similarity_search_with_scores(
        embedder.embed_query(query),
        top_k=top_k,
        filters=filters,
    )
