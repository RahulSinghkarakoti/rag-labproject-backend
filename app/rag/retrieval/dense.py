import asyncio

async def retrieve_dense(vector_store, embedder, query: str, *, top_k: int, filters: dict):
    query_embedding = await asyncio.to_thread(embedder.embed_query, query)
    return await vector_store.similarity_search_with_scores(
        query_embedding,
        top_k=top_k,
        filters=filters,
    )
