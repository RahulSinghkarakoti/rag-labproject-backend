from collections import defaultdict

from app.rag.retrieval.dense import retrieve_dense
from app.rag.retrieval.fusion import weighted_fusion
from app.rag.retrieval.mmr import maximal_marginal_relevance
from app.rag.retrieval.sparse import retrieve_sparse


class RetrievalWorkflow:
    """Coordinate dense, sparse, hybrid, and MMR retrieval for knowledge bases."""

    def __init__(self, vector_store):
        self.vector_store = vector_store



    async def retrieve(self, queries: list[str], knowledge_bases: list, config):
        """Retrieve relevant documents using dense, sparse, hybrid, or MMR.""" 
        if not queries or not knowledge_bases:
            return []
 
        # 1. Group Knowledge Bases by embedding model
        by_model = defaultdict(list)
        for kb in knowledge_bases:
            embedding_config = kb.embedding_config or {}
            model = embedding_config.get("model") or config.dense_model
            by_model[model].append(str(kb.id))
        strategy = config.strategy
        documents = []
        def deduplicate_scored(results):
            """Keep the highest-scoring occurrence of each document."""
            unique = {}
            for document, score in results:
                doc_id = str(document.id)
                if doc_id not in unique or score > unique[doc_id][1]:
                    unique[doc_id] = (document, score)
            return list(unique.values())
        for model, knowledge_base_ids in by_model.items():
            from app.rag.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(model_name=model)
            filters = {"knowledge_base_id": knowledge_base_ids}
            # Gather candidates across ALL queries before selecting final results.
            all_dense = []
            all_sparse = []
            for index, query in enumerate(queries, start=1):
                # 2. Dense retrieval
                if strategy in {"dense", "hybrid", "mmr"}:
                    fetch_k = (
                        config.fetch_k
                        if strategy == "mmr"
                        else config.top_k
                    )
                    dense_results = await retrieve_dense(
                        self.vector_store,
                        embedder,
                        query,
                        top_k=fetch_k,
                        filters=filters,
                    )

                    if config.similarity_threshold is not None:
                        before = len(dense_results)
                        dense_results = [
                            (doc, score)
                            for doc, score in dense_results
                            if score >= config.similarity_threshold
                        ]
 
                    all_dense.extend(dense_results)
                # 3. Sparse retrieval
                if strategy in {"sparse", "hybrid"}:
                    sparse_results = await retrieve_sparse(
                        self.vector_store,
                        query,
                        model=config.sparse_model,
                        top_k=config.top_k,
                        filters=filters,
                    )
                    all_sparse.extend(sparse_results)
            # 4. Deduplicate candidates across queries
            all_dense = deduplicate_scored(all_dense)
            all_sparse = deduplicate_scored(all_sparse)
 
            # 5. Apply the selected strategy globally across these queries
            if strategy == "dense":
                selected = []
                for doc, score in sorted(
                    all_dense,
                    key=lambda item: item[1],
                    reverse=True,
                )[:config.top_k]:
                    doc.score = float(score)
                    selected.append(doc)
            elif strategy == "sparse":
                selected = []
                for doc, score in sorted(
                    all_sparse,
                    key=lambda item: item[1],
                    reverse=True,
                )[:config.top_k]:
                    doc.score = float(score)
                    selected.append(doc)
            elif strategy == "hybrid":
                selected = weighted_fusion(
                    all_dense,
                    all_sparse,
                    alpha=config.alpha,
                    top_k=config.top_k,
                )
            elif strategy == "mmr":
                if not all_dense:
                    selected = []
                else:
                    # A single MMR pass over candidates from every query.
                    combined_query = " ".join(queries)
                    query_embedding = embedder.embed_query(combined_query)
                    selected = maximal_marginal_relevance(
                        query_embedding,
                        all_dense,
                        top_k=config.top_k,
                        lambda_mult=config.mmr_lambda,
                    )

            else:
                raise ValueError(f"Unsupported retrieval strategy: {strategy}")
            documents.extend(selected)
        # 6. Final deduplication and global limit
        unique = {}
        for document in documents:
            unique.setdefault(str(document.id), document)
        final_documents = list(unique.values())[:config.top_k]
 
        return final_documents