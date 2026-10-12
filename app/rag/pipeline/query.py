from datetime import datetime, timezone
from typing import Any, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message
from app.models.run import Run
from app.models.thread import Thread
from app.models.knowledge_base import KnowledgeBase
from app.rag.generation.base import BaseGenerationProvider
from app.rag.vectorstore import PgVectorStore
from app.rag.query.transformer import transform_query
from app.rag.retrieval.workflow import RetrievalWorkflow
from app.rag.reranking import rerank_documents
from app.schemas.thread import ThreadConfig


class QueryPipeline:
    """
    RAG Query Pipeline in the RAG Lab.
    Driven by the latest PipelineVersion config (ThreadConfig) which decides
    the transformation technique, retrieval hyperparameters, reranking, and generation flow.
    """

    def __init__(
        self,
        *,
        db: AsyncSession,
        config: dict | ThreadConfig,
        run: Run,
        thread: Thread,
        message: Message,
        provider: BaseGenerationProvider,
    ):
        self.db = db
        self.config = config
        self.run = run
        self.thread = thread
        self.message = message
        self.provider = provider
        self.vector_store = PgVectorStore(db)
        self.retrieval_workflow = RetrievalWorkflow(self.vector_store)

        # Validate / normalize into ThreadConfig
        if isinstance(config, ThreadConfig):
            self.thread_config = config
        elif isinstance(config, dict):
            self.thread_config = ThreadConfig.model_validate(config)
        else:
            self.thread_config = ThreadConfig()

    async def execute(self) -> dict[str, Any]:
        """
        Executes the query pipeline flow according to the pipeline version configuration.
        For this run, executes query transformation with the configured technique.
        """
        state = await self._prepare()
        context = state["context"]
        response = await self._generate(self.message.content, context)
        return self._build_result(state, response)

    async def execute_stream(self):
        state = await self._prepare()
        response_parts = []
        async for chunk in self._generate_stream(self.message.content, state["context"]):
            response_parts.append(chunk)
            yield {"type": "token", "content": chunk}
        yield {"type": "complete", "result": self._build_result(state, "".join(response_parts))}

    async def _prepare(self) -> dict[str, Any]:
        memory = await self._load_memory()
        transformed_queries = await self._process_query(self.message.content, memory)
        retrieved_documents = await self._retrieve(transformed_queries)
        reranked_documents = await self._rerank(transformed_queries, retrieved_documents)
        context = await self._build_context(transformed_queries, reranked_documents, memory)
        return {
            "memory": memory,
            "transformed_queries": transformed_queries,
            "retrieved_documents": retrieved_documents,
            "reranked_documents": reranked_documents,
            "context": context,
        }

    def _build_result(self, state: dict[str, Any], response: str) -> dict[str, Any]:
        transformed_queries = state["transformed_queries"]
        retrieved_documents = state["retrieved_documents"]
        reranked_documents = state["reranked_documents"]
        return {
            "original_query": self.message.content,
            "transformed_queries": transformed_queries,
            "transformation_mode": self.thread_config.query_transformation.mode,
            "retrieved_documents": [
                {"id": str(doc.id), "content": doc.content} for doc in retrieved_documents
            ],
            "reranked_documents": [
                {"id": str(doc.id), "content": doc.content} for doc in reranked_documents
            ],
            "response": response,
            "answer": response,
            "Processed Query": transformed_queries,
            "config": self.thread_config.model_dump(),
        }

    async def _load_memory(self) -> List[dict[str, Any]]:
        """
        Loads prior conversation messages in chronological sequence for context.
        """
        if not self.thread or not self.thread.id:
            return []

        try:
            stmt = (
                select(Message)
                .where(
                    Message.thread_id == self.thread.id,
                    Message.sequence < (self.message.sequence or 1000000),
                )
                .order_by(Message.sequence.asc())
            )
            result = await self.db.scalars(stmt)
            messages = result.all()
            return [{"role": m.role, "content": m.content} for m in messages if m.content]
        except Exception:
            return []

    async def _process_query(self, query: str, memory: List[dict[str, Any]]) -> List[str]:
        """
        Transforms the user query using the configured technique (none, rewrite, hyde,
        multi_query, step_back, sub_queries).
        """
        qt_config = self.thread_config.query_transformation
        return await transform_query(
            query=query,
            provider=self.provider,
            config=qt_config,
            memory=memory,
        )

    async def _retrieve(self, queries: List[str]):
        """Run configured sparse+dense retrieval across the thread knowledge bases."""
        thread_metadata = getattr(self.thread, "metadata_", None)
        knowledge_base_ids = (
            thread_metadata.get("knowledge_base_ids", [])
            if isinstance(thread_metadata, dict)
            else []
        )
        if not knowledge_base_ids:
            return []

        knowledge_bases = (
            await self.db.scalars(
                select(KnowledgeBase).where(KnowledgeBase.id.in_(knowledge_base_ids))
            )
        ).all()
        retrieval_config = self.thread_config.retrieval
        if self.thread_config.reranking.enabled:
            retrieval_config = retrieval_config.model_copy(
                update={"top_k": max(retrieval_config.top_k, self.thread_config.reranking.top_n)}
            )
        return await self.retrieval_workflow.retrieve(
            queries,
            knowledge_bases,
            retrieval_config,
        )

    async def _rerank(self, queries: List[str], documents):
        """Rerank retrieved candidates when enabled in the pipeline config."""
        config = self.thread_config.reranking
        if not config.enabled or not documents:
            return documents
        # Query execution always needs documents for context; score-only output is
        # still available from the standalone reranker API.
        if not config.return_documents:
            config = config.model_copy(update={"return_documents": True})
        return rerank_documents(queries, documents, config)

    async def _build_context(
        self,
        queries: List[str],
        documents,
        memory: List[dict[str, Any]],
    ) -> str:
        """Build the complete prompt context for the final generation call."""
        query_context = "\n".join(f"- {query}" for query in queries) or "- None"
        document_context = "\n\n".join(
            f"[{index}] {document.content}"
            for index, document in enumerate(documents, start=1)
            if document.content
        ) or "None"
        memory_context = "\n".join(
            f"{item['role']}: {item['content']}" for item in memory
        ) or "None"
        return (
            f"Transformed queries:\n{query_context}\n\n"
            f"Conversation memory:\n{memory_context}\n\n"
            f"Retrieved context:\n{document_context}"
        )

    async def _generate(self, query: str, context: str) -> str:
        """Generate one non-streaming answer using the configured provider."""
        prompt = (
            "Answer the user query using only the provided retrieved context and conversation memory. "
            "If the context is insufficient, say so clearly. Do not invent facts.\n\n"
            f"Query: {query}\n\nContext: {context}"
        )
        return await self.provider.ainvoke_safe(prompt)

    async def _generate_stream(self, query: str, context: str):
        prompt = (
            "Answer the user query using only the provided retrieved context and conversation memory. "
            "If the context is insufficient, say so clearly. Do not invent facts.\n\n"
            f"Query: {query}\n\nContext: {context}"
        )
        async for chunk in self.provider.astream(prompt):
            yield chunk
            

