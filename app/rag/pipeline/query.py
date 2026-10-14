"""
RAG Query Pipeline orchestrator.

``QueryPipeline`` is the top-level coordinator for a single user-query run.
It delegates every distinct concern to a focused module:

    memory loading      → :mod:`app.rag.memory.manager`
    context building    → :mod:`app.rag.memory.context`
    query transformation → :mod:`app.rag.query.transformer`  (via processor)
    retrieval           → :mod:`app.rag.retrieval.workflow`
    reranking           → :mod:`app.rag.reranking`
    step logging/timing → :mod:`app.rag.pipeline.executor`
    result schema       → :mod:`app.rag.pipeline.schemas`

The pipeline is fully driven by :class:`~app.schemas.thread.ThreadConfig`
which is stored on the active ``PipelineVersion`` and configured per-thread
by the user.  Factory/registry selection (provider, transformer, retrieval
strategy) all flow from that config — no hard-coded choices here.
"""
from __future__ import annotations

import logging
from typing import Any, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.run import Run
from app.models.thread import Thread
from app.models.thread_knowledge_base import thread_knowledge_bases
from app.rag.generation.base import BaseGenerationProvider
from app.rag.memory.context import build_generation_prompt, build_rag_context
from app.rag.memory.manager import MemoryManager
from app.rag.pipeline.context import PipelineContext
from app.rag.pipeline.executor import PipelineStepExecutor
from app.rag.pipeline.schemas import PipelineState
from app.rag.query.transformer import transform_query
from app.rag.reranking import rerank_documents
from app.rag.retrieval.workflow import RetrievalWorkflow
from app.rag.vectorstore import PgVectorStore
from app.schemas.thread import ThreadConfig

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    RAG Query Pipeline in the RAG Lab.

    Driven by the latest ``PipelineVersion`` config (:class:`~app.schemas.thread.ThreadConfig`)
    which decides the transformation technique, retrieval hyper-parameters,
    reranking, and generation flow.

    The class itself is now a *thin orchestrator*: each pipeline stage is
    implemented in its own module and simply called from :meth:`_prepare`.
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
    ) -> None:
        self.db = db
        self.config = config
        self.run = run
        self.thread = thread
        self.message = message
        self.provider = provider

        # ── vector store & retrieval ──────────────────────────────────────
        self.vector_store = PgVectorStore(db)
        self.retrieval_workflow = RetrievalWorkflow(self.vector_store)

        # ── validate / normalise config into ThreadConfig ─────────────────
        if isinstance(config, ThreadConfig):
            self.thread_config = config
        elif isinstance(config, dict):
            self.thread_config = ThreadConfig.model_validate(config)
        else:
            self.thread_config = ThreadConfig()

        # ── provider identity (used in logs) ──────────────────────────────
        self.model_name: str = getattr(
            getattr(self.provider, "config", None), "model", "unknown"
        )
        self.provider_name: str = getattr(self.provider, "provider_name", "unknown")

        # ── pipeline infrastructure ───────────────────────────────────────
        self.memory_manager = MemoryManager(db=self.db)

        self._pipeline_context = PipelineContext(
            run_id=str(run.id),
            thread_id=str(thread.id) if thread else "",
            message_id=str(message.id),
            provider_name=self.provider_name,
            model_name=self.model_name,
        )
        self.executor = PipelineStepExecutor(self._pipeline_context)

        self._log_stage("pipeline", "waiting")

    # ------------------------------------------------------------------
    # Internal logging helper (kept for backwards compat with log parsers)
    # ------------------------------------------------------------------

    def _log_stage(self, stage: str, state: str, **details: Any) -> None:
        logger.info(
            "query_pipeline stage=%s state=%s provider=%s model=%s%s",
            stage,
            state,
            self.provider_name,
            self.model_name,
            "".join(f" {k}={v}" for k, v in details.items()),
        )

    # ------------------------------------------------------------------
    # Public execution interface
    # ------------------------------------------------------------------

    async def execute(self) -> dict[str, Any]:
        """
        Executes the full RAG pipeline (non-streaming) and returns the result dict.

        The result dict shape is stable — callers such as ``QueryEngine``
        use ``result.get(...)`` on keys defined in :meth:`_build_result`.
        """
        state = await self._prepare()
        response = await self._generate(self.message.content, state.context)
        return self._build_result(state, response)

    async def execute_stream(self):
        """
        Executes the RAG pipeline in streaming mode.

        Yields ``{"type": "token", "content": chunk}`` events during generation,
        followed by a final ``{"type": "complete", "result": {...}}`` event.
        """
        state = await self._prepare()
        response_parts: list[str] = []
        async for chunk in self._generate_stream(self.message.content, state.context):
            response_parts.append(chunk)
            yield {"type": "token", "content": chunk}
        yield {
            "type": "complete",
            "result": self._build_result(state, "".join(response_parts)),
        }

    # ------------------------------------------------------------------
    # Prepare phase — runs all stages in sequence, returns PipelineState
    # ------------------------------------------------------------------

    async def _prepare(self) -> PipelineState:
        """
        Run all pre-generation stages (memory → transform → retrieve → rerank → context).

        Each stage is wrapped by :class:`~app.rag.pipeline.executor.PipelineStepExecutor`
        so that timing and structured logs are emitted automatically.
        """
        self._log_stage("prepare", "waiting")

        # 1. Memory — load prior conversation turns
        memory: list[dict[str, Any]] = await self.executor.run(
            "memory",
            self.memory_manager.load(
                thread_id=self.thread.id,
                before_sequence=self.message.sequence,
            ),
        )

        # 2. Query Transformation — expand / rewrite / hyde / passthrough
        transformed_queries: list[str] = await self.executor.run(
            "transformation",
            self._process_query(self.message.content, memory),
        )

        # 3. Retrieval — dense / sparse / hybrid / mmr across knowledge bases
        retrieved_documents = await self.executor.run(
            "retrieval",
            self._retrieve(transformed_queries),
        )

        # 4. Reranking — cross-encoder reranking (skipped when disabled)
        reranked_documents = await self.executor.run(
            "reranking",
            self._rerank(transformed_queries, retrieved_documents),
        )

        # 5. Context assembly — build the prompt context block
        context = build_rag_context(transformed_queries, reranked_documents, memory)
        self._log_stage("context", "completed")

        return PipelineState(
            memory=memory,
            transformed_queries=transformed_queries,
            retrieved_documents=retrieved_documents,
            reranked_documents=reranked_documents,
            context=context,
        )

    # ------------------------------------------------------------------
    # Result builder — keeps a stable dict shape for engine.py
    # ------------------------------------------------------------------

    def _build_result(self, state: PipelineState, response: str) -> dict[str, Any]:
        """
        Build the result dict consumed by :class:`~app.rag.engine.QueryEngine`.

        The shape of this dict is stable — do not rename keys without a
        corresponding update to ``engine._persist_run_result``.
        """
        return {
            "original_query": self.message.content,
            "transformed_queries": state.transformed_queries,
            "transformation_mode": self.thread_config.query_transformation.mode,
            "retrieved_documents": [
                {"id": str(doc.id), "content": doc.content}
                for doc in state.retrieved_documents
            ],
            "reranked_documents": [
                {"id": str(doc.id), "content": doc.content}
                for doc in state.reranked_documents
            ],
            "response": response,
            "answer": response,
            "Processed Query": state.transformed_queries,
            "config": self.thread_config.model_dump(),
        }

    # ------------------------------------------------------------------
    # Pipeline stage implementations
    # ------------------------------------------------------------------

    async def _process_query(
        self, query: str, memory: List[dict[str, Any]]
    ) -> List[str]:
        """
        Delegate query transformation to the configured strategy.

        Mode selection (none / rewrite / hyde / multi_query / step_back /
        sub_queries) is resolved by
        :class:`~app.rag.query.processor.QueryTransformationProcessor`
        based on ``ThreadConfig.query_transformation``.
        """
        return await transform_query(
            query=query,
            provider=self.provider,
            config=self.thread_config.query_transformation,
            memory=memory,
        )

    async def _retrieve(self, queries: List[str]):
        """
        Run configured retrieval across all knowledge bases attached to the thread.

        Strategy selection (dense / sparse / hybrid / mmr) is driven by
        ``ThreadConfig.retrieval``.  Top-k is expanded when reranking is
        enabled so the reranker has enough candidates to work with.
        """
        knowledge_bases = (
            await self.db.scalars(
                select(KnowledgeBase)
                .join(
                    thread_knowledge_bases,
                    thread_knowledge_bases.c.knowledge_base_id == KnowledgeBase.id,
                )
                .where(thread_knowledge_bases.c.thread_id == self.thread.id)
            )
        ).all()

        if not knowledge_bases:
            return []

        retrieval_config = self.thread_config.retrieval
        if self.thread_config.reranking.enabled:
            # Expand top_k so the reranker has enough candidates.
            retrieval_config = retrieval_config.model_copy(
                update={
                    "top_k": max(
                        retrieval_config.top_k,
                        self.thread_config.reranking.top_n,
                    )
                }
            )

        return await self.retrieval_workflow.retrieve(
            queries,
            knowledge_bases,
            retrieval_config,
        )

    async def _rerank(self, queries: List[str], documents):
        """
        Rerank retrieved candidates when reranking is enabled in the pipeline config.

        Forces ``return_documents=True`` because the pipeline always needs the
        document objects for context building (score-only mode is available
        from the standalone reranker API endpoint).
        """
        config = self.thread_config.reranking
        if not config.enabled or not documents:
            return documents
        if not config.return_documents:
            config = config.model_copy(update={"return_documents": True})
        return rerank_documents(queries, documents, config)

    async def _generate(self, query: str, context: str) -> str:
        """
        Generate a non-streaming answer using the configured LLM provider.

        Prompt construction is delegated to
        :func:`~app.rag.memory.context.build_generation_prompt`.
        """
        self._log_stage("generation", "waiting")
        prompt = build_generation_prompt(query, context)
        response = await self.provider.ainvoke_safe(prompt)
        self._log_stage("generation", "completed")
        return response

    async def _generate_stream(self, query: str, context: str):
        """
        Stream generation tokens from the configured LLM provider.

        Yields raw text chunks; the caller is responsible for assembling
        them into the final response string.
        """
        self._log_stage("generation", "waiting")
        prompt = build_generation_prompt(query, context)
        async for chunk in self.provider.astream(prompt):
            yield chunk
        self._log_stage("generation", "completed")
