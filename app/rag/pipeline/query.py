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


GREETINGS = {
    "hi", "hello", "hey", "hola", "namaste", "greetings", "howdy",
    "good morning", "good afternoon", "good evening", "good day",
    "hi there", "hello there", "hey there", "what's up", "whats up", "sup",
    "how are you", "how are u", "hi how are you", "hello how are you",
    "hi how are u", "hello how are u", "how do you do"
}


def is_greeting_query(query: str) -> bool:
    cleaned = query.strip().lower().rstrip("!.,?")
    if cleaned in GREETINGS:
        return True
    words = cleaned.split()
    if len(words) <= 3 and any(w in GREETINGS for w in words):
        question_words = {
            "where", "what", "how", "why", "who", "which", "can", "could",
            "would", "is", "are", "tell", "explain", "find", "search", "show"
        }
        if not any(w in question_words for w in words):
            return True
    return False


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
        run_manager: Any = None,
    ) -> None:
        self.db = db
        self.config = config
        self.run = run
        self.thread = thread
        self.message = message
        self.provider = provider
        self.run_manager = run_manager

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
        self.executor = PipelineStepExecutor(self._pipeline_context, run_manager=self.run_manager)

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
        if state.is_greeting:
            response = state.greeting_response
        else:
            response = await self._generate(self.message.content, state.context)
        return self._build_result(state, response)

    async def execute_stream(self):
        """
        Executes the RAG pipeline in streaming mode.

        Yields ``{"type": "token", "content": chunk}`` events during generation,
        followed by a final ``{"type": "complete", "result": {...}}`` event.
        """
        state = await self._prepare()
        if state.is_greeting:
            response = state.greeting_response
            yield {"type": "token", "content": response}
            yield {
                "type": "complete",
                "result": self._build_result(state, response),
            }
            return

        response_parts: list[str] = []
        async for chunk in self._generate_stream(self.message.content, state.context):
            response_parts.append(chunk)
            yield {"type": "token", "content": chunk}
        yield {
            "type": "complete",
            "result": self._build_result(state, "".join(response_parts)),
        }

    def as_runnable(self, stream: bool = False):
        """
        Exposes the pipeline as a standard LangChain Runnable.
        Supports LCEL composition, .ainvoke(), and .astream() while preserving
        all underlying factory configurations and step tracking.
        """
        from langchain_core.runnables import RunnableLambda

        if stream:
            async def _run_stream(inputs: Any):
                query = self.message.content
                if isinstance(inputs, dict):
                    query = inputs.get("query") or inputs.get("question") or query
                elif isinstance(inputs, str) and inputs.strip():
                    query = inputs

                original_content = self.message.content
                self.message.content = query
                try:
                    async for event in self.execute_stream():
                        yield event
                finally:
                    self.message.content = original_content

            return RunnableLambda(_run_stream)

        async def _run(inputs: Any):
            query = self.message.content
            if isinstance(inputs, dict):
                query = inputs.get("query") or inputs.get("question") or query
            elif isinstance(inputs, str) and inputs.strip():
                query = inputs

            original_content = self.message.content
            self.message.content = query
            try:
                return await self.execute()
            finally:
                self.message.content = original_content

        return RunnableLambda(_run)


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

        # Fast path: greeting check to skip expensive LLM/vector search pipeline
        if is_greeting_query(self.message.content):
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

            if knowledge_bases:
                kb_names = ", ".join(f"**{kb.name}**" for kb in knowledge_bases)
                greeting_text = (
                    f"Hi! How are you? What would you like to get info from your "
                    f"knowledge base containing {kb_names}?"
                )
            else:
                greeting_text = "Hi! How are you? What would you like to get info from your knowledge base?"

            self._log_stage("prepare", "greeting_bypassed")
            return PipelineState(
                memory=[],
                transformed_queries=[self.message.content],
                retrieved_documents=[],
                reranked_documents=[],
                context="",
                is_greeting=True,
                greeting_response=greeting_text,
            )

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
            input_data={"query": self.message.content, "memory_length": len(memory)},
            config=self.thread_config.query_transformation.model_dump(),
        )

        # 3. Retrieval — dense / sparse / hybrid / mmr across knowledge bases
        retrieved_documents = await self.executor.run(
            "retrieval",
            self._retrieve(transformed_queries),
            input_data={"queries": transformed_queries},
            config=self.thread_config.retrieval.model_dump(),
        )

        # 4. Reranking — cross-encoder reranking (skipped when disabled)
        reranked_documents = await self.executor.run(
            "reranking",
            self._rerank(transformed_queries, retrieved_documents),
            input_data={"queries": transformed_queries, "document_count": len(retrieved_documents)},
            config=self.thread_config.reranking.model_dump() if self.thread_config.reranking.enabled else None,
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

        results = await self.retrieval_workflow.retrieve(
            queries,
            knowledge_bases,
            retrieval_config,
        )

        if self.run_manager:
            await self.run_manager.record_retrieval(
                query=", ".join(queries),
                retriever_type=retrieval_config.strategy,
                top_k=retrieval_config.top_k,
                documents=results,
            )

        return results

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
        
        results = rerank_documents(queries, documents, config)

        if self.run_manager:
            await self.run_manager.record_rerank(results)

        return results

    async def _generate(self, query: str, context: str) -> str:
        """
        Generate a non-streaming answer using the configured LLM provider.
        """
        self._log_stage("generation", "waiting")
        step = None
        if self.run_manager:
            step = await self.run_manager.start_step(
                "generation",
                input_data={"query": query, "context_length": len(context)},
                config=self.thread_config.generation.model_dump(),
            )
        prompt = build_generation_prompt(
            query,
            context,
            citation_enforced=self.thread_config.generation.citation_enforced,
        )
        try:
            response = await self.provider.ainvoke_safe(prompt)
            self._log_stage("generation", "completed")
            if self.run_manager and step:
                await self.run_manager.complete_step(step, output_data={"response": response})
            return response
        except Exception as e:
            self._log_stage("generation", "failed")
            if self.run_manager and step:
                await self.run_manager.fail_step(step, str(e))
            raise

    async def _generate_stream(self, query: str, context: str):
        """
        Stream generation tokens from the configured LLM provider.
        """
        self._log_stage("generation", "waiting")
        step = None
        if self.run_manager:
            step = await self.run_manager.start_step(
                "generation",
                input_data={"query": query, "context_length": len(context)},
                config=self.thread_config.generation.model_dump(),
            )
        prompt = build_generation_prompt(
            query,
            context,
            citation_enforced=self.thread_config.generation.citation_enforced,
        )
        full_response = ""
        try:
            async for chunk in self.provider.astream(prompt):
                full_response += chunk
                yield chunk
            self._log_stage("generation", "completed")
            if self.run_manager and step:
                await self.run_manager.complete_step(step, output_data={"response": full_response})
        except Exception as e:
            self._log_stage("generation", "failed")
            if self.run_manager and step:
                await self.run_manager.fail_step(step, str(e))
            raise

