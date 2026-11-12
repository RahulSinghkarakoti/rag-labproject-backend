# Query Execution Pipeline Flow

The **Query Execution Pipeline** coordinates the end-to-end RAG workflow for user prompts: message request handling, pipeline configuration validation, progressive conversation memory summarization, query transformation, multi-Knowledge-Base retrieval, cross-encoder reranking, prompt context assembly, LLM token streaming, cancellation management, and durable PostgreSQL audit persistence (`RunStep` & `ModelCall`).

---

## 📊 Query Execution Pipeline Flowchart

```mermaid
flowchart TD
    UserQuery(["💬 User Submits Message"]) --> API["FastAPI Thread Endpoint (/api/v1/threads)"]
    API --> Engine["QueryEngine.run() or run_stream()<br/>(app/rag/engine.py)"]

    subgraph QueryEngine ["Query Engine & Run Lifecycle"]
        Engine --> CreateRun["Set Run.status = 'running'"]
        CreateRun --> CheckCancel1["Check CancellationManager (app/core/cancellation.py)"]
        CheckCancel1 --> InitPipeline["Instantiate QueryPipeline & Validate ThreadConfig<br/>(app/rag/pipeline/query.py)"]

        subgraph Pipeline ["QueryPipeline Execution (_prepare & _generate)"]
            
            %% Stage 1: Memory
            InitPipeline --> Stage1["1. Memory Stage (app/rag/memory/manager.py)<br/>MemoryManager.get_conversation_context()"]
            Stage1 --> SummaryCheck{"Turns > Trigger Threshold?"}
            SummaryCheck -->|Yes| Compaction["ThreadSummarizer updates rolling summary<br/>Persists ThreadSummary record in DB"]
            SummaryCheck -->|No| LoadTurns["Load recent turns verbatim"]
            Compaction --> Stage2
            LoadTurns --> Stage2

            %% Stage 2: Transformation
            Stage2["2. Query Transformation Stage (app/rag/query/)<br/>QueryTransformationProcessor.process()"]
            Stage2 --> TransMode{"Configured Strategy Mode?"}
            TransMode -->|none| PassThru["PassThrough -> [original_query]"]
            TransMode -->|rewrite / hyde / multi_query / step_back / sub_queries| LLMTrans["LLM Transformer Execution"]
            PassThru --> Stage3
            LLMTrans --> Stage3

            %% Stage 3: Retrieval
            Stage3["3. Multi-KB Retrieval Stage (app/rag/retrieval/)<br/>RetrievalWorkflow.retrieve()"]
            Stage3 --> GroupKB["Group KBs by embedding model"]
            GroupKB --> SearchKBs["Dense, Sparse, Hybrid, or MMR Search"]
            SearchKBs --> Stage4

            %% Stage 4: Reranking
            Stage4["4. Reranking Stage (app/rag/reranking/)<br/>CrossEncoderReranker"]
            Stage4 --> RerankCheck{"Reranking Enabled?"}
            RerankCheck -->|Yes| ScoreDocs["Score candidate chunks against anchor query<br/>Truncate candidates to top_n"]
            RerankCheck -->|No| PassDocs["Pass retrieved candidate documents"]
            ScoreDocs --> Stage5
            PassDocs --> Stage5

            %% Stage 5 & 6: Prompt & Generation
            Stage5["5. Context Assembly (app/rag/memory/context.py)<br/>build_rag_context() & build_generation_prompt()"]
            Stage5 --> EmitContextStatus["Emit context_status event with token metrics"]
            EmitContextStatus --> Stage6["6. Generation Stage (app/rag/generation/)"]
            
            Stage6 --> ExecMode{"Execution Mode?"}
            ExecMode -->|run_stream()| StreamTokens["Provider.astream(prompt)<br/>Yield token events to client"]
            ExecMode -->|run()| NonStream["Provider.ainvoke_safe(prompt)"]
        end

        StreamTokens -->|Complete| Persist["7. Durable Persistence (_persist_run_result)"]
        NonStream --> Persist

        subgraph Persistence ["PostgreSQL Audit Trail"]
            Persist --> AddMsg["Create Assistant Message record with citation metadata"]
            Persist --> RecordSteps["Record RunStep records via PipelineStepExecutor:<br/>- memory, transformation, retrieval, reranking, generation"]
            Persist --> RecordCall["Record ModelCall record<br/>(provider, model, tokens, request/response)"]
            RecordCall --> MarkCompleted["Set Run.status = 'completed'"]
        end
    end

    subgraph CancellationAndErrors ["Cancellation & Exception Handling"]
        Pipeline -->|User Stopped / Disconnected| Cancelled["CancellationManager triggers CancelledError<br/>Set Run.status = 'cancelled'"]
        Pipeline -->|Model Provider / System Error| Failed["sanitize_model_error()<br/>Set Run.status = 'failed'<br/>Save run.error JSON"]
    end

    MarkCompleted --> ReturnResult(["Return Final Response & Output Message ID"])
    Cancelled --> ReturnResult
    Failed --> ReturnResult
```

---

## 🔬 Detailed Step-by-Step Breakdown

### 1. Engine & Configuration Validation
- **Classes**: [`QueryEngine`](../app/rag/engine.py), [`QueryPipeline`](../app/rag/pipeline/query.py)
- **Action**:
  - Resolves active `PipelineVersion` configuration into a validated [`ThreadConfig`](../app/schemas/thread.py).
  - Initializes LLM generation provider (`GroqProvider`, `GeminiProvider`, or `OpenAIProvider`) via `ProviderModelFactory`.
  - Instantiates `PgVectorStore`, `RetrievalWorkflow`, `MemoryManager`, and `PipelineStepExecutor`.
  - Sets `Run.status = "running"`.

### 2. Conversation Memory & Progressive Compaction
- **Module**: [`app/rag/memory/manager.py`](../app/rag/memory/manager.py), [`app/rag/memory/summarizer.py`](../app/rag/memory/summarizer.py)
- **Action**:
  - Loads prior chat turns for the thread up to `before_sequence`.
  - Keeps recent dialogue turns (`THREAD_SUMMARY_RECENT_TURNS`) verbatim for immediate context.
  - When unsummarized turns exceed `THREAD_SUMMARY_TRIGGER_THRESHOLD`, `ThreadSummarizer` compacts older turns into a rolling summary.
  - Stores summary in PostgreSQL `ThreadSummary` table (`summary`, `covered_until_message_id`, `token_count`).

### 3. Query Transformation Stage
- **Modules**: [`app/rag/query/processor.py`](../app/rag/query/processor.py), [`app/rag/query/transformer.py`](../app/rag/query/transformer.py)
- **Action**:
  - `QueryTransformationProcessor` inspects `ThreadConfig.query_transformation.mode`.
  - Executes configured technique (`none`, `rewrite`, `hyde`, `multi_query`, `step_back`, or `sub_queries`).
  - Produces an array of target search query strings for the retrieval workflow.

### 4. Multi-KB Candidate Retrieval
- **Module**: [`app/rag/retrieval/workflow.py`](../app/rag/retrieval/workflow.py)
- **Action**:
  - Groups all Knowledge Bases attached to the thread by embedding model.
  - Automatically expands retrieval `top_k` when reranking is enabled (`max(top_k, top_n) // query_count`).
  - Executes dense vector search (`retrieve_dense`), sparse BM25 search (`retrieve_sparse`), hybrid weighted score fusion ($\alpha$), or MMR (`maximal_marginal_relevance`).
  - Deduplicates candidates across transformed queries while preserving highest scores (`deduplicate_scored`).

### 5. Anchor Query Reranking Stage
- **Module**: [`app/rag/reranking/reranker.py`](../app/rag/reranking/reranker.py)
- **Action**:
  - If `ThreadConfig.reranking.enabled=True`, scores candidate chunks against the anchor user query (`message.content`).
  - Clamps candidate document content length to 1500 chars (~384 tokens) to prevent quadratic attention blowup during cross-encoder inference.
  - Sorts candidates by rerank score, applies optional `score_threshold`, and truncates list to top $N$ reranked documents (`top_n`).

### 6. Context & Prompt Assembly
- **Module**: [`app/rag/memory/context.py`](../app/rag/memory/context.py)
- **Action**:
  - Assembles formatted `[Context]` blocks including chunk text, document source metadata, and conversation summary.
  - Builds generation prompt and applies optional citation rules (`citation_enforced=True`).
  - In streaming mode, calculates context token metrics (`total_tokens`, `prompt_tokens`, `is_summarized`) and emits a `context_status` event.

### 7. LLM Response Generation & Token Streaming
- **Module**: [`app/rag/generation/`](../app/rag/generation/)
- **Action**:
  - Sends assembled prompt to generation provider (`GroqProvider`, `GeminiProvider`, or `OpenAIProvider`).
  - In non-streaming mode (`run()`): executes `ainvoke_safe()`.
  - In streaming mode (`run_stream()`): streams tokens chunk-by-chunk using `astream()`, yielding `token` events and stage tracking events (`step_start`, `step_complete`).

### 8. Cancellation Management
- **Module**: [`app/core/cancellation.py`](../app/core/cancellation.py)
- **Action**:
  - Periodically checks `cancellation_manager.is_cancelled(run_id)` at every stage.
  - If user stops execution or disconnects: aborts execution, sets `Run.status = "cancelled"`, and records cancellation details without writing incomplete output messages.

### 9. Durable Persistence & Audit Trail
- **Module**: [`app/rag/engine.py`](../app/rag/engine.py#L198)
- **Action**:
  - Saves assistant `Message` record with citation metadata (`index`, `name`, `mime_type`).
  - Writes structured `RunStep` records via `PipelineStepExecutor` (`memory`, `transformation`, `retrieval`, `reranking`, `generation`).
  - Records `ModelCall` metadata (tokens, model name, provider, request/response).
  - Sets `Run.status = "completed"`.
