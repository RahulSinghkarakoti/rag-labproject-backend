# Document Ingestion Pipeline Flow

The **Document Ingestion Pipeline** processes raw files uploaded to a Knowledge Base. It handles file type detection, content extraction, chunking strategy resolution, vector embedding generation, PostgreSQL `pgvector` persistence, and real-time Server-Sent Events (SSE) status publishing.

---

## 📊 Ingestion Pipeline Flowchart

```mermaid
flowchart TD
    Start(["📄 Document Upload API (/api/v1/documents)"]) --> Step1["1. DB Record Creation<br/>Document status = 'pending'"]
    Step1 --> Step2["2. Dispatch Async Worker Task<br/>ingest_document(ctx, document_id)"]

    subgraph Worker ["Worker Task Context (app/workers/ingestion.py)"]
        Step2 --> FetchDoc["Fetch Document & Parent KnowledgeBase (with row lock)"]
        FetchDoc --> MarkProcessing["Mark Document status = 'processing'<br/>Publish SSE: status='processing' on user channel"]
        
        subgraph Pipeline ["IngestionPipeline Orchestrator (app/rag/pipeline/ingestion.py)"]
            MarkProcessing --> LoadStage["3. Document Loading Stage"]
            LoadStage --> DetectType["FileTypeDetector.detect(storage_path)<br/>(Identifies pdf, text, or csv)"]
            DetectType --> LoadFile["LoaderFactory.load(storage_path)<br/>(PDFLoader, TextFileLoader, or CSVFileLoader)"]
            LoadFile --> SanitizeText["Sanitize NUL bytes ('\x00' -> '') for PostgreSQL compatibility"]
            SanitizeText --> SSELoaded["Publish SSE: status='loading' -> status='loaded'"]

            SSELoaded --> ChunkStage["4. Document Chunking Stage"]
            ChunkStage --> InitChunker["ChunkerFactory.create(<br/>strategy, chunk_size, chunk_overlap, file_type, embedder<br/>)<br/>(Supports recursive, character, markdown, semantic)"]
            InitChunker --> SplitDocs["chunker.split(documents)"]
            SplitDocs --> SSEChunked["Publish SSE: status='chunking' -> status='chunked'"]

            SSEChunked --> EmbedStage["5. Document Embedding Stage"]
            EmbedStage --> InitEmbedder["SentenceTransformerEmbedder(model_name)<br/>(Validated via validate_embedding_model)"]
            InitEmbedder --> BatchLoop{"Batch Chunks (batch_size = 16)"}
            BatchLoop -->|Process Batch| EmbedBatch["embedder.embed_documents(batch)"]
            EmbedBatch --> SSEBatch["Publish SSE: status='embedding'<br/>(progress: embedded_count / total_chunks)"]
            SSEBatch --> BatchLoop
            BatchLoop -->|All Batches Complete| SSEEmbedded["Publish SSE: status='embedded'"]

            SSEEmbedded --> StoreStage["6. Vector Persistence Stage"]
            StoreStage --> PgVector["PgVectorStore.add_documents(<br/>chunks, embeddings, metadata, document_id<br/>)"]
            PgVector --> SSEStored["Publish SSE: status='storing' -> status='stored'"]
        end

        SSEStored --> MarkCompleted["7. Mark Document status = 'completed'<br/>Publish SSE: status='completed'"]
    end

    subgraph ErrorHandling ["Error Handling & Rollback"]
        Pipeline -->|Exception Occurs| Rollback["DB Transaction Rollback"]
        Rollback --> MarkFailed["Mark Document status = 'failed'<br/>Publish SSE: status='failed' with error details"]
    end

    MarkCompleted --> End(["✅ Ingestion Completed"])
    MarkFailed --> EndFailed(["❌ Ingestion Failed"])
```

---

## 🔬 Detailed Step-by-Step Breakdown

### 1. Document Upload & Task Dispatch
- **Endpoint**: [`POST /api/v1/documents`](../app/api/documents.py)
- **Action**: Validates file payload, writes uploaded file to disk storage, creates a database `Document` record with `status="pending"`, and dispatches the background ingestion task.

### 2. Worker Initialization & Lock
- **Module**: [`app/workers/ingestion.py`](../app/workers/ingestion.py)
- **Action**: 
  - Retrieves the `Document` with a row-level DB lock (`with_for_update()`) and resolves its parent `KnowledgeBase`.
  - Determines the target SSE channel (`ingestion_channel:{user_id}` or `ingestion_channel`).
  - Updates document status to `processing` and emits SSE event `ingestion_status` (`status: "processing"`).

### 3. File Type Detection & Loading
- **Modules**: [`app/rag/loaders/detector.py`](../app/rag/loaders/detector.py), [`app/rag/loaders/factory.py`](../app/rag/loaders/factory.py), [`app/rag/loaders/registry.py`](../app/rag/loaders/registry.py)
- **Action**:
  - `FileTypeDetector` inspects the file extension / MIME type to classify the format (`pdf`, `text`, or `csv`).
  - `LoaderFactory` fetches the registered loader (`PDFLoader`, `TextFileLoader`, or `CSVFileLoader`) from `LoaderRegistry`.
  - Content is extracted into LangChain `Document` objects and sanitized by stripping NUL bytes (`\x00`) to prevent PostgreSQL text column errors.
  - Emits SSE status events: `loading` followed by `loaded` (with total `document_count`).

### 4. Text Chunking & Strategy Resolution
- **Modules**: [`app/rag/chunking/factory.py`](../app/rag/chunking/factory.py), [`app/rag/chunking/splitter.py`](../app/rag/chunking/splitter.py)
- **Action**:
  - `ChunkerFactory` evaluates the Knowledge Base `chunking_config` (`strategy`, `chunk_size`, `chunk_overlap`) against document type compatibility rules.
  - Supported strategies:
    - `recursive`: `RecursiveChunker` (default, standard text/PDF hierarchy).
    - `character`: `CharacterChunker` (separator-based splitting).
    - `markdown`: `MarkdownChunker` (header-aware structure splitting).
    - `semantic`: `SemanticChunker` (embedding-based semantic breakpoint splitting).
  - Validates compatibility (e.g. falls back to `recursive` if `markdown` or `semantic` strategies are incompatible with tabular `csv`).
  - Emits SSE status events: `chunking` followed by `chunked` (with total `chunk_count`).

### 5. Vector Embedding Generation
- **Modules**: [`app/rag/embeddings/sentence_transformer.py`](../app/rag/embeddings/sentence_transformer.py), [`app/core/model_config.py`](../app/core/model_config.py)
- **Action**:
  - Validates requested embedding model via `validate_embedding_model()`.
  - Instantiates `SentenceTransformerEmbedder` for the Knowledge Base model configuration (e.g. `all-MiniLM-L6-v2`).
  - Processes chunk text in batches (`batch_size = 16`).
  - Emits SSE progress notifications per batch (`embedded_count` / `total_chunks`) and final `embedded` event.

### 6. Vector Store & Rich Metadata Persistence
- **Module**: [`app/rag/vectorstore/pgvector.py`](../app/rag/vectorstore/pgvector.py)
- **Action**:
  - Inserts document chunks, vector embeddings, and metadata into PostgreSQL using `pgvector`.
  - Attaches rich chunk metadata:
    - `filename`, `type` (MIME type), `file_type`, `size_bytes`, `storage_path`
    - `knowledge_base_id`, `embedding_model`, `embedding_dimension`
    - `chunking_strategy`, `chunking_config`
  - Emits SSE status events: `storing` followed by `stored`.

### 7. Completion & SSE Notification Stream
- Updates database `Document` status to `completed`.
- Emits final SSE event `ingestion_status` (`status: "completed"`).
- In case of failure: rolls back DB transaction, sets `status = "failed"`, and emits SSE event with the exception error message.
