# Document Ingestion Pipeline Flow

The **Ingestion Pipeline** handles raw document processing, text extraction, chunking, vector embedding, and storage into PostgreSQL using `pgvector`. It also provides real-time SSE progress updates to the client.

---

## 📊 Ingestion Pipeline Flowchart

```mermaid
flowchart TD
    Start(["📄 Document Upload API Trigger"]) --> Step1["1. DB Record Creation<br/>Document status = 'pending'"]
    Step1 --> Step2["2. Dispatch Async Worker Task<br/>ingest_document(ctx, document_id)"]
    
    subgraph Worker ["Worker Task Context (app/workers/ingestion.py)"]
        Step2 --> FetchDoc["Fetch Document & KB from DB"]
        FetchDoc --> MarkProcessing["Mark Document status = 'processing'<br/>Publish SSE event: status='processing'"]
        
        subgraph Pipeline ["IngestionPipeline Orchestrator (app/rag/pipeline/ingestion.py)"]
            MarkProcessing --> LoadStage["3. Document Loading Stage"]
            LoadStage --> DetectType["FileTypeDetector.detect(storage_path)"]
            DetectType --> LoadFile["LoaderFactory.load(storage_path)"]
            LoadFile --> SanitizeText["Sanitize NUL bytes ('\x00' -> '')"]
            SanitizeText --> SSELoaded["Publish SSE: status='loaded'"]
            
            SSELoaded --> ChunkStage["4. Document Chunking Stage"]
            ChunkStage --> InitChunker["ChunkerFactory.create(<br/>strategy, chunk_size, chunk_overlap<br/>)"]
            InitChunker --> SplitDocs["chunker.split(documents)"]
            SplitDocs --> SSEChunked["Publish SSE: status='chunked'"]
            
            SSEChunked --> EmbedStage["5. Document Embedding Stage"]
            EmbedStage --> InitEmbedder["SentenceTransformerEmbedder(model_name)"]
            InitEmbedder --> BatchLoop{"Batch Chunks (size=16)"}
            BatchLoop -->|Process Batch| EmbedBatch["embedder.embed_documents(batch)"]
            EmbedBatch --> SSEBatch["Publish SSE: status='embedding' (progress)"]
            SSEBatch --> BatchLoop
            BatchLoop -->|All Batches Complete| SSEEmbedded["Publish SSE: status='embedded'"]
            
            SSEEmbedded --> StoreStage["6. Vector Store Stage"]
            StoreStage --> PgVector["PgVectorStore.add_documents(<br/>chunks, embeddings, metadata<br/>)"]
            PgVector --> SSEStored["Publish SSE: status='stored'"]
        end

        SSEStored --> MarkCompleted["7. Mark Document status = 'completed'<br/>Publish SSE: status='completed'"]
    end

    subgraph ErrorHandling ["Error Handling"]
        Pipeline -->|Exception Occurs| Rollback["DB Rollback"]
        Rollback --> MarkFailed["Mark Document status = 'failed'<br/>Publish SSE: status='failed'"]
    end

    MarkCompleted --> End(["✅ Ingestion Completed"])
    MarkFailed --> EndFailed(["❌ Ingestion Failed"])
```

---

## 🔬 Detailed Step-by-Step Breakdown

### 1. Document Upload & Task Dispatch
- **Endpoint**: [`POST /api/v1/documents`](../app/api/documents.py)
- **Action**: Validates user upload, writes raw file to storage, creates a database record with `status="pending"`, and dispatches the background task to the ingestion worker.

### 2. Worker Initialization & DB Context
- **Module**: [`app/workers/ingestion.py`](../app/workers/ingestion.py)
- **Action**: 
  - Retrieves the `Document` and parent `KnowledgeBase` configuration (embedding model, chunking parameters).
  - Updates status to `processing`.
  - Emits SSE event: `ingestion_status` with `status: "processing"`.

### 3. File Type Detection & Loading
- **Module**: [`app/rag/loaders/`](../app/rag/loaders/)
- **Action**:
  - `FileTypeDetector` inspects file extension/MIME type.
  - `LoaderFactory` instantiates appropriate loader (`PDFLoader`, `DocxLoader`, `TextLoader`, etc.).
  - Extracts text content and sanitizes NULL bytes (`\x00`) to prevent PostgreSQL text column errors.
  - Emits SSE event: `status: "loaded"`.

### 4. Text Chunking
- **Module**: [`app/rag/chunking/`](../app/rag/chunking/)
- **Action**:
  - `ChunkerFactory` instantiates configured strategy (e.g. `RecursiveChunker`, `SemanticChunker`, `TokenChunker`).
  - Splits loaded documents into chunks based on `chunk_size` and `chunk_overlap`.
  - Emits SSE event: `status: "chunked"`.

### 5. Vector Embedding Generation
- **Module**: [`app/rag/embeddings/sentence_transformer.py`](../app/rag/embeddings/sentence_transformer.py)
- **Action**:
  - `SentenceTransformerEmbedder` processes chunks in batches (default `batch_size = 16`).
  - Emits progress SSE events for each batch (`embedded_count` / `total_chunks`).
  - Emits SSE event: `status: "embedded"`.

### 6. PgVector Database Persistence
- **Module**: [`app/rag/vectorstore/pgvector.py`](../app/rag/vectorstore/pgvector.py)
- **Action**:
  - Inserts vector embeddings, chunk text, and rich metadata into PostgreSQL `vector` table.
  - Attached Metadata includes: `filename`, `file_type`, `size_bytes`, `knowledge_base_id`, `embedding_model`, `embedding_dimension`, `chunking_strategy`, and `chunking_config`.
  - Emits SSE event: `status: "stored"`.

### 7. Completion & SSE Event Stream
- Updates database document record `status="completed"`.
- Emits final SSE event `status: "completed"`.
