# RAG Backend Pipeline Architecture & Flows

Welcome to the **RAG Backend Pipeline Documentation**. This directory contains detailed flow charts and technical breakdowns for each pipeline component in the backend.

---

## 📐 System Architecture Overview

```mermaid
flowchart TD
    subgraph Client ["Client / Frontend"]
        User UI
    end

    subgraph API ["FastAPI Endpoints"]
        DocAPI["/api/v1/documents"]
        ThreadAPI["/api/v1/threads"]
        SSEHub["SSE Event Publisher"]
    end

    subgraph Ingestion ["Ingestion Subsystem"]
        Worker["Background Ingestion Worker"]
        Loader["LoaderFactory & Detectors"]
        Chunker["ChunkerFactory (Recursive/Semantic/etc)"]
        Embedder["SentenceTransformerEmbedder"]
        PgVector["PgVector Store"]
    end

    subgraph QueryEngine ["Query Execution Engine"]
        Engine["QueryEngine"]
        QPipeline["QueryPipeline Orchestrator"]
        Greeting["Greeting Bypass Check"]
        Memory["MemoryManager"]
        Transformer["QueryTransformationProcessor"]
        Retrieval["RetrievalWorkflow (Dense/Sparse/Hybrid/MMR)"]
        Reranker["CrossEncoderReranker"]
        PromptBuilder["Context & Prompt Builder"]
        LLM["LLM Generation Provider (Gemini/Groq/OpenAI)"]
    end

    subgraph DB ["PostgreSQL / Database"]
        Postgres["AsyncSession DB"]
    end

    %% Ingestion Flow Connections
    DocAPI -->|1. Upload File & Create Doc| Postgres
    DocAPI -->|2. Dispatch Task| Worker
    Worker -->|3. Load File| Loader
    Worker -->|4. Split Text| Chunker
    Worker -->|5. Generate Vectors| Embedder
    Worker -->|6. Store Embeddings & Chunks| PgVector
    Worker -->|SSE Updates| SSEHub
    SSEHub -.->|Real-time Events| Client

    %% Query Flow Connections
    ThreadAPI -->|1. Send Message| Engine
    Engine -->|2. Execute Pipeline| QPipeline
    QPipeline -->|Check Greetings| Greeting
    QPipeline -->|3. Fetch History| Memory
    QPipeline -->|4. Transform Query| Transformer
    QPipeline -->|5. Search Documents| Retrieval
    Retrieval <-->|Vector & BM25 Query| PgVector
    QPipeline -->|6. Rerank Candidates| Reranker
    QPipeline -->|7. Assemble Prompt| PromptBuilder
    QPipeline -->|8. Generate Tokens| LLM
    Engine -->|9. Persist Run, Message & Steps| Postgres
    Engine -.->|Streaming Tokens| Client
```

---

## 🗂 Pipeline Flow Index

Click on any pipeline below to view its full Mermaid diagram and execution workflow:

1. 📥 [**Document Ingestion Pipeline**](ingestion_pipeline.md)
   - Covers document upload, file type detection, custom loaders, chunking strategies, embedding generation, vector store insertion, and real-time SSE progress reporting.

2. 🔍 [**Query Execution Pipeline**](query_pipeline.md)
   - Covers user query submission, fast greeting bypass, memory context loading, query transformation, multi-knowledge-base retrieval, cross-encoder reranking, prompt context assembly, LLM streaming, and run/step audit persistence.

3. 🔎 [**Retrieval Workflow Pipeline**](retrieval_pipeline.md)
   - Detailed breakdown of dense vector search, sparse BM25 keyword search, hybrid weighted fusion, and Maximal Marginal Relevance (MMR) candidate deduplication.

4. 🔄 [**Query Transformation Pipeline**](query_transformation_pipeline.md)
   - Detailed breakdown of query processing modes: `PassThrough`, `Query Rewriter`, `HyDE` (Hypothetical Document Embeddings), `Multi-Query Expander`, `Step-Back Prompting`, and `Sub-Query Decomposition`.

---

## 🛠 Tech Stack & Key Components

| Subsystem | Components | File Path |
| :--- | :--- | :--- |
| **Ingestion Worker** | Background Worker & Pipeline | [`app/workers/ingestion.py`](../app/workers/ingestion.py), [`app/rag/pipeline/ingestion.py`](../app/rag/pipeline/ingestion.py) |
| **Loaders** | Multi-Format File Loaders | [`app/rag/loaders/`](../app/rag/loaders/) |
| **Chunking** | Text Splitters & Chunker Factory | [`app/rag/chunking/`](../app/rag/chunking/) |
| **Embeddings** | SentenceTransformer Embedder | [`app/rag/embeddings/`](../app/rag/embeddings/) |
| **Vector Store** | PostgreSQL PgVector Integration | [`app/rag/vectorstore/pgvector.py`](../app/rag/vectorstore/pgvector.py) |
| **Query Engine** | Engine & Query Pipeline Orchestrator | [`app/rag/engine.py`](../app/rag/engine.py), [`app/rag/pipeline/query.py`](../app/rag/pipeline/query.py) |
| **Query Transformers**| Processor & Strategy Classes | [`app/rag/query/`](../app/rag/query/) |
| **Retrieval Subsystem**| Dense, Sparse, Hybrid, MMR | [`app/rag/retrieval/`](../app/rag/retrieval/) |
| **Reranker** | CrossEncoder Model Registry & Reranker | [`app/rag/reranking/`](../app/rag/reranking/) |
| **LLM Providers** | Gemini, Groq, OpenAI Providers | [`app/rag/generation/`](../app/rag/generation/) |
