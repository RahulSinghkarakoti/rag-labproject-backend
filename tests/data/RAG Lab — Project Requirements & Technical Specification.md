# RAG Lab
## Interactive RAG Learning & Experimentation Platform

**Working name:** RAG Lab  
**Type:** Educational / Developer Tool / RAG Experimentation Platform  
**Primary goal:** Help students and developers understand how Retrieval-Augmented Generation works by allowing them to visually experiment with different RAG techniques and compare their results.

---

# 1. Project Vision

RAG Lab is an interactive platform where a user can upload or provide a knowledge source and build a RAG pipeline by selecting different techniques.

Instead of treating RAG as a black box:

> Document → RAG → Answer

the platform exposes every important stage:

> Document → Loading → Chunking → Embedding → Vector Store → Retrieval → Reranking → Context → LLM → Answer

Users can change individual components, run the pipeline, inspect intermediate results, and compare different configurations.

The platform should work both as:

1. **A learning tool** — understand how RAG works.
2. **An experimentation tool** — test different RAG configurations.
3. **A practical RAG builder** — discover a configuration suitable for a real use case.

---

# 2. Core User Problem

RAG is often taught as:

```text
Documents
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retriever
    ↓
LLM
```

This hides many important decisions:

- How large should chunks be?
- Should chunks overlap?
- Which chunking strategy should be used?
- Which embedding model should be used?
- Should similarity search or MMR be used?
- How many documents should be retrieved?
- Should retrieval use a similarity threshold?
- Does reranking improve results?
- How much context is actually reaching the LLM?
- Why did the RAG system produce a bad answer?

RAG Lab makes these decisions visible and experimentally testable.

---

# 3. Target Users

## Primary

### Students

Students learning:

- Python
- LangChain
- RAG
- embeddings
- vector databases
- information retrieval
- LLM applications

### Developers

Developers who want to experiment with:

- chunking strategies
- embedding models
- retrievers
- reranking
- RAG architectures

### AI/ML learners

Users who want to understand the relationship between:

```text
Text
 ↓
Chunks
 ↓
Vectors
 ↓
Similarity
 ↓
Retrieval
 ↓
LLM
```

---

# 4. Core Product Concept

The application has a visual pipeline builder.

Example:

```text
┌──────────────┐
│ Data Source  │
└──────┬───────┘
       ↓
┌──────────────┐
│   Loader     │
└──────┬───────┘
       ↓
┌──────────────┐
│  Chunking    │
└──────┬───────┘
       ↓
┌──────────────┐
│  Embeddings  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Vector Store │
└──────┬───────┘
       ↓
┌──────────────┐
│  Retriever   │
└──────┬───────┘
       ↓
┌──────────────┐
│  Reranker    │
└──────┬───────┘
       ↓
┌──────────────┐
│     LLM      │
└──────────────┘
```

Every node can be configured.

---

# 5. Recommended Technology Stack

## Frontend

### Next.js

Use:

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui

Why:

- You already have experience with React/Next.js.
- TypeScript gives good type safety.
- Tailwind makes the UI fast to build.
- shadcn/ui provides ready-made components without forcing a large UI framework.
- Next.js is suitable for a polished developer-tool interface.

### Visualization

Use:

**React Flow**

for the visual RAG pipeline.

Example:

```text
[Document]
     ↓
[Chunker]
     ↓
[Embedding]
     ↓
[FAISS]
     ↓
[Retriever]
     ↓
[LLM]
```

Nodes can display their current configuration.

---

# 6. Backend

## Python + FastAPI

Recommended backend:

```text
Python
   ↓
FastAPI
   ↓
Pydantic
```

FastAPI will expose APIs such as:

```text
POST /documents
POST /experiments
POST /experiments/{id}/run
POST /retrieve
POST /chat
GET  /experiments/{id}
```

Why FastAPI:

- Easy to learn.
- Excellent Python ecosystem.
- Automatic API documentation.
- Async support.
- Pydantic validation.
- Very suitable for AI/RAG applications.

---

# 7. RAG / AI Stack

Use Python libraries directly where possible.

## Core

```text
Python
FastAPI
Pydantic
```

## RAG

```text
LangChain
```

Use LangChain selectively rather than hiding everything behind it.

The project should teach what actually happens underneath.

For example:

```text
SentenceTransformer
        ↓
numpy vector
        ↓
FAISS
        ↓
similarity search
```

should remain understandable even if LangChain provides a convenient wrapper.

---

# 8. Embedding Models

Initial models:

```text
all-MiniLM-L6-v2
BAAI/bge-small-en-v1.5
```

Later:

```text
BGE-base
E5
other open-source embedding models
```

The UI should allow:

```text
Embedding Model

○ all-MiniLM-L6-v2
○ BGE-small-en-v1.5
○ BGE-base
```

The system should display:

- embedding dimension
- model name
- approximate model size
- local/cloud
- CPU/GPU support

---

# 9. Document Sources

## MVP

Support:

```text
PDF
TXT
Markdown
YouTube transcript
```

Later:

```text
CSV
DOCX
HTML
JSON
Web pages
```

YouTube is particularly useful because the project can preserve:

```text
video_id
start
end
```

and display:

> Jump to 04:32

---

# 10. Chunking Module

The chunking module should allow users to experiment with different strategies.

## MVP

### Recursive Character Text Splitter

Settings:

```text
Chunk size: 500
Chunk overlap: 50
```

### Character splitter

### Token-based splitter

---

## Later

### Sentence-based splitting

### Paragraph-based splitting

### Semantic chunking

The UI should show the actual resulting chunks.

Example:

```text
Original document

──────────────────────────────

Chunk 1
"The transformer architecture..."

──────────────────────────────

Chunk 2
"...uses attention mechanisms..."

──────────────────────────────
```

Users should be able to click a chunk and inspect:

- text
- token count
- character count
- source location
- chunk ID

---

# 11. Chunk Visualization

This is one of the most important educational features.

For example:

```text
Document
────────────────────────────────────────────

██████████████████
       Chunk 1

       ██████████████████
              Chunk 2

              ███████████████████
                     Chunk 3
```

For overlap:

```text
Chunk 1
████████████████████

             Chunk 2
             ████████████████████

                         Chunk 3
                         █████████████████
```

This allows users to visually understand chunk overlap.

---

# 12. Vector Store Module

Initial vector stores:

```text
FAISS
Chroma
```

The platform should explain:

```text
Vector Store

FAISS

Documents: 152
Dimensions: 384
Index type: IndexFlatIP
```

Users should be able to inspect:

- vector count
- embedding dimensions
- storage size
- index type

---

# 13. Retrieval Module

Initial retrieval methods:

### Similarity Search

```text
Top K = 5
```

### Similarity Search + Threshold

```text
K = 5
Threshold = 0.70
```

### MMR

Settings:

```text
K = 5
Fetch K = 20
Lambda = 0.5
```

---

# 14. Retrieval Visualization

For every query show:

```text
QUERY

"What is retrieval augmented generation?"

        ↓

Retrieved documents

1. Chunk 42
   Similarity: 0.91

2. Chunk 17
   Similarity: 0.87

3. Chunk 83
   Similarity: 0.81

4. Chunk 12
   Similarity: 0.76

5. Chunk 51
   Similarity: 0.72
```

Each result should be expandable.

Display:

```text
Text
Metadata
Score
Source
Timestamp
Chunk ID
```

---

# 15. Embedding Visualization

For educational purposes, vectors can be projected into 2D.

Pipeline:

```text
384-dimensional vectors
          ↓
        PCA
          ↓
       2D plot
```

Display:

```text
              Chunk 7 •
                    /
           • Chunk 3
                  /
               ★ Query
                  \
                   • Chunk 12
```

The user should be able to click points and inspect the corresponding text.

This is not intended to represent the actual embedding space perfectly; it is a visualization of the projection.

---

# 16. Reranking

Later feature.

Pipeline:

```text
Query
 ↓
Retriever
 ↓
Top 10
 ↓
Reranker
 ↓
Top 3
 ↓
LLM
```

The UI should show:

```text
Before reranking

Chunk A  0.91
Chunk B  0.87
Chunk C  0.84


After reranking

Chunk C  0.96
Chunk A  0.92
Chunk B  0.81
```

This makes the purpose of reranking easy to understand.

---

# 17. Prompt Configuration

Allow users to edit the final prompt.

Default:

```text
You are a question-answering assistant.

Answer only using information from the provided context.

If the context does not contain enough information,
say:

"I don't know based on the provided context."

Context:
{context}

Question:
{question}
```

Show:

```text
Retrieved Context
       ↓
Prompt Template
       ↓
LLM
```

---

# 18. LLM Module

Initial support can be:

```text
Groq
```

Later:

```text
OpenAI
Ollama
local Hugging Face models
other providers
```

The interface should expose:

```text
Model
Temperature
Max tokens
```

For educational experiments, temperature should be configurable.

---

# 19. RAG Pipeline Debugger

This should be one of the major features.

When the user asks:

> What is NLM?

the system should show:

```text
① Query
"What is NLM?"

        ↓

② Query Embedding
384 dimensions

        ↓

③ Retrieval

Chunk 12 — 0.91
Chunk 43 — 0.84
Chunk 19 — 0.79

        ↓

④ Context

3 chunks
1,247 tokens

        ↓

⑤ Prompt

System instructions
+
Context
+
Question

        ↓

⑥ LLM

        ↓

⑦ Answer
```

The student can inspect every stage.

---

# 20. Experiment System

Users should be able to save configurations.

Example:

## Experiment A

```text
Chunker: Recursive
Chunk size: 500
Overlap: 50
Embedding: MiniLM
Retriever: Similarity
K: 5
```

## Experiment B

```text
Chunker: Recursive
Chunk size: 1000
Overlap: 100
Embedding: BGE
Retriever: MMR
K: 5
```

Then run the same questions against both.

---

# 21. Evaluation

Later introduce a basic evaluation system.

Metrics could include:

### Retrieval

```text
Relevant chunks retrieved
Retrieval similarity
Recall@K
Precision@K
```

### Generation

```text
Answer relevance
Context relevance
Faithfulness / groundedness
```

### Performance

```text
Embedding latency
Retrieval latency
LLM latency
Total latency
```

The platform should show the underlying measurements rather than reducing everything to a single unexplained "RAG score."

---

# 22. Experiment Comparison

Example:

```text
                    Experiment A    Experiment B

Chunk size              500             1000

Embedding             MiniLM             BGE

Retriever            Similarity          MMR

Retrieved chunks         5                 5

Relevant chunks         3                 4

Context tokens        1800              2200

Retrieval latency       15ms              22ms
```

Users can determine which configuration behaves better for their particular dataset.

---

# 23. Dataset-Level Configuration

Every dataset should have its own configuration.

Example:

```text
Dataset
│
├── Source
│
├── Chunking
│
├── Embedding
│
├── Vector Store
│
├── Retriever
│
├── Reranker
│
└── Prompt
```

This allows users to build:

```text
My YouTube RAG
My PDF RAG
My Documentation RAG
My Research Paper RAG
```

---

# 24. Recommended Architecture

```text
                    FRONTEND
               Next.js + TypeScript
                        │
                        │ REST API
                        ▼
                    FASTAPI
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Documents      RAG Engine    Experiments
          │             │             │
          │        ┌────┴─────┐       │
          │        │          │       │
          │    Chunking   Retrieval   │
          │        │          │       │
          │    Embeddings  Reranking │
          │        │          │       │
          │        └────┬─────┘       │
          │             │             │
          │        Vector Store       │
          │             │             │
          │          FAISS            │
          │                           │
          └───────────┬───────────────┘
                      │
                  PostgreSQL
```

---

# 25. Storage

## MVP

Use:

```text
SQLite
+
local filesystem
```

This keeps development simple.

Store:

```text
data/
├── datasets/
├── vectorstores/
├── experiments/
└── uploads/
```

## Production

Move to:

```text
PostgreSQL
Object storage
```

FAISS indexes can remain on disk for local deployments.

---

# 26. Database Models

Initial database entities:

### Dataset

```text
id
name
description
source_type
created_at
```

### Document

```text
id
dataset_id
source
content
metadata
```

### Experiment

```text
id
dataset_id
name
configuration
created_at
```

### QueryRun

```text
id
experiment_id
question
answer
latency
created_at
```

### RetrievalResult

```text
id
query_run_id
document_id
rank
score
```

---

# 27. Backend API

Initial API design:

```text
POST /api/datasets
GET  /api/datasets

POST /api/datasets/{id}/ingest

GET  /api/datasets/{id}/chunks

POST /api/experiments

GET  /api/experiments/{id}

POST /api/experiments/{id}/run

POST /api/experiments/{id}/retrieve

POST /api/experiments/{id}/chat

GET  /api/experiments/{id}/runs
```

---

# 28. Frontend Pages

## Dashboard

Show:

```text
Datasets
Experiments
Recent runs
```

## Dataset Explorer

```text
Source
Documents
Chunks
Metadata
Statistics
```

## RAG Builder

Main page:

```text
┌─────────────────────────────────────────────┐
│                 RAG PIPELINE                │
│                                             │
│ [Source] → [Chunker] → [Embedding] → ... │
│                                             │
└─────────────────────────────────────────────┘

Configuration Panel
────────────────────
Chunk size
Overlap
Embedding model
Retriever
K
...
```

## Retrieval Explorer

Show:

```text
Query
 ↓
Retrieved chunks
 ↓
Scores
 ↓
Metadata
```

## Experiment Comparison

Side-by-side comparison of configurations.

## Chat

Normal chatbot interface with an expandable:

> Show RAG process

button.

---

# 29. UI Design Direction

Use a developer-tool style interface.

Recommended:

```text
Next.js
Tailwind
shadcn/ui
React Flow
Recharts
```

Layout:

```text
┌─────────────────────────────────────────────┐
│ RAG Lab                              Run ▶  │
├──────────────┬──────────────────────────────┤
│              │                              │
│ Configuration│       Pipeline               │
│              │                              │
│ Chunking     │ [Source] → [Chunker] → ... │
│ Embedding    │                              │
│ Retriever    │                              │
│ Reranker     │                              │
│ LLM          │                              │
│              │                              │
├──────────────┴──────────────────────────────┤
│ Retrieval / Chunks / Logs / Answer          │
└─────────────────────────────────────────────┘
```

---

# 30. Python Learning Goals

The backend should deliberately expose you to real Python concepts.

You will practice:

```text
Python
├── classes
├── dataclasses
├── type hints
├── async/await
├── generators
├── decorators
├── exception handling
├── file handling
├── JSON
├── Pydantic
├── API development
├── NumPy
├── vector operations
├── embeddings
├── FAISS
├── database access
├── testing
└── background processing
```

Don't hide everything behind LangChain.

For example, implement a basic vector search yourself once:

```text
Embedding vectors
       ↓
Cosine similarity
       ↓
Top K
```

Then compare your implementation with FAISS.

That will teach you considerably more than simply calling:

```python
vectorstore.similarity_search()
```

---

# 31. Recommended Project Structure

```text
rag-lab/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── datasets.py
│   │   │   ├── experiments.py
│   │   │   └── chat.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── dependencies.py
│   │   │
│   │   ├── models/
│   │   │   ├── dataset.py
│   │   │   └── experiment.py
│   │   │
│   │   ├── rag/
│   │   │   ├── loaders.py
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   ├── vectorstores.py
│   │   │   ├── retrievers.py
│   │   │   ├── rerankers.py
│   │   │   └── pipeline.py
│   │   │
│   │   └── services/
│   │       ├── ingestion.py
│   │       ├── experiments.py
│   │       └── evaluation.py
│   │
│   ├── tests/
│   └── pyproject.toml
│
├── frontend/
│   │
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── hooks/
│   ├── types/
│   └── package.json
│
├── data/
│   ├── uploads/
│   ├── datasets/
│   └── vectorstores/
│
└── README.md
```

---

# 32. MVP Scope

Do NOT build everything initially.

The first version should only support:

```text
PDF / TXT / YouTube
        ↓
Recursive chunking
        ↓
MiniLM embeddings
        ↓
FAISS
        ↓
Similarity retrieval
        ↓
LLM
```

UI controls:

```text
Dataset
Chunk size
Chunk overlap
Top K
LLM
```

And visualizations:

```text
Chunks
Retrieved documents
Similarity scores
Final context
Final answer
```

That alone is already a useful project.

---

# 33. Phase 2

Add:

```text
Multiple chunking strategies
Multiple embedding models
MMR
Similarity threshold
Vector visualization
```

---

# 34. Phase 3

Add:

```text
Reranking
Experiment comparison
Evaluation
Latency measurements
Token usage
```

---

# 35. Phase 4

Add:

```text
Semantic chunking
Hybrid search
BM25
Keyword + vector retrieval
Query rewriting
Multi-query retrieval
Parent-child retrieval
Context compression
```

This eventually becomes a serious RAG experimentation platform.

---

# 36. Phase 5 — RAG Recommendation

Given:

```text
Dataset
+
Questions
+
Evaluation results
```

the system can help identify which configuration performs well on that dataset.

For example:

```text
Dataset: Technical Documentation

Tested:
────────────────────────────
Recursive + MiniLM + Similarity
Semantic + BGE + MMR
Recursive + BGE + MMR

Results:
────────────────────────────
Retrieval metrics
Latency
Context size
Answer quality
```

The platform can then explain the observed tradeoffs rather than simply declaring one configuration universally "best."

---

# 37. Important Product Principle

The application should never become a black box itself.

Every result should be explainable.

For example, instead of:

```text
RAG Score: 87%
```

show:

```text
Why did this answer happen?

Retrieved:
✓ Chunk 12
✓ Chunk 48
✓ Chunk 73

Context:
1,842 tokens

Retrieval:
Similarity search

Top K:
3

Embedding:
all-MiniLM-L6-v2

LLM:
...

Prompt:
...

Latency:
1.8 seconds
```

The student should be able to trace:

```text
Question
   ↓
Embedding
   ↓
Retrieved chunks
   ↓
Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

---

# 38. Final Recommended Stack

## Frontend

```text
Next.js
TypeScript
Tailwind CSS
shadcn/ui
React Flow
Recharts
```

## Backend

```text
Python
FastAPI
Pydantic
SQLAlchemy
SQLite → PostgreSQL
```

## RAG

```text
LangChain
Sentence Transformers
FAISS
NumPy
```

## LLM

Start with:

```text
Groq
```

Later:

```text
Ollama
OpenAI
local Hugging Face models
```

## Testing

```text
pytest
Vitest / Playwright
```

## Development

```text
uv
Python 3.12+
Node.js
npm/pnpm
Git
Docker later
```

---

# 39. First Milestone

The first working version should answer this question:

> "Can I upload a document, change the chunk size, run RAG, and visually see exactly which chunks were retrieved and why?"

If yes, **MVP 1 is successful.**

Then every subsequent feature should build on that foundation rather than expanding the scope all at once.

# 40. Long-Term Vision

RAG Lab eventually becomes:

> **An interactive laboratory for understanding, debugging, experimenting with, and designing RAG systems.**

The user doesn't simply build a chatbot.

They learn:

```text
Why chunking matters
Why embeddings matter
Why retrieval fails
Why MMR behaves differently
Why reranking helps
Why context quality matters
Why some RAG systems hallucinate
How different datasets require different strategies
```

That makes the project both a **strong portfolio project** and a practical way to deepen your Python/RAG knowledge.