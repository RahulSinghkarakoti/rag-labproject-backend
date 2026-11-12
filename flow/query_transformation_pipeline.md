# Query Transformation Pipeline Flow

The **Query Transformation Pipeline** prepares user input for optimal vector search and document retrieval. Coordinated by `QueryTransformationProcessor`, it resolves the active strategy from `ThreadConfig.query_transformation.mode` and generates one or more targeted search query strings.

---

## 📊 Query Transformation Flowchart

```mermaid
flowchart TD
    RawQuery(["💬 Raw Input Query + Conversation Memory"]) --> Processor["QueryTransformationProcessor.process()<br/>(app/rag/query/processor.py)"]

    Processor --> CheckEmpty{"Is Query Empty or Whitespace?"}
    CheckEmpty -->|Yes| EmptyReturn["Return []"]

    CheckEmpty -->|No| NormalizeConfig["Normalize QueryTransformationConfig<br/>(Handles legacy hyde_enabled & multi_query_count fields)"]
    NormalizeConfig --> ModeResolve{"Resolve Strategy Mode<br/>(norm_config.mode)"}

    subgraph Strategies ["Transformation Strategies"]
        ModeResolve -->|'none'| PassThrough["PassThroughTransformer<br/>(app/rag/query/passthrough.py)"]
        PassThrough --> ReturnRaw["Return [original_query]"]

        ModeResolve -->|'rewrite'| Rewriter["QueryRewriter<br/>(app/rag/query/rewriter.py)"]
        Rewriter --> LLMRewrite["LLM rephrases ambiguous query into standalone question using memory"]
        LLMRewrite --> ReturnRewrite["Return [rewritten_query]"]

        ModeResolve -->|'hyde'| HyDE["HyDETransformer<br/>(app/rag/query/hyde.py)"]
        HyDE --> LLMHyDE["LLM generates hypothetical answer passage"]
        LLMHyDE --> ReturnHyDE["Return [hypothetical_document_passage]"]

        ModeResolve -->|'multi_query'| Expander["MultiQueryExpander<br/>(app/rag/query/expander.py)"]
        Expander --> LLMExpand["LLM generates query_count alternative phrasings"]
        LLMExpand --> ReturnMulti["Return [query_1, query_2, ...]"]

        ModeResolve -->|'step_back'| StepBack["StepBackTransformer<br/>(app/rag/query/step_back.py)"]
        StepBack --> LLMStepBack["LLM derives broader high-level concept question"]
        LLMStepBack --> ReturnStepBack["Return [original_query, step_back_query]"]

        ModeResolve -->|'sub_queries'| Decomposer["SubQueryDecomposer<br/>(app/rag/query/sub_queries.py)"]
        Decomposer --> LLMDecompose["LLM breaks complex multi-part question into sub-questions"]
        LLMDecompose --> ReturnSub["Return [sub_query_1, sub_query_2, ...]"]
    end

    ReturnRaw --> FallbackCheck
    ReturnRewrite --> FallbackCheck
    ReturnHyDE --> FallbackCheck
    ReturnMulti --> FallbackCheck
    ReturnStepBack --> FallbackCheck
    ReturnSub --> FallbackCheck

    FallbackCheck{"Transformed Array Valid?"} -->|Valid| Output(["📋 Transformed Search Queries Array"])
    FallbackCheck -->|Empty / Error| FallbackRaw["Fallback to [original_query]"] --> Output
```

---

## 🔬 Detailed Strategy Breakdown

### 1. Configuration & Mode Normalization
- **Schema**: [`QueryTransformationConfig`](../app/schemas/thread.py)
- **Parameters**:
  - `mode`: `"none"` | `"rewrite"` | `"hyde"` | `"multi_query"` | `"step_back"` | `"sub_queries"`
  - `query_count`: Integer between 1 and 10 (default: 3).
  - `include_original`: Boolean (default: `True`).
  - `prompt_template`: Optional custom prompt instructions.
- **Legacy Field Mapping**: Automatically maps legacy boolean/integer settings (`hyde_enabled=True` -> `mode="hyde"`, `multi_query_count > 0` -> `mode="multi_query"`).

---

### 2. Strategy Implementations

#### 1. `none` (PassThrough)
- **Module**: [`app/rag/query/passthrough.py`](../app/rag/query/passthrough.py)
- **Behavior**: Performs no transformation. Returns `[original_query]`.
- **Use Case**: Fast retrieval, direct keyword search, or minimal latency requirements.

#### 2. `rewrite` (Query Rewriter)
- **Module**: [`app/rag/query/rewriter.py`](../app/rag/query/rewriter.py)
- **Behavior**: Analyzes conversation memory and reformulates conversational or ambiguous queries into standalone search questions (e.g., *"What did he say about it?"* -> *"What did Einstein say about special relativity in 1905?"*).
- **Use Case**: Multi-turn chat threads containing pronouns, indirect context references, or ellipsis.

#### 3. `hyde` (Hypothetical Document Embeddings)
- **Module**: [`app/rag/query/hyde.py`](../app/rag/query/hyde.py)
- **Behavior**: Prompts the LLM to generate a hypothetical "ideal answer" document passage. The vector embedding of this fake answer is then used during dense vector retrieval to match real documents with similar semantic structure.
- **Use Case**: Abstract questions where query syntax differs significantly from document passage syntax.

#### 4. `multi_query` (Multi-Query Expander)
- **Module**: [`app/rag/query/expander.py`](../app/rag/query/expander.py)
- **Behavior**: Prompts the LLM to generate `query_count` distinct query variants from different viewpoints, phrasing styles, or vocabulary choices.
- **Use Case**: Overcoming single-query keyword sensitivity and maximizing retrieval recall.

#### 5. `step_back` (Step-Back Prompting)
- **Module**: [`app/rag/query/step_back.py`](../app/rag/query/step_back.py)
- **Behavior**: Derives a higher-level step-back concept question (e.g., *"Why did my FastAPI endpoint throw 422?"* -> *"How does FastAPI request validation handle missing body parameters?"*) and returns both original and step-back queries.
- **Use Case**: Technical troubleshooting, deep reasoning, and retrieving underlying domain principles.

#### 6. `sub_queries` (Sub-Query Decomposition)
- **Module**: [`app/rag/query/sub_queries.py`](../app/rag/query/sub_queries.py)
- **Behavior**: Decomposes compound or multi-part questions into individual atomic sub-questions (e.g., *"Compare PostgreSQL and MongoDB scaling"* -> `["PostgreSQL scaling capabilities", "MongoDB scaling capabilities"]`).
- **Use Case**: Comparative analysis and complex multi-part prompts.

---

### 3. Fallback & Safety Guarantees
- If a strategy encounters an LLM provider error or returns an empty list, `QueryTransformationProcessor.process()` falls back to returning `[original_query]` so the retrieval pipeline is never starved of search input.
