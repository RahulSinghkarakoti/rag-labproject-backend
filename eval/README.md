# RAG Evaluation Guide & Command Reference

This guide explains how to run, configure, and alter evaluation benchmarks for this RAG platform using the local benchmark dataset.

---

## 1. Dataset Overview & Directory Structure

All evaluation scripts, dataset files, and reports are located in the dedicated [`eval/`](file:///home/rahulkarakoti/Documents/NOTEBOOKLLM-LITE/backend/eval) directory:

```text
eval/
├── dataset/
│   ├── documents.csv                         # 20 Enter the Gungeon wiki articles (the knowledge corpus)
│   ├── single_passage_answer_questions.csv   # 40 single-hop questions with ground-truth answers
│   ├── multi_passage_answer_questions.csv    # 40 multi-hop synthesis questions with ground-truth answers
│   └── no_answer_questions.csv               # 40 unanswerable questions (tests hallucination refusal)
├── ingest.py                                 # Ingestion script into PostgreSQL/PgVector
├── run_eval.py                               # Evaluation harness & benchmark runner
├── eval_report.md                            # Latest generated markdown scorecard
├── eval_results.csv                          # Latest per-query results CSV
└── README.md                                 # This guide
```

Total: **120 test cases** across 20 source documents.

---

## 2. Ingestion & Database Setup

Before running queries, documents must be chunked and indexed into PostgreSQL/PgVector.

### Check or Ingest Dataset:
```bash
.venv/bin/python eval/ingest.py
```
- If the Knowledge Base **`Eval - Enter The Gungeon`** already has 20 documents, it skips re-indexing automatically.

### Force Re-index (Wipe & Re-embed Chunks):
```bash
.venv/bin/python eval/ingest.py --reindex
```

---

## 3. Evaluation Commands & How to Alter Them

The evaluation harness is executed via:
```bash
.venv/bin/python eval/run_eval.py [OPTIONS]
```

### 3.1. Fast 5-Way Comparison ($0 LLM Cost)
Compare Pure Dense, Pure BM25, Hybrid, Hybrid + Reranker, and MMR side-by-side across all 120 questions:
```bash
.venv/bin/python eval/run_eval.py --compare-matrix
```
*Tip: Runs purely in vector memory in ~20 seconds without consuming API credits.*

---

### 3.2. End-to-End Evaluation (Retrieval + Generation + Judge)

#### Default Hybrid Run (Recommended):
```bash
.venv/bin/python eval/run_eval.py --strategy hybrid --alpha 0.5
```

#### Altering Retrieval Strategy:
Choose between `dense`, `sparse`, `hybrid`, or `mmr`:
```bash
# Pure Dense (PgVector cosine distance)
.venv/bin/python eval/run_eval.py --strategy dense

# Pure Sparse (BM25 keyword search)
.venv/bin/python eval/run_eval.py --strategy sparse

# MMR (Maximal Marginal Relevance for chunk diversity)
.venv/bin/python eval/run_eval.py --strategy mmr --mmr-lambda 0.6
```

#### Altering Hybrid Alpha ($\alpha$):
Alpha controls the balance between dense vector similarity and sparse BM25:
- `1.0`: 100% Dense, 0% BM25
- `0.0`: 0% Dense, 100% BM25
- `0.5`: 50/50 Balanced (Default)

```bash
# Keyword-heavy hybrid (great for exact names and IDs):
.venv/bin/python eval/run_eval.py --strategy hybrid --alpha 0.3

# Semantic-heavy hybrid (great for conceptual questions):
.venv/bin/python eval/run_eval.py --strategy hybrid --alpha 0.7
```

#### Enabling Cross-Encoder Reranking:
To add Stage-2 neural reranking (`ms-marco-MiniLM-L-6-v2`):
```bash
.venv/bin/python eval/run_eval.py --strategy hybrid --rerank
```

---

### 3.3. Speed & Cost Controls

#### Quick Dry Run (`--sample N`):
Test only the first `N` questions of each split to verify settings quickly:
```bash
# Tests only 2 single, 2 multi, and 2 no-answer questions (6 total):
.venv/bin/python eval/run_eval.py --sample 2 --strategy hybrid
```

#### Retrieval-Only Mode (`--retrieval-only`):
Test Hit@K and latency across any custom strategy without generating answers:
```bash
.venv/bin/python eval/run_eval.py --strategy mmr --retrieval-only
```

#### Testing a Specific Question Split (`--split`):
Choose from `single`, `multi`, `no_answer`, or `all`:
```bash
# Test ONLY unanswerable questions to measure hallucination refusal:
.venv/bin/python eval/run_eval.py --split no_answer

# Test ONLY multi-hop questions:
.venv/bin/python eval/run_eval.py --split multi
```

#### Altering Top-K Retrieved Passages (`--top-k`):
Change the number of chunks passed to the LLM:
```bash
# Pass 8 chunks instead of default 5:
.venv/bin/python eval/run_eval.py --strategy hybrid --top-k 8
```

---

### 3.4. Altering LLM Models & Custom Output Paths

```bash
# Use a specific Gemini model:
.venv/bin/python eval/run_eval.py --provider gemini --model gemini-3.8-flash

# Save reports to custom filenames:
.venv/bin/python eval/run_eval.py --report eval/my_experiment_report.md --csv eval/my_experiment_results.csv
```

---

## 4. Complete CLI Options Cheat-Sheet

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--compare-matrix` | flag | `False` | Benchmarks all 5 retrieval configurations side-by-side ($0 LLM cost). |
| `--strategy` | choice | `hybrid` | Retrieval strategy: `dense`, `sparse`, `hybrid`, `mmr`. |
| `--alpha` | float | `0.5` | Dense weight for hybrid search (`0.0` = pure BM25, `1.0` = pure dense). |
| `--mmr-lambda` | float | `0.5` | Diversity trade-off for MMR (`1.0` = maximum relevance, `0.0` = maximum diversity). |
| `--rerank` | flag | `False` | Enables Stage-2 Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`). |
| `--top-k` | int | `5` | Number of context chunks retrieved and passed to the LLM. |
| `--fetch-k` | int | `20` | Pre-ranking candidate pool size before fusion/reranking. |
| `--retrieval-only` | flag | `False` | Skips LLM generation; only evaluates Hit Rate, MRR, and latency. |
| `--sample` | int | `None` | Evaluates only the first N questions per split (great for fast testing). |
| `--split` | choice | `all` | Evaluates a specific split: `all`, `single`, `multi`, `no_answer`. |
| `--provider` | choice | `gemini` | LLM provider for generation & judge (`gemini`, `groq`). |
| `--model` | str | `gemini-3.8-flash` | LLM model identifier. |
| `--report` | path | `eval/eval_report.md` | Destination file for the markdown scorecard report. |
| `--csv` | path | `eval/eval_results.csv` | Destination file for per-query evaluation details. |

---

## 5. Understanding the Output Metrics

After each run, results are printed in the console and written to [`eval/eval_report.md`](file:///home/rahulkarakoti/Documents/NOTEBOOKLLM-LITE/backend/eval/eval_report.md) and [`eval/eval_results.csv`](file:///home/rahulkarakoti/Documents/NOTEBOOKLLM-LITE/backend/eval/eval_results.csv):

1. **Hit@1 / Hit@3 / Hit@5**:
   - Percentage of queries where the true source document appeared in the Top 1, Top 3, or Top 5 retrieved chunks.
2. **MRR (Mean Reciprocal Rank)**:
   - Evaluates rank quality: $\frac{1}{\text{rank}}$. If the correct document is Rank 1, score is 1.0; if Rank 2, 0.5. Higher is better.
3. **Faithfulness (1–5)**:
   - Evaluated by LLM Judge: Checks if all claims made in the generated answer are strictly grounded in the retrieved text (5 = zero hallucination).
4. **Correctness (1–5)**:
   - Evaluated by LLM Judge: Semantic agreement between generated answer and reference ground truth.
5. **Negative Refusal Rate (%)**:
   - Evaluated on `no_answer_questions.csv`: Percentage of unanswerable questions where the model correctly refused or stated missing information instead of fabricating facts.
6. **Average Latency (ms)**:
   - Average millisecond retrieval time per query.
