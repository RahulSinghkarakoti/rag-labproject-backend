#!/usr/bin/env python3
"""
RAG Evaluation Harness for Enter The Gungeon Benchmark Dataset.
Supports:
- Configurable RAG pipeline: Dense, Sparse (BM25), Hybrid, MMR, Reranking, Query Transformation.
- Experiment Matrix Comparison (--compare-matrix).
- Retrieval-only Mode (--retrieval-only, $0 LLM cost).
- Generation & Hallucination Refusal Evaluation with LLM-as-a-Judge (Gemini / Groq).
- Output: Console Scorecard, Markdown Report, CSV Details.
"""

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure project root is in sys.path
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Increase CSV max field size
csv.field_size_limit(sys.maxsize)

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import select

from app.database.database import AsyncSessionLocal
from app.models.knowledge_base import KnowledgeBase
from app.rag.generation.factory import provider_model_factory
from app.rag.reranking.reranker import rerank_documents
from app.rag.retrieval.workflow import RetrievalWorkflow
from app.rag.vectorstore import PgVectorStore
from app.schemas.thread import GenerationConfig, RerankingConfig, RetrievalConfig


DEFAULT_DATASET_DIR = EVAL_DIR / "dataset"
KB_NAME = "Eval - Enter The Gungeon"


# ==============================================================================
# Helper Data Loaders
# ==============================================================================

def load_eval_questions(dataset_dir: Path, split: str = "all", sample: int | None = None) -> list[dict]:
    questions = []

    splits_to_load = []
    if split in ("all", "single"):
        splits_to_load.append(("single", dataset_dir / "single_passage_answer_questions.csv"))
    if split in ("all", "multi"):
        splits_to_load.append(("multi", dataset_dir / "multi_passage_answer_questions.csv"))
    if split in ("all", "no_answer"):
        splits_to_load.append(("no_answer", dataset_dir / "no_answer_questions.csv"))

    for split_type, file_path in splits_to_load:
        if not file_path.exists():
            print(f"⚠️ Warning: {file_path} not found. Skipping {split_type}.")
            continue

        split_qs = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_item = {
                    "split": split_type,
                    "document_index": int(row["document_index"]) if "document_index" in row else None,
                    "question": row["question"].strip(),
                    "ground_truth_answer": row.get("answer", "").strip(),
                }
                split_qs.append(q_item)

        if sample and sample > 0:
            split_qs = split_qs[:sample]

        questions.extend(split_qs)

    return questions


# ==============================================================================
# Retrieval Evaluator (Zero-Cost Metric Computation)
# ==============================================================================

def evaluate_retrieval_hits(retrieved_chunks: list[Any], target_doc_index: int) -> dict:
    retrieved_doc_indices = []
    for c in retrieved_chunks:
        # Check metadata on chunk or document
        meta = getattr(c, "metadata_", {}) or {}
        doc_idx = meta.get("document_index")
        if doc_idx is None and hasattr(c, "document") and c.document:
            doc_meta = getattr(c.document, "metadata_", {}) or {}
            doc_idx = doc_meta.get("document_index")
        retrieved_doc_indices.append(doc_idx)

    hit_1 = 1 if len(retrieved_doc_indices) > 0 and retrieved_doc_indices[0] == target_doc_index else 0
    hit_3 = 1 if target_doc_index in retrieved_doc_indices[:3] else 0
    hit_5 = 1 if target_doc_index in retrieved_doc_indices[:5] else 0

    mrr = 0.0
    for rank_idx, doc_idx in enumerate(retrieved_doc_indices):
        if doc_idx == target_doc_index:
            mrr = 1.0 / (rank_idx + 1)
            break

    return {
        "hit@1": hit_1,
        "hit@3": hit_3,
        "hit@5": hit_5,
        "mrr": mrr,
        "retrieved_indices": retrieved_doc_indices,
    }


# ==============================================================================
# LLM-as-a-Judge Evaluators
# ==============================================================================

JUDGE_QA_PROMPT = """You are an impartial, highly rigorous evaluation judge assessing a RAG (Retrieval-Augmented Generation) answer.

Given:
1. Question: {question}
2. Retrieved Context:
{context}
3. Generated Answer: {answer}
4. Ground Truth Reference Answer: {reference}

Evaluate two criteria on a strict 1 to 5 scale:
1. Faithfulness (1-5): Is every claim in the Generated Answer directly supported by the Retrieved Context? (5 = 100% grounded, no unverified claims; 1 = pure hallucination or contradicts context).
2. Correctness (1-5): How accurately does the Generated Answer convey the key facts compared to the Reference Answer? (5 = fully accurate and complete; 1 = completely wrong or irrelevant).

Output your evaluation in strict JSON format:
```json
{{
  "faithfulness": <int 1-5>,
  "correctness": <int 1-5>,
  "reasoning": "<brief 1-sentence justification>"
}}
```
"""

JUDGE_REFUSAL_PROMPT = """You are an evaluation judge assessing whether a RAG system properly refused to answer an unanswerable question or hallucinated false claims.

The question asked cannot be answered by the corpus. A good RAG system must either explicitly refuse (e.g., "The provided documents do not mention...", "I cannot answer this based on the provided context") or state that information is missing. If the system fabricates facts or makes up an answer, it fails.

Given:
1. Question: {question}
2. Retrieved Context:
{context}
3. Generated Answer: {answer}

Output your evaluation in strict JSON format:
```json
{{
  "proper_refusal": <true if the answer correctly refused or admitted lack of context, false if it hallucinated an answer>,
  "hallucination_detected": <true if it fabricated information, false otherwise>,
  "reasoning": "<brief 1-sentence justification>"
}}
```
"""

RAG_GENERATION_PROMPT = """You are an expert AI assistant answering questions using strictly the provided context.
If the context does not contain sufficient facts to answer the question, state clearly: "I cannot answer this question based on the provided context." Do not make up facts.

Context:
{context}

Question: {question}
Answer:"""


async def call_llm_judge_qa(judge_provider, question: str, context: str, answer: str, reference: str) -> dict:
    prompt = JUDGE_QA_PROMPT.format(
        question=question,
        context=context[:2500],
        answer=answer,
        reference=reference,
    )
    try:
        response_text = await judge_provider.ainvoke(prompt)
        # Parse JSON
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        raw_json = match.group(1) if match else response_text
        # Fallback curly brace regex
        if not match:
            c_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            raw_json = c_match.group(1) if c_match else response_text
        data = json.loads(raw_json)
        return {
            "faithfulness": float(data.get("faithfulness", 3)),
            "correctness": float(data.get("correctness", 3)),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as e:
        return {"faithfulness": 0.0, "correctness": 0.0, "reasoning": f"Judge error: {str(e)[:100]}"}


async def call_llm_judge_refusal(judge_provider, question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_REFUSAL_PROMPT.format(
        question=question,
        context=context[:2500],
        answer=answer,
    )
    try:
        response_text = await judge_provider.ainvoke(prompt)
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        raw_json = match.group(1) if match else response_text
        if not match:
            c_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            raw_json = c_match.group(1) if c_match else response_text
        data = json.loads(raw_json)
        proper_refusal = bool(data.get("proper_refusal", False))
        hallucination = bool(data.get("hallucination_detected", not proper_refusal))
        return {
            "proper_refusal": proper_refusal,
            "hallucination_detected": hallucination,
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as e:
        # Heuristic fallback if judge call fails
        ans_lower = answer.lower()
        refused = any(p in ans_lower for p in ["cannot answer", "not provided", "not mention", "insufficient", "don't know", "does not contain"])
        return {
            "proper_refusal": refused,
            "hallucination_detected": not refused,
            "reasoning": f"Heuristic fallback (Judge error: {str(e)[:50]})",
        }


# ==============================================================================
# Core Evaluation Loop
# ==============================================================================

async def evaluate_single_config(
    kb,
    questions: list[dict],
    retrieval_config: RetrievalConfig,
    rerank_config: RerankingConfig | None = None,
    run_generation: bool = True,
    generator_provider = None,
    judge_provider = None,
    label: str = "Default",
) -> dict:
    results = []
    t_start_total = time.time()

    async with AsyncSessionLocal() as session:
        vs = PgVectorStore(session)
        workflow = RetrievalWorkflow(vs)

        for i, item in enumerate(questions):
            q_text = item["question"]
            split_type = item["split"]
            target_doc_index = item.get("document_index")

            # 1. Retrieval
            t0_ret = time.time()
            retrieved = await workflow.retrieve([q_text], [kb], retrieval_config)
            
            # Apply reranker if enabled
            if rerank_config and rerank_config.enabled:
                retrieved = rerank_documents([q_text], retrieved, rerank_config)

            ret_latency_ms = (time.time() - t0_ret) * 1000

            # Retrieval metric evaluation
            ret_metrics = evaluate_retrieval_hits(retrieved, target_doc_index) if target_doc_index is not None else {
                "hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0.0, "retrieved_indices": [getattr(c, "metadata_", {}).get("document_index") for c in retrieved]
            }

            context_str = "\n\n".join([f"--- Passage {idx+1} ---\n{c.content}" for idx, c in enumerate(retrieved[:retrieval_config.top_k])])

            answer = ""
            gen_latency_ms = 0.0
            judge_metrics = {}

            # 2. Generation & LLM Judge (if enabled)
            if run_generation and generator_provider and judge_provider:
                t0_gen = time.time()
                prompt = RAG_GENERATION_PROMPT.format(context=context_str, question=q_text)
                try:
                    answer = await generator_provider.ainvoke(prompt)
                except Exception as e:
                    answer = f"Generation Error: {e}"
                gen_latency_ms = (time.time() - t0_gen) * 1000

                # Run appropriate judge
                if split_type in ("single", "multi"):
                    judge_metrics = await call_llm_judge_qa(
                        judge_provider,
                        question=q_text,
                        context=context_str,
                        answer=answer,
                        reference=item["ground_truth_answer"],
                    )
                else:  # no_answer
                    judge_metrics = await call_llm_judge_refusal(
                        judge_provider,
                        question=q_text,
                        context=context_str,
                        answer=answer,
                    )

            row_result = {
                "config_label": label,
                "split": split_type,
                "question": q_text,
                "target_doc_index": target_doc_index,
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "retrieved_indices": ret_metrics["retrieved_indices"],
                "hit@1": ret_metrics["hit@1"],
                "hit@3": ret_metrics["hit@3"],
                "hit@5": ret_metrics["hit@5"],
                "mrr": ret_metrics["mrr"],
                "ret_latency_ms": round(ret_latency_ms, 1),
                "generated_answer": answer,
                "gen_latency_ms": round(gen_latency_ms, 1),
                **judge_metrics,
            }
            results.append(row_result)

            status_char = "✓" if (ret_metrics["hit@1"] or judge_metrics.get("proper_refusal")) else "•"
            print(f"[{label}] [{i+1}/{len(questions)}] {status_char} ({split_type[:4]}) Q: {q_text[:50]}... (Ret: {ret_latency_ms:.0f}ms)", flush=True)

    # Aggregate Metrics
    qa_rows = [r for r in results if r["split"] in ("single", "multi")]
    no_ans_rows = [r for r in results if r["split"] == "no_answer"]

    total_qa = len(qa_rows)
    hit1_avg = (sum(r["hit@1"] for r in qa_rows) / total_qa * 100) if total_qa else 0
    hit3_avg = (sum(r["hit@3"] for r in qa_rows) / total_qa * 100) if total_qa else 0
    hit5_avg = (sum(r["hit@5"] for r in qa_rows) / total_qa * 100) if total_qa else 0
    mrr_avg = (sum(r["mrr"] for r in qa_rows) / total_qa) if total_qa else 0
    ret_lat_avg = (sum(r["ret_latency_ms"] for r in results) / len(results)) if results else 0

    faith_avg = (sum(r.get("faithfulness", 0) for r in qa_rows) / total_qa) if total_qa and run_generation else 0
    corr_avg = (sum(r.get("correctness", 0) for r in qa_rows) / total_qa) if total_qa and run_generation else 0

    total_no_ans = len(no_ans_rows)
    refusal_rate = (sum(1 for r in no_ans_rows if r.get("proper_refusal")) / total_no_ans * 100) if total_no_ans and run_generation else 0

    summary = {
        "label": label,
        "strategy": retrieval_config.strategy,
        "rerank": bool(rerank_config and rerank_config.enabled),
        "total_queries": len(results),
        "hit@1": round(hit1_avg, 1),
        "hit@3": round(hit3_avg, 1),
        "hit@5": round(hit5_avg, 1),
        "mrr": round(mrr_avg, 4),
        "faithfulness": round(faith_avg, 2),
        "correctness": round(corr_avg, 2),
        "refusal_rate": round(refusal_rate, 1),
        "avg_ret_latency_ms": round(ret_lat_avg, 1),
        "total_time_sec": round(time.time() - t_start_total, 1),
    }

    return {"summary": summary, "results": results}


# ==============================================================================
# Comparative Experiment Matrix
# ==============================================================================

def get_experiment_matrix() -> list[tuple[str, RetrievalConfig, RerankingConfig | None]]:
    """Define standard configurations to benchmark side-by-side."""
    return [
        (
            "Pure Dense",
            RetrievalConfig(strategy="dense", top_k=5, fetch_k=20),
            None,
        ),
        (
            "Pure Sparse (BM25)",
            RetrievalConfig(strategy="sparse", top_k=5, fetch_k=20),
            None,
        ),
        (
            "Hybrid (alpha=0.5)",
            RetrievalConfig(strategy="hybrid", alpha=0.5, top_k=5, fetch_k=20),
            None,
        ),
        (
            "Hybrid + Cross-Encoder",
            RetrievalConfig(strategy="hybrid", alpha=0.5, top_k=5, fetch_k=20),
            RerankingConfig(enabled=True, model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5),
        ),
        (
            "MMR (Diversity lambda=0.5)",
            RetrievalConfig(strategy="mmr", mmr_lambda=0.5, top_k=5, fetch_k=20),
            None,
        ),
    ]


# ==============================================================================
# Markdown & CSV Export
# ==============================================================================

def export_markdown_report(summaries: list[dict], report_path: Path):
    lines = [
        "# RAG Evaluation Benchmark Report",
        "",
        f"**Dataset**: Enter The Gungeon Benchmark (`/home/rahulkarakoti/Downloads/archive`)  ",
        f"**Knowledge Base**: `{KB_NAME}` | **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "## 1. Executive Summary & Configuration Comparison",
        "",
        "| Configuration | Strategy | Rerank | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | Faithfulness (1-5) | Correctness (1-5) | Refusal Rate (%) | Avg Latency (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summaries:
        faith_str = f"{s['faithfulness']:.2f}" if s["faithfulness"] > 0 else "N/A"
        corr_str = f"{s['correctness']:.2f}" if s["correctness"] > 0 else "N/A"
        ref_str = f"{s['refusal_rate']:.1f}%" if s["refusal_rate"] > 0 else "N/A"
        rerank_str = "✓" if s["rerank"] else "✗"
        lines.append(
            f"| **{s['label']}** | `{s['strategy']}` | {rerank_str} | **{s['hit@1']}%** | {s['hit@3']}% | {s['hit@5']}% | **{s['mrr']}** | {faith_str} | {corr_str} | {ref_str} | {s['avg_ret_latency_ms']} ms |"
        )

    lines.extend([
        "",
        "---",
        "## 2. Key Metrics Explained",
        "- **Hit@K**: Percentage of QA queries where the ground-truth document was retrieved in the top $K$ candidates.",
        "- **MRR (Mean Reciprocal Rank)**: Quality metric rewarding relevant documents ranking higher: $\\frac{1}{\\text{rank}}$.",
        "- **Faithfulness**: LLM-as-a-judge score (1-5) verifying the answer contains no hallucinations unsupported by context.",
        "- **Correctness**: LLM-as-a-judge semantic agreement (1-5) between generated answer and ground-truth answer.",
        "- **Refusal Rate**: Percentage of `no_answer_questions` properly declined without inventing false facts.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Markdown Report generated: {report_path}")


def export_csv_results(all_results: list[dict], csv_path: Path):
    if not all_results:
        return
    # Collect all unique fieldnames preserving order
    fieldnames = []
    for r in all_results:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"📊 CSV Detail Results saved: {csv_path}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Project on Local Benchmark Dataset")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Path to dataset directory")
    parser.add_argument("--strategy", type=str, default="hybrid", choices=["dense", "sparse", "hybrid", "mmr"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid retrieval alpha (dense weight)")
    parser.add_argument("--mmr-lambda", type=float, default=0.5, help="MMR diversity lambda")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks to return")
    parser.add_argument("--fetch-k", type=int, default=20, help="Initial candidate pool size")
    parser.add_argument("--rerank", action="store_true", help="Enable Cross-Encoder Reranking")
    parser.add_argument("--retrieval-only", action="store_true", help="Evaluate retrieval only (zero LLM cost)")
    parser.add_argument("--compare-matrix", action="store_true", help="Benchmark all 5 standard retrieval configs")
    parser.add_argument("--split", type=str, default="all", choices=["all", "single", "multi", "no_answer"])
    parser.add_argument("--sample", type=int, default=None, help="Sample N questions per split (for dry runs)")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--model", type=str, default="gemini-3.8-flash")
    parser.add_argument("--report", type=Path, default=EVAL_DIR / "eval_report.md", help="Path to markdown report")
    parser.add_argument("--csv", type=Path, default=EVAL_DIR / "eval_results.csv", help="Path to CSV details")

    args = parser.parse_args()

    # Load Knowledge Base
    async with AsyncSessionLocal() as session:
        kb_query = await session.execute(select(KnowledgeBase).where(KnowledgeBase.name == KB_NAME))
        kb_record = kb_query.scalars().first()
        if not kb_record:
            print(f"❌ Knowledge Base '{KB_NAME}' not found! Run 'eval/ingest.py' first.")
            sys.exit(1)
        kb = SimpleNamespace(
            id=str(kb_record.id),
            embedding_config=kb_record.embedding_config or {"model": "all-MiniLM-L6-v2"},
        )

    # Load Questions
    questions = load_eval_questions(args.dataset_dir, split=args.split, sample=args.sample)
    if not questions:
        print("❌ No questions found to evaluate!")
        sys.exit(1)

    print(f"🎯 Loaded {len(questions)} evaluation questions (split: '{args.split}', sample: {args.sample})")

    # Initialize Generator / Judge if needed
    generator_provider = None
    judge_provider = None
    if not args.retrieval_only and not args.compare_matrix:
        print(f"🤖 Initializing Generator & Judge: provider='{args.provider}', model='{args.model}'")
        generator_provider = provider_model_factory.create({
            "generation": {"provider": args.provider, "llm_model": args.model, "temperature": 0.1, "max_tokens": 800}
        })
        judge_provider = provider_model_factory.create({
            "generation": {"provider": args.provider, "llm_model": args.model, "temperature": 0.0, "max_tokens": 600}
        })

    all_summaries = []
    all_raw_results = []

    if args.compare_matrix:
        print("\n" + "=" * 75)
        print("📊 RUNNING RETRIEVAL EXPERIMENT MATRIX (5 Configurations)")
        print("=" * 75)
        matrix = get_experiment_matrix()
        for label, ret_cfg, rerank_cfg in matrix:
            print(f"\n▶ Benchmarking: {label} (strategy={ret_cfg.strategy}, rerank={bool(rerank_cfg)})")
            eval_out = await evaluate_single_config(
                kb=kb,
                questions=questions,
                retrieval_config=ret_cfg,
                rerank_config=rerank_cfg,
                run_generation=False,
                label=label,
            )
            all_summaries.append(eval_out["summary"])
            all_raw_results.extend(eval_out["results"])
    else:
        ret_cfg = RetrievalConfig(
            strategy=args.strategy,
            alpha=args.alpha,
            mmr_lambda=args.mmr_lambda,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
        )
        rerank_cfg = RerankingConfig(enabled=True, model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=args.top_k) if args.rerank else None
        label = f"{args.strategy.title()}" + (" + Rerank" if args.rerank else "")

        eval_out = await evaluate_single_config(
            kb=kb,
            questions=questions,
            retrieval_config=ret_cfg,
            rerank_config=rerank_cfg,
            run_generation=not args.retrieval_only,
            generator_provider=generator_provider,
            judge_provider=judge_provider,
            label=label,
        )
        all_summaries.append(eval_out["summary"])
        all_raw_results.extend(eval_out["results"])

    # Output Console Summary Table
    print("\n" + "=" * 80)
    print("🏆 FINAL EVALUATION SCORECARD")
    print("=" * 80)
    for s in all_summaries:
        print(f"📌 {s['label']:<24} | Hit@1: {s['hit@1']:>5.1f}% | Hit@3: {s['hit@3']:>5.1f}% | Hit@5: {s['hit@5']:>5.1f}% | MRR: {s['mrr']:>6.4f} | Latency: {s['avg_ret_latency_ms']:>5.1f}ms")
        if s.get("faithfulness", 0) > 0 or s.get("refusal_rate", 0) > 0:
            print(f"   ↳ Faithfulness: {s['faithfulness']:.2f}/5 | Correctness: {s['correctness']:.2f}/5 | Negative Refusal Rate: {s['refusal_rate']:.1f}%")

    # Export Reports
    export_markdown_report(all_summaries, args.report)
    export_csv_results(all_raw_results, args.csv)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
