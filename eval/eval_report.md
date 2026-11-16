# RAG Evaluation Benchmark Report

**Dataset**: Enter The Gungeon Benchmark (`/home/rahulkarakoti/Downloads/archive`)  
**Knowledge Base**: `Eval - Enter The Gungeon` | **Timestamp**: 2026-11-14 21:44:15  

## 1. Executive Summary & Configuration Comparison

| Configuration | Strategy | Rerank | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | Faithfulness (1-5) | Correctness (1-5) | Refusal Rate (%) | Avg Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Dense** | `dense` | ✗ | **100.0%** | 100.0% | 100.0% | **1.0** | N/A | N/A | N/A | 5387.8 ms |
| **Pure Sparse (BM25)** | `sparse` | ✗ | **100.0%** | 100.0% | 100.0% | **1.0** | N/A | N/A | N/A | 371.2 ms |
| **Hybrid (alpha=0.5)** | `hybrid` | ✗ | **100.0%** | 100.0% | 100.0% | **1.0** | N/A | N/A | N/A | 1035.8 ms |
| **Hybrid + Cross-Encoder** | `hybrid` | ✓ | **100.0%** | 100.0% | 100.0% | **1.0** | N/A | N/A | N/A | 2952.5 ms |
| **MMR (Diversity lambda=0.5)** | `mmr` | ✗ | **100.0%** | 100.0% | 100.0% | **1.0** | N/A | N/A | N/A | 180.9 ms |

---
## 2. Key Metrics Explained
- **Hit@K**: Percentage of QA queries where the ground-truth document was retrieved in the top $K$ candidates.
- **MRR (Mean Reciprocal Rank)**: Quality metric rewarding relevant documents ranking higher: $\frac{1}{\text{rank}}$.
- **Faithfulness**: LLM-as-a-judge score (1-5) verifying the answer contains no hallucinations unsupported by context.
- **Correctness**: LLM-as-a-judge semantic agreement (1-5) between generated answer and ground-truth answer.
- **Refusal Rate**: Percentage of `no_answer_questions` properly declined without inventing false facts.
