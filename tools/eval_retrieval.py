"""Measure retrieval on the evaluation set: recall@5, MRR and latency.

Usage: poetry run python -m tools.eval_retrieval
"""

import json
import time
from pathlib import Path

from llm_engineering.application.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from llm_engineering.application.rag.retriever import ContextRetriever

QUESTIONS_PATH = Path("data/eval/retrieval_questions.json")
K = 5


def unique_in_order(ids: list[str]) -> list[str]:
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    answerable = [q for q in questions if q["type"] != "no_answer"]
    retriever = ContextRetriever(mock=False)

    recalls, rrs, latencies = [], [], []
    for q in answerable:
        start = time.perf_counter()
        chunks = retriever.search(q["question"], k=K)
        latencies.append(time.perf_counter() - start)

        doc_ids = unique_in_order([str(chunk.document_id) for chunk in chunks])

        recalls.append(recall_at_k(doc_ids, q["document_id"], K))
        rrs.append(reciprocal_rank(doc_ids, q["document_id"]))

    mean_recall = sum(recalls) / len(recalls)
    mean_mrr = sum(rrs) / len(rrs)

    print(f"Questions: {len(answerable)}")
    print(f"recall@{K}: {mean_recall:.2f}")
    print(f"MRR:       {mean_mrr:.2f}")
    print(f"Latency:   median {sorted(latencies)[len(latencies) // 2]:.1f} s")


if __name__ == "__main__":
    main()
