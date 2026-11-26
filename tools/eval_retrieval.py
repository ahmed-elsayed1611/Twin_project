"""Measure retrieval on the evaluation set: recall@5, MRR and latency.

Usage: poetry run python -m tools.eval_retrieval
       EVAL_NO_LLM=1 poetry run python -m tools.eval_retrieval   (no API calls; see below)
"""

import json
import os
import time
from pathlib import Path

from loguru import logger

from llm_engineering.application.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from llm_engineering.application.rag.query_expansion import QueryExpansion
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.rag.self_query import SelfQuery

QUESTIONS_PATH = Path("data/eval/retrieval_questions.json")
K = 5


def unique_in_order(ids: list[str]) -> list[str]:
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    answerable = [q for q in questions if q["type"] != "no_answer"]
    retriever = ContextRetriever(mock=False)
    if os.environ.get("EVAL_NO_LLM") == "1":
        # No LLM calls: the question is searched as written, without expansion or author detection. Use it to
        # compare settings that do not involve those steps (e.g. the rerank threshold) without spending API
        # tokens; the numbers are not comparable with a normal run.
        retriever._query_expander = QueryExpansion(mock=True)
        retriever._metadata_extractor = SelfQuery(mock=True)
        logger.info("EVAL_NO_LLM=1: query expansion and self-query are off.")

    recalls, rrs, latencies, n_chunks = {1: [], 3: [], 5: []}, [], [], []
    for q in answerable:
        start = time.perf_counter()
        chunks = retriever.search(q["question"], k=K)
        latencies.append(time.perf_counter() - start)
        n_chunks.append(len(chunks))

        doc_ids = unique_in_order([str(chunk.document_id) for chunk in chunks])

        for k in (1, 3, 5):
            recalls[k].append(recall_at_k(doc_ids, q["document_id"], k))
        rrs.append(reciprocal_rank(doc_ids, q["document_id"]))

    mean_mrr = sum(rrs) / len(rrs)

    logger.info(f"Questions: {len(answerable)}")
    for k in (1, 3, 5):
        logger.info(f"recall@{k}: {sum(recalls[k]) / len(recalls[k]):.2f}")
    logger.info(f"MRR:       {mean_mrr:.2f}")
    logger.info(f"Latency:   median {sorted(latencies)[len(latencies) // 2]:.1f} s")
    logger.info(f"Chunks per answerable question: average {sum(n_chunks) / len(n_chunks):.1f}")

    # Questions the articles cannot answer: the right result is no chunks at all (score threshold).
    no_answer = [q for q in questions if q["type"] == "no_answer"]
    if no_answer:
        empty = sum(1 for q in no_answer if not retriever.search(q["question"], k=K))
        logger.info(f"No-answer questions with no chunks returned: {empty} of {len(no_answer)}")


if __name__ == "__main__":
    main()
