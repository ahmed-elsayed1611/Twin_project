from types import SimpleNamespace

from llm_engineering.application.rag import reranking
from llm_engineering.application.rag.reranking import Reranker
from llm_engineering.domain.queries import Query


def _reranker_with_scores(scores: list[float]) -> Reranker:
    reranker = Reranker()
    reranker._model = lambda pairs: scores  # stands in for the cross-encoder
    return reranker


def _chunks(n: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(content=f"chunk {i}") for i in range(n)]


def test_without_threshold_keeps_the_top_k(monkeypatch) -> None:
    monkeypatch.setattr(reranking.settings, "RAG_RERANK_MIN_SCORE", None)
    chunks = _chunks(3)

    kept = _reranker_with_scores([-5.0, 2.0, -1.0]).generate(Query.from_str("q"), chunks, keep_top_k=2)

    assert kept == [chunks[1], chunks[2]]


def test_threshold_drops_low_scoring_chunks(monkeypatch) -> None:
    monkeypatch.setattr(reranking.settings, "RAG_RERANK_MIN_SCORE", 0.0)
    chunks = _chunks(3)

    kept = _reranker_with_scores([-5.0, 2.0, -1.0]).generate(Query.from_str("q"), chunks, keep_top_k=2)

    assert kept == [chunks[1]]


def test_off_topic_question_gets_no_chunks(monkeypatch) -> None:
    monkeypatch.setattr(reranking.settings, "RAG_RERANK_MIN_SCORE", 0.0)

    kept = _reranker_with_scores([-8.0, -6.0]).generate(Query.from_str("q"), _chunks(2), keep_top_k=3)

    assert kept == []
