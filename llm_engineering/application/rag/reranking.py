import opik

from llm_engineering.application.networks import CrossEncoderModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep


class Reranker(RAGStep):
    def __init__(self, mock: bool = False) -> None:
        super().__init__(mock=mock)

        # Loaded on first use, so mock mode never downloads or loads the model.
        self._model: CrossEncoderModelSingleton | None = None

    @property
    def model(self) -> CrossEncoderModelSingleton:
        if self._model is None:
            self._model = CrossEncoderModelSingleton()

        return self._model

    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if self._mock:
            return chunks

        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self.model(query_doc_tuples)

        # strict: a score list of the wrong length would otherwise pair scores with the wrong chunks.
        scored_query_doc_tuples = list(zip(scores, chunks, strict=True))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        # Score threshold: an off-topic question gets no chunks instead of k irrelevant ones,
        # so the LLM is not handed text it might build a wrong answer from.
        min_score = settings.RAG_RERANK_MIN_SCORE
        if min_score is not None:
            scored_query_doc_tuples = [(score, doc) for score, doc in scored_query_doc_tuples if score >= min_score]

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
