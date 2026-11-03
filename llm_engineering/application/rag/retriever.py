import concurrent.futures

import opik
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

from .query_expansion import QueryExpansion
from .reranking import Reranker
from .self_query import SelfQuery


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    @opik.track(name="ContextRetriever.search")
    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
        if k < 1:
            raise ValueError(f"'k' should be at least 1. Got {k}.")

        query_model = Query.from_str(query)

        # The two LLM calls don't need each other's answers, so they run at the same time. Self-query gets
        # a copy because it writes the author onto its query while expansion is reading the original.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            metadata_task = executor.submit(self._metadata_extractor.generate, query_model.model_copy())
            expansion_task = executor.submit(
                self._query_expander.generate, query_model, expand_to_n=expand_to_n_queries
            )

            query_model = metadata_task.result()
            n_generated_queries = expansion_task.result()
        logger.info(
            f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.",
        )

        for generated_query in n_generated_queries:
            generated_query.author_id = query_model.author_id
            generated_query.author_full_name = query_model.author_full_name
        logger.info(
            f"Successfully generated {len(n_generated_queries)} search queries.",
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        # Each collection may return up to k candidates and the reranker keeps the best k overall.
        # The old `k // 3` per collection ignored the remainder (k=4 and k=5 behaved like 3), failed for k < 3,
        # and gave every collection the same quota whatever the question was about.
        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(
                                value=str(embedded_query.author_id),
                            ),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k,
                query_filter=query_filter,
            )

        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
        repositories_chunks = _search_data_category(EmbeddedRepositoryChunk, embedded_query)

        retrieved_chunks = post_chunks + articles_chunks + repositories_chunks

        return retrieved_chunks

    def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents
