import re

import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=0,
        )

        chain = prompt | model

        response = chain.invoke({"question": query.content})
        result = response.content.strip()

        separator = query_expansion_template.separator
        if separator in result:
            queries_content = result.split(separator)
        else:
            # The model sometimes ignores the separator and writes one question per line, often numbered.
            queries_content = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line) for line in result.splitlines()]

        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        if len(queries) > expand_to_n:
            logger.warning(f"The model returned {len(queries) - 1} rewrites; keeping {expand_to_n - 1}.")

        return queries[:expand_to_n]


if __name__ == "__main__":
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
