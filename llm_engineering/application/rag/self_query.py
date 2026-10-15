import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate

# Answers that mean "no name in the question". The model is told to say "none", but may add a dot or a capital.
NO_AUTHOR_ANSWERS = {"", "none", "null", "n/a"}


class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query.content})
        user_full_name = response.content.strip("\n \"'.")

        if user_full_name.lower() in NO_AUTHOR_ANSWERS:
            return query

        # Reading must not write: an unknown or misspelled name used to create an empty user here,
        # and the search then filtered on a user who owns no chunks.
        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.find(first_name=first_name, last_name=last_name)
        if user is None:
            logger.warning(
                f"Author '{user_full_name}' is not in the data warehouse. Searching without an author filter."
            )

            return query

        query.author_id = user.id
        query.author_full_name = user.full_name

        return query


if __name__ == "__main__":
    query = Query.from_str("I am Paul Iusztin. Write an article about the best types of advanced RAG methods.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
