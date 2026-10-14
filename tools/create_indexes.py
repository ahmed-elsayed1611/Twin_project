from loguru import logger
from pymongo.errors import DuplicateKeyError

from llm_engineering.domain.base.nosql import _database
from llm_engineering.domain.documents import ArticleDocument, RepositoryDocument

# One document per link: MongoDB itself refuses a second insert of the same link,
# even when two crawls pass the "already exists" check at the same time.
UNIQUE_LINK_COLLECTIONS = [ArticleDocument, RepositoryDocument]


def main() -> None:
    for document_class in UNIQUE_LINK_COLLECTIONS:
        collection_name = document_class.get_collection_name()
        try:
            _database[collection_name].create_index("link", unique=True, name="unique_link")
        except DuplicateKeyError:
            logger.error(f"'{collection_name}' already has duplicate links. Remove them, then run this again.")

            raise

        logger.info(f"Unique index on 'link' is in place for '{collection_name}'.")


if __name__ == "__main__":
    main()
