from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)
from llm_engineering.domain.documents import (
    ArticleDocument,
    Document,
    PostDocument,
    RepositoryDocument,
)

from .operations import PARAGRAPH_SEPARATOR, clean_code, clean_text

DocumentT = TypeVar("DocumentT", bound=Document)

# Only these fields hold the article's text; others, such as "language", are metadata.
ARTICLE_TEXT_FIELDS = ("Title", "Subtitle", "Content")


def _clean_and_join(parts: list[str | None]) -> str:
    """Clean each part on its own and join them with a paragraph break, which survives cleaning."""

    cleaned_parts = [clean_text(part) for part in parts if part]

    return PARAGRAPH_SEPARATOR.join(part for part in cleaned_parts if part)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        return CleanedPostDocument(
            id=data_model.id,
            content=_clean_and_join(list(data_model.content.values())),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        text_fields = [data_model.content.get(field) for field in ARTICLE_TEXT_FIELDS]

        return CleanedArticleDocument(
            id=data_model.id,
            content=_clean_and_join(text_fields),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        return CleanedRepositoryDocument(
            id=data_model.id,
            # Code keeps its syntax, and each file starts with its path so the chunks say where they come from.
            content=PARAGRAPH_SEPARATOR.join(
                f"File: {path}\n{clean_code(code)}" for path, code in data_model.content.items() if code
            ),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )
