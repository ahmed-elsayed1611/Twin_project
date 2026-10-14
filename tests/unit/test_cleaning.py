import uuid

from llm_engineering.application.preprocessing.cleaning_data_handlers import (
    ArticleCleaningHandler,
    RepositoryCleaningHandler,
)
from llm_engineering.application.preprocessing.operations import clean_code, clean_text
from llm_engineering.domain.documents import ArticleDocument, RepositoryDocument

AUTHOR = {"author_id": uuid.uuid4(), "author_full_name": "Test Author"}


def test_clean_text_replaces_urls() -> None:  # 4.5
    cleaned = clean_text("Read the docs at https://qdrant.tech/documentation/ today.")

    assert cleaned == "Read the docs at [URL] today."


def test_clean_text_keeps_paragraph_breaks_and_joins_wrapped_lines() -> None:  # 4.5
    cleaned = clean_text("First line of a paragraph\nwrapped here.\n\n\n  Second paragraph!")

    assert cleaned == "First line of a paragraph wrapped here.\n\nSecond paragraph!"


def test_clean_text_removes_symbols() -> None:
    assert clean_text("RAG #1 :) costs $5 -> cheap") == "RAG 1 costs 5 cheap"


def test_clean_code_keeps_syntax() -> None:  # 4.8
    code = "import os\n\n\n\ndef add(a, b):    \n    return a + b\n"

    assert clean_code(code) == "import os\n\ndef add(a, b):\n    return a + b"


def test_article_cleaning_uses_only_text_fields() -> None:  # 4.6, 4.7
    article = ArticleDocument(
        content={"Title": "My title", "Subtitle": None, "Content": "The body.", "language": "en"},
        link="https://example.com/a",
        platform="example.com",
        **AUTHOR,
    )

    cleaned = ArticleCleaningHandler().clean(article)

    assert cleaned.content == "My title\n\nThe body."


def test_repository_cleaning_keeps_file_paths_and_code() -> None:  # 4.7, 4.8
    repository = RepositoryDocument(
        content={"src/app.py": "import os\nprint(os.getcwd())\n"},
        name="repo",
        link="https://github.com/user/repo",
        platform="github",
        **AUTHOR,
    )

    cleaned = RepositoryCleaningHandler().clean(repository)

    assert cleaned.content == "File: src/app.py\nimport os\nprint(os.getcwd())"
