import uuid

from llm_engineering.application.preprocessing.chunking_data_handlers import ArticleChunkingHandler
from llm_engineering.application.preprocessing.operations import chunk_article, count_tokens
from llm_engineering.domain.cleaned_documents import CleanedArticleDocument


def _words(text: str) -> list[str]:
    return text.split()


def test_short_ending_is_kept() -> None:  # 4.11
    # The ending does not fit in the second chunk and is under min_length: the old code dropped it.
    text = "A" * 150 + ". " + "B" * 194 + ". " + "Short end."

    chunks = chunk_article(text, min_length=100, max_length=200)

    assert _words(" ".join(chunks)) == _words(text)


def test_short_article_becomes_one_chunk() -> None:  # 4.11
    assert chunk_article("Only one short sentence.", min_length=1000, max_length=2000) == ["Only one short sentence."]


def test_long_sentence_is_cut_between_words() -> None:  # 4.16
    sentence = " ".join(["word"] * 100)  # 499 characters, no full stop

    chunks = chunk_article(sentence, min_length=50, max_length=100)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert _words(" ".join(chunks)) == _words(sentence)


def test_article_chunks_fit_the_embedding_model() -> None:  # 4.16
    handler = ArticleChunkingHandler()
    text = " ".join(f"Sentence number {i} talks about retrieval augmented generation." for i in range(300))

    chunks = chunk_article(
        text, handler.metadata["min_length"], handler.metadata["max_length"], length_function=count_tokens
    )

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= handler.metadata["max_length"] for chunk in chunks)
    assert _words(" ".join(chunks)) == _words(text)


def test_same_text_in_two_documents_gets_two_ids() -> None:  # 4.13
    text = " ".join(f"Every message will be cleaned and embedded, part {i}." for i in range(20))
    documents = [
        CleanedArticleDocument(
            id=uuid.uuid4(),
            content=text,
            platform="example.com",
            link=f"https://example.com/{i}",
            author_id=uuid.uuid4(),
            author_full_name="Test Author",
        )
        for i in range(2)
    ]

    first, second = (ArticleChunkingHandler().chunk(document) for document in documents)

    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]
    assert {chunk.id for chunk in first}.isdisjoint({chunk.id for chunk in second})
