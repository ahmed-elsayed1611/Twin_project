from llm_engineering.application.networks.sparse import document_sparse_vector, query_sparse_vector, tokenize


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("What is the Bytewax streaming framework?") == ["bytewax", "streaming", "framework"]


def test_same_word_gets_the_same_index_in_documents_and_questions() -> None:
    document = document_sparse_vector("Bytewax processes the queue messages.")
    question = query_sparse_vector("Which framework uses Bytewax?")

    assert set(question.indices) & set(document.indices)


def test_repeating_a_word_adds_less_each_time() -> None:  # BM25 saturation
    def weight(text: str) -> float:
        vector = document_sparse_vector(text)
        return max(vector.values)

    once, twice, ten_times = weight("qdrant"), weight("qdrant " * 2), weight("qdrant " * 10)

    assert once < twice < ten_times
    assert (ten_times - twice) < 8 * (twice - once)


def test_question_of_only_stopwords_has_no_keywords() -> None:
    assert query_sparse_vector("what is the").indices == []
