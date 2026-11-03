import ast
import asyncio
import importlib
import time
from pathlib import Path
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client.http.exceptions import UnexpectedResponse

from llm_engineering.application.rag import query_expansion, reranking, retriever, self_query
from llm_engineering.application.rag.prompt_templates import SelfQueryTemplate
from llm_engineering.application.utils import misc
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk, EmbeddedChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query
from llm_engineering.model.inference.run import ALPACA_TEMPLATE, InferenceExecutor
from llm_engineering.settings import settings

PAUL = UserDocument(first_name="Paul", last_name="Iusztin")


def _fake_chat_model(answer: str, prompts: list[str] | None = None, delay: float = 0.0):
    """Stands in for ChatOpenAI: records the prompt it receives and returns `answer`."""

    def _call(prompt_value):
        if prompts is not None:
            prompts.append(prompt_value.to_string())
        time.sleep(delay)

        return AIMessage(content=answer)

    return lambda **kwargs: RunnableLambda(_call)


def _find_only_paul(**kwargs):
    return PAUL if kwargs == {"first_name": "Paul", "last_name": "Iusztin"} else None


def _chunk(text: str) -> EmbeddedArticleChunk:
    return EmbeddedArticleChunk(
        content=text,
        embedding=None,
        platform="medium.com",
        document_id=PAUL.id,
        author_id=PAUL.id,
        author_full_name=PAUL.full_name,
        link="https://example.com",
    )


class _FakeCrossEncoder:
    loads = 0

    def __init__(self) -> None:
        _FakeCrossEncoder.loads += 1

    def __call__(self, pairs, to_list: bool = True):
        return [float(len(chunk)) for _, chunk in pairs]


@pytest.fixture
def users():
    """MongoDB stand-in: Paul exists; get_or_create would record a new user."""

    created = []
    with mock.patch.multiple(
        UserDocument,
        find=_find_only_paul,
        get_or_create=lambda **kwargs: created.append(kwargs) or UserDocument(**kwargs),
    ):
        yield created


@pytest.fixture
def api():
    with mock.patch("llm_engineering.infrastructure.opik_utils.configure_opik"):
        module = importlib.import_module("llm_engineering.infrastructure.inference_pipeline_api")
    module.get_retriever.cache_clear()
    with mock.patch.object(module.opik_context, "update_current_trace"):
        yield module
    module.get_retriever.cache_clear()


# ---- Step 1: self-query


def test_self_query_sends_only_the_question(users) -> None:  # 9.4
    prompts: list[str] = []
    with mock.patch.object(self_query, "ChatOpenAI", _fake_chat_model("none", prompts)):
        self_query.SelfQuery().generate(Query.from_str("Write a post about RAG."))

    assert prompts[0].rstrip().endswith("User question: Write a post about RAG.")
    assert "author_id" not in prompts[0]


def test_self_query_finds_a_known_author(users) -> None:  # 9.1
    with mock.patch.object(self_query, "ChatOpenAI", _fake_chat_model("Paul Iusztin")):
        query = self_query.SelfQuery().generate(Query.from_str("I am Paul Iusztin. Write about RAG."))

    assert query.author_id == PAUL.id
    assert users == []


@pytest.mark.parametrize("answer", ["Paul Iusztinn", "1345256"])
def test_unknown_author_creates_no_user_and_no_filter(users, answer) -> None:  # 9.1, 9.3
    with mock.patch.object(self_query, "ChatOpenAI", _fake_chat_model(answer)):
        query = self_query.SelfQuery().generate(Query.from_str("Write about RAG."))

    assert query.author_id is None
    assert users == []


@pytest.mark.parametrize("answer", ["none", "None", "none.", "NONE", '"none"', ""])
def test_every_spelling_of_none_means_no_author(users, answer) -> None:  # 9.2
    with mock.patch.object(self_query, "ChatOpenAI", _fake_chat_model(answer)):
        query = self_query.SelfQuery().generate(Query.from_str("Write about RAG."))

    assert query.author_id is None
    assert users == []


def test_self_query_prompt_asks_for_names_only() -> None:  # 9.3
    prompt = SelfQueryTemplate().prompt

    assert "user id" not in prompt
    assert "1345256" not in prompt


# ---- Step 2: query expansion


def test_expansion_sends_only_the_question() -> None:  # 9.4
    prompts: list[str] = []
    with mock.patch.object(query_expansion, "ChatOpenAI", _fake_chat_model("a#next-question#b", prompts)):
        query_expansion.QueryExpansion().generate(Query.from_str("Write a post about RAG."), expand_to_n=3)

    assert prompts[0].rstrip().endswith("Original question: Write a post about RAG.")


def test_expansion_returns_at_most_n_queries() -> None:  # 9.6
    answer = "q1#next-question#q2#next-question#q3#next-question#q4"
    with mock.patch.object(query_expansion, "ChatOpenAI", _fake_chat_model(answer)):
        queries = query_expansion.QueryExpansion().generate(Query.from_str("original"), expand_to_n=3)

    assert [q.content for q in queries] == ["original", "q1", "q2"]


def test_expansion_reads_one_question_per_line() -> None:  # 9.6
    answer = "1. How does RAG work?\n2) Explain retrieval-augmented generation"
    with mock.patch.object(query_expansion, "ChatOpenAI", _fake_chat_model(answer)):
        queries = query_expansion.QueryExpansion().generate(Query.from_str("original"), expand_to_n=3)

    assert [q.content for q in queries] == ["original", "How does RAG work?", "Explain retrieval-augmented generation"]


# ---- Step 3: search


@pytest.mark.parametrize("k", [1, 2, 4, 5, 9])
def test_each_collection_is_asked_for_k_candidates(k) -> None:  # 9.7
    limits: list[int] = []
    embedded = EmbeddedQuery(content="q", embedding=[0.0], author_id=PAUL.id)
    with (
        mock.patch.object(retriever.EmbeddingDispatcher, "dispatch", return_value=embedded),
        mock.patch.object(EmbeddedChunk, "search", side_effect=lambda **kwargs: limits.append(kwargs["limit"]) or []),
    ):
        retriever.ContextRetriever(mock=True)._search(Query.from_str("q"), k)

    assert limits == [k, k, k]


def test_k_below_one_is_refused() -> None:  # 9.7
    with pytest.raises(ValueError, match="at least 1"):
        retriever.ContextRetriever(mock=True).search("q", k=0)


def test_llm_steps_run_at_the_same_time_and_every_query_keeps_the_author(users) -> None:  # 9.13
    searched: list[Query] = []
    with (
        mock.patch.object(self_query, "ChatOpenAI", _fake_chat_model("Paul Iusztin", delay=0.5)),
        mock.patch.object(query_expansion, "ChatOpenAI", _fake_chat_model("q1#next-question#q2", delay=0.5)),
        mock.patch.object(retriever.ContextRetriever, "_search", lambda self, query, k: searched.append(query) or []),
    ):
        start = time.perf_counter()
        retriever.ContextRetriever().search("My name is Paul Iusztin. Write about RAG.", k=3)
        elapsed = time.perf_counter() - start

    assert elapsed < 0.9  # one after the other would take at least 1.0 s
    assert len(searched) == 3
    assert all(query.author_id == PAUL.id for query in searched)


def test_missing_collection_returns_nothing() -> None:  # 9.8
    error = UnexpectedResponse(404, "Not Found", b"", httpx.Headers())
    with mock.patch.object(EmbeddedArticleChunk, "_search", side_effect=error):
        assert EmbeddedArticleChunk.search(query_vector=[0.0], limit=3) == []


def test_other_qdrant_errors_are_raised() -> None:  # 9.8
    error = UnexpectedResponse(500, "Internal Server Error", b"", httpx.Headers())
    with mock.patch.object(EmbeddedArticleChunk, "_search", side_effect=error), pytest.raises(UnexpectedResponse):
        EmbeddedArticleChunk.search(query_vector=[0.0], limit=3)


# ---- Step 5: reranking


def test_reranker_refuses_a_wrong_number_of_scores() -> None:  # 9.9
    class _ShortScores(_FakeCrossEncoder):
        def __call__(self, pairs, to_list: bool = True):
            return [1.0] * (len(pairs) - 1)

    with mock.patch.object(reranking, "CrossEncoderModelSingleton", _ShortScores), pytest.raises(ValueError):
        reranking.Reranker().generate(Query.from_str("q"), [_chunk("a"), _chunk("bb")], keep_top_k=2)


def test_reranker_keeps_the_best_chunks() -> None:
    with mock.patch.object(reranking, "CrossEncoderModelSingleton", _FakeCrossEncoder):
        kept = reranking.Reranker().generate(
            Query.from_str("q"), [_chunk("a"), _chunk("ccc"), _chunk("bb")], keep_top_k=2
        )

    assert [chunk.content for chunk in kept] == ["ccc", "bb"]


def test_mock_reranker_never_loads_the_model() -> None:  # 9.17
    _FakeCrossEncoder.loads = 0
    with mock.patch.object(reranking, "CrossEncoderModelSingleton", _FakeCrossEncoder):
        reranking.Reranker(mock=True).generate(Query.from_str("q"), [_chunk("a")], keep_top_k=1)

    assert _FakeCrossEncoder.loads == 0


# ---- Step 6: the API and the call to the model


def test_no_context_means_no_model_call(api) -> None:  # 9.5
    with (
        mock.patch.object(retriever.ContextRetriever, "search", return_value=[]),
        mock.patch.object(api, "call_llm_service") as call_llm_service,
    ):
        answer = api.rag("Write about RAG.")

    call_llm_service.assert_not_called()
    assert answer == api.NO_CONTEXT_ANSWER


def test_one_retriever_serves_every_request(api) -> None:  # 9.16
    with mock.patch.object(api, "ContextRetriever") as retriever_class:
        retriever_class.return_value.search.return_value = []
        for _ in range(3):
            api.rag("Write about RAG.")

    assert retriever_class.call_count == 1


def test_tokenizer_is_loaded_once() -> None:  # 9.15
    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    misc._get_tokenizer.cache_clear()
    with mock.patch.object(misc.AutoTokenizer, "from_pretrained", return_value=_Tokenizer()) as from_pretrained:
        counts = [misc.compute_num_tokens("one two three") for _ in range(3)]
    misc._get_tokenizer.cache_clear()

    assert counts == [3, 3, 3]
    assert from_pretrained.call_count == 1


def test_error_details_stay_on_the_server(api) -> None:  # 9.10
    with mock.patch.object(api, "rag", side_effect=Exception("mongodb://admin:secret@10.0.0.5")):
        response = TestClient(api.app).post("/rag", json={"query": "x"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_slow_requests_do_not_wait_for_each_other(api) -> None:  # 9.14
    async def two_requests() -> float:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = time.perf_counter()
            responses = await asyncio.gather(*(client.post("/rag", json={"query": "x"}) for _ in range(2)))
            assert all(response.status_code == 200 for response in responses)

            return time.perf_counter() - start

    with mock.patch.object(api, "rag", side_effect=lambda query: time.sleep(0.5) or "ok"):
        elapsed = asyncio.run(two_requests())

    assert elapsed < 0.9  # one after the other would take at least 1.0 s


class _FakeEndpoint:
    def set_payload(self, inputs, parameters=None):
        self.inputs, self.parameters = inputs, parameters

    def inference(self):
        return [{"generated_text": "a post"}]


def test_prompt_uses_the_training_template() -> None:  # 9.12
    endpoint = _FakeEndpoint()
    InferenceExecutor(endpoint, "Write a post about RAG.", "Chunk 1: ...").execute()

    assert endpoint.inputs.startswith("Below is an instruction that describes a task.")
    assert "### Instruction:\nYou are a content creator." in endpoint.inputs
    assert endpoint.inputs.endswith("### Response:\n")


def test_inference_template_matches_the_finetuning_template() -> None:  # 9.12
    finetune = Path(__file__).parents[2] / "llm_engineering" / "model" / "finetuning" / "finetune.py"
    tree = ast.parse(finetune.read_text())
    training_template = next(
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "alpaca_template" for t in node.targets)
    )

    assert ALPACA_TEMPLATE == training_template


def test_answer_length_fits_the_endpoint() -> None:  # 9.11
    assert settings.MAX_NEW_TOKENS_INFERENCE >= 400
    assert settings.MAX_INPUT_LENGTH + settings.MAX_NEW_TOKENS_INFERENCE <= settings.MAX_TOTAL_TOKENS
