from functools import cache
from typing import Generator

import tiktoken
from transformers import AutoTokenizer

from llm_engineering.settings import settings


def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]


def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))


def tiktoken_encoding_for(model_id: str) -> tiktoken.Encoding:
    # Non-OpenAI model ids (e.g. Cerebras' "gpt-oss-120b") are unknown to tiktoken; o200k_base is the encoding of
    # the current OpenAI models and a close enough count for budgeting prompts.
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


@cache
def _get_tokenizer():
    """The tokenizer of the model that answers /rag, loaded once (not three times per request).

    With the OpenAI-compatible backend that is tiktoken: loading HF_MODEL_ID would count with the wrong tokenizer,
    and since HF_MODEL_ID points at my private twin it failed with 401 and broke every /rag request (30 Oct).
    """

    if settings.LLM_INFERENCE_BACKEND == "openai_compatible":
        return tiktoken_encoding_for(settings.OPENAI_MODEL_ID)

    return AutoTokenizer.from_pretrained(settings.HF_MODEL_ID, token=settings.HUGGINGFACE_ACCESS_TOKEN)


def compute_num_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if isinstance(tokenizer, tiktoken.Encoding):
        return len(tokenizer.encode(text))

    return len(tokenizer.encode(text, add_special_tokens=False))
