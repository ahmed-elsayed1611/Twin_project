from __future__ import annotations

from functools import cache

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings

# Where my twin's turns end (ChatML). Generation must stop here or the model goes on to write Contact's next message.
CHATML_END = "<|im_end|>"

# Must stay identical to `alpaca_template` in llm_engineering/model/finetuning/finetune.py: the twin was fine-tuned
# (SFT and DPO) on prompts in this format. That file imports unsloth, so the template is repeated here instead
# of imported; tests/unit/test_rag.py checks the two are the same.
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        instruction = self.prompt.format(query=self.query, context=self.context).strip()

        self.llm.set_payload(
            inputs=ALPACA_TEMPLATE.format(instruction, ""),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer


class TwinChatTemplate:
    """The two tokenizer features a prompt needs (the chat template, special-token ids), read from the files.

    Loading the full tokenizer is not possible here: Colab saved `tokenizer.json` with a newer `tokenizers` than this
    project pins, and the old one fails with "data did not match any variant of untagged enum ModelWrapper" (29 Oct).
    The template is rendered with jinja2 the way transformers' `apply_chat_template` does.
    """

    def __init__(self, template: str, special_token_ids: dict[str, int], eos_token: str | None = None) -> None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)
        self.special_token_ids = special_token_ids
        self.eos_token = eos_token

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        assert not tokenize, "Only text prompts: the endpoint tokenizes."

        return self.template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, eos_token=self.eos_token
        )

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_token_ids[token]


@cache
def twin_tokenizer() -> TwinChatTemplate:
    """My twin's chat template and special tokens, downloaded from its (private) Hub repo."""

    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    def download(filename: str) -> str:
        path = hf_hub_download(settings.HF_MODEL_ID, filename, token=settings.HUGGINGFACE_ACCESS_TOKEN)

        return Path(path).read_text(encoding="utf-8")

    config = json.loads(download("tokenizer_config.json"))
    # transformers >= 4.5x saves the template as its own file instead of inside tokenizer_config.json.
    template = config.get("chat_template") or download("chat_template.jinja")
    added_tokens = json.loads(download("tokenizer.json"))["added_tokens"]

    return TwinChatTemplate(template, {t["content"]: t["id"] for t in added_tokens}, config.get("eos_token"))


class TwinReplyExecutor:
    """Gets my WhatsApp twin's reply to one message from Contact.

    The book's InferenceExecutor writes LinkedIn posts from RAG context in the Alpaca format. My twin was fine-tuned
    on something else: a fixed system message, Contact's message as the user turn and my reply as the assistant turn,
    in ChatML. This builds exactly that prompt with the twin's own chat template and stops at the end of my turn.
    """

    def __init__(self, llm: Inference, message: str, tokenizer=None) -> None:
        self.llm = llm
        self.message = message
        self.tokenizer = tokenizer

    def execute(self) -> str:
        tokenizer = self.tokenizer or twin_tokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": settings.TWIN_SYSTEM_MESSAGE}, {"role": "user", "content": self.message}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The evaluation notebook's generation (greedy, repetition_penalty 1.1, no_repeat_ngram_size 4), so the endpoint
        # answers the way the twin was scored. Sampling with no repetition guard looped ("انا احمد احمد احمد ...").
        parameters = {
            "max_new_tokens": settings.TWIN_MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
        # The two containers differ: TGI (GPU) takes a stop string and has no no_repeat_ngram_size; the transformers
        # pipeline (CPU) passes everything on to generate(), which takes a token id and rejects "stop".
        if settings.DEPLOYMENT_TARGET == "cpu":
            parameters["eos_token_id"] = tokenizer.convert_tokens_to_ids(CHATML_END)
            parameters["no_repeat_ngram_size"] = 4
        else:
            parameters["stop"] = [CHATML_END]
        self.llm.set_payload(inputs=prompt, parameters=parameters)
        reply = self.llm.inference()[0]["generated_text"]

        return reply.split(CHATML_END, 1)[0].strip()
