from __future__ import annotations

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings

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
