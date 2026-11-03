from openai import OpenAI

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings

INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "\n\n### Response:"


class LLMInferenceOpenAICompatible(Inference):
    """Answers through any OpenAI-compatible chat API (OpenAI, Cerebras, ...), with the same interface as
    LLMInferenceSagemakerEndpoint, so the RAG pipeline runs before the fine-tuned model is deployed."""

    def __init__(self, model_id: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__()
        self.model_id = model_id or settings.OPENAI_MODEL_ID
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY, base_url=base_url or settings.OPENAI_BASE_URL)
        self.payload: dict = {}

    def set_payload(self, inputs: str, parameters: dict | None = None) -> None:
        # InferenceExecutor wraps the instruction in the Alpaca template the fine-tuned twin expects. A chat model
        # needs only the instruction itself.
        instruction = inputs
        if INSTRUCTION_MARKER in instruction:
            instruction = instruction.split(INSTRUCTION_MARKER, 1)[1].split(RESPONSE_MARKER, 1)[0]
        parameters = parameters or {}
        self.payload = {
            "messages": [{"role": "user", "content": instruction.strip()}],
            "max_tokens": parameters.get("max_new_tokens", settings.MAX_NEW_TOKENS_INFERENCE),
            "temperature": parameters.get("temperature", settings.TEMPERATURE_INFERENCE),
        }

    def inference(self) -> list[dict]:
        response = self.client.chat.completions.create(model=self.model_id, **self.payload)

        return [{"generated_text": response.choices[0].message.content or ""}]
