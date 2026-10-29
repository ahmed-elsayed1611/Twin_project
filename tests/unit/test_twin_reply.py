from unittest import mock

import pytest

from llm_engineering.domain.inference import Inference
from llm_engineering.model.inference.run import CHATML_END, TwinChatTemplate, TwinReplyExecutor
from llm_engineering.settings import settings

IM_END_ID = 151645


class FakeTokenizer:
    """Renders ChatML the way the twin's chat template does and records the messages it was given."""

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}{CHATML_END}\n" for m in messages)

        return turns + "<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, token):
        assert token == CHATML_END

        return IM_END_ID


class FakeEndpoint(Inference):
    def __init__(self, generated_text: str):
        super().__init__()
        self.generated_text = generated_text
        self.payload = None

    def set_payload(self, inputs, parameters=None):
        self.payload = {"inputs": inputs, "parameters": parameters}

    def inference(self):
        return [{"generated_text": self.generated_text}]


@pytest.fixture
def target(monkeypatch):
    def set_target(value: str) -> None:
        monkeypatch.setattr(settings, "DEPLOYMENT_TARGET", value)

    return set_target


def test_prompt_is_the_training_format(target):
    target("cpu")
    tokenizer, llm = FakeTokenizer(), FakeEndpoint("تمام الحمدلله")

    TwinReplyExecutor(llm, "عامل ايه؟", tokenizer=tokenizer).execute()

    assert tokenizer.messages == [
        {"role": "system", "content": settings.TWIN_SYSTEM_MESSAGE},
        {"role": "user", "content": "عامل ايه؟"},
    ]
    assert llm.payload["inputs"].endswith("<|im_start|>assistant\n")
    assert "### Instruction" not in llm.payload["inputs"]


def test_cpu_stops_with_a_token_id(target):
    target("cpu")
    llm = FakeEndpoint("تمام")

    TwinReplyExecutor(llm, "hi", tokenizer=FakeTokenizer()).execute()

    parameters = llm.payload["parameters"]
    assert parameters["eos_token_id"] == IM_END_ID and "stop" not in parameters
    assert parameters["max_new_tokens"] == settings.TWIN_MAX_NEW_TOKENS
    assert parameters["return_full_text"] is False
    # Same decoding as the evaluation notebook.
    assert parameters["do_sample"] is False and parameters["repetition_penalty"] == 1.1
    assert parameters["no_repeat_ngram_size"] == 4


def test_gpu_stops_with_a_stop_string(target):
    target("gpu")
    llm = FakeEndpoint("تمام")

    TwinReplyExecutor(llm, "hi", tokenizer=FakeTokenizer()).execute()

    assert llm.payload["parameters"]["stop"] == [CHATML_END]
    assert "eos_token_id" not in llm.payload["parameters"]
    assert "no_repeat_ngram_size" not in llm.payload["parameters"]  # TGI does not have it


def test_reply_is_cut_at_the_end_of_my_turn(target):
    target("cpu")
    llm = FakeEndpoint(f" تمام الحمدلله {CHATML_END}\n<|im_start|>user\nوانت؟")  # noqa: RUF001 (Arabic, not a lookalike)

    assert TwinReplyExecutor(llm, "hi", tokenizer=FakeTokenizer()).execute() == "تمام الحمدلله"


def test_chat_template_renders_chatml_without_the_tokenizers_library():
    template = (
        "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}"
        "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
    )
    chat = TwinChatTemplate(template, {CHATML_END: IM_END_ID})

    prompt = chat.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])

    assert prompt == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    assert chat.convert_tokens_to_ids(CHATML_END) == IM_END_ID


@pytest.mark.parametrize(
    ("deployment_target", "image_part", "instance_setting"),
    [("cpu", "huggingface-pytorch-inference:", "CPU_INSTANCE_TYPE"), ("gpu", "tgi", "GPU_INSTANCE_TYPE")],
)
def test_create_endpoint_picks_image_and_instance(target, monkeypatch, deployment_target, image_part, instance_setting):
    from llm_engineering.infrastructure.aws.deploy.huggingface import run

    target(deployment_target)
    monkeypatch.setattr(settings, "AWS_ARN_ROLE", "arn:aws:iam::123456789012:role/test")
    monkeypatch.setattr(run, "get_huggingface_llm_image_uri", lambda *args, **kwargs: "tgi-image")
    monkeypatch.setattr(run, "sagemaker_session", lambda: None)
    monkeypatch.setattr(run, "ResourceManager", mock.Mock())
    monkeypatch.setattr(run, "DeploymentService", mock.Mock())
    strategy = mock.Mock()
    monkeypatch.setattr(run, "SagemakerHuggingfaceStrategy", strategy)

    run.create_endpoint()

    kwargs = strategy.return_value.deploy.call_args.kwargs
    assert image_part in kwargs["llm_image"]
    assert kwargs["gpu_instance_type"] == getattr(settings, instance_setting)
    if deployment_target == "cpu":
        assert kwargs["llm_image"].endswith(settings.CPU_INFERENCE_IMAGE_TAG)
        assert kwargs["config"]["HF_TASK"] == "text-generation" and kwargs["resources"] is None
