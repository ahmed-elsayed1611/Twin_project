"""Runs the real code cells of notebooks/sft_nilechat_v2.ipynb on CPU.

Only what needs Colab or a GPU is replaced: Drive, the Hugging Face login, and Unsloth's model loading
(a tiny random Qwen2 model with a real LoRA adapter from peft). The tokenizer is a real byte-level
tokenizer with NileChat's ChatML chat template, and training uses the real transformers Trainer.
Synthetic chats only: no private data, GPU or token needed.
"""

import contextlib
import getpass
import io
import json
import os
import random
import sys
import tempfile
import types

import huggingface_hub
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "notebooks", "sft_nilechat_v2.ipynb")

CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def make_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tok.normalizer = normalizers.NFC()  # like NileChat's (Qwen) tokenizer: decomposed letters come back composed
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", clean_up_tokenization_spaces=False
    )
    fast.chat_template = CHATML
    return fast


TOKENIZER = make_tokenizer()
MODEL_CONFIG = Qwen2Config(
    vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
)


def make_base():
    torch.manual_seed(0)  # the same base weights every time, as with a real pretrained model
    return Qwen2ForCausalLM(MODEL_CONFIG)


def _save_pretrained_merged(model, path, tokenizer, save_method="merged_16bit"):
    model.merge_and_unload().save_pretrained(path)
    tokenizer.save_pretrained(path)


class FastLanguageModel:
    @staticmethod
    def from_pretrained(model_name, max_seq_length=None, load_in_4bit=False, **kwargs):
        if os.path.exists(os.path.join(str(model_name), "adapter_config.json")):
            model = PeftModel.from_pretrained(make_base(), model_name)
        else:
            model = make_base()
        model.save_pretrained_merged = types.MethodType(_save_pretrained_merged, model)
        return model, TOKENIZER

    @staticmethod
    def get_peft_model(model, r, lora_alpha, lora_dropout, target_modules, random_state=0, **kwargs):
        torch.manual_seed(random_state)
        return get_peft_model(model, LoraConfig(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, task_type="CAUSAL_LM"))

    @staticmethod
    def for_inference(model):
        model.eval()


def install_fakes():
    unsloth = types.ModuleType("unsloth")
    unsloth.__version__ = "fake-for-tests"
    unsloth.FastLanguageModel = FastLanguageModel
    sys.modules["unsloth"] = unsloth

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    sys.modules["google"], sys.modules["google.colab"] = google, colab

    huggingface_hub.login = lambda token=None, **kwargs: None
    getpass.getpass = lambda prompt="": "fake-token"

    kaggle_secrets = types.ModuleType("kaggle_secrets")
    kaggle_secrets.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda name: "fake-token")
    sys.modules["kaggle_secrets"] = kaggle_secrets


# ---------- synthetic data ----------
SYSTEM_MESSAGE = "Continue the conversation as Ahmed. Synthetic test data."
DECOMPOSED_REPLY = "آه تمام"  # "آه تمام" with the آ decomposed
WORDS = ["تمام", "ماشي", "انت فين", "في الطريق", "ربع ساعة", "عامل ايه", "الحمد لله", "يلا", "[NAME]", "ok", "بكرة"]


def make_record(rng, turns, long=False):
    roles = ["user", "assistant"] * 6
    start = rng.choice([0, 0, 0, 1])  # some windows start with one of Ahmed's own turns, as in the real data
    roles = roles[start:start + turns]
    if roles[-1] != "assistant":
        roles = roles[:-1]
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for role in roles:
        messages.append({"role": role, "content": " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 5)))})
    if long:
        messages[-2]["content"] = "كلام طويل " * 400
    return {"messages": messages}


def write_project(drive_root, n_train=60, n_val=12, private_names_count=2, leak_name=None, names_file=True,
                  data_dir=None):
    data_dir = data_dir or os.path.join(drive_root, "ahmed_twin", "redacted")
    os.makedirs(data_dir, exist_ok=True)
    rng = random.Random(0)
    train = [make_record(rng, rng.randint(2, 6)) for _ in range(n_train - 1)] + [make_record(rng, 4, long=True)]
    val = [make_record(rng, rng.randint(2, 6)) for _ in range(n_val)]
    # A reply with a decomposed letter (ا + combining madda, U+0653), as 11 real replies had: the tokenizer
    # returns it composed (آ), so a character-for-character mask check would wrongly fail.
    train[5]["messages"][-1]["content"] = DECOMPOSED_REPLY
    if leak_name:
        train[3]["messages"][-1]["content"] = f"قول ل{leak_name} يكلمني"
    for name, records in (("train", train), ("validation", val)):
        with open(os.path.join(data_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = {
        "samples_train": len(train), "samples_validation": len(val), "pii_redactions": {"NAME": 5},
        "config": {"private_names_count": private_names_count}, "output_validation": {"valid": True, "errors": []},
    }
    with open(os.path.join(data_dir, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f)
    if names_file:
        with open(os.path.join(drive_root, "ahmed_twin", "private_names.txt"), "w", encoding="utf-8") as f:
            f.write("# synthetic\nزكريا\nZoe\n")


# ---------- running the notebook ----------
SMALL_CONFIG = {
    "max_seq_length": 1024, "lora_r": 4, "lora_alpha": 4, "learning_rate": 1e-3, "warmup_steps": 0,
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1, "per_device_eval_batch_size": 4,
    "num_train_epochs": 4, "eval_steps": 3, "early_stopping_patience": 2,
}


KAGGLE_NOTEBOOK = os.path.join(HERE, "..", "notebooks", "sft_nilechat_v2_kaggle.ipynb")
MERGE_NOTEBOOK = os.path.join(HERE, "..", "notebooks", "sft_v2_merge_adapter.ipynb")


def code_cells(notebook=NOTEBOOK):
    with open(notebook, encoding="utf-8") as f:
        notebook = json.load(f)
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def adapt(source, drive_root, replacements=None):
    """Removes pip installs and the GPU check, and points platform paths at temporary folders."""
    replacements = {'"/content/drive/MyDrive"': repr(drive_root), **(replacements or {})}
    lines = []
    for line in source.split("\n"):
        if line.lstrip().startswith("!"):
            continue  # pip install
        if "torch.cuda.is_available(), \"No GPU" in line or "torch.cuda.get_device_name" in line:
            continue  # the GPU check
        for old, new in replacements.items():
            line = line.replace(old, new)
        lines.append(line)
    return "\n".join(lines)


def run_notebook(drive_root, config_overrides=None, constant_overrides=None, trainer_class=None,
                 notebook=NOTEBOOK, replacements=None):
    """Executes every code cell in order. Returns (namespace, stdout, exception or None)."""
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    # Patch the module object that `from transformers import Trainer` reads: with transformers' lazy
    # loading, it is not always the object this file imported.
    imported = sys.modules["transformers"]
    original_trainer = imported.Trainer
    if trainer_class:
        imported.Trainer = trainer_class
    try:
        with contextlib.redirect_stdout(stdout):
            for source in code_cells(notebook):
                exec(compile(adapt(source, drive_root, replacements), "<cell>", "exec"), namespace)
                if "CONFIG = {" in source:
                    namespace["CONFIG"].update({**SMALL_CONFIG, **(config_overrides or {})})
                    namespace.update({"MIN_FREE_DRIVE_GB": 0, "MIN_FREE_DISK_GB": 0, "MIN_MERGED_MODEL_GB": 0,
                                      "MAX_DROPPED_FRACTION": 0.05, **(constant_overrides or {})})
        return namespace, stdout.getvalue(), None
    except Exception as error:  # the tests check which error, and where
        return namespace, stdout.getvalue(), error
    finally:
        imported.Trainer = original_trainer


def new_drive():
    return tempfile.mkdtemp(prefix="sft_v2_test_")


install_fakes()
