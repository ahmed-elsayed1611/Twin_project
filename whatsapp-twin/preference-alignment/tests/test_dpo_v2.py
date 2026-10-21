"""CPU tests for notebooks/dpo_v2.ipynb.

The notebook's real cells run with the real TRL DPOTrainer, a tiny random Qwen2 model with a real LoRA adapter,
a real byte-level tokenizer with NileChat's chat template, and synthetic pairs made by the real pair builder.
Only Colab, the GPU and Unsloth's loading are faked (the harness is shared with the SFT tests).

    pip install -r ../supervised-fine-tuning/tests/requirements-test.txt trl==1.15.0
    python tests/test_dpo_v2.py
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "supervised-fine-tuning" / "tests"))
sys.path.insert(0, str(HERE.parent / "scripts"))

import harness  # noqa: E402  (installs the fake unsloth module)
import build_dpo_pairs_v2 as builder  # noqa: E402
from dpo_test_data import fake_embed, write_data  # noqa: E402

unsloth = sys.modules["unsloth"]
unsloth.PatchDPOTrainer = lambda: None
harness.FastLanguageModel.for_training = staticmethod(lambda m: m.train())

import torch  # noqa: E402
import trl  # noqa: E402, F401  (loads trl so sys.modules['trl'] can be patched below)
import trl.trainer.utils as trl_utils  # noqa: E402


class _ReferenceChunkedLogProb:
    """TRL 1.15's DPO always scores tokens with a Triton GPU kernel, and Triton is not in CPU-only PyTorch. This is
    the same computation in plain PyTorch (the CPU tests only; Colab uses TRL's kernel): per-token log-probabilities
    of `hidden @ weight.T`, plus the optional extras when requested."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, softcap=None, logit_scale=1.0,
              outputs=("log_probs",)):
        logits = (hidden @ weight.t()).float()
        if bias is not None:
            logits = logits + bias.float()
        logits = logits * (logit_scale or 1.0)
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        logits = logits / temperature
        log_softmax = torch.log_softmax(logits, dim=-1)
        log_probs = log_softmax.gather(1, targets[:, None]).squeeze(1)
        entropy = -(log_softmax.exp() * log_softmax).sum(-1) if "entropy" in outputs else None
        log_sum_sq = torch.logsumexp(2 * log_softmax, dim=-1) if "log_sum_sq_probs" in outputs else None
        mean_logits = logits.mean(-1) if "mean_logits" in outputs else None
        is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return log_probs, entropy, log_sum_sq, mean_logits, is_top1


if trl_utils._ChunkedLogProbFunction is None:
    trl_utils._ChunkedLogProbFunction = _ReferenceChunkedLogProb

NOTEBOOK = str(HERE.parent / "notebooks" / "dpo_v2.ipynb")
CONFIG_OVERRIDES = {"max_length": 4096, "learning_rate": 1e-3, "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 1, "per_device_eval_batch_size": 4, "num_train_epochs": 2,
                    "eval_steps": 3, "early_stopping_patience": 2, "warmup_steps": 0, "lora_r": 4, "lora_alpha": 4}
CONSTANTS = {"MIN_FREE_DRIVE_GB": 0, "MIN_ADAPTER_BYTES": 0, "MAX_DROPPED_FRACTION": 0.5, "N_HELDOUT_GENERATIONS": 2,
             "TEST_PAIRS_LIMIT": 30}
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def make_project(loss_on="final_reply_only", mismatch_data=False, names="# synthetic\nزكريا\n"):
    drive = Path(harness.new_drive())
    project = drive / "ahmed_twin"
    data_dir = Path(tempfile.mkdtemp()) / "redacted"
    data_dir.mkdir(parents=True)
    write_data(data_dir)
    builder.main(["--data-dir", str(data_dir), "--output-dir", str(project / "dpo_v2" / "pairs"),
                  "--train-pairs", "40", "--dev-pairs", "12"], embed=fake_embed)
    sft_dir = project / "model_merged_nilechat_v2"
    sft_dir.mkdir(parents=True)
    train_sha = builder.sha256_of(data_dir / "train.jsonl")
    summary = {"config": {"loss_on": loss_on}, "best_eval_loss": 2.65, "best_checkpoint": "/x/checkpoint-1500",
               "data": {"train_sha256": "0" * 64 if mismatch_data else train_sha,
                        "validation_sha256": builder.sha256_of(data_dir / "validation.jsonl")}}
    (sft_dir / "sft_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (project / "private_names.txt").write_text(names, encoding="utf-8")
    return str(drive), project


def run(drive, **kwargs):
    return harness.run_notebook(drive, notebook=NOTEBOOK, config_overrides={**CONFIG_OVERRIDES, **kwargs.pop("config", {})},
                                constant_overrides={**CONSTANTS, **kwargs.pop("constants", {})}, **kwargs)


run_dir = lambda project: project / "dpo_v2" / "runs" / "dpo_v2_beta02_sft05"  # noqa: E731

# 1. A full run
drive, project = make_project()
ns, out, error = run(drive)
check("1.1 the notebook runs end to end", error is None, repr(error))
check("1.2 the anchor is on, in TRL 1.15's API", "loss_type=['sigmoid','sft']" in out)
check("1.3 replies end with exactly one <|im_end|>, prompts unchanged", "Checked: replies end with exactly one <|im_end|>" in out)
summary_path = run_dir(project) / "run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
check("1.4 the summary has the before/after test scores and McNemar",
      {"sft_v2", "dpo", "mcnemar"} <= set(summary.get("test_preference", {})), summary.get("test_preference", {}).keys())
check("1.5 the summary records the base model and the pairs it used",
      summary.get("base", {}).get("sft_model_path", "").endswith("model_merged_nilechat_v2") and "manifest" in summary.get("pairs", {}))
check("1.6 the adapter and the private generations are saved",
      (run_dir(project) / "adapter" / "adapter_config.json").exists() and (run_dir(project) / "generations_private.json").exists())
check("1.6b the saved adapter holds only the trained adapter, not TRL's frozen 'ref' copy",
      not (run_dir(project) / "adapter" / "ref").exists())
summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
check("1.7 the shareable summary holds no conversation text", "النهاردة" not in summary_text and "الشغل" not in summary_text)
check("1.8 v1's DPO folder is never written", not (project.parent / "DPO").exists())
check("1.9 the diagnosis ran", "Preference accuracy" in out or "Improvement within noise" in out or "No measurable" in out)

# 2. Running a finished run again
ns2, out2, error2 = run(drive)
check("2.1 a finished run is not trained again", error2 is None and "Already complete" in out2, repr(error2))

# 3. Inputs that must be refused
_, _, error3 = run(drive, config={"beta": 0.1})
check("3.1 a different CONFIG under the same RUN_NAME is refused", isinstance(error3, ValueError) and "different SFT model" in str(error3), repr(error3))

drive4, _ = make_project(loss_on=None)
_, _, error4 = run(drive4)
check("3.2 a base model that is not SFT v2 is refused", isinstance(error4, ValueError) and "not an SFT v2 model" in str(error4), repr(error4))

drive5, _ = make_project(mismatch_data=True)
_, _, error5 = run(drive5)
check("3.3 pairs built from other data than SFT v2's are refused",
      isinstance(error5, ValueError) and "different train/validation files" in str(error5), repr(error5))

drive6, project6 = make_project(names="النهاردة\n")
_, out6, error6 = run(drive6)
check("3.4 a private name in the pairs stops the notebook", isinstance(error6, ValueError) and "private name" in str(error6), repr(error6))
check("3.5 nothing was trained", not (run_dir(project6) / "run_summary.json").exists())

# 4. A TRL without any anchor support
class _NoAnchorConfig:
    def __init__(self, output_dir=None, beta=0.1):
        pass


original_config = sys.modules["trl"].DPOConfig
sys.modules["trl"].DPOConfig = _NoAnchorConfig
try:
    drive7, _ = make_project()
    _, _, error7 = run(drive7)
finally:
    sys.modules["trl"].DPOConfig = original_config
check("4.1 plain DPO without the anchor is refused", isinstance(error7, RuntimeError) and "anchor" in str(error7), repr(error7))

# 5. A Colab disconnect during training, then a re-run
import transformers  # noqa: E402


class _Disconnect(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 7:
            raise RuntimeError("simulated Colab disconnect")


original_trainer = sys.modules["trl"].DPOTrainer


class _DisconnectingTrainer(original_trainer):
    def __init__(self, *args, callbacks=None, **kwargs):
        super().__init__(*args, callbacks=(callbacks or []) + [_Disconnect()], **kwargs)


drive8, project8 = make_project()
sys.modules["trl"].DPOTrainer = _DisconnectingTrainer
try:
    _, _, error8 = run(drive8)
finally:
    sys.modules["trl"].DPOTrainer = original_trainer
check("5.1 the simulated disconnect stops training", isinstance(error8, RuntimeError) and "disconnect" in str(error8), repr(error8))
_, out9, error9 = run(drive8)
check("5.2 the re-run resumes from the last checkpoint and reuses the baseline",
      "Resuming from" in out9 and "Baseline reused" in out9, out9[-300:])
check("5.3 the resumed run finishes", error9 is None and (run_dir(project8) / "run_summary.json").exists(), repr(error9))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
