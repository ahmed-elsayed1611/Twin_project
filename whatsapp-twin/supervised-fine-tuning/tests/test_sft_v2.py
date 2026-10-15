"""CPU tests for notebooks/sft_nilechat_v2.ipynb. See harness.py for what is real and what is faked.

    pip install -r tests/requirements-test.txt
    python tests/test_sft_v2.py
"""

import hashlib
import json
import os
import sys

import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import new_drive, run_notebook, write_project  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def paths(drive):
    project = os.path.join(drive, "ahmed_twin")
    run = os.path.join(project, "sft_runs", "sft_v2_redacted_final_reply_loss")
    return {
        "summary": os.path.join(run, "run_summary.json"),
        "best_adapter": os.path.join(run, "best_adapter"),
        "merged": os.path.join(project, "model_merged_nilechat_v2"),
        "v1": os.path.join(project, "model_merged_nilechat"),
        "train": os.path.join(project, "redacted", "train.jsonl"),
    }


# 1. A full run from scratch
drive = new_drive()
write_project(drive)
ns, out, error = run_notebook(drive)
p = paths(drive)
check("1.1 full run finishes without an error", error is None, repr(error))
check("1.2 only the final reply and <|im_end|> are trained (made-up chat)",
      ns.get("demo_trained") == "ربع ساعة<|im_end|>", repr(ns.get("demo_trained")))
check("1.3 the real-data mask check ran on every kept sample", "the trained part is exactly the final reply in all 59" in out)
summary = json.load(open(p["summary"], encoding="utf-8")) if os.path.exists(p["summary"]) else {}
check("1.4 the over-long sample is dropped and counted, not truncated",
      summary.get("train_data", {}).get("dropped") == {"longer than max_seq_length": 1}, summary.get("train_data"))
check("1.5 the summary records the best loss, the curve and the step counts",
      isinstance(summary.get("best_eval_loss"), float) and summary.get("eval_history")
      and summary.get("stopped_at_step", 0) > 0 and summary.get("planned_steps", 0) >= summary.get("stopped_at_step", 1))
with open(p["train"], "rb") as f:
    train_hash = hashlib.sha256(f.read()).hexdigest()
check("1.6 the summary fingerprints the exact training file", summary.get("data", {}).get("train_sha256") == train_hash)
check("1.7 the best adapter is saved", os.path.exists(os.path.join(p["best_adapter"], "adapter_config.json")))
check("1.8 the merged v2 model is saved with its run summary",
      os.path.exists(os.path.join(p["merged"], "model.safetensors"))
      and os.path.exists(os.path.join(p["merged"], "sft_run_summary.json")))
check("1.9 the v1 model folder is never written", not os.path.exists(p["v1"]))
check("1.10 the inference check counts private names in the replies", "Replies mentioning a private name:" in out)

# 2. Running the finished notebook again
ns2, out2, error2 = run_notebook(drive)
check("2.1 a finished run is not trained again", "Already complete -- not training again." in out2)
check("2.2 the existing merged model is not overwritten by default", isinstance(error2, FileExistsError), repr(error2))
ns3, out3, error3 = run_notebook(drive, constant_overrides={"OVERWRITE_MERGED": True})
check("2.3 with OVERWRITE_MERGED it re-merges without training", error3 is None and "train_result" not in ns3, repr(error3))

# 3. Same run name, different settings
ns4, out4, error4 = run_notebook(drive, config_overrides={"learning_rate": 5e-4})
check("3.1 a different config under the same RUN_NAME is refused",
      isinstance(error4, ValueError) and "different config" in str(error4), repr(error4))

# 4. A Colab disconnect in the middle of training, then a re-run
class _Disconnect(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 7:
            raise RuntimeError("simulated Colab disconnect")


class _DisconnectingTrainer(transformers.Trainer):
    def __init__(self, *args, callbacks=None, **kwargs):
        super().__init__(*args, callbacks=(callbacks or []) + [_Disconnect()], **kwargs)


drive = new_drive()
write_project(drive)
_, _, error5 = run_notebook(drive, trainer_class=_DisconnectingTrainer)
check("4.1 the simulated disconnect stops training", isinstance(error5, RuntimeError), repr(error5))
ns6, out6, error6 = run_notebook(drive)
check("4.2 the re-run resumes from the last checkpoint", "Resuming from" in out6 and "checkpoint-6" in out6)
check("4.3 the resumed run finishes", error6 is None and os.path.exists(paths(drive)["merged"]), repr(error6))

# 5. Data that must be refused before any training
drive = new_drive()
write_project(drive, leak_name="زكريا")
_, out7, error7 = run_notebook(drive)
check("5.1 a private name in the data stops the notebook",
      isinstance(error7, ValueError) and "private name" in str(error7), repr(error7))
check("5.2 the name itself is never printed", "زكريا" not in out7 and "زكريا" not in str(error7))
check("5.3 nothing was trained", not os.path.exists(paths(drive)["summary"]))

drive = new_drive()
write_project(drive, private_names_count=0)
_, _, error8 = run_notebook(drive)
check("5.4 data built without the names file is refused",
      isinstance(error8, ValueError) and "WITHOUT --redact-names-file" in str(error8), repr(error8))

drive = new_drive()
write_project(drive, names_file=False)
_, _, error9 = run_notebook(drive)
check("5.5 a missing private_names.txt is refused", isinstance(error9, FileNotFoundError), repr(error9))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
