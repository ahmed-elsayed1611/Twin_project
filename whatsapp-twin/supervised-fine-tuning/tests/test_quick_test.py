"""CPU test for notebooks/sft_quick_test_v1_vs_v2.ipynb (version 2), with the same harness as test_sft_v2.py.

Both "models" are the harness's tiny random model, so v1 and v2 score identically: the test checks that the
notebook runs end to end (tokenizer check, both models, the filter, the ready-for-DPO verdict), saves totals only,
and embeds tools/reply_filter.py unchanged. Nothing is downloaded: the tokenizer reference points at a local copy.

    pip install -r tests/requirements-test.txt
    python tests/test_quick_test.py
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import TOKENIZER, new_drive, run_notebook, write_project  # noqa: E402

QUICK_NOTEBOOK = str(HERE.parent / "notebooks" / "sft_quick_test_v1_vs_v2.ipynb")
FILTER_FILE = HERE.parents[1] / "tools" / "reply_filter.py"
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def make_models(drive):
    project = Path(drive) / "ahmed_twin"
    for folder in ("model_merged_nilechat", "model_merged_nilechat_v2"):
        TOKENIZER.save_pretrained(project / folder)
    summary = {"stopped_at_step": 1750, "best_eval_loss": 2.6385, "best_checkpoint": "/x/checkpoint-1550"}
    (project / "model_merged_nilechat_v2" / "sft_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return project / "sft_runs" / "sft_v2_redacted_final_reply_loss" / "test_v1_vs_v2.json"


def run(drive):
    v2_path = str(Path(drive) / "ahmed_twin" / "model_merged_nilechat_v2")
    return run_notebook(drive, notebook=QUICK_NOTEBOOK, replacements={'"Qwen/Qwen2.5-3B"': repr(v2_path)})


# 0. The notebook embeds the repository's filter unchanged
cells = ["".join(c["source"]) for c in json.load(open(QUICK_NOTEBOOK, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
check("0.1 the filter cell is exactly tools/reply_filter.py", FILTER_FILE.read_text(encoding="utf-8").strip() in [c.strip() for c in cells])

# 1. A normal run
drive = new_drive()
write_project(drive)
results_path = make_models(drive)
ns, out, error = run(drive)
check("1.1 the notebook runs end to end", error is None, repr(error))
check("1.2 the tokenizer check reaches a verdict", ns.get("TOKENIZER_CHECK", {}).get("verdict", "").startswith("harmless"),
      ns.get("TOKENIZER_CHECK"))
report = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
check("1.3 the totals are saved, raw and filtered, for both models",
      all(k in report for k in ("likelihood_all", "likelihood_fair_subset", "tokenizer_check", "v1", "v2", "ready_for_dpo"))
      and {"raw", "filtered", "filter"} <= set(report.get("v2", {})), sorted(report))
check("1.4 the same model scores the same (losses compared like for like)",
      report.get("likelihood_all", {}).get("v1_loss") == report.get("likelihood_all", {}).get("v2_loss"))
check("1.5 filtered replies never contain a placeholder", report.get("v2", {}).get("filtered", {}).get("with_placeholder") == 0)
check("1.6 the ready-for-DPO verdict is printed and saved", ("READY FOR DPO v2" in out or "NOT READY" in out)
      and "ready" in report.get("ready_for_dpo", {}))
check("1.7 the made-up chats are printed", "[late, follow-up]" in out and "[privacy probe]" in out)
saved = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
check("1.8 no conversation text is saved", not any(w in saved for w in ("تمام", "ماشي", "في الطريق", "بكرة", "النهاردة")))
check("1.9 real examples stay hidden by default", "Contact:" not in out)
check("1.10 the chat cell answers", "chat_model" in ns)

# 2. A missing v2 model stops at once
drive = new_drive()
write_project(drive)
_, _, error2 = run(drive)
check("2.1 a missing model folder is reported", isinstance(error2, AssertionError) and "Missing" in str(error2), repr(error2))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
