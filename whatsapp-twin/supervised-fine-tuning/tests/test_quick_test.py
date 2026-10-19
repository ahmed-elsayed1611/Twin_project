"""CPU test for notebooks/sft_quick_test_v1_vs_v2.ipynb, with the same harness as test_sft_v2.py.

Both "models" are the harness's tiny random model, so v1 and v2 score identically: the test checks that the
notebook runs end to end, writes its totals, and never writes conversation text, not the scores themselves.

    pip install -r tests/requirements-test.txt
    python tests/test_quick_test.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import new_drive, run_notebook, write_project  # noqa: E402

QUICK_NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks",
                              "sft_quick_test_v1_vs_v2.ipynb")
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def make_models(drive):
    project = os.path.join(drive, "ahmed_twin")
    os.makedirs(os.path.join(project, "model_merged_nilechat"))
    v2 = os.path.join(project, "model_merged_nilechat_v2")
    os.makedirs(v2)
    with open(os.path.join(v2, "sft_run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"stopped_at_step": 1700, "best_eval_loss": 2.65, "best_checkpoint": "/x/checkpoint-1500"}, f)
    return os.path.join(project, "sft_runs", "sft_v2_redacted_final_reply_loss", "quick_test_v1_vs_v2.json")


# 1. A normal run
drive = new_drive()
write_project(drive)
results_path = make_models(drive)
ns, out, error = run_notebook(drive, notebook=QUICK_NOTEBOOK)
check("1.1 the notebook runs end to end", error is None, repr(error))
check("1.2 it reads the v2 run summary", "v2 was trained to step 1700" in out)
report = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else {}
check("1.3 the totals are saved", {"likelihood_all", "likelihood_fair_subset", "median_reply_chars",
                                   "replies_with_a_private_name"} <= set(report), sorted(report))
check("1.4 the same model scores the same (losses compared like for like)",
      report.get("likelihood_all", {}).get("v1_loss") == report.get("likelihood_all", {}).get("v2_loss"))
check("1.5 the private-name count runs", isinstance(report.get("replies_with_a_private_name", {}).get("v2"), int))
saved = open(results_path, encoding="utf-8").read() if os.path.exists(results_path) else ""
check("1.6 no conversation text is saved", not any(w in saved for w in ("تمام", "ماشي", "في الطريق", "بكرة")))
check("1.7 examples stay hidden by default", "Examples hidden" in out)
check("1.8 the chat cell answers", "chat_model" in ns)

# 2. A missing v2 model stops at once
drive = new_drive()
write_project(drive)
_, _, error2 = run_notebook(drive, notebook=QUICK_NOTEBOOK)
check("2.1 a missing model folder is reported", isinstance(error2, AssertionError) and "Missing" in str(error2),
      repr(error2))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
