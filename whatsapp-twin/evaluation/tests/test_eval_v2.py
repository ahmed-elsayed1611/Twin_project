"""CPU test for notebooks/evaluation_v2.ipynb, with the SFT tests' harness and the DPO v2 test data.

Every model is the harness's tiny random Qwen2 (the judge too), the adapters are real random LoRAs, and the pairs
come from the real pair builder. The test checks the mechanics, not the numbers: the v1 leakage check finds planted
copies even when their names were redacted, every system answers every conversation, v1 is only compared on the
fair subset, the judge is gated by its calibration, a run resumes without regenerating, the shareable summary
holds no reply text, and missing optional systems or a disabled judge still give answers.

    python tests/test_eval_v2.py
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "supervised-fine-tuning" / "tests"))
sys.path.insert(0, str(ROOT / "preference-alignment" / "tests"))
sys.path.insert(0, str(ROOT / "preference-alignment" / "scripts"))
from harness import TOKENIZER, make_base, new_drive, run_notebook  # noqa: E402
import build_dpo_pairs_v2 as builder  # noqa: E402
from dpo_test_data import WORDS, fake_embed, write_data  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

NOTEBOOK = str(ROOT / "evaluation" / "notebooks" / "evaluation_v2.ipynb")
FILTER_FILE = ROOT / "tools" / "reply_filter.py"
JUDGE_DIR = Path(tempfile.mkdtemp()) / "judge"
make_base().save_pretrained(JUDGE_DIR)
TOKENIZER.save_pretrained(JUDGE_DIR)
SMALL = {"N_EVAL = 200": "N_EVAL = 16", "N_PREFERENCE_PAIRS = 600": "N_PREFERENCE_PAIRS = 12",
         "max_new_tokens=60": "max_new_tokens=6", "HUMAN_CHECK_SIZE = 30": "HUMAN_CHECK_SIZE = 4",
         '"unsloth/Qwen3-14B-unsloth-bnb-4bit"': repr(str(JUDGE_DIR))}
PLANTED = 3
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def save_model(path):
    make_base().save_pretrained(path)
    TOKENIZER.save_pretrained(path)


def save_adapter(path, seed, changes_model=True):
    import torch
    torch.manual_seed(seed)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
                      init_lora_weights=not changes_model)
    get_peft_model(make_base(), lora).save_pretrained(path)


def make_drive(v1=True, v2_adapter_changes=True):
    drive = Path(new_drive())
    project = drive / "ahmed_twin"
    data_dir = Path(tempfile.mkdtemp()) / "redacted"
    data_dir.mkdir(parents=True)
    write_data(data_dir)
    pairs_dir = project / "dpo_v2" / "pairs"
    builder.main(["--data-dir", str(data_dir), "--output-dir", str(pairs_dir), "--train-pairs", "40", "--dev-pairs", "12"],
                 embed=fake_embed)
    save_model(project / "model_merged_nilechat_v2")
    save_adapter(project / "dpo_v2" / "runs" / "dpo_v2_beta02_sft05" / "adapter", 1, v2_adapter_changes)
    (project / "private_names.txt").write_text("# synthetic\nزكريا\n", encoding="utf-8")
    planted = []
    if v1:
        save_model(project / "model_merged_nilechat")
        save_adapter(drive / "DPO" / "runs" / "dpo_v7_rpo" / "adapter", 2)
        test_pairs = builder.load_jsonl(pairs_dir / "test.jsonl")
        with_name = [p for p in test_pairs if "[NAME]" in json.dumps(p, ensure_ascii=False)]
        planted = (with_name[:1] + [p for p in test_pairs if p not in with_name[:1]])[:PLANTED]
        with open(project / "train.jsonl", "w", encoding="utf-8") as f:
            for pair in planted:   # v1 saw these before redaction: the real name instead of [NAME]
                messages = [*pair["prompt_messages"], {"role": "assistant", "content": pair["chosen"]}]
                messages = [{**m, "content": m["content"].replace("[NAME]", "زكريا")} for m in messages]
                f.write(json.dumps({"id": f"v1-{pair['id']}", "messages": messages}, ensure_ascii=False) + "\n")
    return str(drive), project, [p["id"] for p in planted]


def run(drive, extra=None):
    return run_notebook(drive, notebook=NOTEBOOK, replacements={**SMALL, **(extra or {})})


def eval_dir(project, name="ch7_v2"):
    return project / "eval_v2" / name


# 0. The filter is embedded unchanged
cells = ["".join(c["source"]) for c in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
check("0.1 the filter cell is exactly tools/reply_filter.py", FILTER_FILE.read_text(encoding="utf-8").strip() in [c.strip() for c in cells])

# 1. A full run with all four systems
drive, project, planted_ids = make_drive()
ns, out, error = run(drive)
check("1.1 the notebook runs end to end", error is None, repr(error))
sample_ids = {r["id"] for r in ns.get("GEN", []) + ns.get("PREF", [])}
check("1.2 the v1 leakage check removes the planted copies, even with the name redacted",
      ns.get("V1_LEAK", {}).get("removed") == PLANTED and not sample_ids & set(planted_ids), ns.get("V1_LEAK"))
check("1.3 all four systems answered and scored everything",
      set(ns.get("ACTIVE", {})) == {"sft_v1", "dpo_v1", "sft_v2", "dpo_v2"}
      and all(len(rows) == len(ns["GEN"]) + len(ns["PREF"]) for rows in ns["OUT"].values()))
check("1.4 both adapters are proven active", all(ns.get("ADAPTER_EFFECTS", {}).get(n, 0) > 1e-4 for n in ("dpo_v1", "dpo_v2")),
      ns.get("ADAPTER_EFFECTS"))
fair = sum(r["fair"] for r in ns.get("GEN", []))
check("1.5 the sample has both fair and placeholder conversations", 0 < fair < len(ns.get("GEN", [])), fair)
check("1.6 v1 is compared only on the fair subset",
      ns["OBJECTIVE"]["dpo_v2 vs dpo_v1"]["likelihood"]["n"] == fair and ns["JUDGE"]["Q2 sense"]["n"] == fair
      and ns["OBJECTIVE"]["dpo_v2 vs sft_v2"]["likelihood"]["n"] == len(ns["GEN"]))
check("1.7 every comparison was judged in both orders",
      all(row["n"] == (fair if row["subset"] == "fair" else len(ns["GEN"])) for row in ns["JUDGE"].values()))
check("1.8 the judge is gated by its calibration",
      isinstance(ns.get("JUDGE_TRUSTED"), bool)
      and (ns["JUDGE_TRUSTED"] or not any("judge" in k for k in ns.get("FINDINGS", {}))))
summary_path = eval_dir(project) / "eval_summary.json"
summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
check("1.9 the shareable summary holds no conversation text", summary_text and not any(w in summary_text for w in WORDS if w != "[NAME]"))
private = [json.loads(line) for line in (eval_dir(project) / "eval_private.jsonl").read_text(encoding="utf-8").splitlines()] \
    if (eval_dir(project) / "eval_private.jsonl").exists() else []
check("1.10 the private file has every system's reply", len(private) == len(ns["GEN"])
      and all(all(n in row for n in ns["ACTIVE"]) for row in private))
check("1.11 the four questions are answered", all(q in out for q in ("Q1 Did DPO v2", "Q2 Is v2 better", "Q3 Is DPO v2 safe")))

# 2. Running again resumes and regenerates nothing
sizes = {p.name: p.stat().st_size for p in eval_dir(project).glob("outputs_*")}
ns2, out2, error2 = run(drive)
check("2.1 a second run resumes without generating or judging again",
      error2 is None and "complete" in out2 and "0 to go" in out2
      and sizes == {p.name: p.stat().st_size for p in eval_dir(project).glob("outputs_*")}, repr(error2))

# 3. Different settings under the same name are refused
_, _, error3 = run(drive, {"SEED = 0": "SEED = 1"})
check("3.1 changed settings under the same EVAL_RUN_NAME are refused",
      isinstance(error3, ValueError) and "different settings" in str(error3), repr(error3))

# 4. A trusted judge counts in the findings
ns4, _, error4 = run(drive, {"MIN_JUDGE_CALIBRATION = 0.65": "MIN_JUDGE_CALIBRATION = 0.0",
                             "MIN_OPTION_MASS = 0.5": "MIN_OPTION_MASS = 0.0"})
check("4.0 the tiny random judge fails the format gate (its probability is not on A/B)",
      ns["JUDGE_TRUSTED"] is False and ns["ANSWERS_IN_FORMAT"] is False)
check("4.1 a judge that passes calibration is used in the findings",
      error4 is None and ns4["JUDGE_TRUSTED"] and any("judge" in k for k in ns4["FINDINGS"]), repr(error4))

# 5. Your own votes
votes = "A" * len(ns["HUMAN_SAMPLE"])
ns5, out5, error5 = run(drive, {'VOTES = ""': f"VOTES = {votes!r}"})
check("5.1 votes are counted for the right model and compared with the judge",
      error5 is None and ns5["HUMAN"]["dpo_v2"] == sum(ns5["DPO_IS_A"]) and "The judge agreed with you" in out5, repr(error5))

# 5b. A disconnect while saving left half a line at the end of a file
with open(eval_dir(project) / "outputs_dpo_v2_private.jsonl", "a", encoding="utf-8") as f:
    f.write('{"id": "half-writ')
with open(eval_dir(project) / f"judge_{ns['JUDGE_TAG']}_private.jsonl", "a", encoding="utf-8") as f:
    f.write('{"comparison": "Q1 se')
ns5b, out5b, error5b = run(drive)
check("5.2 a half-written last line is dropped and the run continues",
      error5b is None and out5b.count("Dropping a half-written last line") == 2
      and all(json.loads(line) for line in (eval_dir(project) / "outputs_dpo_v2_private.jsonl").read_text(encoding="utf-8").splitlines()),
      repr(error5b))

# 6. Without the v1 models
drive6, project6, _ = make_drive(v1=False)
ns6, out6, error6 = run(drive6)
check("6.1 missing v1 models are skipped, and v2 is still evaluated",
      error6 is None and set(ns6["ACTIVE"]) == {"sft_v2", "dpo_v2"} and "Skipping sft_v1" in out6
      and ns6["V1_LEAK"]["checked"] is False and "not measured" in ns6["ANSWERS"]["Q2 Is v2 better than the old final model (DPO v1)?"],
      repr(error6))

# 6b. The adapter must be the one its training run recorded
from safetensors.torch import load_file  # noqa: E402

adapter_dir = project6 / "dpo_v2" / "runs" / "dpo_v2_beta02_sft05" / "adapter"
fingerprint = sum(t.float().abs().sum().item() for k, t in load_file(adapter_dir / "adapter_model.safetensors").items() if "lora_" in k)
(adapter_dir.parent / "run_summary.json").write_text(json.dumps({"lora_fingerprint": {"after": fingerprint * 1.01}}), encoding="utf-8")
_, _, error6b = run(drive6)
check("6.2 an adapter that is not the recorded one is refused",
      isinstance(error6b, AssertionError) and "not the one its training run recorded" in str(error6b), repr(error6b))
(adapter_dir.parent / "run_summary.json").write_text(json.dumps({"lora_fingerprint": {"after": fingerprint}}), encoding="utf-8")
ns6c, out6c, error6c = run(drive6)
check("6.3 the recorded adapter passes", error6c is None and ns6c["ADAPTER_IDENTITY"]["dpo_v2"] is True, repr(error6c))

# 7. Without the judge, then a judge that returns NaN
drive7, project7, _ = make_drive(v1=False)
ns7, out7, error7 = run(drive7, {"RUN_JUDGE = True": "RUN_JUDGE = False"})
check("7.1 with the judge off, the objective evidence still answers", error7 is None and ns7["JUDGE"] == {} and "Q1 Did DPO v2" in out7,
      repr(error7))
nan = {"p_first = torch.sigmoid(log_a - log_b)": "p_first = torch.full_like(log_a, float('nan'))"}
ns7b, _, error7b = run(drive7, nan)
judge_file = eval_dir(project7) / f"judge_{ns7b['JUDGE_TAG']}_private.jsonl"
saved7 = [json.loads(line) for line in judge_file.read_text(encoding="utf-8").splitlines()] if judge_file.exists() else []
check("7.2 NaN answers are saved as 'failed', never as a win, and stop the run",
      isinstance(error7b, AssertionError) and "NaN" in str(error7b)
      and any(r["winner"] == "failed" for r in saved7) and not any(r["winner"] in ("a", "b") for r in saved7), repr(error7b))
ns7c, _, error7c = run(drive7)
check("7.3 the next run judges the failed ones again",
      error7c is None and all(row["failed"] == 0 for row in ns7c["JUDGE"].values())
      and all(row["n"] == len(ns7c["GEN"]) for row in ns7c["JUDGE"].values()), repr(error7c))

# 8. An adapter that changes nothing is refused
drive8, _, _ = make_drive(v1=False, v2_adapter_changes=False)
_, _, error8 = run(drive8)
check("8.1 an adapter that changes nothing is refused", isinstance(error8, AssertionError) and "changes nothing" in str(error8),
      repr(error8))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
