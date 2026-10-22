"""CPU test for notebooks/dpo_v2_test_30_messages.ipynb, with the SFT tests' harness.

"SFT v2" is the harness's tiny random model and the "DPO adapter" a random LoRA on it, so the test checks the
mechanics: both models answer all 30 messages, the comparison is blind until the votes are in, the votes are
counted for the right model, an adapter that changes nothing is refused, and the filter is embedded unchanged.

    python tests/test_dpo_v2_test30.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "supervised-fine-tuning" / "tests"))
from harness import SYSTEM_MESSAGE, TOKENIZER, make_base, new_drive, run_notebook  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

NOTEBOOK = str(HERE.parent / "notebooks" / "dpo_v2_test_30_messages.ipynb")
FILTER_FILE = HERE.parents[1] / "tools" / "reply_filter.py"
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def make_drive(adapter_changes_model=True):
    drive = Path(new_drive())
    project = drive / "ahmed_twin"
    sft = project / "model_merged_nilechat_v2"
    make_base().save_pretrained(sft)
    TOKENIZER.save_pretrained(sft)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
                      init_lora_weights=not adapter_changes_model)  # True = B is zero = the adapter does nothing
    get_peft_model(make_base(), lora).save_pretrained(project / "dpo_v2" / "runs" / "dpo_v2_beta02_sft05" / "adapter")
    pairs = project / "dpo_v2" / "pairs"
    pairs.mkdir(parents=True)
    pair = {"prompt_messages": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": "x"}]}
    (pairs / "dev.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
    (project / "private_names.txt").write_text("# synthetic\nزكريا\n", encoding="utf-8")
    return str(drive), project / "dpo_v2" / "runs" / "dpo_v2_beta02_sft05" / "test_30_messages_private.json"


# 0. Static checks
cells = ["".join(c["source"]) for c in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
check("0.1 the filter cell is exactly tools/reply_filter.py", FILTER_FILE.read_text(encoding="utf-8").strip() in [c.strip() for c in cells])

# 1. A normal run, before voting
drive, results_path = make_drive()
ns, out, error = run_notebook(drive, notebook=NOTEBOOK)
check("1.1 the notebook runs end to end", error is None, repr(error))
cases = ns.get("CASES", [])
check("1.2 30 messages: 15 alone, 15 with earlier messages",
      [c["kind"] for c in cases].count("single") == 15 and [c["kind"] for c in cases].count("context") == 15)
check("1.3 every context alternates them/you and ends with them",
      all([m["role"] for m in c["messages"][1:]] == (["user", "assistant"] * 5)[: len(c["messages"]) - 1] for c in cases))
check("1.4 the system message is the one DPO was trained with", all(c["messages"][0]["content"] == SYSTEM_MESSAGE for c in cases))
check("1.5 both models answered all 30", all(len(ns.get("results", {}).get(m, [])) == 30 for m in ("sft_v2", "dpo_v2")))
blind = out.split("## blind end")[0] if "## blind end" in out else out[: out.find("Totals per model")]
check("1.6 the blind section shows A and B without model names",
      "   A: " in blind and "   B: " in blind and "sft_v2:" not in blind and "dpo_v2:" not in blind)
check("1.7 without votes, nothing is revealed",
      "Fill VOTES" in out and "You preferred" not in out and "   sft_v2: " not in out and "Model names hidden" in out)
check("1.8 the filtered replies never show a placeholder",
      not any(ns["PLACEHOLDER"].search(r["filtered"]) for m in ("sft_v2", "dpo_v2") for r in ns["results"][m]))
saved = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
check("1.9 the private results file is saved with all 30", len(saved.get("cases", [])) == 30 and saved.get("votes") is None)

# 2. Votes are counted for the right model
dpo_is_a = ns["DPO_IS_A"]
all_dpo = "".join("A" if a else "B" for a in dpo_is_a)
ns2, out2, error2 = run_notebook(drive, notebook=NOTEBOOK, replacements={'VOTES = ""': f"VOTES = {all_dpo!r}"})
check("2.1 voting for DPO every time gives DPO 30 wins", error2 is None and ns2.get("VOTE_RESULT", {}).get("dpo_v2") == 30,
      ns2.get("VOTE_RESULT"))
check("2.1b after voting, the names are shown", "   dpo_v2: " in out2 and "Model names hidden" not in out2)
mixed = "=" * 10 + "".join("B" if a else "A" for a in dpo_is_a[10:])
ns3, _, error3 = run_notebook(drive, notebook=NOTEBOOK, replacements={'VOTES = ""': f"VOTES = {mixed!r}"})
check("2.2 ties and SFT votes are counted", error3 is None and ns3.get("VOTE_RESULT", {}).get("ties") == 10
      and ns3["VOTE_RESULT"]["sft_v2"] == 20, ns3.get("VOTE_RESULT"))
saved3 = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
check("2.3 the votes are saved", saved3.get("votes", {}) and saved3["votes"].get("ties") == 10)
_, _, error4 = run_notebook(drive, notebook=NOTEBOOK, replacements={'VOTES = ""': 'VOTES = "AB"'})
check("2.4 the wrong number of votes is refused", isinstance(error4, AssertionError) and "votes for" in str(error4), repr(error4))

# 3. An adapter that changes nothing is refused
drive5, _ = make_drive(adapter_changes_model=False)
_, _, error5 = run_notebook(drive5, notebook=NOTEBOOK)
check("3.1 an adapter that changes nothing is refused", isinstance(error5, AssertionError) and "did not load" in str(error5), repr(error5))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
