import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import io, contextlib  # noqa: F401 (kept as in the original harness test; it cannot run in this CI, so left unchanged)
root = tempfile.mkdtemp()
fp = setup_drive(root)
OV = {"NUM_EVAL_SAMPLES": "30", "JUDGE_REQUESTS_PER_MINUTE": "100000", "GENERATION": "dict(max_new_tokens=12, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=4)"}

# Make DPO texts available to the fake judge after generation: patch via a hook cell order -> run until judge, then set.
ns = run_all(root, overrides=OV, stop_before="if RUN_LLM_JUDGE:\n    from concurrent.futures")
FakeJudge.dpo_texts = set(ns["df"]["dpo"])
ident = int(ns["df"]["identical"].sum()); different = len(ns["df"]) - ident
# run remaining cells
started = False
for i, c in enumerate(CELLS):
    if "if RUN_LLM_JUDGE:\n    from concurrent.futures" in c: started = True
    if started:
        src = prep(c, root, OV)
        if src: exec(compile(src, f"<cell {i}>", "exec"), ns)

print("\n================ TEST ASSERTIONS ================")
df = ns["df"]
print("1. full pipeline ran, rows:", len(df), "| summary saved:", os.path.exists(ns["SUMMARY_PATH"]))
print("2. API calls == 2 x non-identical pairs:", FakeJudge.calls == 2 * different, f"({FakeJudge.calls} calls, {different} non-identical, {ident} identical)")
print("3. identical pairs are ties, never sent:", all(df.loc[df.identical, "judge"] == "tie"))
dev_prompts = {json.loads(l)["prompt"] for l in open(f"{root}/DPO/dpo_eval_hard.jsonl", encoding="utf-8")}
print("4. no DPO-dev conversation in sample:", not any(TOK.apply_chat_template(r["messages"][:-1]) in dev_prompts for r in ns["eval_records"]))
print("5. ids unique:", df["id"].is_unique)
s = json.load(open(ns["SUMMARY_PATH"], encoding="utf-8"))
print("6. summary verdict:", s["verdict"])
print("   judge:", s["llm_judge"]["counts"], "p=", round(s["llm_judge"]["p_value"], 4))
