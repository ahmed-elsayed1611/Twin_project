import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import contextlib, io
OV = {"NUM_EVAL_SAMPLES": "20", "JUDGE_REQUESTS_PER_MINUTE": "100000",
      "GENERATION": "dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=4)"}
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)
def cell_idx(marker): return next(i for i, c in enumerate(CELLS) if marker in c)
def run_range(ns, root, start_marker, end_marker=None, ov=OV):
    s = cell_idx(start_marker); e = cell_idx(end_marker) if end_marker else len(CELLS)
    for i in range(s, e):
        src = prep(CELLS[i], root, ov)
        if src: exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns
def lines(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---- A. disconnect mid-SFT, then mid-DPO; fresh "runtime" each time ----
root = tempfile.mkdtemp(); setup_drive(root)
ns = quiet(run_range, {}, root, "from google.colab import drive", "assert not isinstance(model, PeftModel)")
real_gen = ns["generate_reply"]; calls = {"n": 0}
def crashing(prompt):
    calls["n"] += 1
    if calls["n"] == 8: raise ConnectionError("Colab disconnected")
    return real_gen(prompt)
ns["generate_reply"] = crashing
try: quiet(run_range, ns, root, "assert not isinstance(model, PeftModel)", "from safetensors.torch import load_file")
except ConnectionError: pass
print("A1. after crash, SFT rows saved:", len(lines(ns["SFT_OUTPUTS_PATH"])), "(expected 7)")
ns2 = quiet(run_range, {}, root, "from google.colab import drive", "from safetensors.torch import load_file")   # new runtime: A, B, C2
sft_ids = [r["id"] for r in lines(ns2["SFT_OUTPUTS_PATH"])]
print("A2. resumed SFT complete, no duplicates:", len(sft_ids) == 20 and len(set(sft_ids)) == 20)
print("A3. same frozen sample after restart:", [r["id"] for r in ns["eval_records"]] == [r["id"] for r in ns2["eval_records"]])
quiet(run_range, ns2, root, "from safetensors.torch import load_file", "import pandas as pd")              # C3 + C4
print("A4. DPO complete:", len(lines(ns2["DPO_OUTPUTS_PATH"])) == 20)
try:
    run_range(ns2, root, "assert not isinstance(model, PeftModel)", "from safetensors.torch import load_file")
    print("A5. re-running SFT with adapter attached: NOT blocked  <-- BAD")
except AssertionError as e:
    print("A5. re-running SFT with adapter attached is blocked:", str(e)[:60], "...")

# ---- B. settings changed mid-run without START_FRESH ----
try:
    quiet(run_range, {}, root, "from google.colab import drive", "JUDGE_API_KEY = None", ov={**OV, "NUM_EVAL_SAMPLES": "25"})
    print("B1. changed settings: NOT detected  <-- BAD")
except RuntimeError as e:
    print("B1. changed settings detected:", str(e)[:70], "...")
ns_b = quiet(run_range, {}, root, "from google.colab import drive", "JUDGE_API_KEY = None", ov={**OV, "NUM_EVAL_SAMPLES": "25", "START_FRESH": "True"})
print("B2. START_FRESH wipes old progress:", not os.path.exists(ns_b["SFT_OUTPUTS_PATH"]))

# ---- C. wrong adapter in the run folder ----
root_c = tempfile.mkdtemp(); setup_drive(root_c)
s = json.load(open(f"{root_c}/DPO/runs/dpo_v7_rpo/run_summary.json")); s["lora_fingerprint"]["after"] *= 1.5
json.dump(s, open(f"{root_c}/DPO/runs/dpo_v7_rpo/run_summary.json", "w"))
ns_c = quiet(run_range, {}, root_c, "from google.colab import drive", "from safetensors.torch import load_file")
try:
    quiet(run_range, ns_c, root_c, "from safetensors.torch import load_file", "assert isinstance(model, PeftModel)")
    print("C1. wrong adapter: NOT detected  <-- BAD")
except AssertionError as e:
    print("C1. wrong adapter detected:", str(e)[:70], "...")

# ---- D. judge API failing for some conversations, then retry ----
root_d = tempfile.mkdtemp(); setup_drive(root_d)
ns_d = quiet(run_range, {}, root_d, "from google.colab import drive", "if RUN_LLM_JUDGE:\n    from concurrent.futures")
FakeJudge.dpo_texts = set(ns_d["df"]["dpo"]); FakeJudge.mode = "prefers_dpo"
nonident = ns_d["df"][~ns_d["df"]["identical"]]["id"].tolist()
# make failures targetable: inject id marker into prompts via anonymize hook
orig_ctx = ns_d["judge_context"]
ns_d["judge_context"] = lambda rec: f"[[{rec['id']}]]\n" + orig_ctx(rec)
import time as _tm; _real_sleep = _tm.sleep; ns_d["time"].sleep = lambda s: None                                     # don't actually wait on backoff
FakeJudge.fail_ids = set(nonident[:3])
quiet(run_range, ns_d, root_d, "if RUN_LLM_JUDGE:\n    from concurrent.futures", "JUDGE_RESULT = None")
j = {r["id"]: r for r in lines(ns_d["JUDGMENTS_PATH"])}
print("D1. failing conversations marked failed:", sum(1 for i in nonident[:3] if j[i]["winner"] == "failed") == min(3, len(nonident)),
      "| errors logged:", len(ns_d["error_log"]) > 0)
FakeJudge.fail_ids = set(); FakeJudge.calls = 0
quiet(run_range, ns_d, root_d, "if RUN_LLM_JUDGE:\n    from concurrent.futures", "JUDGE_RESULT = None")
print("D2. retry only re-judges the failed ones:", FakeJudge.calls == 2 * min(3, len(nonident)), f"({FakeJudge.calls} calls)")
ns_d = quiet(run_range, ns_d, root_d, "JUDGE_RESULT = None")
counts = ns_d["JUDGE_RESULT"]["counts"]
print("D3. final counts cover each conversation once:", sum(counts.values()) == 20, counts)

# ---- E. parse_verdict and rate limiter ----
ns_d["time"].sleep = _real_sleep   # restore real sleep for the timing test
pv = ns_d["parse_verdict"]
cases = {"A": "A", " b ": "B", "TIE": "TIE", "**A**": "A", "B.": "B", "Candidate B": "B", "Answer: A": "A",
         "CANDIDATE B is closer": "B", "A or B": None, "": None, None: None, "I think TIE": "TIE", "ABBA": None, "tie.": "TIE"}
bad = {k: (pv(k), v) for k, v in cases.items() if pv(k) != v}
print("E1. verdict parsing:", "all", len(cases), "cases correct" if not bad else f"WRONG: {bad}")
import time as _t
RL = ns_d["RateLimiter"](120); t0 = _t.monotonic()
import threading as th
ts = [th.Thread(target=RL.wait) for _ in range(5)]; [t.start() for t in ts]; [t.join() for t in ts]
el = _t.monotonic() - t0
print(f"E2. rate limiter: 5 calls at 120/min took {el:.2f}s (expected ~2.0s):", 1.9 < el < 2.4)

# ---- G. private-name matching: whole words, comments ignored ----
ns_g = ns_d
print("G1. comment lines ignored:", all(not n.startswith("#") for n in ns_g["PRIVATE_NAMES"]), "| names:", len(ns_g["PRIVATE_NAMES"]))
mp, an = ns_g["mentions_private_name"], ns_g["anonymize"]
cases = {"فرح ماشي": False, "قولي ل رح": True, "ورح قالت": True, "فلانه جت": True, "Zed said": True, "وZed": True, "Zedd": False}
wrong = {t: mp(t) for t, e in cases.items() if mp(t) != e}
print("G2. whole-word detection:", "all", len(cases), "correct" if not wrong else f"WRONG {wrong}")
got = an("قولي ل رح وفلانه فرح وZed")
print("G3. anonymization keeps real words:", got == "قولي ل [NAME] و[NAME] فرح و[NAME]", "->", got)

# ---- F. context trimming keeps system + latest turns ----
long_msgs = [{"role": "system", "content": "SYS"}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": "كلام " * 150} for i in range(12)]
ns_d["MAX_CONTEXT_TOKENS"] = 1024
prompt, n = ns_d["fit_context"](long_msgs)
print("F1. long context trimmed:", n <= 1024, "| system kept:", prompt.startswith("<|im_start|>system\nSYS"),
      "| latest turn kept:", prompt.rstrip().endswith("<|im_start|>assistant"))
