import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import contextlib, io
OV = {"NUM_EVAL_SAMPLES": "20", "JUDGE_REQUESTS_PER_MINUTE": "100000",
      "GENERATION": "dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=4)"}
def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        r = fn(*a, **k)
    return r, buf.getvalue()
def idx(marker): return next(i for i, c in enumerate(CELLS) if marker in c)
def run_range(ns, root, start, end=None, ov=OV):
    for i in range(idx(start), idx(end) if end else len(CELLS)):
        src = prep(CELLS[i], root, ov)
        if src: exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns
def lines(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

class StatusError(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.status_code = code

root = tempfile.mkdtemp(); setup_drive(root)
ns, _ = quiet(run_range, {}, root, "from google.colab import drive", "if RUN_LLM_JUDGE:\n    from concurrent.futures")
FakeJudge.dpo_texts = set(ns["df"]["dpo"]); n_diff = int((~ns["df"]["identical"]).sum())

# H1. A 402 (payment) stops judging at once: no retry storm, nothing saved as judged/failed
orig_create = FakeJudge.create
def pay_wall(self, **kw):
    FakeJudge.calls += 1
    raise StatusError(402, "Error code: 402 - Payment required")
FakeJudge.create = pay_wall; FakeJudge.calls = 0
_, out = quiet(run_range, ns, root, "if RUN_LLM_JUDGE:\n    from concurrent.futures", "JUDGE_RESULT = None")
print(f"H1. 402 stops judging: {FakeJudge.calls} API calls (was 1350 before the fix) ->", FakeJudge.calls <= 3 * ns["JUDGE_WORKERS"])
print("H2. clear message shown:", "JUDGING STOPPED" in out and "402" in out)
saved = lines(ns["JUDGMENTS_PATH"])
print("H3. only identical (tie) pairs saved, nothing marked failed:", all(j["winner"] == "tie" for j in saved), f"({len(saved)} saved)")

# H4. after fixing the key/billing, re-running the cell judges everything that's left
FakeJudge.create = orig_create; FakeJudge.calls = 0
quiet(run_range, ns, root, "if RUN_LLM_JUDGE:\n    from concurrent.futures", "JUDGE_RESULT = None")
saved = {j["id"]: j for j in lines(ns["JUDGMENTS_PATH"])}
print("H4. re-run completes all 20, no failed:", len(saved) == 20 and not any(j["winner"] == "failed" for j in saved.values()),
      f"({FakeJudge.calls} calls = 2 x {n_diff} non-identical)")

# H5. switching judge model: no config error, no re-generation, separate verdict file
gen_before = (lines(ns["SFT_OUTPUTS_PATH"]), lines(ns["DPO_OUTPUTS_PATH"]))
ov2 = {**OV, "JUDGE_PROMPT_VERSION": "3"}
ns2, out2 = quiet(run_range, {}, root, "from google.colab import drive", "import pandas as pd", ov=ov2)
print("H5. switching judge settings keeps generations (no config error, nothing regenerated):",
      (lines(ns2["SFT_OUTPUTS_PATH"]), lines(ns2["DPO_OUTPUTS_PATH"])) == gen_before and "0 to go" in out2)
print("H6. new judge gets its own verdict file:", ns2["JUDGMENTS_PATH"] != ns["JUDGMENTS_PATH"])

# H7. a run started with the OLD notebook (judge keys inside eval_config.json) still resumes
cfg_path = ns["CONFIG_PATH"]; cfg = json.load(open(cfg_path)); cfg.update(judge_model="gpt-oss-120b", judge_prompt_version=2)
json.dump(cfg, open(cfg_path, "w"))
try:
    quiet(run_range, {}, root, "from google.colab import drive", "CEREBRAS_API_KEY = None" if any("CEREBRAS_API_KEY = None" in c for c in CELLS) else "JUDGE_API_KEY = None")
    print("H7. old-format eval_config.json (extra judge keys) resumes without error: True")
except RuntimeError as e:
    print("H7. old-format eval_config.json resumes without error: False ->", e)
