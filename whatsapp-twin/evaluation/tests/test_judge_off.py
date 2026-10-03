import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import contextlib, io
root = tempfile.mkdtemp(); setup_drive(root)
OV = {"NUM_EVAL_SAMPLES": "20", "RUN_LLM_JUDGE": "False",
      "GENERATION": "dict(max_new_tokens=8, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=4)"}
FakeJudge.calls = 0
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    ns = run_all(root, overrides=OV)
out = buf.getvalue()
s = json.load(open(ns["SUMMARY_PATH"], encoding="utf-8"))
print("I1. judge off: full notebook runs, summary saved:", os.path.exists(ns["SUMMARY_PATH"]))
print("I2. no API calls and no key requested:", FakeJudge.calls == 0 and ns["JUDGE_API_KEY"] is None)
print("I3. judge reported as 'not run', verdict still produced:", s["evidence"]["LLM judge, both orders (E)"] == "not run" and bool(s["verdict"]))
print("I4. no judgments file written:", not any(f.startswith("judgments") for f in os.listdir(ns["EVAL_DIR"])))
