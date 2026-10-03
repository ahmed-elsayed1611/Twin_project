"""CPU smoke tests for notebooks/evaluation.ipynb.

Executes the notebook's real code cells end to end with a tiny random Qwen2 model, a real
LoRA adapter (peft), synthetic conversations, and a fake judge API. Only GPU/Unsloth model
loading is mocked. No private data, GPU, or API key needed.

    pip install -r tests/requirements-test.txt
    python tests/run_all.py
"""
import os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
failed = False
for script in ["test_full.py", "test_failures.py", "test_judge_errors.py", "test_judge_off.py"]:
    print(f"\n######## {script}")
    out = subprocess.run([sys.executable, os.path.join(here, script)], capture_output=True, text=True)
    report = [l for l in out.stdout.splitlines() if re.match(r"^[1-9A-I][0-9]?\. ", l)]
    print("\n".join(report) if report else out.stdout[-2000:])
    bad = out.returncode != 0 or any(re.search(r"\bFalse\b|BAD|WRONG", l) for l in report)
    if bad:
        failed = True
        print(out.stderr[-3000:])
print("\nALL SMOKE TESTS PASSED" if not failed else "\nSOME TESTS FAILED")
sys.exit(1 if failed else 0)
