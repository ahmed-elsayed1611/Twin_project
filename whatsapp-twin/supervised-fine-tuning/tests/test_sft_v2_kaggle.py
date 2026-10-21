"""CPU tests for notebooks/sft_nilechat_v2_kaggle.ipynb and notebooks/sft_v2_merge_adapter.ipynb.

The Kaggle notebook shares its training code with the Colab one (tested in test_sft_v2.py); these tests cover
what differs: finding the private dataset, Kaggle Secrets, no names file, and the zip, then merging that zip
on "Colab".

    python tests/test_sft_v2_kaggle.py
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import KAGGLE_NOTEBOOK, MERGE_NOTEBOOK, new_drive, run_notebook, write_project  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


def kaggle_dirs(datasets=("ahmed-twin-redacted",), **project_kwargs):
    """A fake /kaggle/input with the given private datasets (no names file) and an empty /kaggle/working."""
    root = tempfile.mkdtemp(prefix="sft_v2_kaggle_")
    input_root, working = os.path.join(root, "input"), os.path.join(root, "working")
    os.makedirs(working)
    for name in datasets:
        write_project(root, names_file=False, data_dir=os.path.join(input_root, name), **project_kwargs)
    os.makedirs(input_root, exist_ok=True)
    return input_root, working


def run_kaggle(input_root, working, **kwargs):
    return run_notebook(new_drive(), notebook=KAGGLE_NOTEBOOK,
                        replacements={'"/kaggle/input"': repr(input_root), '"/kaggle/working"': repr(working)}, **kwargs)


# 1. A full Kaggle run
input_root, working = kaggle_dirs()
ns, out, error = run_kaggle(input_root, working)
zip_path = os.path.join(working, "sft_v2_best_adapter.zip")
check("1.1 the Kaggle run finishes without an error", error is None, repr(error))
check("1.2 the private dataset is found by its audit_report.json",
      ns.get("DATA_DIR") == os.path.join(input_root, "ahmed-twin-redacted"), ns.get("DATA_DIR"))
kaggle_code = "".join("".join(c["source"]) for c in json.load(open(KAGGLE_NOTEBOOK, encoding="utf-8"))["cells"]
                      if c["cell_type"] == "code")
check("1.3 the token comes from Kaggle Secrets (no typed prompt)",
      'get_secret("HF_TOKEN")' in kaggle_code and "getpass(" not in kaggle_code)
check("1.4 the name scan is skipped with an explanation, not silently",
      "Private-name scan skipped here: it ran on your PC before upload" in out)
check("1.5 only the final reply is trained (same code as Colab)", ns.get("demo_trained") == "ربع ساعة<|im_end|>")
names_in_zip = zipfile.ZipFile(zip_path).namelist() if os.path.exists(zip_path) else []
check("1.6 the zip holds the run summary and the best adapter",
      "run_summary.json" in names_in_zip and "best_adapter/adapter_config.json" in names_in_zip, names_in_zip[:5])
check("1.7 nothing is merged on Kaggle", not any("merged" in name for name in os.listdir(working)))
summary = json.loads(zipfile.ZipFile(zip_path).read("run_summary.json")) if names_in_zip else {}
check("1.8 the summary in the zip records the final-reply loss and the dropped sample",
      summary.get("config", {}).get("loss_on") == "final_reply_only"
      and summary.get("train_data", {}).get("dropped") == {"longer than max_seq_length": 1})

# 2. Datasets that must be refused
_, _, error2 = run_kaggle(*kaggle_dirs(datasets=("a", "b")))
check("2.1 two datasets with an audit_report.json are refused",
      isinstance(error2, AssertionError) and "found 2" in str(error2), repr(error2))
_, _, error3 = run_kaggle(*kaggle_dirs(datasets=()))
check("2.2 no dataset attached is refused with a hint",
      isinstance(error3, AssertionError) and "Add Input" in str(error3), repr(error3))
_, _, error4 = run_kaggle(*kaggle_dirs(private_names_count=0))
check("2.3 a build without names redaction is still refused, even without the names file",
      isinstance(error4, ValueError) and "WITHOUT --redact-names-file" in str(error4), repr(error4))

# 3. Merging the zip on Colab
drive = new_drive()
os.makedirs(os.path.join(drive, "ahmed_twin"))
merge_replacements = {'"/content/sft_v2_adapter"': repr(os.path.join(drive, "local_adapter")),
                      "MIN_MERGED_MODEL_GB = 3": "MIN_MERGED_MODEL_GB = 0"}
_, _, error5 = run_notebook(drive, notebook=MERGE_NOTEBOOK, replacements=merge_replacements)
check("3.1 a missing zip is refused with a hint",
      isinstance(error5, AssertionError) and "upload sft_v2_best_adapter.zip" in str(error5), repr(error5))

shutil.copy(zip_path, os.path.join(drive, "ahmed_twin", "sft_v2_best_adapter.zip"))
_, out6, error6 = run_notebook(drive, notebook=MERGE_NOTEBOOK, replacements=merge_replacements)
merged = os.path.join(drive, "ahmed_twin", "model_merged_nilechat_v2")
check("3.2 the merge finishes without an error", error6 is None, repr(error6))
check("3.3 the merged v2 model and its run summary are on Drive",
      os.path.exists(os.path.join(merged, "model.safetensors"))
      and os.path.exists(os.path.join(merged, "sft_run_summary.json")))
check("3.4 the merge reports the run it came from", "best eval loss" in out6 and "stopped at step" in out6)
check("3.5 the v1 model folder is never written", not os.path.exists(os.path.join(drive, "ahmed_twin", "model_merged_nilechat")))
_, _, error7 = run_notebook(drive, notebook=MERGE_NOTEBOOK, replacements=merge_replacements)
check("3.6 an existing merged model is not overwritten by default", isinstance(error7, FileExistsError), repr(error7))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
