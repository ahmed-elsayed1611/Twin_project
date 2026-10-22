"""Tests for scripts/build_dpo_pairs_v2.py. Synthetic conversations, and a small deterministic embedding in place
of the sentence-transformers model, so nothing is downloaded and no private data is needed.

    python tests/test_build_dpo_pairs_v2.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import build_dpo_pairs_v2 as builder  # noqa: E402
from dpo_test_data import WORDS, fake_embed, make_session_records, write_data  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not condition else ""))


# 1. Grouping conversations
rng = random.Random(0)
session_a, session_b = make_session_records(rng, 10, "a"), make_session_records(rng, 10, "b")
groups = builder.group_conversations(session_a + session_b)
check("1.1 two conversations in file order become two groups", len(groups) == 2 and len(groups[0]) == len(session_a),
      [len(g) for g in groups])
check("1.2 a conversation's overlapping windows stay in one group", len(builder.group_conversations(session_a)) == 1)

# 2. Small rules
check("2.1 placeholders do not count against an Arabic reply", builder.arabic_char_ratio("[NAME] قال كده") == 1.0)
check("2.2 an English reply is not Arabic", builder.arabic_char_ratio("ok see you tomorrow") == 0.0)
exchanges = builder.dedupe_exchanges(session_a + session_a)
check("2.3 a repeated exchange is kept once", len(exchanges) == len(session_a))

# 3. A full build
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    data_dir, out_dir = tmp / "redacted", tmp / "pairs"
    data_dir.mkdir()
    train, validation = write_data(data_dir)
    names_file = tmp / "private_names.txt"
    names_file.write_text("# synthetic\nزكريا\n", encoding="utf-8")
    manifest = builder.main(["--data-dir", str(data_dir), "--output-dir", str(out_dir), "--train-pairs", "60",
                             "--dev-pairs", "20", "--private-names", str(names_file)], embed=fake_embed)
    pairs = {n: builder.load_jsonl(out_dir / f"{n}.jsonl") for n in ("train", "dev", "test")}

    check("3.1 the three files and the manifest are written", all(pairs.values()) and (out_dir / "manifest.json").exists())
    check("3.2 the manifest fingerprints the exact input files",
          manifest["source"]["train_sha256"] == builder.sha256_of(data_dir / "train.jsonl")
          and manifest["source"]["validation_sha256"] == builder.sha256_of(data_dir / "validation.jsonl"))
    check("3.3 the requested sizes are respected", len(pairs["train"]) <= 60 and len(pairs["dev"]) <= 20,
          manifest["counts"])
    check("3.4 test takes the rest of validation", len(pairs["test"]) > len(pairs["dev"]), manifest["counts"])
    dev_groups, test_groups = {p["group"] for p in pairs["dev"]}, {p["group"] for p in pairs["test"]}
    check("3.5 no conversation is in both dev and test", not dev_groups & test_groups)
    val_texts = {m["content"] for r in validation for m in r["messages"][1:]}
    check("3.6 train pairs use only training conversations",
          not any(p["chosen"] in val_texts or p["rejected"] in val_texts for p in pairs["train"]))
    dev_replies = {p["chosen"] for p in pairs["dev"]}
    check("3.7 test negatives never come from dev replies", not any(p["rejected"] in dev_replies for p in pairs["test"]))
    check("3.8 chosen and rejected always differ", all(p["chosen"] != p["rejected"] for p in sum(pairs.values(), [])))
    check("3.9 similarity limits hold", all(p["similarity"] <= 0.5 and p["prompt_similarity"] <= 0.5
                                            for p in sum(pairs.values(), [])))
    check("3.10 prompts are messages ending before the reply",
          all(p["prompt_messages"][0]["role"] == "system" and p["prompt_messages"][-1]["role"] == "user"
              for p in sum(pairs.values(), [])))
    check("3.11 the private-name check ran", manifest["private_name_check"] == "0 pairs with a listed name")

    # 4. Data that must be refused
    names_file.write_text(WORDS[0] + "\n", encoding="utf-8")  # a word that is in every conversation
    try:
        builder.main(["--data-dir", str(data_dir), "--output-dir", str(tmp / "p2"), "--train-pairs", "30",
                      "--dev-pairs", "10", "--private-names", str(names_file)], embed=fake_embed)
        check("4.1 a listed name in the pairs stops the build", False)
    except ValueError as error:
        check("4.1 a listed name in the pairs stops the build", "private name" in str(error))

    write_data(data_dir, private_names_count=0)
    try:
        builder.main(["--data-dir", str(data_dir), "--output-dir", str(tmp / "p3")], embed=fake_embed)
        check("4.2 data built without the names file is refused", False)
    except ValueError as error:
        check("4.2 data built without the names file is refused", "WITHOUT --redact-names-file" in str(error))

    # 5. Different conversations that open with the very same prompt (found on the real data)
    write_data(data_dir)
    opening = [{"role": "system", "content": "Continue the conversation as Ahmed. Synthetic test data."},
               {"role": "user", "content": "صباح الخير يا جماعه كلكم"}]
    for name, reply in (("train", "صباح النور عاملين ايه النهارده"), ("validation", "صباح الفل كله تمام عندكم")):
        with open(data_dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
            for copy in range(3):  # several sessions, so the shared prompt is in both splits
                record = {"id": f"shared-{name}-{copy}", "messages": [*opening, {"role": "assistant", "content": f"{reply} {copy}"}]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = builder.main(["--data-dir", str(data_dir), "--output-dir", str(tmp / "p4"), "--train-pairs", "1000",
                             "--dev-pairs", "10"], embed=fake_embed)
    pairs = {n: builder.load_jsonl(tmp / "p4" / f"{n}.jsonl") for n in ("train", "dev", "test")}
    train_keys = {json.dumps(p["prompt_messages"], ensure_ascii=False) for p in pairs["train"]}
    check("5.1 a shared opening prompt no longer stops the build", "prompt_overlap_removed" in manifest)
    check("5.2 no dev or test prompt is also in train",
          not any(json.dumps(p["prompt_messages"], ensure_ascii=False) in train_keys for p in pairs["dev"] + pairs["test"]))
    check("5.3 the removed pairs are counted", sum(manifest["prompt_overlap_removed"].values()) >= 1, manifest["prompt_overlap_removed"])

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
