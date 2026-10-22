"""Synthetic data and a small deterministic embedding, shared by the DPO v2 tests."""

import hashlib
import json
import random

import numpy as np

SYSTEM = "Continue the conversation as Ahmed. Synthetic test data."
WORDS = ["النهاردة", "بكرة", "الشغل", "البيت", "العربية", "الامتحان", "الفلوس", "المطعم", "الماتش", "السفر",
         "الدكتور", "الجامعة", "الموبايل", "القهوة", "الاجتماع", "المشروع", "[NAME]", "الصبح", "بالليل", "الساعة"]


def fake_embed(texts):
    """Bag of character trigrams hashed into 128 dimensions, L2-normalized: similar texts score high."""
    rows = []
    for text in texts:
        vec = np.zeros(128, dtype="float32")
        padded = f"  {text}  "
        for i in range(len(padded) - 2):
            vec[int(hashlib.md5(padded[i:i + 3].encode()).hexdigest(), 16) % 128] += 1
        rows.append(vec / max(np.linalg.norm(vec), 1e-9))
    return np.stack(rows)


def sentence(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 6)))


def make_session_records(rng, n_turns, tag):
    """Sliding windows exactly like the data builder: up to 5 previous turns, then one reply of Ahmed's."""
    turns = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append({"role": role, "content": f"{sentence(rng)} {tag}{i}"})
    records = []
    for index, target in enumerate(turns):
        if target["role"] != "assistant" or turns[index - 1]["role"] != "user":
            continue
        messages = [{"role": "system", "content": SYSTEM}, *turns[max(0, index - 5):index], target]
        record_id = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:20]
        records.append({"id": record_id, "messages": messages})
    return records


def write_data(directory, private_names_count=76):
    rng = random.Random(1)
    train = [r for s in range(30) for r in make_session_records(rng, rng.randint(6, 14), f"t{s}_")]
    validation = [r for s in range(40) for r in make_session_records(rng, rng.randint(6, 14), f"v{s}_")]
    for name, records in (("train", train), ("validation", validation)):
        with open(directory / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = {"output_validation": {"valid": True}, "config": {"private_names_count": private_names_count}}
    (directory / "audit_report.json").write_text(json.dumps(audit), encoding="utf-8")
    return train, validation
