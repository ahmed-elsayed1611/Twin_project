"""Builds the DPO v2 preference pairs from the REDACTED dataset that SFT v2 was trained on.

For each of Ahmed's real replies, `chosen` is that reply and `rejected` is a real reply he sent in a different
conversation: same voice, wrong context (the hard negatives of v4, see build_hard_negative_dpo_dataset.py).

What changed from the v4 builder, and why:
  - Input is the redacted build (`--data-dir`), checked through its audit_report.json: data built without the
    private names file is refused. v4 read the unredacted files.
  - Pairs store the conversation as messages (`prompt_messages`), not a templated string. The training notebook
    applies the SFT v2 model's own chat template, so DPO prompts match SFT v2 exactly. No tokenizer and no
    Hugging Face login are needed here anymore.
  - Dev and test are split by CONVERSATION. validation.jsonl keeps each conversation's samples together and in
    order, and consecutive samples that share no message belong to different conversations. v4 drew dev and
    test from the same file with the same seed, so test repeated dev (500 -> 300 after overlap removal) and
    windows of one conversation could land in both.
  - Each split draws its negatives from its own replies, so a dev or test reply never appears in another split.
  - A rejected reply must also be unlike the last message it is paired against (`--max-prompt-similarity`):
    a reply that answers that message is a plausible reply, not a clear negative.
  - Placeholders such as [NAME] are ignored when measuring how Arabic a reply is.
  - Every output comes with manifest.json: SHA-256 of the input files (the notebook checks them against SFT v2's
    run summary), settings, counts and skip reasons. Nothing private is printed.

Usage (CPU, local):
    pip install -r requirements.txt
    python scripts/build_dpo_pairs_v2.py --data-dir ../data-preparation/outputs/redacted \
        --output-dir dpo_v2_pairs --private-names ~/private_names.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

BUILDER_VERSION = 2
MIN_CANDIDATE_LEN = 15          # characters: drops universal replies (اه / لا / تمام) that fit any conversation
MIN_ARABIC_CHAR_RATIO = 0.5     # of the letters, after removing placeholders
LENGTH_RATIO_BANDS = [(0.4, 2.5), (0.25, 4.0), (0.15, 6.5)]  # rejected/chosen length, tried narrowest first
MAX_ATTEMPTS_PER_BAND = 20
MAX_SIMILARITY = 0.5            # rejected vs chosen, by meaning: above this, both replies say the same thing
MAX_PROMPT_SIMILARITY = 0.5     # rejected vs the last message: above this, rejected may answer it well
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PLACEHOLDER_PATTERN = re.compile(r"\[(NAME|EMAIL|URL|PHONE|NUMBER|CREDENTIAL|REDACTED)\]")

Embed = Callable[[Sequence[str]], np.ndarray]  # texts -> L2-normalized rows


# ---------------------------------------------------------------- loading and grouping


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _message_keys(record: dict) -> set[tuple[str, str]]:
    return {(m["role"], m["content"]) for m in record["messages"] if m["role"] != "system"}


def group_conversations(records: Sequence[dict]) -> list[list[dict]]:
    """Splits records (in file order) into conversations: a new group starts where a record shares no message
    with the one before it. Sliding windows of one conversation always overlap, so text is never shared across
    groups, which is what makes a dev/test split by group leak-free."""
    groups: list[list[dict]] = []
    previous: set[tuple[str, str]] = set()
    for record in records:
        keys = _message_keys(record)
        if not groups or not (keys & previous):
            groups.append([])
        groups[-1].append(record)
        previous = keys
    return groups


def last_user_message(record: dict) -> Optional[str]:
    return next((m["content"] for m in reversed(record["messages"][:-1]) if m["role"] == "user"), None)


def dedupe_exchanges(records: Sequence[dict]) -> list[dict]:
    """One record per (last user message, reply): the same exchange repeats as the window grows. Keeps the last
    (longest-context) version."""
    kept: dict[tuple[str, str], dict] = {}
    for record in records:
        messages = record["messages"]
        user = last_user_message(record)
        if messages[-1]["role"] != "assistant" or user is None:
            continue
        kept[(user.strip(), messages[-1]["content"].strip())] = record
    return list(kept.values())


# ---------------------------------------------------------------- negatives


def arabic_char_ratio(text: str) -> float:
    letters = [c for c in PLACEHOLDER_PATTERN.sub(" ", text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters)


def build_pool(records: Sequence[dict]) -> tuple[list[str], Counter]:
    """Distinct replies that can serve as a negative."""
    pool, seen, dropped = [], set(), Counter()
    for record in records:
        text = record["messages"][-1]["content"].strip()
        if text in seen:
            continue
        seen.add(text)
        if len(text) < MIN_CANDIDATE_LEN:
            dropped["too_short"] += 1
        elif arabic_char_ratio(text) < MIN_ARABIC_CHAR_RATIO:
            dropped["not_arabic"] += 1
        else:
            pool.append(text)
    return pool, dropped


def pick_negative(chosen: str, chosen_vec: np.ndarray, prompt_vec: np.ndarray, pool: Sequence[str],
                  pool_vecs: np.ndarray, rng: random.Random, max_similarity: float,
                  max_prompt_similarity: float) -> tuple[Optional[dict], str]:
    """Narrowest length band first; within a band, random candidates until one is unlike both the chosen reply
    and the last message. Returns (negative, "") or (None, reason)."""
    reason = "no_candidate_in_length_bands"
    for low, high in LENGTH_RATIO_BANDS:
        for _ in range(MAX_ATTEMPTS_PER_BAND):
            index = rng.randrange(len(pool))
            text = pool[index]
            if text == chosen or not low <= len(text) / max(len(chosen), 1) <= high:
                continue
            similarity = float(chosen_vec @ pool_vecs[index])
            if similarity > max_similarity:
                reason = "candidates_too_similar_to_chosen"
                continue
            prompt_similarity = float(prompt_vec @ pool_vecs[index])
            if prompt_similarity > max_prompt_similarity:
                reason = "candidates_too_similar_to_last_message"
                continue
            return {"text": text, "similarity": similarity, "prompt_similarity": prompt_similarity}, ""
    return None, reason


def build_pairs(records: Sequence[dict], embed: Embed, rng: random.Random, limit: Optional[int],
                max_similarity: float, max_prompt_similarity: float, group_of: Optional[dict] = None
                ) -> tuple[list[dict], dict]:
    """Pairs for one split, with negatives drawn from that split's own replies."""
    exchanges = dedupe_exchanges(records)
    pool, pool_dropped = build_pool(exchanges)
    if len(pool) < 20:
        raise ValueError(f"Only {len(pool)} usable negatives -- too few to build pairs.")
    pool_vecs = embed(pool)
    order = list(range(len(exchanges)))
    rng.shuffle(order)
    if limit is not None:
        order = order[: limit * 3]  # enough candidates for the limit after skips, without embedding everything
    chosen_texts = [exchanges[i]["messages"][-1]["content"].strip() for i in order]
    prompt_texts = [last_user_message(exchanges[i]) for i in order]
    chosen_vecs, prompt_vecs = embed(chosen_texts), embed(prompt_texts)

    pairs, skipped = [], Counter()
    for position, index in enumerate(order):
        if limit is not None and len(pairs) >= limit:
            break
        record = exchanges[index]
        negative, reason = pick_negative(chosen_texts[position], chosen_vecs[position], prompt_vecs[position],
                                         pool, pool_vecs, rng, max_similarity, max_prompt_similarity)
        if negative is None:
            skipped[reason] += 1
            continue
        pairs.append({
            "id": record.get("id"),
            "group": group_of.get(record.get("id")) if group_of else None,
            "prompt_messages": record["messages"][:-1],
            "chosen": chosen_texts[position],
            "rejected": negative["text"],
            "similarity": round(negative["similarity"], 4),
            "prompt_similarity": round(negative["prompt_similarity"], 4),
        })

    stats = {
        "exchanges": len(exchanges),
        "negative_pool": len(pool),
        "pool_dropped": dict(pool_dropped),
        "pairs": len(pairs),
        "skipped": dict(skipped),
    }
    if pairs:
        stats.update({
            "mean_chars_chosen": round(float(np.mean([len(p["chosen"]) for p in pairs])), 1),
            "mean_chars_rejected": round(float(np.mean([len(p["rejected"]) for p in pairs])), 1),
            "mean_similarity": round(float(np.mean([p["similarity"] for p in pairs])), 3),
            "max_similarity": round(max(p["similarity"] for p in pairs), 3),
            "mean_prompt_similarity": round(float(np.mean([p["prompt_similarity"] for p in pairs])), 3),
        })
    return pairs, stats


def split_validation(records: Sequence[dict], dev_records_target: int, rng: random.Random
                     ) -> tuple[list[dict], list[dict], dict]:
    """Whole conversations go to dev until it holds `dev_records_target` records; the rest go to test."""
    groups = group_conversations(records)
    order = list(range(len(groups)))
    rng.shuffle(order)
    dev, test, group_of = [], [], {}
    for group_index in order:
        target = dev if len(dev) < dev_records_target else test
        for record in groups[group_index]:
            target.append(record)
            group_of[record.get("id")] = group_index
    return dev, test, {"groups": len(groups), "group_of": group_of}


# ---------------------------------------------------------------- privacy check


def count_records_with_names(pairs: Sequence[dict], names_file: Path, tools_dir: Path) -> int:
    sys.path.insert(0, str(tools_dir))
    from private_names_lib import compile_name_patterns, find_private_names, load_private_names

    patterns = compile_name_patterns(load_private_names(names_file))
    return sum(
        any(find_private_names(text, patterns) for text in
            [m["content"] for m in pair["prompt_messages"]] + [pair["chosen"], pair["rejected"]])
        for pair in pairs
    )


# ---------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None, embed: Optional[Embed] = None) -> dict:
    parser = argparse.ArgumentParser(description="Build the DPO v2 pairs from the redacted dataset.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Redacted build: train.jsonl, validation.jsonl, audit_report.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=1500, help="Same as v7, so training cost is comparable")
    parser.add_argument("--dev-pairs", type=int, default=200, help="Dev only picks the checkpoint; the rest of validation goes to test")
    parser.add_argument("--max-similarity", type=float, default=MAX_SIMILARITY)
    parser.add_argument("--max-prompt-similarity", type=float, default=MAX_PROMPT_SIMILARITY)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--private-names", type=Path, help="private_names.txt: fail if any listed name is in a pair")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_path, val_path, audit_path = (args.data_dir / n for n in ("train.jsonl", "validation.jsonl", "audit_report.json"))
    for path in (train_path, val_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("output_validation", {}).get("valid") is not True:
        raise ValueError("audit_report.json says the build did not pass its own validation.")
    if audit.get("config", {}).get("private_names_count", 0) < 1:
        raise ValueError("This data was built WITHOUT --redact-names-file. Use the redacted build.")

    if embed is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.embedding_model)

        def embed(texts: Sequence[str]) -> np.ndarray:
            return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype("float32")

    rng = random.Random(args.seed)
    train_records, val_records = load_jsonl(train_path), load_jsonl(val_path)
    dev_records, test_records, split_info = split_validation(val_records, dev_records_target=args.dev_pairs * 2, rng=rng)

    settings = {"max_similarity": args.max_similarity, "max_prompt_similarity": args.max_prompt_similarity}
    train_pairs, train_stats = build_pairs(train_records, embed, rng, args.train_pairs, **settings)
    dev_pairs, dev_stats = build_pairs(dev_records, embed, rng, args.dev_pairs, group_of=split_info["group_of"], **settings)
    test_pairs, test_stats = build_pairs(test_records, embed, rng, None, group_of=split_info["group_of"], **settings)

    dev_groups = {p["group"] for p in dev_pairs}
    test_groups = {p["group"] for p in test_pairs}
    assert not dev_groups & test_groups, "A conversation landed in both dev and test."

    # Different conversations can still open with the very same short prompt (e.g. only "صباح الخير"). A dev or
    # test pair whose exact prompt is also in train (or, for test, in dev) is not unseen, so it is removed.
    def key(pair):
        return json.dumps(pair["prompt_messages"], ensure_ascii=False)

    train_keys = {key(p) for p in train_pairs}
    dev_before, test_before = len(dev_pairs), len(test_pairs)
    dev_pairs = [p for p in dev_pairs if key(p) not in train_keys]
    seen_keys = train_keys | {key(p) for p in dev_pairs}
    test_pairs = [p for p in test_pairs if key(p) not in seen_keys]
    prompt_overlap_removed = {"dev": dev_before - len(dev_pairs), "test": test_before - len(test_pairs)}
    for pair in train_pairs + dev_pairs + test_pairs:
        assert pair["chosen"] and pair["rejected"] and pair["chosen"] != pair["rejected"]

    names_found = None
    if args.private_names:
        names_found = count_records_with_names(train_pairs + dev_pairs + test_pairs, args.private_names,
                                               Path(__file__).resolve().parents[2] / "tools")
        if names_found:
            raise ValueError(f"{names_found} pairs contain a private name. Rebuild the data with the names file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)):
        with open(args.output_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    manifest = {
        "builder_version": BUILDER_VERSION,
        "source": {
            "train_sha256": sha256_of(train_path),
            "validation_sha256": sha256_of(val_path),
            "redacted": True,
            "private_names_count": audit["config"]["private_names_count"],
        },
        "settings": {**settings, "train_pairs": args.train_pairs, "dev_pairs": args.dev_pairs, "seed": args.seed,
                     "embedding_model": args.embedding_model, "min_candidate_len": MIN_CANDIDATE_LEN,
                     "min_arabic_char_ratio": MIN_ARABIC_CHAR_RATIO, "length_ratio_bands": LENGTH_RATIO_BANDS},
        "split": {"validation_conversations": split_info["groups"], "dev_conversations": len(dev_groups),
                  "test_conversations": len(test_groups)},
        "counts": {"train": len(train_pairs), "dev": len(dev_pairs), "test": len(test_pairs)},
        "prompt_overlap_removed": prompt_overlap_removed,
        "stats": {"train": train_stats, "dev": dev_stats, "test": test_stats},
        "private_name_check": "not run" if names_found is None else "0 pairs with a listed name",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pairs -> train {len(train_pairs)} | dev {len(dev_pairs)} | test {len(test_pairs)} "
          f"(validation: {split_info['groups']} conversations, {len(dev_groups)} in dev, {len(test_groups)} in test)")
    print(f"  Removed because the same prompt is in train (or, for test, in dev): {prompt_overlap_removed}")
    for name, stats in (("train", train_stats), ("dev", dev_stats), ("test", test_stats)):
        print(f"  {name}: {json.dumps({k: v for k, v in stats.items()}, ensure_ascii=False)}")
    print("Private-name check:", manifest["private_name_check"])
    print("Saved to", args.output_dir)
    return manifest


if __name__ == "__main__":
    main()
