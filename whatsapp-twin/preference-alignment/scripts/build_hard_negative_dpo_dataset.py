"""
build_hard_negative_dpo_dataset.py

Builds a DPO preference dataset using HARD negatives instead of AI-generated
ones: for each real (prompt, chosen) pair from train.jsonl, `rejected` is a
different REAL reply Ahmed actually sent -- just to a different conversation.

Why this is harder than an AI-generated rejected reply:
  - Same voice, same casual register, same person -- no stylistic shortcut
    (no emoji/bullet-point tell) for the model to lean on.
  - The model has to learn actual contextual judgment (is this reply
    relevant to *this* conversation?) rather than "avoid formal AI phrasing."

Version history (each fix found by manually reviewing the previous version's
real output on train.jsonl):
  v2 - A very short `chosen` reply could get paired against a multi-paragraph
       `rejected`, since the length fallback ignored length entirely once no
       close match was found. Fixed with progressively wider length-ratio
       bands, skipping the row if none work. Also filtered non-Arabic-script
       outliers (English/garbled messages) out of the candidate pool.
  v3 - Random sampling occasionally landed on a `rejected` that's topically
       close enough to `chosen` to read as ambiguous rather than clearly
       wrong. Added a TF-IDF cosine-similarity check to skip such candidates.
  v4 (this version) - TF-IDF only catches shared VOCABULARY, not shared
       MEANING. Testing v3 found pairs where `chosen` and `rejected` were
       two different reactions to the same situation -- genuinely ambiguous
       as a preference pair, but sharing literally zero words, so TF-IDF
       similarity came out as 0.0000 and didn't catch it. Replaced TF-IDF with a multilingual
       sentence-embedding model (meaning-based similarity, not word-overlap
       based), which does catch this. Trade-off: a one-time ~470MB model
       download via `sentence-transformers`, still CPU-only and fast
       (no GPU required), but a heavier dependency than v3's TF-IDF.

Usage:
    python build_hard_negative_dpo_dataset.py \
        --train-path train.jsonl \
        --output-path dpo_dataset_hard.jsonl \
        --model-name UBC-NLP/NileChat-3B \
        --num-samples 1500

Requires `transformers` (tokenizer only, no model weights for the target
model) and `sentence-transformers` (downloads its own small multilingual
embedding model, ~470MB, CPU-only) -- no GPU needed to run this script.
"""

import argparse
import json
import random
from getpass import getpass

MIN_CANDIDATE_LEN = 15              # chars -- filters out weak/universal negatives (اه/لا/تمام...)
MIN_ARABIC_CHAR_RATIO = 0.5         # candidates must be at least this Arabic-script by character count
LENGTH_RATIO_BANDS = [              # tried in order, widest last -- avoids an unconstrained fallback
    (0.4, 2.5),
    (0.25, 4.0),
    (0.15, 6.5),
]
MAX_ATTEMPTS_PER_BAND = 20

# Cosine similarity ceiling on MEANING (sentence-embedding based), not shared
# vocabulary. Typical multilingual-MiniLM cosine similarity for genuinely
# unrelated short sentences sits roughly in the 0.0-0.3 range; related/
# overlapping-topic sentences are usually 0.4+. 0.5 is a moderately strict
# default -- lower it (e.g. 0.4) for stricter filtering, raise it if too many
# rows get skipped for your dataset's phrasing style.
MAX_TOPIC_SIMILARITY = 0.5
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dedupe_by_last_turn(records):
    """Sliding-window data means the same exchange can appear many times as
    the window grows. Keep only one row per unique (last user msg, chosen
    reply) pair, so a single real exchange doesn't dominate the dataset."""
    seen = {}
    for r in records:
        messages = r["messages"]
        if len(messages) < 2:
            continue
        last_user = next((m["content"] for m in reversed(messages[:-1]) if m["role"] == "user"), None)
        chosen = messages[-1]["content"]
        if messages[-1]["role"] != "assistant" or last_user is None:
            continue
        key = (last_user.strip(), chosen.strip())
        seen[key] = r  # keep the most recently seen (longest context) version
    return list(seen.values())


def arabic_char_ratio(text: str) -> float:
    """Fraction of alphabetic-ish characters that fall in the Arabic Unicode
    block. Filters out candidates that are mostly English/garbled text --
    these are trivially distinguishable from a normal Arabic `chosen` reply
    by script alone, reintroducing a stylistic shortcut."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return arabic / len(letters)


def build_candidate_pool(records):
    """All (index, reply text) pairs eligible to be used as a rejected
    negative -- excludes short/generic replies and non-Arabic-script
    outliers."""
    pool = []
    dropped_short = 0
    dropped_non_arabic = 0
    for i, r in enumerate(records):
        text = r["messages"][-1]["content"].strip()
        if len(text) < MIN_CANDIDATE_LEN:
            dropped_short += 1
            continue
        if arabic_char_ratio(text) < MIN_ARABIC_CHAR_RATIO:
            dropped_non_arabic += 1
            continue
        pool.append((i, text))
    print(f"  Dropped {dropped_short} candidates for being too short (< {MIN_CANDIDATE_LEN} chars)")
    print(f"  Dropped {dropped_non_arabic} candidates for being mostly non-Arabic-script (garbled/English outliers)")
    return pool


def build_embedding_index(pool_texts, model):
    """Encodes the whole candidate pool once into normalized embeddings, so
    per-pair cosine similarity later is just a single dot product -- much
    cheaper than re-encoding on every comparison."""
    embeddings = model.encode(
        pool_texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def pick_hard_negative(chosen_text, chosen_embedding, pool, pool_embeddings, rng):
    """Tries progressively wider length-ratio bands, and within each band,
    rejects a candidate if it's too semantically similar to `chosen`
    (embedding cosine similarity above MAX_TOPIC_SIMILARITY) -- this catches
    cases like chosen='forget about him' / rejected='go finish the fight',
    which share no vocabulary but are clearly about the same kind of
    situation. Returns None if nothing reasonable found at any band."""
    chosen_len = len(chosen_text)

    for lo, hi in LENGTH_RATIO_BANDS:
        for _ in range(MAX_ATTEMPTS_PER_BAND):
            pool_idx = rng.randrange(len(pool))
            _, text = pool[pool_idx]
            if text.strip() == chosen_text.strip():
                continue
            ratio = len(text) / max(chosen_len, 1)
            if not (lo <= ratio <= hi):
                continue

            similarity = float(chosen_embedding @ pool_embeddings[pool_idx])
            if similarity > MAX_TOPIC_SIMILARITY:
                continue  # too semantically close -- try another candidate

            return text

    return None  # no reasonable match found even at the widest band -- skip this row


def main():
    global MAX_TOPIC_SIMILARITY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="train.jsonl")
    parser.add_argument("--output-path", default="dpo_dataset_hard.jsonl")
    parser.add_argument("--model-name", default="UBC-NLP/NileChat-3B",
                         help="Used only to load the tokenizer's chat template -- no model weights downloaded.")
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--max-similarity", type=float, default=MAX_TOPIC_SIMILARITY,
                         help="Embedding cosine-similarity ceiling for a candidate negative (lower = stricter).")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME,
                         help="sentence-transformers model used for semantic similarity filtering.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    MAX_TOPIC_SIMILARITY = args.max_similarity

    rng = random.Random(args.seed)

    print(f"Loading {args.train_path} ...")
    records = load_jsonl(args.train_path)
    print(f"  {len(records)} raw rows")

    records = dedupe_by_last_turn(records)
    print(f"  {len(records)} unique exchanges after sliding-window dedup")

    print("Building candidate pool ...")
    pool = build_candidate_pool(records)
    print(f"  {len(pool)} eligible negative candidates")

    if len(pool) < 50:
        raise ValueError("Candidate pool too small to build meaningful hard negatives.")

    print(f"Loading embedding model ({args.embedding_model}) -- one-time download, CPU-only ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embedding_model)

    print("Encoding candidate pool (this is the slowest one-time step) ...")
    pool_texts = [text for _, text in pool]
    pool_embeddings = build_embedding_index(pool_texts, embed_model)

    from huggingface_hub import login
    from transformers import AutoTokenizer

    login(token=getpass("Hugging Face token (needed once, to fetch the gated tokenizer): "))
    print(f"Loading tokenizer for {args.model_name} (chat template only, no model weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{args.model_name}'s tokenizer has no chat_template -- can't build matching prompts.")

    rng.shuffle(records)
    target_count = min(args.num_samples, len(records))

    output_rows = []
    skipped_no_negative = 0
    length_stats = []
    similarity_stats = []

    for r in records:
        if len(output_rows) >= target_count:
            break

        messages = r["messages"]
        prompt_messages = messages[:-1]
        chosen = messages[-1]["content"].strip()
        chosen_embedding = embed_model.encode(chosen, normalize_embeddings=True, convert_to_numpy=True)

        rejected = pick_hard_negative(chosen, chosen_embedding, pool, pool_embeddings, rng)
        if rejected is None:
            skipped_no_negative += 1
            continue

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        output_rows.append({
            "id": r.get("id"),
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
        })
        length_stats.append((len(chosen), len(rejected)))
        rejected_embedding = embed_model.encode(rejected, normalize_embeddings=True, convert_to_numpy=True)
        similarity_stats.append(float(chosen_embedding @ rejected_embedding))

    print()
    print(f"Built {len(output_rows)} hard-negative preference pairs")
    print(f"Skipped (no valid negative found, even at widest length band): {skipped_no_negative}")

    if length_stats:
        avg_chosen_len = sum(c for c, _ in length_stats) / len(length_stats)
        avg_rejected_len = sum(rj for _, rj in length_stats) / len(length_stats)
        max_ratio = max(max(c, rj) / max(min(c, rj), 1) for c, rj in length_stats)
        print(f"Avg chosen length: {avg_chosen_len:.1f} chars")
        print(f"Avg rejected length: {avg_rejected_len:.1f} chars")
        print(f"Worst-case length ratio in this dataset: {max_ratio:.1f}x (lower is better)")

    if similarity_stats:
        avg_sim = sum(similarity_stats) / len(similarity_stats)
        max_sim = max(similarity_stats)
        print(f"Avg chosen/rejected semantic similarity: {avg_sim:.3f}")
        print(f"Max semantic similarity in this dataset: {max_sim:.3f} (should be <= {MAX_TOPIC_SIMILARITY}; lower is better)")

    # Sanity checks -- same standard as before.
    empty = sum(1 for row in output_rows if not row["chosen"].strip() or not row["rejected"].strip())
    identical = sum(1 for row in output_rows if row["chosen"].strip() == row["rejected"].strip())
    print(f"Empty chosen/rejected: {empty}")
    print(f"Identical pairs: {identical}")
    assert empty == 0, "Found empty chosen/rejected -- fix before saving."
    assert identical == 0, "Found identical pairs -- fix before saving."

    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Saved to {args.output_path}")
    print()
    print("Sample pair:")
    print("PROMPT (tail):", output_rows[0]["prompt"][-200:])
    print("CHOSEN:", output_rows[0]["chosen"])
    print("REJECTED:", output_rows[0]["rejected"])


if __name__ == "__main__":
    main()
