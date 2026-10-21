# Preference Alignment (DPO)

Direct Preference Optimization on top of the [supervised fine-tuning](../supervised-fine-tuning) SFT model. It teaches the model to prefer Ahmed's real reply over a real-but-wrong one, using **hard negatives** mined from his own messages.

Follows the preference alignment chapter of the [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook), applied to real personal chat data.

**Previous:** [Supervised fine-tuning](../supervised-fine-tuning) · **Next:** [Evaluation](../evaluation)

```
preference-alignment/
├── scripts/build_hard_negative_dpo_dataset.py   # v1: train / dev / test preference pairs (local, CPU)
├── scripts/build_dpo_pairs_v2.py                # v2: pairs from the redacted data, split by conversation
├── notebooks/dpo_training.ipynb                 # v1: self-diagnosing DPO training on SFT v1 (run v7 selected)
├── notebooks/dpo_v2.ipynb                       # v2: DPO on top of SFT v2
├── tests/                                       # CPU tests for the v2 builder and notebook
├── requirements.txt
└── README.md
```

## Version 2: DPO on top of SFT v2

The v1 results below (v7) stay as they are: that model was trained on the unredacted data, on top of SFT v1. v2 is the DPO stage of the retrained pipeline: [SFT v2](../supervised-fine-tuning#version-2-notebookssft_nilechat_v2ipynb) (redacted data, loss on the final reply only) → DPO v2 → the evaluation.

### The pairs: `build_dpo_pairs_v2.py`

| Problem in v1 | v2 |
|---|---|
| Pairs built from the unredacted data | Reads the **redacted** build and refuses data built without the names file; `--private-names` re-scans every pair |
| Nothing tied the pairs to the SFT model | `manifest.json` records the SHA-256 of `train.jsonl` and `validation.jsonl`; the notebook requires them to equal **SFT v2's** run summary, so the test conversations are unseen by this exact model |
| Dev and test from the same file with the same seed: 500 test pairs, 300 after removing repeats of dev; windows of one conversation could land in both | Validation is split **by conversation** (consecutive samples sharing no message start a new one); dev ≈ 200 pairs, **test = every other validation conversation** |
| Dev/test negatives could come from the other split's replies | Each split draws negatives from its own replies |
| A rejected reply could still answer the last message | Also refused if too similar **to the last message** (`--max-prompt-similarity 0.5`) |
| `[NAME]` placeholders made Arabic replies look non-Arabic | Placeholders are ignored when measuring the Arabic ratio |
| Prompts pre-templated with the original tokenizer (needed a Hugging Face login) | Pairs store **messages**; the notebook applies SFT v2's own chat template, so prompts match SFT v2's training exactly. No tokenizer or login needed here |

### The notebook: `dpo_v2.ipynb`

It keeps what worked in v8 (step-0 baseline, separate dev/test, McNemar, LoRA fingerprints, verified best-checkpoint restore, automatic diagnosis) and fixes what a review against **today's library versions** found:

| Found | v2 |
|---|---|
| **TRL 1.15** (what `pip install unsloth` brings on Colab) removed `rpo_alpha` and `max_prompt_length`. v8 would quietly fall back to **plain DPO** (no anchor), then crash on `max_prompt_length` | The anchor is `loss_type=["sigmoid", "sft"]` with `loss_weights=[1, 0.5]` (the same loss as `rpo_alpha=0.5`), or `rpo_alpha` on older TRL. Plain DPO is **refused**, and any setting TRL does not accept stops the notebook instead of being dropped |
| TRL appends the tokenizer's EOS to each reply, which may not be `<\|im_end\|>`, the token SFT v2 learned to end replies with | The tokenizer's EOS is set to `<\|im_end\|>`, and the **tokenized** data is checked: every reply ends with exactly one `<\|im_end\|>`, prompts unchanged |
| TRL's default `keep_start` truncation cuts the end of a long pair, which is the reply | Pairs over `max_length` are dropped and counted before training |
| TRL ≥ 1 adds a frozen copy of the adapter (`ref`) as DPO's reference: fingerprints counted both adapters, and saves stored both | Fingerprints and saving use only the trained adapter (`default`). The `ref` copy of a fresh adapter is exactly SFT v2, so the reference is correct |
| Resumed from any checkpoint in the folder | Resumes only if SFT model, pairs and `CONFIG` match `run_config.json`; the SFT baseline is saved and reused |
| Head-to-head loaded any run's adapter, including v1 adapters trained on a different base model | Only runs on the same SFT model; v1 vs v2 belongs in the evaluation notebook |
| Real conversations and generations in `run_summary.json` | Totals in `run_summary.json`; real text in `generations_private.json`; names in generations counted, never printed |

Starting settings are v7's (`beta=0.2`, anchor 0.5, learning rate 1e-5, batch 1 × 16, evaluation every 25 steps, early stopping).

### Running v2

```bash
pip install -r requirements.txt
python scripts/build_dpo_pairs_v2.py --data-dir ../data-preparation/outputs/redacted \
    --output-dir dpo_v2_pairs --private-names ~/private_names.txt
```

Upload `train.jsonl`, `dev.jsonl`, `test.jsonl` and `manifest.json` from `dpo_v2_pairs/` to `MyDrive/ahmed_twin/dpo_v2/pairs/`, then run `notebooks/dpo_v2.ipynb` on a T4 (Run all; after a disconnect, Run all again). Results go to `MyDrive/ahmed_twin/dpo_v2/runs/<RUN_NAME>/`; v1's `MyDrive/DPO/` is never written.

### Tests (CPU, no GPU, no private data)

```bash
pip install -r ../supervised-fine-tuning/tests/requirements-test.txt trl==1.15.0
python tests/test_build_dpo_pairs_v2.py   # 21 checks: grouping, splits, leaks, refusals, shared openings
python tests/test_dpo_v2.py               # 22 checks: the notebook's real cells with the real TRL DPOTrainer
python tests/test_dpo_v2_test30.py        # 16 checks: the 30-message blind test notebook
```

`test_dpo_v2.py` runs a tiny random model with a real LoRA adapter through the real TRL 1.15 trainer, on synthetic pairs made by the real builder: a full run, a re-run, refused inputs (not SFT v2, mismatched data, a private name, a changed config, a TRL without an anchor), and a simulated Colab disconnect with resume. TRL 1.15 scores tokens with a Triton GPU kernel; the CPU test substitutes the same computation in plain PyTorch. Not covered on CPU: Unsloth itself and the GPU kernel, so the first Colab run is the real test for those.

### v2 result (run `dpo_v2_beta02_sft05`, T4, 2026-10-21)

Pairs: train 1,500 | dev 200 | test 1,629 (88 validation conversations: 11 dev, 77 test), 0 private names, fingerprints equal to SFT v2's `sft_run_summary.json`. Used after the length filter: 1,497 / 200 / 1,616.

| | SFT v2 | DPO v2 |
|---|---|---|
| Test preference accuracy (1,616 unseen pairs) | 66.4% | **70.5%** (+4.1 pts) |
| McNemar (pairs fixed / broken) | | 84 / 17, p ≈ 8e-12 |
| Real-reply log-prob (dev) | -40.22 | -41.22 (-2.5%: the anchor held) |
| Wrong-reply log-prob (dev) | -45.59 | -48.90 |
| Dev loss | 2.007 (step 0) | 1.867 (step 282) |
| Private names in generations | 0 | 0 |

Anchor: `loss_type=['sigmoid','sft'], loss_weights=[1.0, 0.5]`. All 282 steps ran (3 epochs, no early stop) and the best checkpoint was the last one: the model was still improving. A next run should try `learning_rate=2e-5` under a new `RUN_NAME`.

**30-message blind test** (`notebooks/dpo_v2_test_30_messages.ipynb`: 15 single messages, 15 with earlier context, same seed for both models, `tools/reply_filter.py` on both, A/B shown blind until the user votes). 13 of 30 replies identical; of the rest, a read-through favoured DPO about 8 to 2, mostly on messages with context (it uses what was said before). Both models stay short and generic and sometimes invent facts: a 3B limit, not a DPO one.

### What went wrong on the first Colab runs (and the fix)

| Problem | Cause | Fix |
|---|---|---|
| "TRL supports neither loss_type nor rpo_alpha" | Unsloth's `PatchDPOTrainer` replaces `DPOConfig.__init__` with one that hides `loss_type` behind `**kwargs` | Read accepted settings from the dataclass fields, then assert the anchor values really arrived (tests 4.2, 4.3) |
| "Only 1.50 GB free on Drive" | SFT v2's old checkpoints were in the Trash, which still counts | Empty the Trash (the notebook's space check stopped it safely) |
| Pairs not found | Uploaded to `dpo_v2/` instead of `dpo_v2/pairs/` | Moved |
| Colab disconnect at step ~55 | Session ended | Run all again: resumed from `checkpoint-50`, baseline reused |
| `peft` refused to load in the test notebook | Colab's old `torchao` (0.10) | `pip uninstall torchao` in the install cell |
| No GPU ("usage limits") | Free daily quota used by training | Ran the test with the second Google account, which owns the shared `ahmed_twin` folder |

### Next improvements (in order)
1. Chapter 7 evaluation (v1 vs SFT v2 vs DPO v2 on more messages, with a judge) as the fixed yardstick.
2. Generate several replies and keep the most relevant one (no training).
3. Data clean-up (merge consecutive messages, split conversations on time gaps) and SFT v3.
4. DPO round 2: higher learning rate, more pairs, the model's own bad replies as negatives.
5. RAG memory for facts; a 7B model with QLoRA.

## Final result

On **300 test pairs** that were never used for training or for choosing a checkpoint, measured as the model's own preference (does it give Ahmed's real reply a higher per-token likelihood than a wrong one?):

| Model | Test preference accuracy | vs. SFT (paired McNemar) |
|---|---|---|
| SFT (checkpoint-1000) | 68.7% | — |
| DPO v8 (`beta=0.1`, RPO 0.5) | 71.0% | +2.3 pts, p = 0.065 |
| **DPO v7 (`beta=0.2`, RPO 0.5) — selected** | **71.3%** | **+2.6 pts, p ≈ 0.02–0.04** |

v7 and v8 gave the same answer on 299 of 300 pairs, so halving `beta` changed nothing measurable. v7 was chosen by a rule fixed *before* the result: with no significant difference, keep the more conservative run. Read the result as **likely real but small**. The behavioural effect shows more clearly in [evaluation](../evaluation): an LLM judge prefers DPO 71% of the time, and DPO's replies are close to Ahmed's real length.

## Part 1 — The hard-negative dataset

**Why:** the first DPO run used AI-generated `rejected` replies (formal, emoji, bullet points). The model hit 100% preference accuracy by step 36 of 180, because it only had to learn "avoid AI style", not Ahmed's judgment.

**What the builder does:** for each real `(conversation, reply)`, `rejected` is a **different real reply Ahmed sent in another conversation**. Same voice, wrong context, so the style shortcut is gone. Each version fixed a problem found by reading real output:

| Version | Problem found | Fix |
|---|---|---|
| v2 | A one-word `chosen` could be paired with a paragraph-long `rejected` (length became the shortcut) | Progressively wider length-ratio bands; skip the row if none fit |
| v2 | A few candidates were mostly English or garbled (Arabic typed on an English keyboard) | Arabic-script ratio filter |
| v3 | Some negatives were close in topic to `chosen`, so the pair was ambiguous | TF-IDF similarity filter |
| v4 | TF-IDF can't see meaning: the ambiguous pairs shared **zero words** (similarity 0.0000) | Multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`); confirmed fixed on the exact flagged pairs |

On the real data: 24,915 rows → 24,907 unique exchanges → 17,970 candidates. 1,500 pairs built with average similarity 0.31, maximum ≤ 0.5, and chosen/rejected lengths 36.6 / 35.1 characters.

```bash
pip install -r requirements.txt
python scripts/build_hard_negative_dpo_dataset.py --train-path train.jsonl      --output-path dpo_dataset_hard.jsonl --num-samples 1500  # train
python scripts/build_hard_negative_dpo_dataset.py --train-path validation.jsonl --output-path dpo_eval_hard.jsonl    --num-samples 200   # dev
python scripts/build_hard_negative_dpo_dataset.py --train-path validation.jsonl --output-path dpo_test_hard.jsonl    --num-samples 500   # test
```
It needs the NileChat tokenizer (for the chat template) and a small embedding model, both on CPU. You'll be asked for a Hugging Face token once.

## Part 2 — The training notebook

The notebook is **self-diagnosing**: each run measures itself against a clean baseline, keeps its own artifacts, and ends with a diagnosis that names the setting to change next.

| Set | Source | Used for | Never used for |
|---|---|---|---|
| train | `train.jsonl` | learning | measuring |
| dev | `validation.jsonl` | early stopping / best checkpoint | reporting |
| test | `validation.jsonl` minus anything in train/dev | before vs. after, run vs. run | any training decision |

### How the setup evolved (every change came from a real run)
| Run | What the metrics showed | What changed |
|---|---|---|
| AI-generated negatives | 100% accuracy by step 36; generations unchanged | Hard negatives (Part 1) |
| Hard negatives, `beta=0.5` | Healthy accuracy (85–91%), **but** real replies became *less* likely (`logps/chosen` -8.7%) | SFT anchor on `chosen` (`rpo_alpha=0.5`), `beta=0.2`, held-out eval, early stopping, automatic diagnosis |
| v7 (`beta=0.2`) | Anchor worked (-2.2%); +2.7 pts on 150 pairs, within noise | Significance test, all eval pairs scored, `beta=0.1` |
| Design review | The same pairs chose the checkpoint *and* reported the result (optimistic bias) | Separate dev/test sets, head-to-head between runs, verified best-checkpoint restore, step-0 baseline |
| v8 (`beta=0.1`) | Same as v7 on 299/300 test pairs | Experiments concluded; v7 selected |

### Built-in safeguards
| Check | Guards against |
|---|---|
| One folder per `RUN_NAME`, overwrite guard | Losing a finished experiment |
| Disk-space check, file-size check on every save | Silent save failures on a full Drive |
| LoRA weight fingerprint before/after | "Successful" runs that changed nothing |
| Best checkpoint verified against the file on disk | A restore that silently didn't happen |
| Dev/test overlap removal | Inflated held-out numbers |
| Prompt-format normalization | Training and evaluation seeing different formats |
| Tokenizer compared with the original NileChat-3B | The load-time regex warning hiding a real tokenization change |
| EOS check on the tokenized data | Missing or doubled end-of-turn tokens across TRL versions |
| Auto-resume from the latest checkpoint | Losing progress to a Colab disconnect |
| Paired McNemar tests | Mistaking noise for improvement |

### Running (Colab, GPU)
1. Upload the three `*.jsonl` files to `MyDrive/DPO/`; the merged SFT model must exist at `ahmed_twin/model_merged_nilechat` ([supervised fine-tuning](../supervised-fine-tuning), Section 8).
2. Set a new `RUN_NAME` and run top to bottom.
3. Read the diagnosis (F2) and the head-to-head (G3).

## Data and privacy
Preference files are derived from private conversations and are never committed. Generated replies can contain real names, so the model and its outputs stay private too. Clear notebook outputs before committing, and run [`tools/check_private_names.py`](../tools) before pushing.

`UBC-NLP/NileChat-3B` is licensed for **research only** (Qwen-Research).
