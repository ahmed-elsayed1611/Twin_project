# Preference Alignment (DPO)

Direct Preference Optimization on top of the [supervised fine-tuning](../supervised-fine-tuning) SFT model. It teaches the model to prefer Ahmed's real reply over a real-but-wrong one, using **hard negatives** mined from his own messages.

Follows the preference alignment chapter of the [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook), applied to real personal chat data.

**Previous:** [Supervised fine-tuning](../supervised-fine-tuning) · **Next:** [Evaluation](../evaluation)

```
preference-alignment/
├── scripts/build_hard_negative_dpo_dataset.py   # builds train / dev / test preference pairs (local, CPU)
├── notebooks/dpo_training.ipynb                 # self-diagnosing DPO training (Colab, GPU)
├── requirements.txt
└── README.md
```

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
