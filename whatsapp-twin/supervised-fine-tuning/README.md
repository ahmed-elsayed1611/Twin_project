# Supervised Fine-Tuning (SFT)

Fine-tunes [`UBC-NLP/NileChat-3B`](https://huggingface.co/UBC-NLP/NileChat-3B) on real Egyptian-Arabic WhatsApp conversations to create the first version of the Ahmed Twin chat model. Uses LoRA via [Unsloth](https://github.com/unslothai/unsloth) and `trl`'s `SFTTrainer`, on Google Colab (T4).

Follows the supervised fine-tuning chapter of the [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook). The book's article/repo instruction dataset is replaced by real personal chat data, prepared by the [dataset builder](../data-preparation) (multi-turn sliding context, session-based train/validation split, PII redaction).

**Previous:** [Data preparation](../data-preparation) · **Next:** [Preference alignment](../preference-alignment) · [Evaluation](../evaluation)

## Results

| Run | Base model | Best validation loss | Notes |
|---|---|---|---|
| 1 | Qwen2.5-3B-Instruct | 1.734 @ step ~1550 | Fallback while NileChat-3B access was pending |
| 2 | **NileChat-3B** | **1.625 @ steps 900–1200** | Lower loss in about half the steps (Egyptian-dialect pretraining pays off); selected |

Both runs overfit (validation loss reverses) around steps 1550–1650, well short of the planned 2 epochs / 3116 steps. Training was stopped at **checkpoint-1000**, and that checkpoint was merged into the standalone model used by the DPO and evaluation stages.

## What a later review found in the v1 run

Found on 15 Oct, by reading the notebook and auditing the data. They do not invalidate the results above, but they change what those numbers mean.

| Finding | Effect |
|---|---|
| **The loss counted every token.** `SFTTrainer` received the whole chat-template string (`dataset_text_field="text"`) with no response-only masking | The model was also trained to write the system message and Contact's messages. The 1.625 validation loss mixes "predict Ahmed" with "predict Contact" and the system message, which is identical in every sample and therefore easy to predict |
| **Real names were in the training data.** The data was built without name redaction: 24.1% of training samples contained a private name | Because of the first finding, the model learned to *write* the names in all of them, not only in Ahmed's replies (6.1%). The evaluation measured the result: a private name in 23 of 150 SFT replies, against 10 in Ahmed's real replies |
| **Overfitting started at the second pass over the data.** 24,915 samples ÷ 16 per step ≈ 1,558 steps per epoch, the same place validation loss turned up | Memorising the training set is also how exact private text can come back out. Stopping at checkpoint-1000 helped, but by hand |
| **Over-long samples were never checked.** Truncation at 2,048 tokens cuts the end of a sample, which is the reply | Not measured for v1; v2 counts and drops them |

## Version 2: `notebooks/sft_nilechat_v2.ipynb`

v1 is kept unchanged: it produced the model that DPO and the evaluation use today. v2 is the retraining notebook.

| v1 | v2 |
|---|---|
| Data with names | Only the redacted build. Refuses data whose `audit_report.json` shows no names file, and scans for names again (counts only, never prints a name) |
| Loss on every token | **Loss on the final reply only.** Labels are built by hand: `-100` on everything before Ahmed's final reply, so each reply is a target exactly once. Checked on a made-up chat and on every real sample |
| `SFTTrainer` | `transformers.Trainer` with the hand-built labels (an `SFTTrainer` would rebuild labels from the text) |
| Packing on | Packing off: packing would let one sample attend to another and mix their labels |
| Stopped by hand | Early stopping (4 evaluations without improvement) and `load_best_model_at_end` |
| `save_total_limit=2` could delete the best checkpoint | The best checkpoint is always kept, and saved again as `best_adapter` |
| Truncation not checked | Over-long samples dropped and counted; more than 1% dropped stops the notebook |
| No resume | Resumes from the last checkpoint, only if config and data fingerprints match `run_config.json` |
| Writes `model_merged_nilechat` | Writes `model_merged_nilechat_v2`; refuses to overwrite it unless asked |
| — | `run_summary.json`: config, data SHA-256, eval curve, dropped samples, library versions |

**Its validation loss is not comparable with v1's 1.625:** it averages over Ahmed's final replies only. Compare v1 and v2 with the evaluation notebook (likelihood of the real reply, judge, private-name rate).

**Running it:** build the data with `PRIVATE_NAMES_FILE` (see [data preparation](../data-preparation)), upload `train.jsonl`, `validation.jsonl` and `audit_report.json` to `MyDrive/ahmed_twin/redacted/`, then run top to bottom. After a disconnect, run top to bottom again. In a fresh session after training: Sections 1, 2, 3, then 9 and 10.

**Tests (CPU, no GPU or private data):** `tests/test_sft_v2.py` runs the notebook's real cells with a tiny random Qwen2 model, a real LoRA adapter, a real byte-level tokenizer with NileChat's chat template and the real `Trainer`, on synthetic chats. It covers a full run, re-running a finished run, a changed config, a simulated Colab disconnect and resume, and data that must be refused (a private name, no names file, an audit without name redaction).

```bash
pip install -r supervised-fine-tuning/tests/requirements-test.txt
python supervised-fine-tuning/tests/test_sft_v2.py
```

Not tested on CPU: Unsloth itself and the GPU. The first Colab run is the real test for those.

## Configuration

| Setting | Value |
|---|---|
| LoRA | r = 32, alpha = 32, dropout 0, all attention + MLP projections |
| Effective batch | 16 (2 per device × 8 accumulation) |
| Learning rate | 3e-4, linear decay, 10 warmup steps |
| Sequence length | 2048, packing on |
| Evaluation | every 50 steps on `validation.jsonl` |
| Precision | fp16 on T4 (bf16 where supported) |

## Notebook v1: `notebooks/sft_nilechat.ipynb` (produced the current model)

1. **Setup:** Drive, GPU check, dependencies, Hugging Face login (`getpass`, never stored)
2. **Load base model and tokenizer**
3. **Data:** format chats with the tokenizer's chat template; tokenizer sanity check
4. **LoRA adapters**
5. **Train:** `SFTTrainer`, optional old-run cleanup (off by default), disk check
6. **Inference check:** the project's decoding settings (explicit end-of-turn token, `temperature=0.4`)
7. **Save merged model** (only if training ran to completion)
8. **Recover checkpoint-1000:** merges the checkpoint this project actually used into `ahmed_twin/model_merged_nilechat`, works in a fresh session

### Fixes in this version
| Problem | Fix |
|---|---|
| The data cell used the tokenizer before the model was loaded (a top-to-bottom run crashed with `NameError`) | Model loads first; verified with `tools/check_notebook_order.py` |
| The "clear broken folders" cell deleted the whole checkpoints folder on every run, including checkpoint-1000 | Opt-in flag, off by default |
| Recovery saved to a different folder than the one the DPO and evaluation stages load | Both save paths now write `ahmed_twin/model_merged_nilechat` |
| Recovery merged through `peft.PeftModel`, the pattern that failed in the DPO stage | Loads the checkpoint folder directly with Unsloth, and frees GPU memory first |
| Inference check had no end-of-turn token (runaway generation) and broke on transformers 5.x | Explicit `eos_token_id`; prompt built as text, then tokenized |

## Running (Colab, GPU)
1. Put `train.jsonl` and `validation.jsonl` (from the [dataset builder](../data-preparation)) in `MyDrive/ahmed_twin/`.
2. Accept the NileChat-3B license on Hugging Face (it's a gated model) with the account whose token you'll paste.
3. Run top to bottom. To reproduce this project, interrupt training once validation loss bottoms out (around step 1000), then run Section 8.

## Known limitations
- `save_steps=500` + `save_total_limit=2` keeps only the 2 newest checkpoints, so an unattended run past step ~2000 deletes checkpoint-1000. This is kept as-is because it's the configuration that produced the model. The DPO stage fixes it with best-model tracking and early stopping.
- Technical-topic prompts are comparatively weak, which is likely real data scarcity in the conversations.
- `UBC-NLP/NileChat-3B` is licensed for **research only** (Qwen-Research). Review it before any other use of a derived model.

## Data and privacy
`train.jsonl` / `validation.jsonl` come from a real private conversation and are never committed (see `.gitignore`). Clear notebook outputs before committing, and run [`tools/check_private_names.py`](../tools) before pushing.
