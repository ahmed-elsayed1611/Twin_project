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

## Configuration

| Setting | Value |
|---|---|
| LoRA | r = 32, alpha = 32, dropout 0, all attention + MLP projections |
| Effective batch | 16 (2 per device × 8 accumulation) |
| Learning rate | 3e-4, linear decay, 10 warmup steps |
| Sequence length | 2048, packing on |
| Evaluation | every 50 steps on `validation.jsonl` |
| Precision | fp16 on T4 (bf16 where supported) |

## Notebook: `notebooks/sft_nilechat.ipynb`

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
