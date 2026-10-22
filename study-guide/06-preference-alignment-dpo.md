# Chapter 6 — Preference alignment (DPO), applied to the WhatsApp twin

Notes from studying the chapter and running DPO v2 on top of SFT v2 (October 2026).
Code: `whatsapp-twin/preference-alignment/` (notebook `notebooks/dpo_v2.ipynb`).

## The idea in one picture

```
prompt:   the conversation so far
chosen:   what Ahmed really replied        <- "say more like this"
rejected: Ahmed's reply from ANOTHER chat  <- "say less like this"
```

SFT teaches *how* to talk (imitation). DPO teaches *which* reply fits better (relevance), by comparing two replies to the same prompt.
**Hard negatives** (my own replies from other conversations) mean the style is the same on both sides, so the model can only win by being relevant.

## Key terms

| Term | Simple meaning |
|---|---|
| Reference model | SFT v2 frozen. With LoRA it is just the adapter switched off (`ref_model=None`) |
| Reward | `beta × (log-prob now − log-prob in the reference)` |
| DPO loss | Make reward(chosen) − reward(rejected) bigger |
| `beta` | Leash length: bigger = stay closer to SFT v2 |
| Likelihood displacement | DPO widens the gap by pushing *both* replies down, so real replies get less likely too |
| Anchor (RPO) | Add the normal SFT loss on `chosen` (weight 0.5) so real replies stay likely. TRL ≥ 1: `loss_type=["sigmoid","sft"], loss_weights=[1, 0.5]` (was `rpo_alpha`) |
| Preference accuracy | % of pairs where the model gives the real reply a higher *average* per-token log-prob than the wrong one (50% = coin flip) |
| McNemar's test | Count only pairs that flipped (fixed vs broken); if training did nothing they'd split about evenly |

## The notebook in 6 stops

1. **Setup (A):** Drive + GPU check; `UNSLOTH_RETURN_LOGITS=1` (scoring needs real logits); `PatchDPOTrainer()`; `CONFIG`; one folder per `RUN_NAME`.
2. **Checks (B, C):** base model really is SFT v2 (`loss_on = final_reply_only`); pairs' SHA-256 = SFT v2's data (so test is truly unseen); no test prompt in train; dev/test never share a conversation; no private names; resume / skip / refuse via `run_config.json`; at least 3 GB free on Drive.
3. **Prepare (D, E):** load SFT v2; end token = `<|im_end|>` (same ending SFT learned); fresh LoRA (starts at zero, so "before" = exactly SFT v2); LoRA fingerprint; format pairs with SFT v2's own chat template; drop (never cut) pairs over 1024 tokens.
4. **Baseline (F):** preference accuracy of SFT v2 on the test pairs (average, not total, log-prob: fair to long replies); greedy sample replies; saved to `baseline.json` and reused on resume.
5. **Train (G):** config with the anchor (verified it really arrived); check TRL's tokens (exactly one `<|im_end|>` per reply, prompt unchanged); step-0 eval; train with eval/save every 25 steps; verify the best checkpoint was restored and the LoRA changed.
6. **After (H, I):** same test again; McNemar; automatic diagnosis; save only the `default` adapter (not TRL's `ref` copy); `run_summary.json` = numbers only (shareable), `generations_private.json` = text (never share).

## My results

| | SFT v2 | DPO v2 |
|---|---|---|
| Preference accuracy, 1,616 unseen pairs | 66.4% | **70.5%** |
| Pairs fixed / broken | | 84 / 17 (p ≈ 8e-12) |
| Real reply log-prob | -40.22 | -41.22 (-2.5%, healthy) |
| Wrong reply log-prob | -45.59 | -48.90 (pushed down much more) |

30-message blind test: 13/30 identical; DPO better mostly when there is earlier context. Both still short/generic and sometimes invent facts (3B limit).

## Lessons from running it

- A library can silently change your settings (Unsloth hid `loss_type`): always assert the setting arrived.
- Trash counts against Drive storage; old checkpoints filled it.
- Resume works only because every run has a fingerprinted config and checkpoints on Drive.
- Free Colab has a daily GPU quota; a second account (owner of the shared folder) can run the same notebook.
- Never compare two models without a significance test on the same items.

## Interview answer

> "I aligned my SFT model with DPO on hard negatives — my own replies from other conversations — so it learns relevance, not style. I added an SFT anchor against likelihood displacement. On 1,616 unseen pairs, preference accuracy went from 66.4% to 70.5%, significant by McNemar (84 fixed vs 17 broken), while real-reply likelihood dropped only 2.5%."

## What's next
Chapter 7 evaluation (fixed yardstick), then reranking several candidates, data clean-up + SFT v3, DPO round 2 (lr 2e-5), RAG memory, a 7B model.
