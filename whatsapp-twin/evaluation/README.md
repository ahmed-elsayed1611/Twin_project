# Evaluation — SFT vs. DPO

Evaluates the final models of the Ahmed Twin project, the **SFT** model ([supervised fine-tuning](../supervised-fine-tuning)) and the **DPO** model `dpo_v7_rpo` ([preference alignment](../preference-alignment)), on real WhatsApp conversations that neither model was trained on.

**Previous:** [Supervised fine-tuning](../supervised-fine-tuning) · [Preference alignment](../preference-alignment)

Follows the evaluation chapter of the [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook), adapted to a personal Egyptian-Arabic chat model.

## Version 2: `notebooks/evaluation_v2.ipynb`

Evaluates the v2 pipeline (redacted data, SFT on the final reply only, DPO v2) against v1, and answers four questions: did DPO v2 improve on SFT v2; is v2 better than the old final model (`dpo_v7_rpo`); is it safe to show; how far is it from a real person. The v1 notebook below and its results are unchanged.

| | v1 notebook | v2 notebook |
|---|---|---|
| Systems | SFT v1, DPO v7 | SFT v1, DPO v1 (v7), SFT v2, DPO v2; v1 optional |
| Test conversations | `validation.jsonl`, minus DPO dev | DPO v2's **test** split (unseen by every v2 stage), **minus anything found in v1's `train.jsonl`** (a name-, placeholder- and digit-insensitive match) |
| Fairness to v1 | — | v1 is compared only on conversations without `[NAME]`-style placeholders, which it never saw |
| Decoding | Greedy | The deployed one (temperature 0.4 + `tools/reply_filter.py`), **same seed per conversation for every system** |
| Judge | `gpt-oss-120b` over an API (anonymized excerpts leave Colab) | **Local** Qwen3-14B (4-bit) on the Colab GPU: **nothing leaves the runtime** |
| How the judge answers | Writes A / B / TIE, parsed | Its probabilities for A and B are read (one forward pass, no parsing); both orders, a win needs both |
| Judge validity | Swap consistency only | **Calibration gate**: it must prefer the real reply over a hard negative in ≥ 65% of conversations, or its verdicts are reported but not used; optional blind human check measures judge-human agreement |
| Judge questions | Closer to the real reply | **sense** (better next message, no reference) and **like** (closer to the real reply) |
| Statistics | Sign tests | Sign / McNemar tests, Wilson intervals, **Holm correction** per question |
| Outputs | `eval_summary.json` with real text | `eval_summary.json` **numbers only** (checked), real text only in `*_private*` files |

How it was designed: three rounds, each fixing the previous one's weaknesses (the table is in the notebook's introduction). CPU test: `python tests/test_eval_v2.py` (19 checks: v1 leakage with redacted names, all systems complete, fair subset, calibration gate, resume without regenerating, refused setting changes, missing v1, judge off, a dead adapter, the votes).

## Evaluation design

Three independent kinds of evidence, all **paired** (both models answer the same conversations) and all with **significance tests**:

| Evidence | What it measures | Why it's trustworthy |
|---|---|---|
| **Likelihood of the real reply** (primary) | Which model assigns higher probability to what Ahmed *actually* wrote | Objective: no judge, no prompt, no opinion |
| **LLM-as-judge** (primary, optional) | Which reply sounds more like the real person, in context | Each pair judged in **both A/B orders**; a win needs both to agree, so position bias cancels |
| **Automatic checks** (supporting) | Length, rambling, generic-assistant phrases, wrong dialect, private-name leaks, character overlap with the real reply | Cheap, transparent, compared against Ahmed's real replies as the baseline |

The verdict uses only the two primary metrics, and says "no significant difference" whenever p ≥ 0.05.

### Data hygiene
- Conversations come from `validation.jsonl`, never used in SFT or DPO training.
- Sliding-window duplicates are removed.
- Conversations used to **choose** the DPO checkpoint (the DPO dev set) are excluded.
- The sample is frozen to disk, so a resumed run evaluates exactly the same conversations.

### Reliability features
| Feature | Guards against |
|---|---|
| SFT replies generated first, then the adapter is attached to the same model | Two model copies on a T4, and adapter on/off toggling errors |
| Adapter verified three ways: matches the file, matches the fingerprint the DPO stage recorded, and is actually active | Evaluating the wrong adapter (the previous version silently evaluated an outdated one) |
| Resume keyed by a unique conversation id | Different conversations colliding on a shared last message ("تمام") |
| Config fingerprint per eval run | Silently mixing results from different settings |
| Identical SFT/DPO replies count as ties and are never judged | Coin-flip noise from judging two identical texts |
| Strict verdict parsing (exactly one of A / B / TIE) | `"CANDIDATE B"` being read as "A" |
| Shared rate limiter; temporary errors retried; **permanent errors (bad key, payment, unknown model) stop at once** | Silent rate-limit failures, and retry storms (a 402 was once retried 1,350 times) |
| Judge settings kept out of the generation fingerprint; one verdict file per judge | Switching or fixing the judge forcing a full re-generation |
| Report rebuilt from saved files, never from memory | Wrong totals after a resume |

## Folder structure

```
evaluation/
├── notebooks/evaluation.ipynb      # the evaluation (Colab, GPU)
├── tests/                          # CPU smoke tests: run the notebook's real cells end to end
│   ├── run_all.py
│   ├── harness.py                  # tiny model, real LoRA adapter, synthetic chats, fake judge
│   ├── test_full.py                # full pipeline
│   ├── test_failures.py            # disconnects, changed settings, wrong adapter, API errors, name matching
│   ├── test_judge_errors.py        # permanent API errors stop at once; switching judges never regenerates
│   ├── test_judge_off.py           # the whole notebook with RUN_LLM_JUDGE = False
│   └── requirements-test.txt
├── requirements.txt
└── README.md
```
The privacy tools (`check_private_names.py`, `private_names_lib.py`) are shared by all chapters and live in [`tools/`](../tools) at the repo root.

## Running the evaluation (Colab)

1. Make sure these exist on Drive: the SFT model, `DPO/runs/dpo_v7_rpo/adapter` (with its `run_summary.json`), and `ahmed_twin/validation.jsonl`.
2. Optional: create `ahmed_twin/private_names.txt` (real names, one per line) for leak detection and judge anonymization. **Never commit this file.**
3. Judge: Cerebras (`gpt-oss-120b`) is the default; switching to Groq or any OpenAI-compatible provider is one line in A2. Add the key to Colab's 🔑 Secrets (`CEREBRAS_API_KEY`), or you'll be prompted. To skip the judge, set `RUN_LLM_JUDGE = False`.
4. Open `notebooks/evaluation.ipynb` with a GPU runtime and run top to bottom (about 30–40 minutes for 150 conversations, including the judge).
5. After a disconnect, re-run from the top: generation and judging continue where they stopped.

Results are saved to `DPO/eval_runs/<EVAL_RUN_NAME>/` on Drive (`eval_summary.json`, `eval_table.csv`). They contain real messages, so they stay on Drive.

## Testing the evaluation code (no GPU, no private data)

```bash
pip install -r tests/requirements-test.txt
python tests/run_all.py
```

This executes the notebook's actual code cells on CPU with a tiny random model, a real LoRA adapter, synthetic conversations, and a fake judge. Only GPU/Unsloth model loading is mocked. It covers the full pipeline plus disconnect-and-resume, changed settings, a wrong adapter, judge API failures and retries, verdict parsing, rate limiting, and context trimming.

## Privacy

### Private names (applies to the whole project)
`private_names.txt` lists the real people in the WhatsApp data (one name or spelling per line, `#` for comments). It lives **only** at `MyDrive/ahmed_twin/private_names.txt` and on your machine, never in any repo. It's used to:
- **detect leaks** in generated replies (evaluation notebook, part D4)
- **anonymize** anything sent to an external API (evaluation notebook, part E)
- **scan repos before pushing**, so no name ends up on GitHub:
  ```bash
  python tools/check_private_names.py --names ~/private_names.txt .
  ```
  Run it from the repo root. Exit code 1 means a name was found, and the scan includes notebook outputs. To run it automatically before every push, put this in `.git/hooks/pre-push` of each repo and `chmod +x` it:
  ```bash
  #!/bin/sh
  python tools/check_private_names.py --names ~/private_names.txt .
  ```

Matching is **whole-word**, with the attached prefixes و / ب / ل, so a short name never matches inside a longer everyday word (ف and ك aren't treated as prefixes, because ف/ك + a short name often spells a real word). Everyday words that are also names (منه، لينا، كريم، جمال، امل…) are left out on purpose to avoid constant false alarms.

### Other notes

- The LLM judge sends conversation excerpts to a third-party API (Cerebras by default). It's opt-in, and names listed in `private_names.txt` are replaced with `[NAME]`, but other personal details can still be sent.
- Notebook outputs contain real messages: **Edit → Clear all outputs before committing**, or run `nbstripout --install` once.
- An earlier version of this notebook contained a hardcoded API key. If you used that key, revoke it.

## Results

150 held-out conversations from `validation.jsonl` (DPO dev conversations excluded). SFT = NileChat-3B checkpoint-1000; DPO = `dpo_v7_rpo`. Judge: `gpt-oss-120b`, each pair judged in both A/B orders.

| Evidence | SFT | DPO | Result |
|---|---|---|---|
| **LLM judge** (decided pairs) | 23 | **56** | **DPO preferred 71%** (95% CI 60–80%), p = 0.0003. 68 ties, 3 not judged (API quota) |
| **Likelihood of the real reply** (NLL / token, lower is better) | **2.700** | 2.739 | **SFT better**, p < 0.0001. Small effect: perplexity 14.9 → 15.5 (+4%) |
| Median reply length (real replies: 19.5 chars) | 48 | **29** | DPO is much closer to real length |
| Average reply length (real: 37 chars) | 98 | **55** | |
| Replies that ramble to the token limit | 13 | **7** | |
| Replies mentioning a private name (real replies: 10) | 23 | **11** | DPO matches the real rate; SFT more than doubles it |
| Character overlap with the real reply | 39 closer | 32 closer | No significant difference (p = 0.48) |
| Identical SFT/DPO replies | | | 15 / 150 |

Judge consistency under A/B swap: 104/135 (77%).

### Interpretation
The two primary metrics disagree, because they measure different things:

- **Likelihood** asks: how probable are Ahmed's *exact words* under the model? DPO slightly lowers this. That's a known DPO effect: it widens the gap between real and wrong replies mostly by pushing wrong replies down, and real replies drop a little too. It also showed during training (`logps/chosen` about -2% even with the RPO anchor).
- **The judge and the behavioral checks** ask: does the model *behave* like Ahmed? On that, DPO is clearly better: preferred about 2.4 : 1, replies close to his real length, less rambling, and private names mentioned at his real rate instead of twice as often.

For a "digital twin" the goal is behaving like the person, not reproducing exact wording, so **DPO (`dpo_v7_rpo`) is the final model**. The trade-off is a small (+4% perplexity) loss in exact-wording likelihood, reported here rather than hidden.

### Caveats
- One LLM judge. Its 77% swap-consistency means about a quarter of non-identical pairs were effectively guesses; those count as ties, never as wins.
- Part of the judge's preference may come from DPO's shorter replies. That's legitimate here, since the real replies are short, but it means "closer length" and "better reply" aren't fully separated.
- 150 conversations from a single contact. The results describe this data, not chat in general.

## License

See the repository's root [`LICENSE`](../../LICENSE). `UBC-NLP/NileChat-3B` is licensed for **research only** (Qwen-Research).
