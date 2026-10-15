# Data Preparation — Raw WhatsApp to Fine-Tuning Dataset

This is a **complete, standalone data-preparation project**. It starts with a raw WhatsApp `.txt` export and produces validated local training files. It does **not** download a model, require a GPU, use ZenML, or require Poetry.

The project is deliberately limited to the data stage. Its job is to produce a reliable dataset for the fine-tuning stage in [Supervised fine-tuning](../supervised-fine-tuning).

**Next:** [Supervised fine-tuning](../supervised-fine-tuning) · [Preference alignment](../preference-alignment) · [Evaluation](../evaluation)

## What the Project Does

The pipeline reads the raw export, parses Arabic- and English-locale message headers, merges multiline text and short same-sender bursts, anonymizes every non-Ahmed sender as `Contact`, reduces common PII, creates multi-turn Contact-to-Ahmed training examples, creates session-level train/validation splits, and validates its own outputs before completing.

> Ahmed is represented as a person with a natural conversational style. The project does not impose an “engineer,” “mentor,” or “assistant” identity on the training data.

## Project Structure

```text
data-preparation/
├── DATA_CONTRACT.md                  # Rules, privacy boundary, and output contract
├── README.md                         # This guide
├── src/
│   └── twin_dataset_builder.py       # Complete parser, cleaner, splitter, validator, and CLI
├── scripts/
│   ├── run_from_raw.sh               # One command for tests + dataset build + audit
│   └── audit_outputs.py              # Post-build quality audit of the generated files
├── tests/
│   └── test_twin_dataset_builder.py  # Dependency-free unit tests
├── data/
│   └── raw/                          # Optional private location for your chat export
└── outputs/                          # Created training artifacts; ignored by Git
```

## Requirements

| Requirement | Why it is needed | Check command |
|---|---|---|
| Linux / WSL / macOS | The included launcher is a standard shell script. | `uname -s` |
| Python 3.10+ | The parser and test suite use only the Python standard library. | `python3 --version` |
| Your exported WhatsApp `.txt` file | The private raw input. | `ls -lh /path/to/whatsapp_chat.txt.txt` |

No `pip install`, Poetry environment, database, model, API key, or internet connection is required for this pipeline.

## Exact Local Steps

### 1. Go to the project folder

```bash
cd whatsapp-twin/data-preparation
```

### 2. Run the complete process with one command

Use an **absolute** path to your raw WhatsApp export.

```bash
./scripts/run_from_raw.sh /absolute/path/to/whatsapp_chat.txt.txt
```

For example, if the chat export is in your Windows Downloads folder:

```bash
./scripts/run_from_raw.sh /mnt/c/Users/<WindowsUser>/Downloads/whatsapp_chat.txt.txt
```

If WhatsApp uses a different display name for you, override the default `Ahmed` value:

```bash
SELF_SENDER='Ahmed Mohamed' ./scripts/run_from_raw.sh /absolute/path/to/whatsapp_chat.txt.txt
```

To replace the real names inside messages with `[NAME]` (strongly recommended; the file is private and never committed):

```bash
PRIVATE_NAMES_FILE=~/private_names.txt ./scripts/run_from_raw.sh /absolute/path/to/whatsapp_chat.txt.txt
```

The command runs the unit tests first. It stops immediately if a test fails or if the raw file cannot be read. It then builds and validates the dataset. Without `PRIVATE_NAMES_FILE` it prints a warning: only the sender labels are anonymized, not names written inside messages.

## Output Files

A successful run writes these files under `outputs/` by default.

| File | What it contains | Use it for |
|---|---|---|
| `train.jsonl` | Role-based chat records: `system`, `user`, and final `assistant` reply. | Main fine-tuning input. |
| `validation.jsonl` | Same format, from separate complete sessions. | Training evaluation. |
| `all_samples.json` | Human-readable `instruction` / `response` records. | Spot checks and instruction-format compatibility work. |
| `audit_report.json` | Counts for parsing, cleaning, redaction, sessions, splits, and validation. | Your first file to inspect. |
| `rejections.jsonl` | Rejection reasons and source line numbers only; no rejected text. | Debugging and tuning filters. |

## How to Verify the Build

Run these commands after the build completes:

```bash
# Read summary counts and the validation result
python3 -m json.tool outputs/audit_report.json | head -n 80

# View one redacted, role-based training record
head -n 1 outputs/train.jsonl

# Confirm output files exist and inspect their sizes
ls -lh outputs/
```

A valid build must report `"valid": true` in `audit_report.json` under `output_validation`. The exact number of samples is not fixed: it depends on the raw chat and configured merging, session, and filtering rules.

## Privacy Checklist

The project maps third-party sender labels to `Contact`, replaces detected emails, URLs, and phone-like strings with placeholders, and skips common system/media/deleted-message events. It does **not** guarantee that all personal information has been removed from free-form text.

Before you upload any data to a training service, build with the private names list and inspect random lines in `train.jsonl`. Use `--redact-names-file` for names: it matches whole words and keeps the و/ب/ل prefix. `--redact-term` is for other literal terms (an address, a company): it also matches inside longer words, so a short name passed there would break everyday words.

```bash
python3 src/twin_dataset_builder.py \
  --input /absolute/path/to/whatsapp_chat.txt.txt \
  --output-dir outputs \
  --self-sender Ahmed \
  --redact-names-file ~/private_names.txt \
  --redact-term 'A private address'
```

Measured on the real export (15 Oct): without the names file, 24.1% of training samples contained a private name, 6.1% in the reply the model learns to write. With it, 0, confirmed by `tools/check_private_names.py`.

Keep the raw export and the `outputs/` directory private. Do not commit them to GitHub.

## Configuration You Can Change

| Setting | Default | Effect |
|---|---:|---|
| `--context-turns` | `5` | Number of preceding messages kept as context. |
| `--max-gap-hours` | `4` | Starts a new conversation session after a long silence. |
| `--burst-gap-seconds` | `300` | Joins short back-to-back messages from the same speaker, including a multi-part reply typed with a short pause in the middle. |
| `--min-response-chars` | `2` | Excludes extremely short content. |
| `--max-duplicate-responses` | `3` | Keeps at most this many samples with the exact same response text; set to `0` to disable. |
| `--validation-ratio` | `0.10` | Allocates whole sessions to evaluation. |

Read [`DATA_CONTRACT.md`](DATA_CONTRACT.md) before changing settings; it explains why each rule exists and what it changes.

## Hand-off to Fine-Tuning

When you are satisfied with `train.jsonl` and `validation.jsonl`, use those files as the dataset inputs for [Supervised fine-tuning](../supervised-fine-tuning). Do not mix data processing and model training in the same command. Keeping them separate makes the dataset reproducible and easier to audit.

## References

[1] [WhatsApp Help Center — Export your chat history](https://faq.whatsapp.com/1180414079177245/)
[2] [Hugging Face TRL — Dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
[3] [NIST — De-identification of Personal Information](https://csrc.nist.gov/publications/detail/sp/800-188/final)
