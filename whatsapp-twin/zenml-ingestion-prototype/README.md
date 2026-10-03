# ZenML WhatsApp Ingestion (early prototype)

The first version of the data stage: a WhatsApp extractor written as an extension to the
[*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook) codebase, in the
same style as the book's own ZenML data pipelines.

It was **superseded by the standalone [dataset builder](../data-preparation)**, which is what produced the
data used in the fine-tuning and evaluation stages. It is kept here because it shows where the project started and why the data
stage was rewritten.

## What it does

`WhatsAppExtractor` streams a raw export and produces `instruction` / `response` samples:

- parses Arabic- and English-locale message headers (Arabic-Indic digits, ص / م markers)
- maps every other sender to `Contact`, and redacts phone numbers, emails and URLs
- resets the context after a long silence (4 hours by default)
- keeps up to 5 previous turns as context, and prepends a persona prompt

Two ZenML steps (`parse_whatsapp_data` → `create_instruction_dataset`) wrap it in `ingest_whatsapp_pipeline`.

## Why it was replaced

| Limitation here | In the dataset builder |
|---|---|
| Context is flattened into one `instruction` string | Role-based chat `messages`, formatted with the model's own chat template |
| No train/validation split | Whole sessions assigned to train or validation, so sliding windows never leak across the split |
| Back-to-back messages from one sender stay separate turns | Short same-sender bursts are merged into one turn |
| A fixed persona prompt ("software engineer and AI enthusiast") | A neutral system message; the persona comes from the data |
| An unparseable timestamp falls back to the current time | Parse failures are counted and reported |
| No record of what was dropped | `audit_report.json` and `rejections.jsonl`, plus output validation |
| Needs the whole book stack (ZenML, Poetry, MongoDB) | Python standard library only |

## Layout

The files mirror their location inside the book repository's `llm_engineering/` package:

```
llm_engineering/
├── application/dataset/whatsapp_extractor.py   # parser, cleaner, sample builder
├── steps/whatsapp/ingest_whatsapp_steps.py     # ZenML steps
├── pipelines/ingest_whatsapp.py                # ZenML pipeline
├── configs/ingest_whatsapp.yaml                # pipeline parameters
└── tests/test_whatsapp_extraction.py
```

## Tests

The extractor tests need only `pydantic` and `pytest`:

```bash
cd whatsapp-twin/zenml-ingestion-prototype
python -m pytest llm_engineering/tests
```

The steps and the pipeline need ZenML and the book's environment. They are not covered by these tests.
