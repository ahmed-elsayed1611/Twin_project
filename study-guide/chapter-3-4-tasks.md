# Chapters 3 and 4 — fixes and improvements task list

Work in `~/Twin_project`, on its own branch:

```bash
cd ~/Twin_project && git checkout -b chapter-3-4-fixes
```

Rule: change one thing at a time, run again, and record the numbers. Every fix needs a "before" and an "after". Entry numbers (3.x, 4.x) refer to [improvements-to-the-book-code.md](improvements-to-the-book-code.md).

## Step 0 — record the baseline (before changing anything)

- [x] Run the crawl and the feature pipeline exactly as they are.
- [ ] Write these numbers in `baseline.md` (the numbers are in the 14 Oct Notion run log and in `improvements-to-the-book-code.md`):
  - articles in MongoDB
  - points in Qdrant `cleaned_articles`
  - points in Qdrant `embedded_articles`
  - chunks per article (from the chunk-and-embed step's metadata in the ZenML dashboard)

## Part 1 — make failures loud

| # | Done | File | Change | Entry |
|---|---|---|---|---|
| 1 | [x] | `steps/etl/crawl_links.py`, after the loop (~line 29) | Raise an error if fewer than ~80% of links succeeded, instead of only logging | 3.7 |
| 2 | [x] | `steps/feature_engineering/query_data_warehouse.py`, the `except` in `fetch_all_data` (~line 51) | Re-raise the error instead of using an empty list | 4.3 |
| 3 | [x] | `steps/feature_engineering/load_to_vector_db.py` (~line 21) | Raise instead of `return False`, and check what `bulk_insert` returns | 4.9 |
| 4 | [x] | `llm_engineering/application/networks/embeddings.py`, the `except` in `__call__` (~line 100), and `embedding_data_handlers.py` | Re-raise; use `zip(..., strict=True)` | 4.14 |

Test: `docker stop llm_engineering_qdrant`, run the feature pipeline, and check ZenML now shows the run as failed (it showed green before). Then `docker start llm_engineering_qdrant`.

## Part 2 — stop data being lost or overwritten

| # | Done | File | Change | Entry |
|---|---|---|---|---|
| 5 | [x] | `llm_engineering/application/preprocessing/chunking_data_handlers.py` (the three `md5` lines) | Hash the document id with the text: `md5(f"{data_model.id}:{chunk}")` | 4.13 |
| 6 | [x] | `llm_engineering/application/preprocessing/operations/chunking.py`, end of `chunk_article` | A last basket under `min_length` joins the previous chunk; with no previous chunk, keep it anyway | 4.11 |
| 7 | [x] | `steps/feature_engineering/query_data_warehouse.py`, line 23 | Use `UserDocument.find(...)` and raise if the author doesn't exist | 4.2 |
| 8 | [x] | A small script, or `llm_engineering/domain/base/nosql.py` | Unique index on `link` for `articles` and `repositories` | 3.6 |

Test: `embedded_articles` should have more points than the baseline, and a misspelled author in `configs/feature_engineering.yaml` should stop the run.

## Part 3 — better cleaning, then measure it

| # | Done | File | Change | Entry |
|---|---|---|---|---|
| 9 | [x] | `llm_engineering/application/preprocessing/operations/cleaning.py` | Replace URLs with `[URL]` first; squeeze spaces but keep paragraph breaks (`\n\n`) | 4.5 |
| 10 | [x] | `llm_engineering/application/preprocessing/cleaning_data_handlers.py` | For articles, join only `Title`, `Subtitle`, `Content`, with a separator that survives cleaning | 4.6, 4.7 |

Test (experiment B): change `chunk_size` for one type and count the chunks. Before fix 9 nothing changes; after fix 9 it does. That confirms entry 4.12.

## Part 4 — try a better embedding model

The book uses `sentence-transformers/all-MiniLM-L6-v2` (`settings.py`, line 63): small, CPU-friendly, English-only, from 2021, 384 dimensions, 256-token input limit.

| # | Done | Task |
|---|---|---|
| 11 | [ ] | Check the current MTEB leaderboard and pick candidates (for English: `BAAI/bge-small-en-v1.5`; for Arabic, in week 5: `intfloat/multilingual-e5-base`, `BAAI/bge-m3`) |
| 12 | [ ] | Switch `TEXT_EMBEDDING_MODEL_ID` in `.env` to one candidate |
| 13 | [ ] | Delete the `embedded_*` collections in Qdrant (the vector size changes, e.g. 384 to 768) and run the feature pipeline again |
| 14 | [ ] | Record: vector size, chunks per article (chunk size follows the model's input limit), and how long the pipeline took on CPU |
| 15 | [ ] | Compare retrieval quality on the same small question set for both models (the full evaluation comes in week 5) |

Optional: compare an open model with a paid API, on the public book data only.

| # | Done | Task |
|---|---|---|
| 16 | [ ] | Get a free Cohere trial key; add it to `.env` (never commit it) |
| 17 | [ ] | Write a `CohereEmbeddingModel` class with the same interface as `EmbeddingModelSingleton` (`__call__`, `embedding_size`, `max_input_length`, `model_id`), calling Cohere's multilingual embed model |
| 18 | [ ] | Pass `input_type="search_document"` for chunks and `input_type="search_query"` for questions (the book embeds both the same way, so this needs a small change in the query handler) |
| 19 | [ ] | Replace the token splitter in `chunk_text`, which needs a local Hugging Face model name, with a splitter that works without one |
| 20 | [ ] | Compare with the open model on the same questions: quality, speed, and cost per run |

No paid embedding API (OpenAI, Cohere, Voyage) for the private WhatsApp data: it would send the text to their servers.

## Part 5 — protect each fix with a test

- [x] One small `pytest` test per fix. Done 14 Oct: 24 tests in `tests/unit/` (`test_cleaning.py`, `test_chunking.py`, `test_crawling.py`, `test_pipeline_steps.py`), all passing. Written with Claude, not by me.
- [ ] Read every test and be able to explain what it proves and which entry it protects (interview prep).

## Part 6 — write it up

- [x] Mark each entry in `improvements-to-the-book-code.md` as fixed, with before and after numbers.
- [ ] One commit per fix, naming the entry. Parts 1, 2 and 4.16/4.17 committed; the last 8 commits (cleaning, crawlers, tests, README) still to run, then push `chapter-3-4-fixes`.
- [x] A short "What I changed in the book's code" section in the repo README.
- [x] Run log in Notion: "Chapters 3 and 4 — Re-run, Data Bugs and 25 Fixes (14 Oct)", under 2 · Book Code — Run Logs.

## Later — week 5 (3 to 9 November)

- [ ] WhatsApp conversations as a new data type in the feature pipeline, under the privacy rules: real data local only, synthetic data for anything public, aggregate numbers only, and consent from the other person (or only my own messages).
- [ ] Retrieval evaluation: book version vs. fixed version vs. a multilingual embedding model, measured with recall@5 on 30–50 questions.

## Time

Step 0 is the run session. Parts 1–5 take about two 6-hour weekend days, or about five weekday sessions. Part 6 takes one session.
