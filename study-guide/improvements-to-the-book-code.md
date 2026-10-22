# What I would improve in the book's code

One section per chapter, in the book's order. Every entry has the same four lines, so they can be read quickly and said aloud in an interview:

- **Where:** the file and the line or function.
- **Problem:** what goes wrong, and for whom.
- **Change:** what I would do instead.
- **Source:** how I found it — *by reading* (I noticed it while tracing the code) or *by running* (it broke on my machine; details in my run logs).

## Overview

| # | Chapter | Entries | Status |
|---|---|---|---|
| 1 | LLM Twin concept and architecture | 0 | Nothing to improve: no code |
| 2 | Tooling and installation | 4 | All closed on 14 Oct: 2.4 fixed, 2.3 documented, 2.1 and 2.2 needed no change |
| 3 | Data engineering | 8 | All fixed on 14 Oct |
| 4 | RAG feature pipeline | 17 | All fixed on 14 Oct except 4.4 (kept by choice); 24 unit tests in `tests/unit/` |
| 5 | Supervised fine-tuning | 6 | Dataset generation run in August; training code to read |
| 6 | Preference alignment | 0 | To fill in |
| 7 | Evaluation | 0 | To fill in |
| 8 | Inference optimisation | 0 | To fill in |
| 9 | RAG inference pipeline | 19 | 18 fixed on 15 Oct, 9.19 to measure; 34 unit tests in `tests/unit/test_rag.py`; live run still to do |
| 10 | Deployment | 0 | To fill in |
| 11 | MLOps and LLMOps | 1 | To fill in |

**The three I would mention first in an interview:** 3.1 (the GitHub crawler damages the code it stores), 5.3 (a concurrency cap is not a rate limiter), 2.2 (MongoDB authentication is not enforced).

---

## 1 · LLM Twin concept and architecture

No code in this chapter, so nothing to improve. The design ideas to keep are in study guide 0.

---

## 2 · Tooling and installation

### 2.1 `poetry install` does not install everything the commands need
- **Where:** `pyproject.toml`
- **Problem:** after a clean install, running the tasks failed one missing package at a time: `poethepoet`, `zenml[server]`, `sqlalchemy_utils`, `sqlmodel`.
- **Change:** declare them as dependencies, so one install command is enough.
- **Source:** by running.
- **Checked 14 Oct, no change needed:** `poethepoet` and `zenml[server]` are now declared in `pyproject.toml` (`zenml[server]` brings `sqlalchemy_utils` and `sqlmodel`), and all four import.

### 2.2 MongoDB authentication is not enforced
- **Where:** `docker-compose.yml`, the `mongo` service.
- **Problem:** the root user is created, but `mongod` runs without `--auth`. Anyone on the machine can connect with no password, while the connection string suggests the database is protected.
- **Change:** add `command: ["--auth"]` to the service.
- **Source:** by running (found while connecting MongoDB Compass).
- **Checked 14 Oct, this entry was wrong:** `mongosh` without credentials is refused with "Unauthorized". The official image adds `--auth` itself when `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD` are set. (Compass probably had the credentials saved.) No change.

### 2.3 The ZenML timeout setting resets itself
- **Where:** the local ZenML server (`zenml login --local`).
- **Problem:** the server regenerates its config on every start and sets `http_timeout` back to 30 seconds, so a long pipeline fails. Editing the config file by hand does not last.
- **Change:** set `ZENML_STORE_HTTP_TIMEOUT=300` in the environment (I put it in `~/.bashrc`), and document it in the README.
- **Source:** by running.
- **Done 14 Oct:** documented in the Twin_project README.

### 2.4 A value in `.env` silently beats a change in `settings.py`
- **Where:** `llm_engineering/settings.py` (Pydantic `BaseSettings`) and `.env`.
- **Problem:** I changed a default in the class and nothing happened, because `.env` still held the old value and `.env` wins. Nothing warns you.
- **Change:** keep each setting in one place only, and print the active values at the start of a run.
- **Source:** by running.
- **Fixed 14 Oct:** `load_settings` logs "Active settings: {...}" for a fixed list of settings without secrets (`LOGGED_SETTINGS`). The connection strings and keys are never printed.

---

## 3 · Data engineering

### 3.1 The GitHub crawler removes every space from every file
- **Where:** `llm_engineering/application/crawlers/github.py`, `tree[file_path] = f.read().replace(" ", "")`
- **Problem:** `import os` is stored as `importos`. The stored code is damaged for everything that reads it later: chunking, embedding, dataset generation.
- **Change:** store the content unchanged, or only collapse runs of blank space.
- **Source:** by reading.
- **Fixed 14 Oct:** the file is stored unchanged. Not run on real data (no repositories in the data set); covered by the repository cleaning test.

### 3.2 The LinkedIn crawler is switched off, so there are never any posts
- **Where:** `llm_engineering/application/crawlers/linkedin.py`, `is_deprecated=True` raises on the first line of `extract`.
- **Problem:** the `posts` collection is never filled. This is the root cause of the `cleaned_posts` 404 error I hit later, in dataset generation.
- **Change:** remove it from the default chain in `crawl_links.py`, or replace it with a source that works.
- **Source:** by reading, confirmed by running.
- **Fixed 14 Oct, differently:** kept registered (otherwise LinkedIn links would fall through to the generic crawler and save a login page), but it now raises in `__init__` before starting Chrome. `_crawl_link` creates the crawler inside its `try`, so this counts as one failed link instead of stopping the step.

### 3.3 A branch that can never run
- **Where:** `llm_engineering/application/utils/split_user_full_name.py`, `if len(name_tokens) == 0`
- **Problem:** `"".split(" ")` returns `[""]`, so the list is never empty. An empty name passes through as first name `""` and last name `""`.
- **Change:** check `if not user.strip()` before splitting.
- **Source:** by reading.
- **Fixed 14 Oct:** `user.split()` (no argument) returns `[]` for `""` and `"   "`, so the empty branch now works; double spaces between names are also handled.

### 3.4 Adding a website means editing three places
- **Where:** a new crawler file, a `register_...` method in `dispatcher.py`, and the chain in `steps/etl/crawl_links.py`.
- **Problem:** the list of active crawlers is hardcoded in the step.
- **Change:** let each crawler declare its own domain, and have the dispatcher register every crawler it finds.
- **Source:** by reading.
- **Fixed 14 Oct:** `BaseCrawler.domain`, and `CrawlerDispatcher.build_default()` registers every subclass (at any depth) that sets it. The old `register_...` methods still work.

### 3.5 Medium extraction depends on Medium's CSS class names
- **Where:** `medium.py`, `soup.find_all("h1", class_="pw-post-title")`
- **Problem:** if Medium renames the class, the title is silently stored as `None`. Nothing fails.
- **Change:** log a warning when the title or subtitle is missing, and count it in the step's metadata.
- **Source:** by reading.
- **Fixed 14 Oct (warning only):** a warning naming the CSS class that stopped matching. Not counted in metadata.

### 3.6 The "already exists" check and the insert are two separate calls
- **Where:** every crawler: `self.model.find(link=link)`, then `instance.save()`.
- **Problem:** two runs at the same time could both pass the check and insert the same link twice.
- **Change:** a unique index on `link` in MongoDB, so the database itself refuses duplicates.
- **Source:** by reading.
- **Fixed 14 Oct:** `tools/create_indexes.py` adds a unique index on `link` for `articles` and `repositories` (run once per fresh database). Test: saving a second article with an existing link returns `None` and the count stays at 76.

### 3.7 A failed crawl looks the same as a successful run
- **Where:** `steps/etl/crawl_links.py`, the `try/except` in `_crawl_link`.
- **Problem:** the pipeline finishes "successfully" even if most links failed. The counts are only in the metadata.
- **Change:** fail the step, or warn loudly, when the success rate falls under a threshold.
- **Source:** by reading.
- **Fixed 14 Oct (commit 9b1a78d):** the step raises `RuntimeError` under 80% success. Test: 1 good link out of 3 → run failed with "Only 1/3 links crawled (33%)". Before: green.

### 3.8 An error page is saved as a real article
- **Where:** `llm_engineering/application/crawlers/custom_article.py`, `extract`.
- **Problem:** found on the 14 Oct run. All 45 of Paul Iusztin's `decodingml.substack.com` links now land on a Substack "Page not found" page. The crawler saved each one as an article (472 characters each) and counted it as a success. Only the chapter 4 chunker's 1,000-character minimum kept them out of `embedded_articles`; they are still in `cleaned_articles`, which the training pipeline reads.
- **Change:** check the HTTP status and reject pages that are too short or contain "Page not found"; count them as failed crawls.
- **Source:** by running.
- **Fixed 14 Oct (commit 51407b3):** `extract` raises `ValueError` for pages under 1,000 characters or showing "Page not found" / "404 Not Found" (`AsyncHtmlLoader` gives no status code, so the check is on the content). Test: a dead Substack page is rejected (290 chars); a real Maxime article is still saved (23,849 chars). Data: Paul's articles now come from the book's saved JSON (76 real articles in total).

---

## 4 · RAG feature pipeline

### 4.1 The embedding may not reach the final object (to verify)
- **Where:** the vector base class, `llm_engineering/domain/base/vector.py`, as printed in the book.
- **Problem:** the `attributes` dictionary is built before the embedding is added to `payload`. As written in the book, the embedding might never make it into the final object.
- **Change:** check the real repository code against the book's listing first. If it is real, build `attributes` after the embedding is set.
- **Source:** by reading (my note "Mistakes in the Book").
- **Checked 8 Oct:** the repository code is correct. In `from_record` (line 34), `attributes` is built from the payload first, and the embedding is then added to `attributes` itself (line 43). So this is a problem in the book's printed listing only, not in the code I run.

### 4.2 An unknown author name creates an empty user and the run still succeeds
- **Where:** `steps/feature_engineering/query_data_warehouse.py`, line 23: `UserDocument.get_or_create(...)`.
- **Problem:** a pipeline that only reads should not create data. A typo in `author_full_names` creates a new user with no documents, the three queries return nothing, and the pipeline ends green with that author missing from Qdrant.
- **Change:** use `UserDocument.find(...)`, and stop or warn when the user does not exist or has no documents.
- **Source:** by reading.
- **Fixed 14 Oct:** `find` plus a `ValueError` when the user is missing, and a warning when the user has no documents. Test: "Paul Iusztinn" in the config → run failed with "was not found"; users in MongoDB stayed at 2 (before: a third, empty user was created and the run was green).

### 4.3 A failed MongoDB query is turned into an empty list
- **Where:** `fetch_all_data` in the same file, the `try/except` around `future.result()` (lines 49–54).
- **Problem:** if the articles query fails, the step logs one line and continues with zero articles. Training and retrieval would then run on incomplete data with no failed step to show it.
- **Change:** let the error stop the step, or record the failure in the output metadata and fail when a whole category is missing.
- **Source:** by reading.
- **Fixed 14 Oct (commit 0614143):** the error is logged and re-raised.

### 4.4 The documents are mixed into one list, then grouped again twice
- **Where:** `query_data_warehouse` flattens the per-type dictionary (line 27); `load_to_vector_db` calls `group_by_class`; `EmbeddingDispatcher.dispatch` asserts one category per batch.
- **Problem:** the grouping by type is thrown away and rebuilt later. The step outputs are plain `list`, so nothing checks what is inside, and one data type cannot be re-run alone.
- **Change:** pass a dictionary keyed by data category between the steps. `fetch_all_data` already produces one.
- **Trade-off:** the mixed list keeps every step to one input and one output, and a new data type needs no change to the pipeline file. A dictionary costs one extra loop in each step.
- **Source:** my own question while reading.
- **Decision 14 Oct: kept as it is.** It is a design choice, not a bug, and the trade-off above favours the mixed list for a project this size.

### 4.5 The cleaning in the code is much simpler than the cleaning the chapter describes
- **Where:** `llm_engineering/application/preprocessing/operations/cleaning.py`, `clean_text`.
- **Problem:** the chapter talks about removing emojis and replacing URLs with a placeholder. The code is two regexes: drop every character that is not a letter, digit, underscore, whitespace or `. , ! ?`, then squeeze whitespace. A URL is not replaced; it only loses its punctuation (`https example.com rag`). Paragraph breaks are lost too, which matters later for chunking.
- **Change:** replace URLs with a `[URL]` token before stripping symbols, and keep paragraph breaks (squeeze spaces and newlines separately).
- **Source:** by reading.
- **Fixed 14 Oct:** links → `[URL]`; blank lines → one paragraph break (`\n\n`); a single line break inside a paragraph (html2text wraps lines) → a space. On the 76 articles: `https ` fragments 43 → 1, `[URL]` 0 → 46, articles with paragraph breaks 0 → 76.

### 4.6 The ` #### ` separator is added and then removed by the next call
- **Where:** `cleaning_data_handlers.py`: `clean_text(" #### ".join(...))`.
- **Problem:** `#` is not in the allowed characters, so the separator between title, subtitle and content disappears at once. The parts end up joined by a single space with nothing marking where the title ends.
- **Change:** join with a separator that survives cleaning (for example a newline that is kept), or clean each part first and join afterwards.
- **Source:** by reading.
- **Fixed 14 Oct:** each part is cleaned on its own, then joined with `\n\n`, so the title ends with a paragraph break.

### 4.7 Every value of `content` is joined, including ones that are not text of the document
- **Where:** `ArticleCleaningHandler.clean` joins `data_model.content.values()`.
- **Problem:** the generic article crawler stores `language` inside `content`, so a stray `en` is glued onto the end of the article. For repositories the file names are the dictionary keys, so they are dropped and only file contents are kept.
- **Change:** pick the fields by name (`Title`, `Subtitle`, `Content`) for articles; for repositories include each file's path before its content.
- **Source:** by reading, confirmed by running.
- **Fixed 14 Oct:** articles use `ARTICLE_TEXT_FIELDS`; repositories write `File: <path>` before each file. Articles ending in ` en`: 71 of 76 → 0.

### 4.8 Code is cleaned with the same rule as prose
- **Where:** `RepositoryCleaningHandler.clean` calls the same `clean_text`.
- **Problem:** brackets, quotes, operators, colons and slashes are all removed. Together with 3.1 (the crawler already removed every space), a line such as `def add(a, b): return a + b` reaches the embedding model as `defadd a,b returna b`.
- **Change:** a separate cleaning function for code that keeps its syntax, and fix 3.1 first.
- **Source:** by reading.
- **Fixed 14 Oct:** `clean_code` only removes trailing spaces and extra blank lines. Covered by unit tests (no repositories in the data set).

### 4.9 A failed load to Qdrant does not fail the pipeline
- **Where:** `steps/feature_engineering/load_to_vector_db.py`, and how `pipelines/feature_engineering.py` uses it.
- **Problem:** two layers hide a failure. (1) The step returns `False` when a batch raises, but the pipeline only uses the step's `invocation_id`, so nobody reads that `False` and the run ends green. (2) `bulk_insert` itself returns `False` when the insert fails a second time, and the step does not check that return value, so it carries on and returns `True`.
- **Change:** raise an exception when a batch cannot be inserted, so ZenML marks the step as failed.
- **Source:** by reading.
- **Fixed 14 Oct (commit 5ff2751):** no `try/except` in the step, and it raises when `bulk_insert` returns `False`. Tests: Qdrant stopped → run failed with "Connection refused" (before: green with nothing saved). The fix also caught a real Qdrant timeout during collection creation on the first test run.

### 4.10 The step's metadata counter may write into a real chunk (to verify)
- **Where:** `steps/feature_engineering/rag.py`, `_add_embeddings_metadata`: `metadata[category] = embedded_chunk.metadata`, then `metadata[category]["authors"] = ...`.
- **Problem:** the first line does not copy the dictionary; it points at the first embedded chunk's own `metadata`. Adding `authors` to it would then also add `authors` to that chunk, which is saved to Qdrant. If so, the first chunk of each category carries an extra `authors` list that the others do not.
- **Change:** copy first: `metadata[category] = dict(embedded_chunk.metadata)`.
- **How to verify:** after a run, open the first point of `embedded_articles` in the Qdrant dashboard and look for an `authors` key inside `metadata`.
- **Source:** by reading. **Confirmed 14 Oct:** exactly 1 of the 981 chunks in `embedded_articles` has `'authors': ['Maxime Labonne', 'Paul Iusztin']` inside its metadata; the other 980 do not. **Fixed 14 Oct:** `dict(...)` copies in both helpers → 0 chunks with the extra key. `num_chunks` now counts chunks before embedding.
- Related: `num_chunks` and `num_embedded_chunks` in the same step are both set to `len(embedded_chunks)`, so they are always equal and one of them tells you nothing.

### 4.11 Short articles and article endings are dropped
- **Where:** `operations/chunking.py`, `chunk_article`: a chunk is kept only `if len(current_chunk) >= min_length` (1,000 characters).
- **Problem:** the last part of an article is thrown away when it is under 1,000 characters, and an article shorter than 1,000 characters produces no chunks at all, so it can never be retrieved. Nothing is logged.
- **Change:** attach a short tail to the previous chunk, keep a short article as one chunk, and count dropped text in the step metadata.
- **Source:** by reading.
- **Fixed 14 Oct:** a short piece (in the middle when the next sentence does not fit, or at the end) joins the previous chunk, or stands alone if it is the first. Before: 39 of 76 articles lost text, 25,841 characters in total (97.2% kept). After: 0 articles lose text (100% kept). Still 500 chunks, since the pieces join existing chunks. Nothing is dropped any more, so the metadata counter is not needed.

### 4.12 The first stage of `chunk_text` probably does nothing (to verify)
- **Where:** `operations/chunking.py`, `chunk_text`: `RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, ...)`.
- **Problem:** it splits on blank lines, but `clean_text` has already turned every newline into a space (see 4.5). So the text arrives as one section and the real cutting is done only by the token splitter, at the embedding model's limit. If so, `chunk_size` (250 for posts, 1,500 for repositories) has no effect.
- **Change:** keep paragraph breaks in cleaning, or split on sentences as `chunk_article` does.
- **How to verify:** change `chunk_size` for one type, run again, and compare the number of chunks.
- **Source:** by reading. **Confirmed and fixed by 4.5 on 14 Oct:** on an 8-paragraph text, the old cleaning gives 2 chunks with `chunk_size` 250 and 2 with 1,500 (no effect); the new cleaning gives 8 and 3.

### 4.13 The chunk id comes from the text alone
- **Where:** `chunking_data_handlers.py`: `chunk_id = hashlib.md5(chunk.encode()).hexdigest()`.
- **Problem:** two different documents that contain the same chunk text get the same id, and the second one overwrites the first in Qdrant, so one of them loses that chunk. (The good side of the same design: re-running the pipeline creates no duplicates.)
- **Change:** hash the document id together with the text.
- **Source:** by reading, confirmed by running.
- **Confirmed and fixed 14 Oct (commit 2133155):** 500 chunks sent, 497 stored. Cause: 4 different Paul articles start with the same series intro ("Every message will be cleaned, chunked, embedded…"), so 4 chunks shared one id. After hashing `f"{document_id}:{chunk}"`: 500 sent, 500 stored (Paul 290 → 293).

### 4.14 A failed embedding call makes a whole batch of chunks disappear
- **Where:** `application/networks/embeddings.py`, `EmbeddingModelSingleton.__call__` returns `[]` when `encode` raises; `embedding_data_handlers.py`, `embed_batch` then does `zip(data_model, embeddings, strict=False)`.
- **Problem:** `zip` stops at the shorter list. With an empty list of embeddings it produces nothing, so all 10 chunks of that batch are dropped. One error line is logged and the pipeline continues and ends green.
- **Change:** let the error stop the step, or use `strict=True` so a length mismatch raises.
- **Source:** by reading.
- **Fixed 14 Oct (commit 4685036):** both: `__call__` re-raises and `zip(..., strict=True)`.

### 4.15 The singleton ignores new arguments after the first call
- **Where:** `application/networks/base.py`, `SingletonMeta.__call__`.
- **Problem:** `EmbeddingModelSingleton(model_id="another-model")` returns the first model that was created, without any warning, if one already exists.
- **Change:** raise when different arguments are passed, or key the stored instances by the arguments.
- **Source:** by reading (the file's own comment says this).
- **Fixed 14 Oct:** `SingletonMeta` stores the first call's arguments and raises `ValueError` for different ones; a call with no arguments still returns the instance. Limit: passing the default value explicitly also counts as "different".

### 4.16 Article chunks are longer than the embedding model can read
- **Where:** `ArticleChunkingHandler` (`min_length` 1,000, `max_length` 2,000 characters) with `all-MiniLM-L6-v2`, which reads at most 256 tokens and silently cuts the rest.
- **Problem:** measured on 14 Oct over the 500 article chunks: median 425 tokens, max 869. 484 chunks (97%) are over 256 tokens, and 41% of all article tokens are never embedded. The text is stored in Qdrant but a question about the second half of a chunk cannot find it. Posts and repositories do not have this problem: `chunk_text` cuts at the model's limit.
- **Change:** size article chunks by tokens from `embedding_model.max_input_length` (about 1,000 characters), or use a model with a longer input (for example `bge-small-en-v1.5` reads 512 tokens). Measure retrieval before and after (task list, part 4).
- **Source:** by running (token count with the model's own tokenizer).
- **Fixed 14 Oct:** `chunk_article` takes a `length_function`. Articles are measured with `count_tokens` (the model's tokenizer), with `max_length` = model limit − 2 = 254 tokens (room for the start and end tokens) and `min_length` = 127. A sentence longer than the limit (often code, which has no full stops) is cut between words. A short piece joins the previous chunk only if the result still fits. `chunk_document` (chapter 5 dataset generation) still measures in characters. Before: 500 chunks, median 425 tokens, 97% over the limit, 41% of tokens never embedded. After: 981 chunks, median 239 tokens, max 256, 0% over, every word kept. Cost: twice the chunks, so embedding takes about twice as long. Not measured yet: whether retrieval answers improve (needs the question set from part 4).

### 4.17 The Qdrant client waits only 5 seconds
- **Where:** `llm_engineering/infrastructure/db/qdrant.py`: `QdrantClient(host=..., port=...)` with no `timeout`, so the HTTP client's 5-second default applies.
- **Problem:** found on the 14 Oct runs. With the Windows disk at over 100% busy, Qdrant took 22 to 75 seconds to create a collection, so `bulk_insert` timed out in the create-and-retry path. Before fix 4.9 this would have ended green with an empty collection; after it, the run failed loudly, which is how I found it.
- **Change / fixed 14 Oct:** a `QDRANT_TIMEOUT` setting (default 120 seconds, can be changed in `.env`) passed to both client constructors. When re-running tests, empty the collections (`POST /points/delete` with an empty filter) instead of deleting them, so Qdrant does not have to recreate them.
- **Source:** by running.

---

## 5 · Supervised fine-tuning

These are all from the instruction-dataset generation, which I ran in August.

### 5.1 The token counter only works with OpenAI model names
- **Where:** `llm_engineering/application/dataset/generation.py`, `tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)`
- **Problem:** any other model name raises `KeyError` before a single API call is made.
- **Change:** `tiktoken.get_encoding("cl100k_base")`. The count is only an estimate, so it does not need the exact tokenizer.
- **Source:** by running.

### 5.2 The LLM client cannot point at another provider
- **Where:** `generation.py`, `ChatOpenAI(...)` has no `base_url`.
- **Problem:** it always calls OpenAI. With a Groq or Gemini key it fails in a confusing way.
- **Change:** add an `OPENAI_BASE_URL` setting and pass it to the client.
- **Source:** by running.

### 5.3 A concurrency cap is used where a rate limiter is needed
- **Where:** `generation.py`, `chain.batch(batch, stop=None)` sends 24 requests at once.
- **Problem:** a free-tier limit is per minute. Lowering concurrency to 1 still hit the limit, because it bounds requests in flight, not requests over time.
- **Change:** an `InMemoryRateLimiter` on the client (I used 0.2 requests per second). It gave a full run with zero rate-limit errors.
- **Source:** by running.

### 5.4 The config points at the author's Hugging Face account
- **Where:** `configs/generate_instruct_datasets.yaml`, `dataset_id: pauliusztin/llmtwin`
- **Problem:** a push fails with a permissions error that looks unrelated to the real cause.
- **Change:** build the dataset ID from a setting holding the user's own account name.
- **Source:** by reading, before running.

### 5.5 No way to choose which data categories to generate from
- **Where:** `steps/generate_datasets/query_feature_store.py`, `fetch_all_data()`
- **Problem:** it always fetches articles, posts and repositories. To test on a small batch I had to comment out code.
- **Change:** a `categories` parameter in the YAML.
- **Source:** by running.

### 5.6 One wrong capital letter drops a sample
- **Where:** the output parser for `InstructDatasetSample`.
- **Problem:** the model returned the key `"Answer"` instead of `"answer"`, and the sample was rejected.
- **Change:** normalise key names to lowercase before validation.
- **Source:** by running.

_To add after reading `llm_engineering/model/finetuning/`: the training code itself._

---

## 6 · Preference alignment

_To fill in. My own version of this stage, and why it differs from the book's, is in `whatsapp-twin/preference-alignment/README.md`._

---

## 7 · Evaluation

_To fill in. My own version is in `whatsapp-twin/evaluation/README.md`._

---

## 8 · Inference optimisation

_To fill in._

---

## 9 · RAG inference pipeline

All fixed on 15 Oct on the branch `chapter-9-fixes` in `~/Twin_project`. Docker was off, so every before and after was measured with fakes in place of OpenAI, MongoDB, Qdrant and SageMaker (the same script run on the old and the new code). 34 unit tests in `tests/unit/test_rag.py`. Still to do: one live run with `tools/rag.py`.

### 9.1 Asking a question can create a user in MongoDB
- **Where:** `application/rag/self_query.py`, line 32: `UserDocument.get_or_create(...)`.
- **Problem:** the same bug as 4.2, on the read path. A misspelled or unknown name ("Paul Iusztinn") creates an empty user, the query is filtered on that user, and the search silently returns nothing.
- **Change:** `find`; if the author is unknown, search without a filter and log a warning.
- **Source:** by reading.
- **Fixed 15 Oct:** "Paul Iusztinn" → before: 1 user created, filter on an empty user. After: 0 users created, search with no filter, warning in the log. "Paul Iusztin" still gets his filter.

### 9.2 Only the exact word "none" means "no author"
- **Where:** `self_query.py`, line 28: `if user_full_name == "none"`.
- **Problem:** "None", "none." or "NONE" from the model are treated as a name, and a user called "None None" is created.
- **Change:** strip quotes, dots and spaces, compare in lower case, and accept "", "none", "null", "n/a".
- **Source:** by reading.
- **Fixed 15 Oct:** "None" and "none." → before: a user created for each. After: no user, no filter.

### 9.3 The prompt asks for a user id that the code cannot handle
- **Where:** `prompt_templates.py`, `SelfQueryTemplate` (example 3: "1345256"), and `self_query.py`.
- **Problem:** the code treats every answer as a name, so an id becomes the user "1345256 1345256". Real user ids are UUIDs anyway.
- **Change:** ask for the name only and drop the id example.
- **Source:** by reading.
- **Fixed 15 Oct:** the prompt no longer mentions ids. "1345256" → before: a user created. After: no user, no filter.

### 9.4 The LLM receives the whole `Query` object instead of the question
- **Where:** `self_query.py` line 25 and `query_expansion.py` line 26: `chain.invoke({"question": query})`.
- **Problem:** the prompt ends with `Query(id=UUID('…') content='…' author_id=None author_full_name=None metadata={})`. Both LLM steps read ids and field names as part of the question.
- **Change:** pass `query.content`.
- **Source:** by reading, confirmed by running (the prompt captured from a fake model).
- **Fixed 15 Oct:** before: the prompt contained the whole object. After: it ends with "User question: Write a post about RAG.".

### 9.5 With no context, the model answers from memory
- **Where:** `infrastructure/inference_pipeline_api.py`, `rag()`.
- **Problem:** when retrieval finds nothing, the model is still called with an empty context and makes up an answer — the problem RAG is there to solve.
- **Change:** return a fixed "I couldn't find anything…" answer without calling the model, and tag the trace `no_context`.
- **Source:** by reading.
- **Fixed 15 Oct:** 3 questions with nothing in Qdrant → before: the model called 3 times. After: 0 calls.

### 9.6 Query expansion trusts the model's format and count
- **Where:** `query_expansion.py`, after the split.
- **Problem:** 4 rewrites gave 5 queries instead of 3. A model that ignores `#next-question#` and writes one numbered question per line gave one giant "rewrite" holding both questions.
- **Change:** fall back to one question per line (numbering removed) when the separator is missing, and keep at most `expand_to_n` queries.
- **Source:** by reading.
- **Fixed 15 Oct:** 4 rewrites → before 5 queries, after 3. One per line → before 2 queries (one with both questions), after 3 clean queries.

### 9.7 `k // 3` per collection
- **Where:** `retriever.py`, `_search`: `assert k >= 3` and `limit=k // 3`.
- **Problem:** `k=4` and `k=5` behave like 3; `k=2` crashes inside a thread; every collection gets the same quota whatever the question is about.
- **Change:** each collection returns up to `k` candidates and the reranker keeps the best `k`; refuse `k < 1` before starting threads.
- **Source:** by reading.
- **Fixed 15 Oct:** limits per collection, before → after: k=2 AssertionError → 2; k=4 1 → 4; k=5 1 → 5; k=9 3 → 9. Cost: up to 3× more candidates for the reranker (27 pairs for `k=3`; it runs on CPU).

### 9.8 Every Qdrant error looks like "no results"
- **Where:** `domain/base/vector.py`, `search`: `except UnexpectedResponse: return []`.
- **Problem:** a missing collection (no posts were crawled) and a real server error look the same: an empty list and one log line. Same family as 4.9.
- **Change:** only "collection not found" (404) returns `[]`, with one warning per collection; any other error is raised.
- **Source:** by reading.
- **Fixed 15 Oct:** a 500 from Qdrant → before: `[]`. After: raised. A 404 still returns `[]`.

### 9.9 The reranker pairs scores and chunks with `strict=False`
- **Where:** `reranking.py`, line 24.
- **Problem:** if the model returned fewer scores than chunks, the last chunks would be dropped and the scores could belong to the wrong chunks, silently. Same as 4.14.
- **Change:** `zip(..., strict=True)`.
- **Source:** by reading.
- **Fixed 15 Oct:** 3 chunks and 2 scores → before: 2 chunks out, no error. After: `ValueError`.

### 9.10 The 500 response shows internal error messages to the caller
- **Where:** `inference_pipeline_api.py`, the `except` in `rag_endpoint`.
- **Problem:** `detail=str(e)` can send hosts, URLs or credentials to whoever called the API.
- **Change:** log the full error on the server, return a generic message.
- **Source:** by reading.
- **Fixed 15 Oct:** an error mentioning `mongodb://admin:secret@…` → before: in the response. After: "Internal error while answering the query."; the details are in the server log.

### 9.11 Answers are cut after 150 tokens
- **Where:** `settings.py`, `MAX_NEW_TOKENS_INFERENCE = 150`.
- **Problem:** about 110 words, so LinkedIn posts end mid-sentence.
- **Change:** 600, checked against the endpoint: `MAX_INPUT_LENGTH` (2,048) + 600 stays under `MAX_TOTAL_TOKENS` (4,096).
- **Source:** by reading.
- **Fixed 15 Oct:** 150 → 600, with a test that keeps it inside the endpoint's limits.

### 9.12 The inference prompt is not the format the model was trained on
- **Where:** `model/inference/run.py`, `InferenceExecutor`.
- **Problem:** the twin was fine-tuned (SFT and DPO) on the Alpaca format (`### Instruction:` / `### Response:`, `alpaca_template` in `finetune.py`), but inference sent plain text. A fine-tuned model usually answers worse outside its training format.
- **Change:** wrap the instruction in the same template. It is repeated in `run.py` because `finetune.py` imports unsloth; a test checks the two templates are identical.
- **Source:** by reading (compared with `finetune.py`).
- **Fixed 15 Oct:** before: the prompt started "You are a content creator…". After: "Below is an instruction that describes a task…", ending with "### Response:". Not measured yet: answer quality before and after (needs the endpoint).

### 9.13 Self-query and expansion wait for each other
- **Where:** `retriever.py`, `search`.
- **Problem:** the two OpenAI calls run one after the other although neither needs the other's answer, so every question pays two round trips.
- **Change:** run both at once (self-query gets a copy of the query, because it writes to it), then copy the author onto every rewrite.
- **Source:** by reading.
- **Fixed 15 Oct:** with each fake LLM call taking 1 s → before: 2.1 s before the first search. After: 1.0 s. Paul's filter is still on 3 of 3 queries.

### 9.14 A slow `rag()` inside `async def` blocks the server
- **Where:** `inference_pipeline_api.py`, `async def rag_endpoint`.
- **Problem:** `rag()` blocks for seconds (OpenAI, Qdrant, SageMaker), and in an `async` endpoint it blocks the event loop, so requests are answered one at a time.
- **Change:** a plain `def` endpoint, which FastAPI runs in a thread pool.
- **Source:** by reading.
- **Fixed 15 Oct:** two requests at once, 1 s each → before: 2.0 s. After: 1.0 s.

### 9.15 The tokenizer is loaded three times per request
- **Where:** `application/utils/misc.py`, `compute_num_tokens`.
- **Problem:** `AutoTokenizer.from_pretrained` on every call, and `rag()` calls it 3 times only to log token counts.
- **Change:** load it once (`functools.cache`).
- **Source:** by reading.
- **Fixed 15 Oct:** 3 calls → before: 3 loads. After: 1 load.

### 9.16 A new retriever for every request
- **Where:** `inference_pipeline_api.py`, `rag()`: `ContextRetriever(mock=False)`.
- **Change:** one retriever for the whole app (`get_retriever()`, cached).
- **Source:** by reading.
- **Fixed 15 Oct:** 3 requests → before: 3 retrievers built. After: 1.

### 9.17 The reranker model loads even in mock mode
- **Where:** `reranking.py`, `__init__`.
- **Problem:** `ContextRetriever(mock=True)` is meant for testing without models, but it still loads (and the first time downloads) the cross-encoder.
- **Change:** load the model on first use.
- **Source:** by reading.
- **Fixed 15 Oct:** mock mode → before: 1 model load. After: 0.

### 9.18 Misspelled file name and prompt
- **Where:** `application/rag/query_expanison.py`; "seperated" in `QueryExpansionTemplate`.
- **Change:** renamed to `query_expansion.py` (with `git mv`, so the history is kept); fixed the spelling.
- **Source:** by reading.
- **Fixed 15 Oct.**

### 9.19 Query expansion uses `temperature=0` (to measure)
- **Where:** `query_expansion.py`.
- **Problem:** the goal of expansion is variety, and temperature 0 pushes the model toward its safest, most similar rewrites.
- **Change:** compare 0 with 0.5–0.7 on the retrieval question set (recall@5).
- **Source:** by reading.
- **Open:** needs the question set from the "My Touch" plan. (The other open question from reading, whether chunks were longer than the reranker reads, is closed by 4.16: article chunks are now at most 256 tokens.)

---

## 10 · Deployment

_To fill in._

---

## 11 · MLOps and LLMOps

### 11.1 The deploy workflow fails on every push
- **Where:** `.github/workflows/cd.yaml`
- **Problem:** it runs on every push to `main` and tries to deploy to AWS with secrets a fork does not have, so every commit shows a red mark.
- **Change:** run it only when started by hand, or only when the AWS secrets exist.
- **Source:** by running (seen on my own repository).

---

## Template for a new entry

```
### N.x Short title that states the problem
- **Where:** file, and the line or function.
- **Problem:** what goes wrong, and for whom.
- **Change:** what I would do instead.
- **Source:** by reading / by running.
```
