# Study guide 0c — The RAG inference pipeline (book chapter 9)

This is the book's code, not mine. The goal is to explain it as someone who studied and ran it.

## The one-sentence version

The inference pipeline takes a user's question, finds out whose content it is about, rewrites it into several versions, searches Qdrant with each version, keeps the best few chunks with a reranker, and passes them to the LLM as context.

## Where it picks up from chapter 4

Chapter 4 filled Qdrant's `embedded_*` collections. Chapter 9 is the reading side. The book splits RAG into three modules: **retrieve**, **augment** (put the chunks in the prompt), **generate**. Almost all of the chapter's code is the retrieve module.

## The whole flow

`llm_engineering/infrastructure/inference_pipeline_api.py`, `rag()` (line 37):

```
documents = ContextRetriever().search(query, k=3)     # retrieve
context   = EmbeddedChunk.to_context(documents)       # augment
answer    = call_llm_service(query, context)          # generate (SageMaker, chapter 10)
```

And inside `search` (`llm_engineering/application/rag/retriever.py`, line 29):

```
query  = Query.from_str(query)
query  = SelfQuery.generate(query)              # LLM call 1: extract the author → author_id
queries = QueryExpansion.generate(query, n=3)   # LLM call 2: original + 2 rewrites
for each query, in parallel:                    # thread pool
    embed it                                    # same model as the chunks
    search embedded_posts, _articles, _repositories   # k // 3 each, filtered by author_id
flatten, remove duplicates
Reranker.generate(original query, chunks, keep_top_k=k)   # cross-encoder
```

### One request, counted (with `k=3`)

| What | How many |
|---|---|
| OpenAI calls (`gpt-4o-mini`) | 2, one after the other, before any search |
| Query embeddings | 3 |
| Qdrant searches | 3 queries × 3 collections = 9, each with `limit = 1` |
| Candidate chunks | at most 9 |
| Cross-encoder scores | one per candidate |
| Chunks in the prompt | 3 |
| Generation calls | 1 (the fine-tuned TwinLlama on SageMaker) |

## The advanced RAG techniques, by stage

| Stage | Technique | File | What it fixes |
|---|---|---|---|
| Pre-retrieval | Self-querying | `self_query.py` | The author's name is metadata, not meaning. Searching for it by vector works badly; filtering on it works exactly |
| Pre-retrieval | Query expansion | `query_expanison.py` (sic) | One question is one point in vector space. Several phrasings cover more of the relevant area |
| Retrieval | Filtered vector search | `retriever.py`, `_search` | Only chunks by that author are searched |
| Post-retrieval | Reranking | `reranking.py` | The vector search is fast but rough; a cross-encoder rereads every candidate next to the question |

## Step 1 — the `Query` object

File: `llm_engineering/domain/queries.py`.

- `Query` is a `VectorBaseDocument` like the chunks: `content`, `author_id`, `author_full_name`, `metadata`.
- `EmbeddedQuery` adds `embedding`. That is how the chapter 4 `EmbeddingDispatcher` can embed a question: `QueryEmbeddingHandler` (`embedding_data_handlers.py`, line 45) is just one more handler.
- `replace_content` (line 20) makes a copy with new text but the **same** `author_id`. This is why self-querying runs *before* expansion: every rewritten query inherits the author filter.

## Step 2 — self-querying

File: `llm_engineering/application/rag/self_query.py`.

1. `SelfQueryTemplate` (`prompt_templates.py`, line 29) is a few-shot prompt: return the user name, the user id, or the word `none`.
2. `temperature=0`, so the same question gives the same answer.
3. If the answer is `none`, the query goes on with no filter (line 28).
4. Otherwise the name is split into first and last (`utils/split_user_full_name.py`: last word is the last name), and `UserDocument.get_or_create` (line 32) looks it up in **MongoDB**.
5. The user's id and full name are written into the query.

## Step 3 — query expansion

File: `llm_engineering/application/rag/query_expanison.py`.

- It asks for `expand_to_n - 1` rewrites (line 21) and puts the original question first (line 31). So `expand_to_n=3` means the original plus 2 rewrites.
- The rewrites come back in one string separated by `#next-question#` (`prompt_templates.py`, line 16), then are split and stripped.
- `prompt | model` is a LangChain chain: fill the template, send it to the model.

## Step 4 — filtered vector search

File: `llm_engineering/application/rag/retriever.py`, `_search` (line 63).

- `EmbeddingDispatcher.dispatch(query)` (line 89) embeds the query with `all-MiniLM-L6-v2`, the same model as the chunks. This is the reason chapter 4 stored the model id on every chunk.
- If there is an `author_id`, a Qdrant `Filter` with a `FieldCondition` on `author_id` is built (lines 69–79). The filter runs inside Qdrant, together with the vector search, not afterwards in Python.
- Each of the three collections is searched with `limit = k // 3` (line 85). So the candidates are split evenly between posts, articles and repositories, whatever the question.
- `search` in `domain/base/vector.py` (line 138) is the chapter 4 OVM method; `from_record` turns the results back into `EmbeddedArticleChunk` etc.

### Running the searches in parallel

`search`, lines 47–52:

- One thread per query with `ThreadPoolExecutor`. The work is waiting on Qdrant (I/O), so threads help even with Python's GIL.
- `as_completed` returns results in finishing order, not query order. That does not matter because the reranker sorts everything afterwards.
- `list(set(...))` removes duplicates. `VectorBaseDocument` defines `__eq__` and `__hash__` on `id` (`vector.py`, lines 24–31), and chunk ids are hashes of the chunk text (chapter 4), so the same chunk found by two queries counts once.

## Step 5 — reranking

File: `llm_engineering/application/rag/reranking.py`.

- The model is `cross-encoder/ms-marco-MiniLM-L-4-v2` (`settings.py`, line 64), loaded once by `CrossEncoderModelSingleton` (`networks/embeddings.py`, line 111).
- It scores `(question, chunk)` pairs, sorts by score, and keeps `keep_top_k`.
- The reranker uses the **original** question (`retriever.py`, line 57), not the rewrites.

### Bi-encoder vs cross-encoder

| | Bi-encoder (embedding model) | Cross-encoder (reranker) |
|---|---|---|
| Input | One text at a time | Question and chunk together |
| Output | A vector | One relevance score |
| Can precompute? | Yes, chunk vectors are stored in Qdrant | No, it must run at query time for every pair |
| Speed | Fast over millions of chunks | Slow, so only for a short list |
| Quality | Rougher | Better: it sees the words of both texts at once |

That is why the order is "search many cheaply, then rerank a few carefully".

## Step 6 — augment and generate

- `EmbeddedChunk.to_context` (`domain/embedded_chunks.py`, line 20) turns the chunks into one string: chunk number, type, platform, author, content.
- `InferenceExecutor` (`model/inference/run.py`, line 20) puts the query and the context into a short "You are a content creator" prompt and calls the SageMaker endpoint. Deployment is chapter 10.
- `rag()` also records token counts for query, context and answer in Opik (lines 44–54).

## Supporting pieces

| Piece | Where | Why |
|---|---|---|
| `RAGStep` base class | `rag/base.py` | Every step has `generate(query, ...)` and a `mock` flag |
| `mock=True` | each step | Skips the OpenAI calls, so the pipeline can be tested for free |
| `@opik.track` | each `generate` | Every step's input and output shows up as one trace in Opik (prompt monitoring, chapter 11) |
| `PromptTemplateFactory` | `rag/base.py` | Prompts live in one file, `prompt_templates.py`, not inside the logic |

## Design patterns to name in an interview

| Pattern | Where | What it gives |
|---|---|---|
| Template method / common interface | `RAGStep.generate` | All steps look the same; each can be swapped or mocked |
| Facade | `ContextRetriever.search` | One call hides four steps |
| Singleton | `CrossEncoderModelSingleton` | The reranker loads once |
| Factory | `PromptTemplateFactory` | Prompts are built in one place |
| Dispatcher, reused from chapter 4 | `EmbeddingDispatcher` for queries | The question is embedded exactly like the chunks |
| Fan-out / fan-in | Thread pool + `set` | N searches in parallel, merged and deduplicated |

## Things I noticed that are worth knowing

These come from reading the code. The ones marked "to verify" need a run.

1. **Searching can write to MongoDB.** `SelfQuery` uses `get_or_create` (`self_query.py`, line 32). A name that is not in the database, misspelled, or made up by the LLM creates a new empty user. The filter then points at a user with no chunks, so the search returns nothing, and the run still looks normal. Same problem as entry 4.2, now on the read path.
2. **`none` must match exactly.** Line 28 compares with `"none"`. If the model answers `None` or `none.`, that string is treated as a name, and a user called "None" is created.
3. **The user-id branch is never handled.** The prompt can return an id like `1345256`, but the code always treats the answer as a full name, so it creates a user whose first and last name are both `1345256`. Real ids are UUIDs anyway.
4. **The LLM may see the whole `Query` object, not just the question (to verify).** Both steps call `chain.invoke({"question": query})` with the `Query` model, not `query.content` (`self_query.py` line 25, `query_expanison.py` line 26). If LangChain formats it with `str()`, the prompt contains `content='...' author_id=... metadata={}`. `tools/rag.py` turns on verbose mode, so one run will show the real prompt.
5. **The quota is `k // 3` per collection.** `k=3` gives 1 chunk per type per query; `k=4` and `k=5` give the same. `k` below 3 fails the `assert` inside a thread (line 64). A question about code still gets post and article candidates; the reranker can drop them, but it cannot add more repository chunks.
6. **A missing collection is silently empty.** `search` in `vector.py` (lines 141–144) logs and returns `[]` on an error. Posts are never crawled (entry 3.2), so a third of every search finds nothing, without anyone seeing it.
7. **Two LLM calls happen in sequence before any search.** Self-query and expansion could run at the same time, with the author copied into the rewrites afterwards. That saves one round trip on every request.
8. **Two English-only models.** `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-4-v2` are both English. This matters for Arabic in week 5.
9. **The reranker's input limit (to verify).** Article chunks go up to 2,000 characters. With the question added, the pair may be longer than the cross-encoder reads, so the end of a chunk might not count.
10. **Small things.** `query_expanison.py` is misspelled. `Reranker` loads the model even when `mock=True`. `zip(..., strict=False)` (`reranking.py`, line 24) would hide a length mismatch, like entry 4.14.

## Run it

Needs: MongoDB and Qdrant running (`poetry run poe local-infrastructure-up`), the chapter 4 feature pipeline already run, and `OPENAI_API_KEY` in `.env`.

```bash
poetry run poe call-rag-retrieval-module
```

This runs `tools/rag.py`: the question "My name is Paul Iusztin… draft a LinkedIn post discussing RAG systems", with `k=9`. It prints each step's prompt (verbose mode) and the 9 chunks.

Write down:

- the author that was extracted, and the 3 search queries
- how many chunks came back before reranking ("N documents retrieved")
- the type and the first line of each of the 9 final chunks

### Experiments (change one thing, run again, write down what changed)

| Experiment | Change | What to record |
|---|---|---|
| A | In `tools/rag.py`, remove "My name is Paul Iusztin." | Whether the filter disappears and chunks by other authors appear |
| B | Change the name to one that does not exist, e.g. "Paul Iusztinn" | Number of chunks returned, and whether a new user now exists in MongoDB `users` (tests observation 1) |
| C | `ContextRetriever(mock=True)` | Which chunks come back with no expansion, no author, no reranking. Compare with the normal run |
| D | `expand_to_n_queries=1`, then 5 | Candidates before reranking; whether the final list changes |
| E | Look at the verbose output of the self-query prompt | Whether the prompt shows the question only, or the whole `Query` object (tests observation 4) |

Delete any user created by experiment B afterwards, and undo each change (`git checkout -- <file>`).

## For my WhatsApp twin

| | Book's inference pipeline | What I would need |
|---|---|---|
| Self-query filter | Author | Contact or chat, and a date range |
| Models | English-only embedding and reranker | Multilingual ones (week 5 candidates in `chapter-3-4-tasks.md`, Part 4) |
| Where the question goes | To OpenAI twice (self-query, expansion) | My questions may contain private details, so a local model for those two steps, or no expansion |
| Where the context goes | To my own SageMaker endpoint | Must stay local or on my own endpoint, never a third-party API |

## Explain-back questions

The flow:

1. Name the steps between the user's question and the 3 chunks in the prompt, in order.
2. Which steps call an LLM, and which call no model at all?
3. Why does self-querying run before query expansion?

Pre-retrieval:

4. Why filter on the author instead of leaving the name in the text and letting the vector search find it?
5. What does query expansion fix? What does it cost per request?
6. What happens when the question names no author?

Retrieval:

7. Why must the question be embedded with the same model as the chunks?
8. Why are threads useful here despite the GIL?
9. The same chunk is found by two of the queries. Why does it end up in the list once?
10. With `k=3`, how many candidates can there be at most? Where does that number come from?

Post-retrieval:

11. Explain bi-encoder vs cross-encoder. Why not use the cross-encoder for the whole search?
12. Why does the reranker score the original question and not the rewrites?

Judgement:

13. Pick one of the ten observations above. Explain the problem and how I would fix it.
14. How would I measure whether query expansion and reranking actually help? (Hint: week 5, recall@5.)
