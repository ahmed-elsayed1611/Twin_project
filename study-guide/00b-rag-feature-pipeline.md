# Study guide 0b — The RAG feature pipeline (book chapter 4)

This is the book's code, not mine. The goal is to explain it as someone who studied and ran it.

## The one-sentence version

The feature pipeline reads raw documents from MongoDB, cleans them, cuts them into chunks, turns each chunk into a vector, and stores both the cleaned documents and the embedded chunks in Qdrant.

## The whole flow is 10 lines

`pipelines/feature_engineering.py`:

```
raw_documents     = query_data_warehouse(author_full_names)      # MongoDB  → Python objects
cleaned_documents = clean_documents(raw_documents)               # remove noise
                    load_to_vector_db(cleaned_documents)         # → Qdrant, no vectors
embedded_documents = chunk_and_embed(cleaned_documents)          # cut + embed
                    load_to_vector_db(embedded_documents)        # → Qdrant, with vectors
```

Notice the two branches after cleaning. Cleaned documents are saved once as they are, and once more after chunking and embedding.

### Why Qdrant holds both

| Stored | Collections | Vector index | Who reads it |
|---|---|---|---|
| Cleaned documents | `cleaned_posts`, `cleaned_articles`, `cleaned_repositories` | No (`use_vector_index = False`) | The training pipeline, to build the instruction dataset |
| Embedded chunks | `embedded_posts`, `embedded_articles`, `embedded_repositories` | Yes, cosine distance | The inference pipeline, for retrieval |

Training needs whole clean texts. Retrieval needs small pieces that can be searched by meaning. Both come from the same cleaning step, so training and inference never see differently-cleaned data. Qdrant is used as the single feature store for both.

## The data changes shape four times

```
ArticleDocument          (MongoDB)     content is a dict: Title, Subtitle, Content
      │ clean
CleanedArticleDocument   (Qdrant)      content is one clean string
      │ chunk
ArticleChunk             (in memory)   one piece of the text + document_id
      │ embed
EmbeddedArticleChunk     (Qdrant)      the piece + its vector + embedding metadata
```

The same four states exist for posts and repositories. That is 3 data types × 4 states. The classes live in `llm_engineering/domain/`: `documents.py`, `cleaned_documents.py`, `chunks.py`, `embedded_chunks.py`.

## Step 1 — `query_data_warehouse`

File: `steps/feature_engineering/query_data_warehouse.py`.

- For each author name it finds the user, then fetches that user's articles, posts and repositories.
- `fetch_all_data` (line 37) runs the three MongoDB queries at the same time with a thread pool. If one query fails, it logs the error and uses an empty list for that type (lines 51–54).
- The step also records metadata (how many documents, which authors) that shows up in the ZenML dashboard.

## Step 2 — `clean_documents` and the dispatcher pattern

This pattern is used three times (cleaning, chunking, embedding), so learn it once here.

File: `llm_engineering/application/preprocessing/dispatchers.py`.

```
CleaningDispatcher.dispatch(document)
    1. data_category = the document's collection name      → "articles"
    2. handler = CleaningHandlerFactory.create_handler(...) → ArticleCleaningHandler()
    3. return handler.clean(document)                       → CleanedArticleDocument
```

Three roles:

| Role | Class | Job |
|---|---|---|
| Handler | `ArticleCleaningHandler` etc. (`cleaning_data_handlers.py`) | Knows how to clean **one** data type |
| Factory | `CleaningHandlerFactory` (line 27) | Picks the right handler for a category |
| Dispatcher | `CleaningDispatcher` (line 40) | The one entry point the pipeline calls |

The pipeline step (`steps/feature_engineering/clean.py`, line 14) only calls `CleaningDispatcher.dispatch(document)`. It has no `if article ... elif post ...`. All type-specific logic is inside the handlers.

What cleaning actually does (`operations/cleaning.py`, 5 lines of logic):

1. The handler joins the values of the `content` dict with `" #### "`.
2. `clean_text` replaces every character that is not a letter, digit, underscore, whitespace or `. , ! ?` with a space.
3. It collapses all whitespace, including newlines, into single spaces.

## Step 3 — `chunk_and_embed`

File: `steps/feature_engineering/rag.py`.

For each cleaned document: chunk it, then embed the chunks in batches of 10 (line 21).

### Chunking: different strategy per data type

File: `chunking_data_handlers.py`.

| Data type | Function | Settings | Why |
|---|---|---|---|
| Posts | `chunk_text` | `chunk_size` 250, overlap 25 | Posts are short |
| Articles | `chunk_article` | `min_length` 1000, `max_length` 2000 characters | Articles are long prose; split on sentence ends so no sentence is cut |
| Repositories | `chunk_text` | `chunk_size` 1500, overlap 100 | Code needs more context per piece |

`chunk_article` (`operations/chunking.py`, line 32) splits the text into sentences with a regex, then adds sentences to the current chunk until the next one would pass `max_length`. A chunk is kept only if it is at least `min_length` long.

`chunk_text` (line 10) has two stages: first split on blank lines, then split by token count so no chunk is longer than the embedding model can read.

**Chunk ids are a hash of the chunk's text** (`hashlib.md5(chunk.encode())`, line 55). The same text always gets the same id.

Each chunk keeps `document_id`, the id of the document it came from, so a retrieved chunk can be traced back to its source.

### Embedding

File: `embedding_data_handlers.py`.

- `embed_batch` (line 29) takes the `content` of each chunk, sends the whole batch through the embedding model, and pairs each chunk with its vector.
- The model is `sentence-transformers/all-MiniLM-L6-v2` (`settings.py`, line 63). It runs on CPU and produces 384 numbers per chunk.
- `EmbeddingModelSingleton` loads the model once and reuses it.
- Each embedded chunk stores which model made its vector (`embedding_model_id`, `embedding_size`).
- There is also a `QueryEmbeddingHandler` (line 45). The inference pipeline uses the same dispatcher to embed the user's question, so questions and chunks are always embedded by the same model.

## Step 4 — `load_to_vector_db` and the Qdrant layer

File: `steps/feature_engineering/load_to_vector_db.py`.

- Groups the documents by class, because each class has its own collection.
- Inserts in batches of 4. If a batch fails, the step returns `False`.

`llm_engineering/domain/base/vector.py` is the Qdrant twin of the MongoDB layer from chapter 3. The book calls it an OVM (object-vector mapper).

- `to_point` (line 47): turns an object into a Qdrant point. The `embedding` field becomes the vector; every other field becomes the payload.
- `from_record` (line 34): the reverse.
- `bulk_insert` (line 80): tries to insert; if the collection does not exist, creates it and tries again.
- `_bulk_insert` uses **upsert** (line 103): insert, or overwrite if the id already exists.
- `_create_collection` (line 188): uses cosine distance and the embedding model's vector size.
- `search` (line 138): vector similarity search, used at inference time.
- Each document class has an inner `Config` with `name` (collection), `category` and `use_vector_index`.

Upsert plus hash-based chunk ids means running the pipeline twice does not create duplicates.

## Design patterns to name in an interview

| Pattern | Where | What it gives |
|---|---|---|
| Strategy | The handler classes | One class per way of cleaning, chunking or embedding |
| Factory | `...HandlerFactory.create_handler` | One place that decides which handler to use |
| Dispatcher | `...Dispatcher.dispatch` | The pipeline code stays free of type checks |
| Singleton | `EmbeddingModelSingleton`, the Qdrant connection | The model and the connection are created once |
| Template method | `EmbeddingDataHandler.embed_batch` calls the abstract `map_model` | Shared batching logic; each type only says how to build its own result object |
| OVM with generics | `VectorBaseDocument(Generic[T])` | `EmbeddedArticleChunk.search(...)` returns `EmbeddedArticleChunk` objects |

### Adding a new data type, honestly

To support a new type, for example "videos", I would add: four domain classes, three handlers, one branch in each of the three factories, and one value in `DataCategory`. The pipeline file and the four steps stay untouched. So it is not "one class", but none of the orchestration changes.

## Things I noticed that are worth knowing

These come from reading the code. Check the first one by experiment in Block 2.

1. **The first stage of `chunk_text` probably does nothing.** It splits on `"\n\n"`, but `clean_text` has already turned every newline into a space. So the real splitting is done by the token splitter, at the embedding model's limit. If that is right, changing `chunk_size` for posts or repositories will not change the chunks.
2. **Short articles disappear.** `chunk_article` drops any chunk under 1,000 characters, including the last part of a long article and the whole of an article shorter than that.
3. **Cleaning is harsh on code.** `clean_text` removes brackets, quotes, operators and slashes. A repository's code loses most of its syntax before it is embedded.
4. **Identical chunks collapse.** Two documents containing the same chunk text get the same chunk id, and the second overwrites the first.
5. **Chunking and embedding are one step.** If embedding fails, chunking is redone too. The book separates them in the code (two dispatchers) but not in the pipeline.

## Run it

```bash
poetry run poe run-feature-engineering-pipeline
```

Then open the Qdrant dashboard at `http://localhost:6333/dashboard`:

- Open `cleaned_articles` and read one document's `content`. Compare it with the same article in MongoDB.
- Open `embedded_articles` and look at one point: its payload, its `document_id`, and its vector length.
- Count how many chunks one article produced.

### Experiments (change one thing, run again, write down what changed)

| Experiment | Change | What to record |
|---|---|---|
| A | In `ArticleChunkingHandler.metadata`, set `max_length` to 1000 and `min_length` to 500 | Number of points in `embedded_articles` before and after |
| B | In `PostChunkingHandler.metadata`, set `chunk_size` to 100 | Whether the number of post chunks changes at all (tests observation 1) |
| C | Run the pipeline twice with no change | Whether the point counts stay the same (tests upsert + hash ids) |

Undo each change after recording the result (`git checkout -- <file>`).

## Compared with my own data preparation

| | Book's feature pipeline | My `twin_dataset_builder.py` |
|---|---|---|
| Unit of data | A chunk of an article | A conversation window ending in my reply |
| Cleaning | One regex for every type | Rules for WhatsApp: system notices, edit tags, PII redaction |
| Splitting | By sentence or token count | By turns, with a session boundary so context does not cross conversations |
| Ids | Hash of the chunk text | Hash of the whole sample (context + reply) |
| Self-check | Metadata counts in ZenML | Re-reads its own output and fails the build if a record is invalid |
| Output | Vectors for retrieval, clean text for training | Training files only; no retrieval yet |

## Explain-back questions

The flow:

1. Name the four states a document passes through, and where each one is stored.
2. Why does the pipeline save cleaned documents to Qdrant with no vector index?
3. What would go wrong if training read raw documents from MongoDB and inference read cleaned chunks from Qdrant?

The dispatcher pattern:

4. Explain handler, factory and dispatcher in one sentence each.
5. `clean_documents` receives a list with articles and posts mixed. How does each one reach the right handler?
6. What exactly do I add to support a new data type, and what stays untouched?

Chunking and embedding:

7. Why are articles chunked by sentence and posts by size?
8. What is chunk overlap for? What is lost with zero overlap?
9. Why must the question at inference time be embedded with the same model as the chunks?
10. Why does each embedded chunk store the embedding model's name?
11. What does `document_id` on a chunk allow me to do?

Qdrant:

12. What is the difference between a point's vector and its payload?
13. I run the pipeline twice on the same data. Why are there no duplicates?
14. Why cosine distance, and what does it measure?

Judgement:

15. Pick one of the five observations above. Explain the problem and how I would fix it.
16. After `clean_text`, what does a line of Python code look like? Is that a good input for an embedding model?
