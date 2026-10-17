# Study guide 0 — Architecture, tooling and data collection (book chapters 1–3)

This is the book's code, not mine. The goal is to explain it as someone who studied and ran it.

## Block 1 — Architecture (chapter 1)

### The one-sentence version

The system is three separate pipelines (feature, training, inference) that only talk to each other through stored data, never by calling each other's code.

### The picture to draw from memory

```
links ──► Data collection (ETL) ──► MongoDB          "data warehouse": raw documents
                                      │
                                      ▼
                    Feature pipeline: clean ─► chunk ─► embed ──► Qdrant   "feature store"
                                      │                              │
                                      ▼                              │
                    Training pipeline: build dataset ─► fine-tune ─► model registry
                                                                     │
                                                                     ▼
                    Inference pipeline: question ─► retrieve from Qdrant ─► prompt ─► model ─► answer
```

### Why three pipelines and not one script

| Reason | What it means in practice |
|---|---|
| Each part can change alone | A new crawler does not touch training. A new model does not touch crawling. |
| Each part runs on different hardware | Crawling needs a CPU and a browser. Training needs a GPU for hours. Inference needs a GPU that answers in seconds. |
| Training and inference read the same features | Both read from Qdrant, so the model is not trained on data prepared one way and served with data prepared another way. This problem is called training–serving skew. |
| A failure stays local | If crawling fails, the last good data is still in the stores and the other pipelines keep working. |

### Where my own work fits

My WhatsApp Twin replaces the **training pipeline's data and training** (dataset, fine-tuning, DPO, evaluation). The data collection, feature and inference pipelines are the book's.

## Block 2 — Tooling (chapter 2)

### What each tool does, in one sentence

| Tool | Job |
|---|---|
| Poetry | Installs the exact library versions the project needs, in its own environment. |
| Poe the Poet | Short names for long commands. `poe run-digital-data-etl` is defined in `pyproject.toml` (line 69 onward). |
| Docker Compose | Starts MongoDB and Qdrant on my machine with one command (`docker-compose.yml`). |
| MongoDB | Stores raw crawled documents as JSON-like records. The "data warehouse". |
| Qdrant | Stores cleaned documents and embedded chunks, and searches them by meaning. The "feature store". |
| ZenML | Runs a pipeline as ordered steps, saves each step's output, and shows the runs in a dashboard. |
| Comet ML / Opik | Comet records training runs. Opik records prompts and answers at inference time. |
| Hugging Face | Hosts datasets and models. The "model registry". |
| AWS SageMaker | Rents GPUs for training and for serving the model. |

### Settings: `llm_engineering/settings.py`

- One `Settings` class reads values from the `.env` file (line 8).
- The API keys default to `None` (lines 14, 17, 20), so the data collection pipeline runs without them.
- MongoDB and Qdrant default to the local Docker containers (lines 27 and 32–33). The MongoDB username and password in the default are the book's public local-only values, the same ones in `docker-compose.yml`.

### Setup steps for today

1. In Docker Desktop on Windows: Settings → Resources → WSL integration → turn on **Ubuntu** → Apply. Then check in the WSL terminal that `docker --version` prints a version.
2. In the repository: `cp .env.example .env`. Leave the placeholder values for now.
3. `poetry install` if the environment for this folder does not exist yet. The project needs Python 3.11 (`.python-version` says 3.11.8). The system Python in WSL is 3.14, so Poetry must be pointed at a 3.11 interpreter. An older environment named `llm-engineering-...-py3.11` exists on this machine, which suggests 3.11 was available before.
4. `poetry run poe local-infrastructure-up` starts MongoDB, Qdrant and the ZenML server.
5. Open the ZenML dashboard at the address the command prints.

If step 1 or 3 fails, send me the error. This block can take longer than two hours the first time.

## Block 3 — Data collection (chapter 3)

### The whole flow

`pipelines/digital_data_etl.py` is 11 lines and has two steps:

```
digital_data_etl(user_full_name, links)
    user = get_or_create_user(user_full_name)     # steps/etl/get_or_create_user.py
    crawl_links(user=user, links=links)           # steps/etl/crawl_links.py
```

The inputs come from a config file, for example `configs/digital_data_etl_paul_iusztin.yaml`: one author name and a list of links.

### Step 1 — `get_or_create_user`

- Splits the full name into first and last name.
- Calls `UserDocument.get_or_create(...)`: find the user in MongoDB, or create them.
- Every crawled document is later stored with this user's id as `author_id`.

### Step 2 — `crawl_links` and the dispatcher

`CrawlerDispatcher` (`llm_engineering/application/crawlers/dispatcher.py`) answers one question: **which crawler class handles this link?**

- `register(domain, crawler)` (line 38) stores a pattern such as `https://(www\.)?medium\.com/*` mapped to a crawler **class**.
- `get_crawler(url)` (line 44) tries each pattern. The first match returns a new crawler object. No match returns `CustomArticleCrawler`, the default.
- `crawl_links` builds it by chaining: `CrawlerDispatcher.build().register_linkedin().register_medium().register_github()` (line 14).

Then for each link (`_crawl_link`, line 34): get the crawler, call `crawler.extract(link=link, user=user)`, and catch any exception so one bad link does not stop the run.

### The crawlers

| Class | File | Used for | How it reads the page |
|---|---|---|---|
| `BaseCrawler` | `base.py` line 17 | The contract: every crawler has a `model` and an `extract` method | — |
| `BaseSeleniumCrawler` | `base.py` line 24 | Sites that need a real browser | Headless Chrome, with scrolling |
| `MediumCrawler` | `medium.py` | Medium articles | Selenium |
| `LinkedInCrawler` | `linkedin.py` | LinkedIn posts | Selenium with login |
| `GithubCrawler` | `github.py` | Repositories | Clones the repository |
| `CustomArticleCrawler` | `custom_article.py` | Everything else, for example Substack | Plain HTML download, converted to text |

`CustomArticleCrawler.extract` (line 18) shows the pattern all crawlers follow:

1. Check whether this link is already stored (`self.model.find(link=link)`). If yes, stop.
2. Download and convert the page.
3. Build an `ArticleDocument` with the content, the link, the platform and the author.
4. `instance.save()` writes it to MongoDB.

### The document classes

`llm_engineering/domain/base/nosql.py` is a small layer between Python objects and MongoDB. The book calls it an ODM (object-document mapper).

- `NoSQLBaseDocument` (line 19) is a Pydantic model with an `id`.
- `to_mongo` (line 42) turns an object into a dictionary and renames `id` to `_id`. `from_mongo` (line 32) does the reverse.
- `save`, `find`, `bulk_find`, `get_or_create`, `bulk_insert` are the database operations.
- `get_collection_name` (line 133) reads the collection name from an inner `Settings` class. A document class without it raises an error.

`llm_engineering/domain/documents.py` defines the real documents:

```
NoSQLBaseDocument
├── UserDocument            collection "users"
└── Document (abstract)     content, platform, author_id, author_full_name
    ├── RepositoryDocument  collection "repositories"
    ├── PostDocument        collection "posts"
    └── ArticleDocument     collection "articles"
```

The connection itself is in `llm_engineering/infrastructure/db/mongo.py`: `MongoDatabaseConnector` creates one `MongoClient` and returns the same one every time (the Singleton pattern).

### Design patterns to name in an interview

| Pattern | Where | What it gives |
|---|---|---|
| Dispatcher / Strategy | `CrawlerDispatcher` + crawler classes | The pipeline does not know which site it is crawling. |
| Builder-style chaining | `.build().register_...()` | Readable setup of which crawlers are active. |
| Template (abstract base class) | `BaseCrawler`, `Document` | Every crawler and document has the same shape. |
| Singleton | `MongoDatabaseConnector` | One database connection for the whole program. |
| ODM with generics | `NoSQLBaseDocument(Generic[T])` | `ArticleDocument.find(...)` returns an `ArticleDocument`, not a plain dictionary. |

### Run it

```bash
poetry run poe run-digital-data-etl
```

Then look in MongoDB (database `twin`): the `users`, `articles` and `repositories` collections. Open one article and find `author_id`, `platform`, `link` and `content`.

Things that may go wrong: Medium and LinkedIn crawling needs Chrome inside WSL, and paywalled or changed pages fail. A run that reports fewer successes than links is normal. If crawling fails badly, `poetry run poe run-import-data-warehouse-from-json` loads the book's saved copy of the data into MongoDB, which is enough for tomorrow.

### Compared with my own data preparation

| | Book's data collection | My `twin_dataset_builder.py` |
|---|---|---|
| Source | Public web pages and repositories | One private WhatsApp export |
| Output | Raw documents in MongoDB | Finished train and validation files |
| Cleaning | Almost none here; it happens in the feature pipeline | Done in the same program: redaction, merging, filtering |
| Structure | Many small classes, orchestrated by ZenML | One class, standard library only |
| Why | Built to grow: new sources, many authors | Built to be auditable and to keep private data local |

## Things I noticed that are worth knowing

- `crawl_links` returns **all** links (line 31), not only the ones that succeeded. The success count is only in the step's metadata.
- Deduplication is by exact link. The same article under two different URLs is stored twice.
- `get_crawler` uses `for ... else`. The `else` part runs only when the loop finishes without returning.
- The first import of `nosql.py` opens the MongoDB connection (line 13), so importing a document class fails if the database is down.

## Explain-back questions

Architecture:

1. What are the three pipelines, and what does each one read and write?
2. Give two reasons the pipelines are separate programs and not one script.
3. What is training–serving skew, and how does the shared feature store prevent it?

Tooling:

4. What is the difference between MongoDB's job and Qdrant's job in this system?
5. What does ZenML add over running the steps as plain Python functions?
6. Why can the data collection pipeline run with no API keys in `.env`?

Data collection:

7. A new link `https://example.com/post` arrives. Follow it through `get_crawler`: which crawler handles it, and why?
8. What would I change to support a new site, for example Dev.to? Which files, and what stays untouched?
9. How does `ArticleDocument` know which MongoDB collection to use?
10. What happens when the same link is crawled a second time?
11. One link out of fifty raises an exception. What happens to the other forty-nine, and where would I see that one failed?
12. Why does `MongoDatabaseConnector` use a Singleton? What would go wrong without it?
13. Name one thing the book's data collection does better than my data preparation, and one thing mine does better.
