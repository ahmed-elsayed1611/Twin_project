# LLM Twin

An end-to-end LLM system that learns how a person writes and generates new text in that voice — a production-style RAG + fine-tuning pipeline, not a notebook demo.

This repository has two parts:

| Part | What it is | Where |
|---|---|---|
| **The book's system** | The reference implementation from *LLM Engineer's Handbook*: data collection, feature pipeline, training and inference. I studied it line by line and run it as written. | the repository root |
| **My work: the WhatsApp Twin** | The fine-tuning track, rebuilt on a real private Egyptian-Arabic conversation in place of the book's public articles: data preparation, supervised fine-tuning, preference alignment, and evaluation. | [`whatsapp-twin/`](whatsapp-twin) |

> **A note on origin**: this project's architecture and implementation are based on *LLM Engineer's Handbook* (Iusztin & Labonne). I didn't design this system from scratch — I studied it in depth, line by line, rebuilt it, and understood the reasoning behind every design decision, before extending it with my own contributions (see below). I'm sharing it as proof of that understanding, not as original design work.

Studied and built by [Ahmed Elsayed](https://github.com/ahmed-elsayed1611) · [LinkedIn](https://linkedin.com/in/ahmed-elsayed16112002)

## My work: the WhatsApp Twin

A raw WhatsApp export becomes a 3B-parameter chat model that replies the way I do. Everything in [`whatsapp-twin/`](whatsapp-twin) is my own code.

| Stage | What I built | Result |
|---|---|---|
| [Data preparation](whatsapp-twin/data-preparation) | Parser, cleaner, PII redaction, session-level split, self-validation. Python standard library only | 135,289 raw lines → 26,989 multi-turn samples (24,915 train / 2,074 validation) |
| [Supervised fine-tuning](whatsapp-twin/supervised-fine-tuning) | LoRA fine-tuning of NileChat-3B on multi-turn chat data | Best validation loss 1.625, against 1.734 for Qwen2.5-3B |
| [Preference alignment](whatsapp-twin/preference-alignment) | DPO with hard negatives mined from my own real replies | Held-out preference accuracy 68.7% → 71.3% (paired McNemar, p ≈ 0.02–0.04) |
| [Evaluation](whatsapp-twin/evaluation) | SFT vs. DPO on 150 unseen conversations, with an LLM judge run in both A/B orders | Judge prefers DPO in 71% of decided pairs (p = 0.0003) |
| [Deployment](#deployment-and-monitoring) | The DPO model on a SageMaker CPU endpoint behind `POST /reply`, with CI/CD and Opik tracing | Serves the twin without a GPU quota |

Three things I would point an interviewer to:

- **I changed the method when the metric was too good.** The first DPO run reached 100% preference accuracy by step 36 of 180. The model had only learned to avoid AI-sounding text. I rebuilt the preference dataset from my own real replies in the wrong context, which removed the shortcut.
- **Every comparison is paired and tested.** The test set is never used for a training decision, and each result comes with a significance test. One metric goes against the final model (perplexity 14.9 → 15.5), and it is reported with the rest.
- **The data is private, and the repository is built around that.** No data, model weights, or generated outputs are committed. The example messages in the code and tests are synthetic.

The full write-up, the pipeline diagram, and the history of what failed are in the [WhatsApp Twin README](whatsapp-twin/README.md).

## Why this repo matters

I came into this project self-taught, transitioning into AI/ML engineering. Rather than stopping at tutorials, I chose a genuinely production-grade codebase and went through it the way an engineer maintaining it would have to: understanding every design pattern (Factory, Strategy, Singleton, Template Method, Builder), every type-safety decision (Python generics across a 9-class domain hierarchy), and every architectural trade-off (batch vs. streaming, why a feature store exists, why cleaning/chunking/embedding are separated into their own layers) — not just running cells top to bottom.

That's the skill I'd bring to a team: the ability to drop into an unfamiliar, non-trivial codebase, actually understand it at the architecture level, and extend it correctly — which is most of what real ML engineering work looks like day to day, far more than greenfield model-building.

## Architecture

The system follows the **FTI (Feature / Training / Inference) pipeline** pattern — three independently deployable pipelines connected through a shared data layer, not one monolithic script.

```
Data Collection (ETL)  →  MongoDB (data warehouse)
        ↓
RAG Feature Pipeline   →  Qdrant (vector DB / feature store)
        ↓
Training Pipeline      →  Fine-tuned LLM
        ↓
Inference Pipeline     →  RAG-augmented generation API
```

### 1. Data Collection (ETL)
Crawls articles, posts, and code repositories from various platforms, standardizes them, and loads them into a MongoDB data warehouse. Built around a `CrawlerDispatcher` that routes each source to the right crawler implementation.

### 2. RAG Feature Pipeline
A batch pipeline, orchestrated with ZenML, that:
- Queries the data warehouse for new/updated documents
- Cleans, chunks, and embeds them
- Loads both the cleaned documents and the embedded chunks into Qdrant, which serves as the logical feature store

Built on a custom object-vector mapping (OVM) layer (`VectorBaseDocument`) and a dispatcher/handler system (Abstract Factory + Strategy patterns) that applies different cleaning, chunking, and embedding logic per data category (posts, articles, repositories) without branching logic scattered through the pipeline.

### 3. Training Pipeline
Builds instruction datasets from the cleaned documents in the feature store and fine-tunes an LLM (via PEFT/TRL) to reproduce a specific writing style and voice.

This is the stage I rebuilt on my own data. See [`whatsapp-twin/`](whatsapp-twin).

### 4. Inference Pipeline
Serves the fine-tuned model behind an API, using RAG at query time — embedding the user's question, retrieving relevant chunks from Qdrant, and augmenting the prompt before generation.

## Repository structure

```
.
├── whatsapp-twin/      # my work: data preparation, fine-tuning, preference alignment, evaluation
├── llm_engineering/    # the book's core package: domain, application, infrastructure, model
├── pipelines/          # ZenML pipelines
├── steps/              # ZenML steps
├── configs/            # pipeline configurations
├── tools/              # CLI entry points: run pipelines, RAG, model service
├── data/               # the book's sample data (public articles and repositories)
└── tests/
```

## Stack

| Layer | Tools |
|---|---|
| Orchestration | ZenML |
| Data warehouse | MongoDB |
| Vector DB / feature store | Qdrant |
| Experiment & prompt tracking | Comet ML, Opik |
| Training infra | AWS SageMaker (book), Google Colab T4 (WhatsApp Twin) |
| Model & fine-tuning | Hugging Face, TRL, PEFT, Unsloth |
| RAG / retrieval | LangChain |
| Serving | FastAPI, AWS SageMaker (CPU endpoint) |
| CI/CD | GitHub Actions, Docker, Amazon ECR |

## Design principles

- **Single source of truth**: the feature store (Qdrant + ZenML artifacts) is the only thing training and inference pipelines read from — never the raw data warehouse directly.
- **Category-agnostic core, category-specific handlers**: adding a new data source means writing one new handler class, not touching the pipeline logic.
- **Batch over streaming, by choice**: data volume and latency requirements don't justify streaming complexity here — documented trade-off, not a limitation.

## Status

This project is still in progress.

| Stage | Status |
|---|---|
| Architecture, tooling, data collection, RAG feature pipeline | Studied, run, and fixed (see below) |
| Data preparation | Done — my own implementation |
| Supervised fine-tuning | Done — my own implementation |
| Preference alignment | Done — my own implementation |
| Evaluation | Done — my own implementation |
| RAG inference pipeline | Studied, run, and fixed (see below) |
| Inference pipeline deployment | Done: my DPO twin on a SageMaker CPU endpoint behind `POST /reply` |
| MLOps and LLMOps | Done: CI (lint, format, secret scan, tests), CD (Docker image to Amazon ECR), Opik tracing |
| RAG quality | In progress: evaluation set (recall@5, MRR), hybrid search, query latency |

### Next
- A retrieval evaluation set, so every RAG change is measured instead of guessed
- Hybrid search (BM25 + dense vectors, merged with reciprocal rank fusion)
- Cut the 61-second self-query step that Opik traces showed as the main source of latency

## What I changed in the book's code (data collection and RAG feature pipeline)

I ran the pipelines, checked the data in MongoDB and Qdrant, and fixed what I found. Each fix has a unit test in `tests/unit/`. Data: 76 articles by the two authors.

**The pipeline reported success when it had failed**
- A crawl with most links failing now fails the step (under 80% success). A crawler that cannot even start counts as one failed link.
- An error page ("Page not found") is rejected instead of saved as an article. All 45 of one author's newsletter links had become error pages and were stored as articles.
- A failed MongoDB query, Qdrant insert or embedding call now stops the run. Before, each one was logged and turned into empty data, and the run ended green. With Qdrant stopped, the old pipeline reported success with nothing saved.
- A misspelled author name stops the run instead of creating a new empty user.

**Data was lost without any message**
- Chunk ids were the hash of the text alone. Four articles share the same intro paragraph, so their chunks overwrote each other: 500 chunks sent, 497 stored. Ids now include the document id: 500 of 500 stored.
- Article chunks were up to 2,000 characters, but the embedding model (`all-MiniLM-L6-v2`) reads only 256 tokens and silently cuts the rest: 97% of chunks were too long and 41% of the text was never embedded. Chunks are now measured in the model's tokens: 0% over the limit.
- Text under 1,000 characters at the end of an article was dropped (39 of 76 articles lost text). Now every word is kept.

**Cleaning**
- Links become `[URL]` instead of noise like `https example.com`, and paragraph breaks are kept. This also makes the post and repository splitters' `chunk_size` work: before, it had no effect because there were no breaks left to split on.
- Only the title, subtitle and body are used: a stray `en` (the language field) was glued to the end of 71 of 76 articles.
- Code from GitHub keeps its spaces and syntax (`import os` was stored as `importos`), and each file starts with its path.

**Smaller fixes**
- A unique index on `link` in MongoDB (`python -m tools.create_indexes`), so the database refuses a second copy of a page.
- The step's metadata counter no longer writes an extra `authors` field into one real chunk.
- A configurable Qdrant timeout (`QDRANT_TIMEOUT`). The default was 5 seconds, and creating a collection took up to 75 seconds on a busy disk.
- Crawlers declare their own domain and are registered automatically. The switched-off LinkedIn crawler fails before starting Chrome.
- The active settings are printed at start-up, because a value in `.env` silently overrides the code's default.
- On a local ZenML server, set `export ZENML_STORE_HTTP_TIMEOUT=300`: the server resets its timeout to 30 seconds on every start, which long steps exceed.

## What I changed in the RAG inference pipeline

Each fix has a unit test in `tests/unit/test_rag.py`.

- **Self-query** never creates users, accepts any spelling of "none", asks the LLM for names only, and sends only the question.
- **Query expansion and search**: rewrites are checked, `k` is applied per collection, and the LLM steps run in parallel.
- **Qdrant** errors other than a missing collection are raised instead of returning empty results.
- **Reranker**: a strict `zip` (no silently dropped results), and the model loads on first use.
- **API**: no model call when nothing was retrieved, no internal error details in responses, one shared retriever, and a cached tokenizer.
- **Generation** uses the same prompt template the model was trained on, with room for 600 new tokens.
- **Any OpenAI-compatible provider** (`OPENAI_BASE_URL`, `LLM_INFERENCE_BACKEND`), so the RAG service runs before the twin is deployed.

## Deployment and monitoring

**Serving my twin on AWS SageMaker.** My account has no GPU quota, so the DPO twin runs on Hugging Face's CPU inference container on `ml.m5.2xlarge` with one model-server worker (eight workers ran out of memory). The book's GPU path stays behind `DEPLOYMENT_TARGET=gpu`. `POST /reply` uses the training prompt (ChatML and my system message) and stops at `<|im_end|>`. The endpoint is deleted after every session, because it bills by the hour.

**CI/CD with GitHub Actions.** Every pull request runs a secret scan (gitleaks), lint and format checks (ruff), and the tests. Every merge into `main` builds the Docker image and pushes it to Amazon ECR, using an IAM user that can only push to that one repository.

**Monitoring with Opik.** Every RAG request is traced step by step. Getting traces to arrive at all took two fixes: the old client created trace timestamps in the year 2202, which the server rejected, and the project name had to be passed in code. The traces then showed that one step, the self-query LLM call, took about 61 seconds.

## Credits

Architecture and core implementation based on [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook) (Iusztin & Labonne). The base model for the WhatsApp Twin is [`UBC-NLP/NileChat-3B`](https://huggingface.co/UBC-NLP/NileChat-3B), licensed for research only.
