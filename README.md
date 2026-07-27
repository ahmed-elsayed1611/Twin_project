# LLM Twin

An end-to-end LLM system that learns a person's writing style from their own content (articles, posts, code) and generates new content in that voice — a production-style RAG + fine-tuning pipeline, not a notebook demo.

> **A note on origin**: this project's architecture and implementation are based on *LLM Engineer's Handbook* (Iusztin & Labonne). I didn't design this system from scratch — I studied it in depth, line by line, rebuilt it, and understood the reasoning behind every design decision, before extending it with my own contributions (see below). I'm sharing it as proof of that understanding, not as original design work.

Studied and built by [Ahmed Elsayed](https://github.com/ahmed-elsayed1611) · [LinkedIn](https://linkedin.com/in/ahmed-elsayed16112002)

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

### 4. Inference Pipeline
Serves the fine-tuned model behind an API, using RAG at query time — embedding the user's question, retrieving relevant chunks from Qdrant, and augmenting the prompt before generation.

## Stack

| Layer | Tools |
|---|---|
| Orchestration | ZenML |
| Data warehouse | MongoDB |
| Vector DB / feature store | Qdrant |
| Experiment & prompt tracking | Comet ML, Opik |
| Training infra | AWS SageMaker |
| Model & fine-tuning | Hugging Face, TRL, PEFT |
| RAG / retrieval | LangChain |
| Serving | FastAPI |

## Design principles

- **Single source of truth**: the feature store (Qdrant + ZenML artifacts) is the only thing training and inference pipelines read from — never the raw data warehouse directly.
- **Category-agnostic core, category-specific handlers**: adding a new data source means writing one new handler class, not touching the pipeline logic.
- **Batch over streaming, by choice**: data volume and latency requirements don't justify streaming complexity here — documented trade-off, not a limitation.

## Status

Actively in development. Current focus: RAG feature pipeline and retrieval/generation.

### My planned contributions
- WhatsApp data crawler, extending the existing crawler dispatcher to a new source
- Streamlit UI for interacting with the twin
- RAG evaluation metrics

## Why this repo matters

I came into this project self-taught, currently working as a Technical Support Representative at AT&T, transitioning into AI/ML engineering. Rather than stopping at tutorials, I chose a genuinely production-grade codebase and went through it the way an engineer maintaining it would have to: understanding every design pattern (Factory, Strategy, Singleton, Template Method, Builder), every type-safety decision (Python generics across a 9-class domain hierarchy), and every architectural trade-off (batch vs. streaming, why a feature store exists, why cleaning/chunking/embedding are separated into their own layers) — not just running cells top to bottom.

That's the skill I'd bring to a team: the ability to drop into an unfamiliar, non-trivial codebase, actually understand it at the architecture level, and extend it correctly — which is most of what real ML engineering work looks like day to day, far more than greenfield model-building.

## Credits

Architecture and core implementation based on *LLM Engineer's Handbook* (Iusztin & Labonne).
