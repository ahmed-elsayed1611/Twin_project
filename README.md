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
| Serving | FastAPI |

## Design principles

- **Single source of truth**: the feature store (Qdrant + ZenML artifacts) is the only thing training and inference pipelines read from — never the raw data warehouse directly.
- **Category-agnostic core, category-specific handlers**: adding a new data source means writing one new handler class, not touching the pipeline logic.
- **Batch over streaming, by choice**: data volume and latency requirements don't justify streaming complexity here — documented trade-off, not a limitation.

## Status

This project is still in progress.

| Stage | Status |
|---|---|
| Architecture, tooling, data collection, RAG feature pipeline | Studied and run as written in the book |
| Data preparation | Done — my own implementation |
| Supervised fine-tuning | Done — my own implementation |
| Preference alignment | Done — my own implementation |
| Evaluation | Done — my own implementation |
| RAG inference pipeline | In progress |
| Inference pipeline deployment | In progress |
| MLOps and LLMOps | In progress |

### Planned contributions
- Streamlit UI for interacting with the twin
- RAG evaluation metrics

## Credits

Architecture and core implementation based on [*LLM Engineer's Handbook*](https://github.com/PacktPublishing/LLM-Engineers-Handbook) (Iusztin & Labonne). The base model for the WhatsApp Twin is [`UBC-NLP/NileChat-3B`](https://huggingface.co/UBC-NLP/NileChat-3B), licensed for research only.
