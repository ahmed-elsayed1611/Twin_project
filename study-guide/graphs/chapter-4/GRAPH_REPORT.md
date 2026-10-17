# Graph Report - g4  (2026-10-14)

## Corpus Check
- Corpus is ~4,114 words - fits in a single context window. You may not need a graph.

## Summary
- 260 nodes · 643 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 73 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Embedding Handlers
- Database Connections and Utils
- Chunking Handlers
- Embedding Model Singletons
- Qdrant Object-Vector Mapper
- Cleaning Handlers and Text Ops
- Pipeline Steps and Dispatchers
- Raw Documents and Warehouse Query
- MongoDB Object-Document Mapper

## God Nodes (most connected - your core abstractions)
1. `VectorBaseDocument` - 38 edges
2. `DataCategory` - 34 edges
3. `NoSQLBaseDocument` - 25 edges
4. `EmbeddingModelSingleton` - 15 edges
5. `EmbeddingDataHandler` - 13 edges
6. `ChunkingDataHandler` - 11 edges
7. `CleaningDataHandler` - 10 edges
8. `Chunk` - 10 edges
9. `PostChunk` - 10 edges
10. `ArticleChunk` - 10 edges

## Surprising Connections (you probably didn't know these)
- `fetch_all_data()` --uses--> `NoSQLBaseDocument`  [INFERRED]
  steps/feature_engineering/query_data_warehouse.py → llm_engineering/domain/base/nosql.py
- `__fetch_articles()` --uses--> `NoSQLBaseDocument`  [INFERRED]
  steps/feature_engineering/query_data_warehouse.py → llm_engineering/domain/base/nosql.py
- `__fetch_posts()` --uses--> `NoSQLBaseDocument`  [INFERRED]
  steps/feature_engineering/query_data_warehouse.py → llm_engineering/domain/base/nosql.py
- `__fetch_repositories()` --uses--> `NoSQLBaseDocument`  [INFERRED]
  steps/feature_engineering/query_data_warehouse.py → llm_engineering/domain/base/nosql.py
- `_add_chunks_metadata()` --uses--> `Chunk`  [INFERRED]
  steps/feature_engineering/rag.py → llm_engineering/domain/chunks.py

## Import Cycles
- None detected.

## Communities (10 total, 3 thin omitted)

### Community 0 - "Embedding Handlers"
Cohesion: 0.12
Nodes (20): EmbeddingHandlerFactory, ArticleEmbeddingHandler, EmbeddingDataHandler, PostEmbeddingHandler, QueryEmbeddingHandler, RepositoryEmbeddingHandler, ArticleChunk, Config (+12 more)

### Community 1 - "Database Connections and Utils"
Cohesion: 0.07
Nodes (3): flatten(), MongoDatabaseConnector, QdrantDatabaseConnector

### Community 2 - "Chunking Handlers"
Cohesion: 0.12
Nodes (15): ArticleChunkingHandler, ChunkingDataHandler, PostChunkingHandler, RepositoryChunkingHandler, ChunkingHandlerFactory, chunk_text(), Chunk, PostChunk (+7 more)

### Community 3 - "Embedding Model Singletons"
Cohesion: 0.08
Nodes (3): SingletonMeta, CrossEncoderModelSingleton, EmbeddingModelSingleton

### Community 5 - "Cleaning Handlers and Text Ops"
Cohesion: 0.15
Nodes (10): ArticleCleaningHandler, CleaningDataHandler, PostCleaningHandler, RepositoryCleaningHandler, CleaningHandlerFactory, chunk_article(), chunk_document(), clean_text() (+2 more)

### Community 6 - "Pipeline Steps and Dispatchers"
Cohesion: 0.13
Nodes (11): ChunkingDispatcher, CleaningDispatcher, EmbeddingDispatcher, CleanedDocument, feature_engineering(), clean_documents(), _get_metadata(), load_to_vector_db() (+3 more)

### Community 7 - "Raw Documents and Warehouse Query"
Cohesion: 0.19
Nodes (15): ArticleDocument, Settings, Document, PostDocument, Settings, RepositoryDocument, Settings, UserDocument (+7 more)

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VectorBaseDocument` connect `Qdrant Object-Vector Mapper` to `Embedding Handlers`, `Database Connections and Utils`, `Chunking Handlers`, `Embedding Model Singletons`, `Cleaning Handlers and Text Ops`, `Pipeline Steps and Dispatchers`, `MongoDB Object-Document Mapper`?**
  _High betweenness centrality (0.245) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `VectorBaseDocument` (e.g. with `EmbeddingModelSingleton` and `DataCategory`) actually correct?**
  _`VectorBaseDocument` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Embedding Handlers` be split into smaller, more focused modules?**
  _Cohesion score 0.12233285917496443 - nodes in this community are weakly interconnected._
- **Why does `EmbeddingModelSingleton` connect `Embedding Model Singletons` to `Embedding Handlers`, `Database Connections and Utils`, `Qdrant Object-Vector Mapper`, `Cleaning Handlers and Text Ops`?**
  _High betweenness centrality (0.166) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `DataCategory` (e.g. with `ChunkingHandlerFactory` and `CleaningDispatcher`) actually correct?**
  _`DataCategory` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Should `Database Connections and Utils` be split into smaller, more focused modules?**
  _Cohesion score 0.07301587301587302 - nodes in this community are weakly interconnected._
- **Why does `DataCategory` connect `Embedding Handlers` to `Database Connections and Utils`, `Chunking Handlers`, `Qdrant Object-Vector Mapper`, `Cleaning Handlers and Text Ops`, `Pipeline Steps and Dispatchers`, `Raw Documents and Warehouse Query`, `MongoDB Object-Document Mapper`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._