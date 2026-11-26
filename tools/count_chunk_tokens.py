"""Check that every article chunk in Qdrant fits in the embedding model (256 tokens, 254 for text).

Usage: poetry run python -m tools.count_chunk_tokens
"""

from loguru import logger

from llm_engineering.application.preprocessing.operations.chunking import count_tokens
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk

LIMIT = 254  # 256 minus the [CLS] and [SEP] tokens the model adds


def main() -> None:
    # 1. Get all the chunks from Qdrant.
    chunks, _ = EmbeddedArticleChunk.bulk_find(limit=2000)

    # 2. One token count per chunk.
    counts = [count_tokens(chunk.content) for chunk in chunks]

    # 3. Only the counts that are too long.
    over = [n for n in counts if n > LIMIT]

    # 4 and 5. Report.
    logger.info(f"Chunks:        {len(counts)}")
    logger.info(f"Biggest chunk: {max(counts)} tokens")
    logger.info(f"Average chunk: {sum(counts) / len(counts):.0f} tokens")
    logger.info(f"Over {LIMIT}:     {len(over)} ({len(over) / len(counts):.0%})")


if __name__ == "__main__":
    main()
