"""BM25 sparse vectors for keyword search in Qdrant (the keyword half of hybrid search).

Each word is one dimension: its index is a stable hash of the word, and its value is the BM25
term-frequency weight. The rare-word weight (IDF) is applied by Qdrant itself, because the sparse
vector is created with `modifier=IDF` (see `VectorBaseDocument._create_collection`).
"""

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

BM25_K1 = 1.2  # how quickly repeating a word stops adding to the score
BM25_B = 0.75  # how much a long chunk is penalised for having more words
BM25_AVG_LENGTH = 170  # average chunk length in words (article chunks are about 222 tokens)

TOKEN_PATTERN = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or that the "
    "this to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def _token_index(token: str) -> int:
    # crc32 gives the same number on every run; Python's hash() changes between runs.
    return zlib.crc32(token.encode())


def document_sparse_vector(text: str) -> SparseVector:
    """BM25 weights for a stored chunk: frequent words count more, with diminishing returns."""

    tokens = tokenize(text)
    length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / BM25_AVG_LENGTH)

    weights: dict[int, float] = {}
    for token, frequency in Counter(tokens).items():
        index = _token_index(token)
        weights[index] = weights.get(index, 0.0) + frequency * (BM25_K1 + 1) / (frequency + length_norm)

    return SparseVector(indices=list(weights), values=list(weights.values()))


def query_sparse_vector(text: str) -> SparseVector:
    """A question's words, each with weight 1: the score is then the sum of the chunk's BM25 weights."""

    indices = sorted({_token_index(token) for token in tokenize(text)})

    return SparseVector(indices=indices, values=[1.0] * len(indices))
