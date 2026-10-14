import re
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""

    return chunk_article(text, min_length, max_length)


def count_tokens(text: str) -> int:
    """Number of tokens the embedding model sees for this text, without the special start and end tokens."""

    return len(embedding_model.tokenizer.tokenize(text))


def chunk_article(
    text: str, min_length: int, max_length: int, length_function: Callable[[str], int] = len
) -> list[str]:
    """Group whole sentences into chunks. Lengths are measured with length_function: characters by default,
    or tokens with count_tokens, so a chunk never exceeds what the embedding model can read."""

    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for piece in _split_long_sentence(sentence, max_length, length_function):
            if length_function(current_chunk + piece) <= max_length:
                current_chunk += piece + " "
            else:
                _add_extract(extracts, current_chunk, min_length, max_length, length_function)
                current_chunk = piece + " "

    _add_extract(extracts, current_chunk, min_length, max_length, length_function)

    return extracts


def _split_long_sentence(sentence: str, max_length: int, length_function: Callable[[str], int]) -> list[str]:
    """A sentence longer than max_length (often code, which has no full stops) is cut between words."""

    if length_function(sentence) <= max_length:
        return [sentence]

    pieces = []
    current_piece = ""
    for word in sentence.split():
        if current_piece and length_function(f"{current_piece} {word}") > max_length:
            pieces.append(current_piece)
            current_piece = word
        else:
            current_piece = f"{current_piece} {word}" if current_piece else word
    if current_piece:
        pieces.append(current_piece)

    return pieces


def _add_extract(
    extracts: list[str], chunk: str, min_length: int, max_length: int, length_function: Callable[[str], int]
) -> None:
    """Keep every piece of text: a piece under min_length joins the previous chunk when the result still fits
    in max_length, and otherwise stays as its own chunk."""

    chunk = chunk.strip()
    if not chunk:
        return

    joined = f"{extracts[-1]} {chunk}" if extracts else ""
    if extracts and length_function(chunk) < min_length and length_function(joined) <= max_length:
        extracts[-1] = joined
    else:
        extracts.append(chunk)
