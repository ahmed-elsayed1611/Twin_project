import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
URL_PLACEHOLDER = "urlplaceholder"
PARAGRAPH_SEPARATOR = "\n\n"


def clean_text(text: str) -> str:
    """Clean prose: links become [URL], symbols are removed, and paragraph breaks are kept."""

    # Replace links before symbols are stripped, otherwise they turn into noise such as "https example.com".
    text = URL_PATTERN.sub(f" {URL_PLACEHOLDER} ", text)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    # Paragraph breaks are kept for chunking; a single line break inside a paragraph becomes a space.
    text = re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)
    text = re.sub(r"\n{2,}", PARAGRAPH_SEPARATOR, text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = text.replace(URL_PLACEHOLDER, "[URL]")

    return text.strip()


def clean_code(text: str) -> str:
    """Clean source code: keep its symbols and indentation, only drop trailing spaces and extra blank lines."""

    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", PARAGRAPH_SEPARATOR, text)

    return text.strip()
