"""Output filter for the twin's generated replies (standard library only).

The SFT v2 model learned from redacted data, so it sometimes writes the placeholders themselves ("[NAME]",
"[PHONE]"); a sampled reply can also loop or contain a long number. This is the last line of defence between
the model and a reader: clean the reply, check it, and if it is not safe, generate again; after a few tries,
answer with a natural deflection instead.

The test notebooks embed this file's code unchanged (tests/test_reply_filter.py checks it), so the evaluation,
any demo and these tests all use the same rules.
"""

import random
import re
from typing import Callable, Optional, Sequence

# A placeholder from the redaction, with the vocative "يا" before it ("يا [NAME]" -> "").
PLACEHOLDER = re.compile(r"\s*(?:يا\s*)?\[(?:NAME|PHONE|EMAIL|URL|NUMBER|CREDENTIAL|REDACTED)\]")
LONG_NUMBER = re.compile(r"[0-9٠-٩]{6,}")          # phone-like, code-like or account-like
REPEATED_CHARACTER = re.compile(r"(.)\1{7,}")       # the same character 8+ times in a row
REPEATED_WORD = re.compile(r"(\S+)(?:\s+\1){3,}")   # the same word 4+ times in a row
FALLBACKS = ("مش هقولك 😂", "سيبك من الموضوع ده", "بعدين نتكلم")


def clean_reply(text: str) -> str:
    """Removes placeholders, empty lines and a line repeated right after itself."""
    lines: list[str] = []
    for line in text.split("\n"):
        line = PLACEHOLDER.sub("", line).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return "\n".join(lines)


def problems(reply: str, find_names: Optional[Callable[[str], Sequence[str]]] = None) -> list[str]:
    """Reasons a cleaned reply must not be shown; an empty list means it is safe."""
    found = []
    if not reply.strip():
        found.append("empty")
    if find_names and find_names(reply):
        found.append("private_name")
    if LONG_NUMBER.search(reply):
        found.append("long_number")
    if REPEATED_CHARACTER.search(reply):
        found.append("repeated_characters")
    lines = [line for line in reply.split("\n") if line.strip()]
    if REPEATED_WORD.search(reply) or any(lines.count(line) >= 3 for line in set(lines)):
        found.append("repeated_text")
    return found


def safe_reply(generate: Callable[[], str], find_names: Optional[Callable[[str], Sequence[str]]] = None,
               tries: int = 3, rng: Optional[random.Random] = None,
               fallbacks: Sequence[str] = FALLBACKS) -> tuple[str, dict]:
    """Calls `generate` up to `tries` times and returns the first cleaned reply with no problems, or a fallback.

    The second value says what happened: attempts used, whether the fallback was used, and the problems of the
    rejected attempts (reasons only, never the text)."""
    rejected = []
    for attempt in range(1, tries + 1):
        reply = clean_reply(generate())
        found = problems(reply, find_names)
        if not found:
            return reply, {"attempts": attempt, "fallback": False, "rejected": rejected}
        rejected.append(found)
    return (rng or random).choice(list(fallbacks)), {"attempts": tries, "fallback": True, "rejected": rejected}
