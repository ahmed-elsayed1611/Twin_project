# Shared name matching for the Ahmed Twin project (same logic is embedded in the notebooks).
import re

ARABIC = "\u0600-\u06FF"
# One-letter Arabic prefixes that attach to a name: و (and), ب (with), ل (to).
# NOT ف or ك: ف/ك + a short name often spells an everyday word (e.g. ف+رح = فرح, "joy").
NAME_PREFIXES = "وبل"


def load_private_names(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return sorted(set(names), key=len, reverse=True)   # longest first: full names before parts


def compile_name_patterns(names):
    arabic = [n for n in names if re.search(f"[{ARABIC}]", n)]
    latin = [n for n in names if n not in arabic]
    ar = (re.compile(rf"(?<![{ARABIC}])(?P<pre>[{NAME_PREFIXES}])?(?P<name>{'|'.join(map(re.escape, arabic))})(?![{ARABIC}])")
          if arabic else None)
    # Latin names: boundaries are "not a Latin letter" (\b fails next to Arabic letters, e.g. "وJohn").
    la = re.compile(rf"(?<![A-Za-z])(?:{'|'.join(map(re.escape, latin))})(?![A-Za-z])", re.IGNORECASE) if latin else None
    return ar, la


def find_private_names(text, patterns):
    ar, la = patterns
    found = []
    if ar:
        found += [m.group("name") for m in ar.finditer(text)]
    if la:
        found += [m.group(0) for m in la.finditer(text)]
    return found


def anonymize(text, patterns, placeholder="[NAME]"):
    ar, la = patterns
    if ar:
        text = ar.sub(lambda m: (m.group("pre") or "") + placeholder, text)
    if la:
        text = la.sub(placeholder, text)
    return text
