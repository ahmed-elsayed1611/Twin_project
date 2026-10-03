#!/usr/bin/env python3
"""Scan files or folders for private names before committing or publishing.

    python check_private_names.py --names ~/private_names.txt path/to/repo [more paths...]

Exits with code 1 if any private name is found, so it also works as a git pre-push hook.
Scans text files (.py .md .ipynb .json .jsonl .txt .csv .yaml .yml .cfg .toml .sh). For notebooks,
both cell sources and saved outputs are scanned -- outputs are the usual way names leak.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from private_names_lib import load_private_names, compile_name_patterns, find_private_names

TEXT_EXT = {".py", ".md", ".ipynb", ".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml", ".cfg", ".toml", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules"}


def iter_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if os.path.splitext(f)[1].lower() in TEXT_EXT:
                    yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, help="path to private_names.txt")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    names_file = os.path.abspath(args.names)
    patterns = compile_name_patterns(load_private_names(names_file))
    findings = 0
    for path in iter_files(args.paths):
        if os.path.abspath(path) == names_file:
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            hits = find_private_names(line, patterns)
            if hits:
                findings += 1
                where = "output/cell content" if path.endswith(".ipynb") else f"line {lineno}"
                print(f"{path} ({where}): {sorted(set(hits))}")
    print(f"\n{findings} line(s) with private names." if findings else "No private names found.")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
