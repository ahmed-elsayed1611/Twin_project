# Project tools

Shared by every chapter. Run them from the repository root.

| Tool | What it does |
|---|---|
| `check_private_names.py` | Scans files and folders, **including notebook outputs**, for the real names in `private_names.txt`. Exit code 1 if any are found. |
| `private_names_lib.py` | The matching rules used everywhere (the evaluation notebook embeds the same logic): whole-word matching with the attached prefixes و / ب / ل, so a short name never matches inside a longer everyday word. |
| `check_notebook_order.py` | Walks a notebook's code cells in top-to-bottom execution order and flags any name used before a cell defines it. This is the kind of bug that only shows up on a fresh run. |
| `reply_filter.py` | The output filter for generated replies: removes redaction placeholders (`[NAME]`, and the `يا` before it), rejects private names, long numbers and loops, generates again, and after a few tries answers with a natural deflection. Standard library only; notebooks embed it unchanged (their tests check this). Tests: `python -m unittest discover -s tools/tests` |

## Before every push
```bash
python tools/check_private_names.py --names ~/private_names.txt .
python tools/check_notebook_order.py */notebooks/*.ipynb
```
To run the name scan automatically, save this as `.git/hooks/pre-push` and `chmod +x` it:
```bash
#!/bin/sh
python tools/check_private_names.py --names ~/private_names.txt . || exit 1
```

`private_names.txt` itself (real names, one per line, `#` for comments) lives only on your machine and at `MyDrive/ahmed_twin/private_names.txt`. It is never committed.
