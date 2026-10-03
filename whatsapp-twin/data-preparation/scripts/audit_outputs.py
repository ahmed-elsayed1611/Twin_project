#!/usr/bin/env python3
"""Perform a dependency-free quality audit of generated WhatsApp dataset files."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twin_dataset_builder import CREDENTIAL_LIKE_PATTERN, EDIT_TAG_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, URL_PATTERN


def records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "outputs").expanduser().resolve()
    expected = [output_dir / name for name in ("train.jsonl", "validation.jsonl", "audit_report.json")]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        print(json.dumps({"valid": False, "error": "missing_expected_files", "files": missing}, indent=2))
        return 1

    ids: set[str] = set()
    response_lengths: list[int] = []
    context_turn_counts: list[int] = []
    roles = Counter()
    pii_matches: list[dict[str, object]] = []
    malformed: list[dict[str, object]] = []

    for split in ("train", "validation"):
        for line_number, record in records(output_dir / f"{split}.jsonl"):
            message_list = record.get("messages")
            if not isinstance(message_list, list) or not message_list:
                malformed.append({"split": split, "line": line_number, "reason": "missing_messages"})
                continue
            identifier = record.get("id")
            if identifier in ids:
                malformed.append({"split": split, "line": line_number, "reason": "duplicate_id"})
            ids.add(identifier)
            if message_list[0].get("role") != "system":
                malformed.append({"split": split, "line": line_number, "reason": "missing_system_message"})
            if message_list[-1].get("role") != "assistant":
                malformed.append({"split": split, "line": line_number, "reason": "final_role_not_assistant"})
            response_lengths.append(len(message_list[-1].get("content", "")))
            context_turn_counts.append(max(0, len(message_list) - 2))
            for message in message_list:
                roles[message.get("role", "missing")] += 1
                content = message.get("content", "")
                matched = [
                    label
                    for label, pattern in (
                        ("EMAIL", EMAIL_PATTERN),
                        ("URL", URL_PATTERN),
                        ("PHONE", PHONE_PATTERN),
                        ("CREDENTIAL", CREDENTIAL_LIKE_PATTERN),
                    )
                    if pattern.search(content)
                ]
                if matched:
                    pii_matches.append({"split": split, "line": line_number, "patterns": matched})
                if EDIT_TAG_PATTERN.search(content):
                    malformed.append({"split": split, "line": line_number, "reason": "unstripped_edit_tag"})

    source_audit = json.loads((output_dir / "audit_report.json").read_text(encoding="utf-8"))
    result = {
        "valid": not malformed and not pii_matches and bool(source_audit.get("output_validation", {}).get("valid")),
        "records_total": len(ids),
        "role_counts": dict(roles),
        "assistant_response_characters": {
            "minimum": min(response_lengths, default=0),
            "median": statistics.median(response_lengths) if response_lengths else 0,
            "maximum": max(response_lengths, default=0),
        },
        "context_turns": {
            "minimum": min(context_turn_counts, default=0),
            "median": statistics.median(context_turn_counts) if context_turn_counts else 0,
            "maximum": max(context_turn_counts, default=0),
        },
        "redactions_applied_during_build": source_audit.get("pii_redactions", {}),
        "format_errors_sample": malformed[:10],
        "residual_pii_matches_sample": pii_matches[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
