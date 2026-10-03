#!/usr/bin/env python3
"""Build a privacy-reduced, human-persona fine-tuning dataset from a WhatsApp text export.

This module deliberately uses only the Python standard library. It can run on a normal
Python 3.10+ installation without Poetry, ZenML, PyTorch, or a model-download step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

LOGGER = logging.getLogger("twin_dataset_builder")

ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
INVISIBLE_MARKS = "\u200e\u200f\u061c\ufeff"

# WhatsApp formats supported by this project:
#   22/3/2023، 12:25 ص - Ahmed: message
#   22/03/2023, 12:25 AM - Ahmed: message
#   [22/03/2023, 12:25] Ahmed: message
HEADER_PATTERN = re.compile(
    r"^\[?"
    r"(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"
    r"\s*[،,]\s*"
    r"(?P<time>\d{1,2}:\d{2})"
    r"(?:\s*(?P<period>ص|م|AM|PM|am|pm))?"
    r"\]?\s*-?\s*"
    r"(?:(?P<sender>[^:\n]{1,160}):\s*)?"
    r"(?P<text>.*)$"
)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
# A phone number must either be 8–15 uninterrupted digits or use multiple structured
# space/hyphen/parenthesis groups. Dots are not allowed within either pattern, so decimals
# and versions are not falsely redacted; ordinary punctuation after a number is allowed.
PHONE_PATTERN = re.compile(
    r"(?<![\w.])(?:\+|00)?[0-9٠-٩]{8,15}(?![0-9٠-٩])"
    r"|(?<![\w.])(?:\+|00)?(?:[0-9٠-٩]{2,4}[ \t()\-]){2,4}[0-9٠-٩]{3,5}(?![0-9٠-٩])"
)
# Deliberately separate from PHONE_PATTERN: a word glued directly to a long digit
# run (e.g. a username-plus-PIN like "Example12345678") isn't a phone number, so
# PHONE_PATTERN's lookbehind excludes it by design (that exclusion exists so
# version-like tokens, e.g. "python3.10", aren't falsely redacted). But a
# letter-token immediately followed by 6+ digits is far more often a shared
# login/credential/reference code than natural prose, so it gets its own
# narrower pattern and its own placeholder rather than loosening PHONE_PATTERN.
CREDENTIAL_LIKE_PATTERN = re.compile(r"\b[A-Za-z\u0600-\u06FF]{3,}[0-9٠-٩]{6,}\b")

# WhatsApp appends this marker directly onto the text of a message the sender
# edited after sending. It is not part of what the person actually said, so
# it must be stripped before the content is treated as real conversation —
# left unstripped, it leaks verbatim into ~11% of records (measured on the
# real export: 3,142 of 27,877 training records before this fix).
EDIT_TAG_PATTERN = re.compile(r"\s*<تم تعديل هذه الرسالة>\s*")

SYSTEM_MARKERS = (
    "الرسائل والمكالمات مشفرة تمامًا",
    "messages and calls are end-to-end encrypted",
    "<تم استبعاد الوسائط>",
    "<media omitted>",
    "تم حذف هذه الرسالة",
    "you deleted this message",
    "this message was deleted",
)

DEFAULT_SYSTEM_MESSAGE = (
    "Continue the conversation as Ahmed. Match the language, tone, and context of "
    "the conversation. Do not claim to know facts that are not supported by the dialogue."
)


@dataclass(frozen=True)
class ParsedMessage:
    """A parsed, redacted WhatsApp message that passed the basic content filter."""

    timestamp: datetime
    sender: str
    content: str
    first_line: int


@dataclass(frozen=True)
class ConversationSession:
    """A contiguous sequence of messages separated from other sessions by a time gap."""

    session_id: int
    messages: tuple[ParsedMessage, ...]


@dataclass(frozen=True)
class TrainingSample:
    """A clean supervised example in both chat and instruction/response representations."""

    sample_id: str
    session_id: int
    messages: tuple[dict[str, str], ...]
    instruction: str
    response: str


@dataclass
class BuildStats:
    """Machine-readable counts explaining what the pipeline accepted or rejected."""

    lines_read: int = 0
    headers_seen: int = 0
    continuation_lines: int = 0
    orphan_lines: int = 0
    system_events_skipped: int = 0
    timestamp_parse_failures: int = 0
    messages_before_cleaning: int = 0
    messages_after_cleaning: int = 0
    burst_merges: int = 0
    sessions_total: int = 0
    samples_total: int = 0
    samples_train: int = 0
    samples_validation: int = 0
    pii_redactions: Counter = field(default_factory=Counter)
    rejection_reasons: Counter = field(default_factory=Counter)

    def to_json(self, config: dict[str, object]) -> dict[str, object]:
        data = asdict(self)
        data["pii_redactions"] = dict(self.pii_redactions)
        data["rejection_reasons"] = dict(self.rejection_reasons)
        data["config"] = config
        return data


class WhatsAppDatasetBuilder:
    """Parse, clean, split, validate, and export a WhatsApp training dataset."""

    def __init__(
        self,
        *,
        self_sender: str,
        context_turns: int = 5,
        max_gap_hours: float = 4.0,
        burst_gap_seconds: int = 300,
        min_response_chars: int = 2,
        max_duplicate_responses: int = 3,
        validation_ratio: float = 0.10,
        redact_terms: Sequence[str] = (),
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        if not self_sender.strip():
            raise ValueError("self_sender cannot be empty")
        if context_turns < 1:
            raise ValueError("context_turns must be at least 1")
        if max_duplicate_responses < 0:
            raise ValueError("max_duplicate_responses must be 0 (unlimited) or a positive integer")
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")

        self.self_sender = self._normalize_sender(self_sender)
        self.context_turns = context_turns
        self.max_gap = timedelta(hours=max_gap_hours)
        self.burst_gap = timedelta(seconds=burst_gap_seconds)
        self.min_response_chars = min_response_chars
        self.max_duplicate_responses = max_duplicate_responses
        self.validation_ratio = validation_ratio
        self.system_message = system_message.strip()
        self.user_redaction_patterns = [
            re.compile(re.escape(term.strip()), re.IGNORECASE)
            for term in redact_terms
            if term.strip()
        ]
        self.stats = BuildStats()
        self._rejections: list[dict[str, object]] = []

    @staticmethod
    def _remove_invisible_marks(value: str) -> str:
        return value.translate({ord(character): None for character in INVISIBLE_MARKS})

    @classmethod
    def _normalize_header_text(cls, value: str) -> str:
        return cls._remove_invisible_marks(value).translate(ARABIC_DIGIT_TRANSLATION).strip()

    @classmethod
    def _normalize_sender(cls, value: str) -> str:
        return " ".join(cls._remove_invisible_marks(value).split())

    @staticmethod
    def _role_for_sender(sender: str, self_sender: str) -> str:
        return "assistant" if sender == self_sender else "user"

    def _parse_timestamp(self, date_text: str, time_text: str, period: Optional[str]) -> Optional[datetime]:
        normalized_date = self._normalize_header_text(date_text).replace(".", "/").replace("-", "/")
        normalized_time = self._normalize_header_text(time_text)
        normalized_period = (period or "").upper()
        if normalized_period == "ص":
            normalized_period = "AM"
        elif normalized_period == "م":
            normalized_period = "PM"

        try:
            first, second, third = (int(piece) for piece in normalized_date.split("/"))
            hour, minute = (int(piece) for piece in normalized_time.split(":"))
        except (TypeError, ValueError):
            return None

        if len(str(first)) == 4:
            year, month, day = first, second, third
        else:
            # The supplied export is Arabic locale / day-first. This is intentional and explicit.
            day, month, year = first, second, third
        if year < 100:
            year += 2000

        if normalized_period:
            if hour < 1 or hour > 12:
                return None
            if normalized_period == "AM" and hour == 12:
                hour = 0
            elif normalized_period == "PM" and hour != 12:
                hour += 12
        elif hour > 23:
            return None

        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    @staticmethod
    def _is_system_event(sender: Optional[str], content: str) -> bool:
        lowered = content.casefold()
        return sender is None or any(marker in lowered for marker in SYSTEM_MARKERS)

    def _redact_content(self, content: str) -> tuple[str, bool]:
        """Return redacted text and a flag indicating whether it can be retained."""
        content = self._remove_invisible_marks(content)
        content = EDIT_TAG_PATTERN.sub(" ", content)
        content = " ".join(content.split())
        if not content:
            return "", False
        if any(marker in content.casefold() for marker in SYSTEM_MARKERS):
            return "", False

        replacements = (
            ("EMAIL", EMAIL_PATTERN),
            ("URL", URL_PATTERN),
            ("PHONE", PHONE_PATTERN),
            ("CREDENTIAL", CREDENTIAL_LIKE_PATTERN),
        )
        for label, pattern in replacements:
            content, count = pattern.subn(f"[{label}]", content)
            if count:
                self.stats.pii_redactions[label] += count

        for pattern in self.user_redaction_patterns:
            content, count = pattern.subn("[REDACTED]", content)
            if count:
                self.stats.pii_redactions["USER_TERM"] += count

        content = " ".join(content.split())
        return content, bool(content)

    def _reject(self, reason: str, line: int) -> None:
        self.stats.rejection_reasons[reason] += 1
        # Never retain rejected message content in the debug file.
        self._rejections.append({"reason": reason, "source_line": line})

    def parse_messages(self, raw_file: Path) -> list[ParsedMessage]:
        """Parse the export and merge valid continuation lines into messages."""
        if not raw_file.is_file():
            raise FileNotFoundError(f"Raw WhatsApp export does not exist: {raw_file}")

        parsed: list[ParsedMessage] = []
        current: Optional[dict[str, object]] = None

        def flush_current() -> None:
            nonlocal current
            if current is None:
                return
            self.stats.messages_before_cleaning += 1
            content, keep = self._redact_content(str(current["content"]))
            if not keep:
                self._reject("empty_or_noise", int(current["first_line"]))
            elif len(content) < self.min_response_chars:
                self._reject("below_minimum_message_length", int(current["first_line"]))
            else:
                parsed.append(
                    ParsedMessage(
                        timestamp=current["timestamp"],  # type: ignore[arg-type]
                        sender=str(current["sender"]),
                        content=content,
                        first_line=int(current["first_line"]),
                    )
                )
                self.stats.messages_after_cleaning += 1
            current = None

        with raw_file.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, original_line in enumerate(handle, start=1):
                self.stats.lines_read += 1
                raw_line = original_line.rstrip("\r\n")
                candidate = self._normalize_header_text(raw_line)
                match = HEADER_PATTERN.match(candidate)

                if match:
                    self.stats.headers_seen += 1
                    flush_current()
                    timestamp = self._parse_timestamp(
                        match.group("date"), match.group("time"), match.group("period")
                    )
                    if timestamp is None:
                        self.stats.timestamp_parse_failures += 1
                        self._reject("unparseable_timestamp", line_number)
                        continue
                    sender = match.group("sender")
                    content = match.group("text")
                    if self._is_system_event(sender, content):
                        self.stats.system_events_skipped += 1
                        continue
                    normalized_sender = self._normalize_sender(sender or "")
                    if not normalized_sender:
                        self.stats.system_events_skipped += 1
                        continue
                    current = {
                        "timestamp": timestamp,
                        "sender": self.self_sender if normalized_sender == self.self_sender else "Contact",
                        "content": content,
                        "first_line": line_number,
                    }
                elif current is not None:
                    self.stats.continuation_lines += 1
                    current["content"] = f"{current['content']} {raw_line.strip()}"
                elif raw_line.strip():
                    self.stats.orphan_lines += 1

        flush_current()
        return self._merge_bursts(parsed)

    def _merge_bursts(self, messages: Sequence[ParsedMessage]) -> list[ParsedMessage]:
        """Merge adjacent same-sender messages sent in a short time window."""
        merged: list[ParsedMessage] = []
        for message in messages:
            if (
                merged
                and message.sender == merged[-1].sender
                and message.timestamp >= merged[-1].timestamp
                and message.timestamp - merged[-1].timestamp <= self.burst_gap
            ):
                previous = merged[-1]
                merged[-1] = ParsedMessage(
                    timestamp=previous.timestamp,
                    sender=previous.sender,
                    content=f"{previous.content}\n{message.content}",
                    first_line=previous.first_line,
                )
                self.stats.burst_merges += 1
            else:
                merged.append(message)
        return merged

    def sessionize(self, messages: Sequence[ParsedMessage]) -> list[ConversationSession]:
        """Split messages into independent sessions using a configurable time gap."""
        if not messages:
            return []
        sessions: list[ConversationSession] = []
        current: list[ParsedMessage] = []
        session_id = 0
        previous_time: Optional[datetime] = None
        for message in messages:
            if previous_time is not None and (
                message.timestamp < previous_time or message.timestamp - previous_time > self.max_gap
            ):
                sessions.append(ConversationSession(session_id=session_id, messages=tuple(current)))
                session_id += 1
                current = []
            current.append(message)
            previous_time = message.timestamp
        if current:
            sessions.append(ConversationSession(session_id=session_id, messages=tuple(current)))
        self.stats.sessions_total = len(sessions)
        return sessions

    def _instruction(self, context: Sequence[ParsedMessage]) -> str:
        history = "\n".join(f"{message.sender}: {message.content}" for message in context)
        return f"{self.system_message}\n\nConversation history:\n{history}\n\nAhmed:"

    def build_samples(self, sessions: Sequence[ConversationSession]) -> list[TrainingSample]:
        """Build chat and instruction/response records from valid Contact -> Ahmed transitions."""
        samples: list[TrainingSample] = []
        seen_hashes: set[str] = set()
        # Tracks how many times each exact response TEXT has been kept so far,
        # independent of its surrounding context. Real chat data contains
        # hundreds of exact-duplicate short acknowledgements ("اه", "لا",
        # "تمام", ...) with different context each time, so the full-sample
        # hash dedup above does not catch this — a separate cap is needed to
        # avoid the response distribution being dominated by a handful of
        # generic one-word replies.
        response_occurrences: Counter[str] = Counter()

        for session in sessions:
            turns = session.messages
            for index, target in enumerate(turns):
                if target.sender != self.self_sender:
                    continue
                if index == 0 or turns[index - 1].sender != "Contact":
                    self._reject("no_immediate_contact_prompt", target.first_line)
                    continue
                if len(target.content) < self.min_response_chars:
                    self._reject("response_below_minimum_length", target.first_line)
                    continue
                if (
                    self.max_duplicate_responses > 0
                    and response_occurrences[target.content] >= self.max_duplicate_responses
                ):
                    self._reject("duplicate_response_capped", target.first_line)
                    continue

                context = turns[max(0, index - self.context_turns) : index]
                conversation_messages = [
                    {"role": "system", "content": self.system_message},
                    *[
                        {
                            "role": self._role_for_sender(message.sender, self.self_sender),
                            "content": message.content,
                        }
                        for message in context
                    ],
                    {"role": "assistant", "content": target.content},
                ]
                content_fingerprint = json.dumps(conversation_messages, ensure_ascii=False, sort_keys=True)
                sample_id = hashlib.sha256(content_fingerprint.encode("utf-8")).hexdigest()[:20]
                if sample_id in seen_hashes:
                    self._reject("duplicate_sample", target.first_line)
                    continue
                seen_hashes.add(sample_id)
                response_occurrences[target.content] += 1
                samples.append(
                    TrainingSample(
                        sample_id=sample_id,
                        session_id=session.session_id,
                        messages=tuple(conversation_messages),
                        instruction=self._instruction(context),
                        response=target.content,
                    )
                )

        self.stats.samples_total = len(samples)
        return samples

    def _is_validation_session(self, session_id: int) -> bool:
        """Deterministically allocate whole sessions to validation without random state."""
        score = int(hashlib.sha256(f"session:{session_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return score < self.validation_ratio

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validate_chat_record(self, record: object) -> Optional[str]:
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            return "record_has_no_messages_list"
        messages = record["messages"]
        if len(messages) < 3:
            return "record_has_insufficient_messages"
        if messages[0] != {"role": "system", "content": self.system_message}:
            return "system_message_mismatch"
        if messages[-1].get("role") != "assistant":
            return "final_message_is_not_assistant"
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                return "invalid_message_role"
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                return "empty_message_content"
            text = message["content"]
            if (
                EMAIL_PATTERN.search(text)
                or URL_PATTERN.search(text)
                or PHONE_PATTERN.search(text)
                or CREDENTIAL_LIKE_PATTERN.search(text)
            ):
                return "possible_unredacted_pii"
            if EDIT_TAG_PATTERN.search(text):
                return "unstripped_edit_tag"
        return None

    def _validate_outputs(self, train_path: Path, validation_path: Path) -> dict[str, object]:
        errors: list[dict[str, object]] = []
        counts: dict[str, int] = {}
        for split_name, path in (("train", train_path), ("validation", validation_path)):
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    count += 1
                    reason = self._validate_chat_record(json.loads(line))
                    if reason:
                        errors.append({"split": split_name, "line": line_number, "reason": reason})
            counts[split_name] = count
        return {"valid": not errors, "records": counts, "errors": errors[:20]}

    def export(self, samples: Sequence[TrainingSample], output_dir: Path) -> dict[str, object]:
        """Write all output files atomically after validating the JSONL output."""
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix="twin_build_", dir=output_dir))
        try:
            train = [sample for sample in samples if not self._is_validation_session(sample.session_id)]
            validation = [sample for sample in samples if self._is_validation_session(sample.session_id)]
            if not train and validation:
                train, validation = validation, []
            if not validation and len(train) > 1:
                validation = [train.pop()]

            self.stats.samples_train = len(train)
            self.stats.samples_validation = len(validation)
            train_path = temporary_dir / "train.jsonl"
            validation_path = temporary_dir / "validation.jsonl"
            all_samples_path = temporary_dir / "all_samples.json"
            rejection_path = temporary_dir / "rejections.jsonl"
            audit_path = temporary_dir / "audit_report.json"

            self._write_jsonl(train_path, ({"id": s.sample_id, "messages": list(s.messages)} for s in train))
            self._write_jsonl(validation_path, ({"id": s.sample_id, "messages": list(s.messages)} for s in validation))
            with all_samples_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "id": sample.sample_id,
                            "instruction": sample.instruction,
                            "response": sample.response,
                            "source": "whatsapp",
                        }
                        for sample in samples
                    ],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            self._write_jsonl(rejection_path, self._rejections)

            validation_result = self._validate_outputs(train_path, validation_path)
            if not validation_result["valid"]:
                raise RuntimeError(f"Output validation failed: {validation_result['errors']}")

            config = {
                "self_sender": self.self_sender,
                "context_turns": self.context_turns,
                "max_gap_hours": self.max_gap.total_seconds() / 3600,
                "burst_gap_seconds": self.burst_gap.total_seconds(),
                "min_response_chars": self.min_response_chars,
                "max_duplicate_responses": self.max_duplicate_responses,
                "validation_ratio": self.validation_ratio,
                "system_message": self.system_message,
                "explicit_redaction_terms_count": len(self.user_redaction_patterns),
            }
            audit = self.stats.to_json(config)
            audit["output_validation"] = validation_result
            with audit_path.open("w", encoding="utf-8") as handle:
                json.dump(audit, handle, ensure_ascii=False, indent=2)

            for generated in (train_path, validation_path, all_samples_path, rejection_path, audit_path):
                shutil.move(str(generated), output_dir / generated.name)
            return audit
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    def run(self, raw_file: Path, output_dir: Path) -> dict[str, object]:
        messages = self.parse_messages(raw_file)
        sessions = self.sessionize(messages)
        samples = self.build_samples(sessions)
        if not samples:
            raise RuntimeError(
                "No training samples were generated. Confirm --self-sender exactly matches your WhatsApp sender name."
            )
        return self.export(samples, output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local WhatsApp dataset for an Ahmed digital-twin fine-tuning experiment."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to a UTF-8 WhatsApp .txt export.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated dataset artifacts.")
    parser.add_argument("--self-sender", required=True, help="Your exact sender name in the WhatsApp export, e.g. Ahmed.")
    parser.add_argument("--context-turns", type=int, default=5, help="Maximum prior message turns in each sample.")
    parser.add_argument("--max-gap-hours", type=float, default=4.0, help="Start a new session after this silence period.")
    parser.add_argument(
        "--burst-gap-seconds",
        type=int,
        default=300,
        help=(
            "Merge same-sender bursts inside this interval. Also determines whether "
            "a multi-part reply typed with pauses still counts as one continuous "
            "assistant turn (see samples_dropped_due_to_burst_gap in the audit report)."
        ),
    )
    parser.add_argument("--min-response-chars", type=int, default=2, help="Reject messages shorter than this length.")
    parser.add_argument(
        "--max-duplicate-responses",
        type=int,
        default=3,
        help=(
            "Keep at most this many samples with the exact same response text "
            "(e.g. repeated short acknowledgements like 'اه'/'لا'/'تمام'). "
            "Set to 0 to disable the cap and keep every occurrence."
        ),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10, help="Session-level validation fraction.")
    parser.add_argument(
        "--redact-term",
        action="append",
        default=[],
        help="Literal sensitive term to replace with [REDACTED]. Repeat this option as needed.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        builder = WhatsAppDatasetBuilder(
            self_sender=args.self_sender,
            context_turns=args.context_turns,
            max_gap_hours=args.max_gap_hours,
            burst_gap_seconds=args.burst_gap_seconds,
            min_response_chars=args.min_response_chars,
            max_duplicate_responses=args.max_duplicate_responses,
            validation_ratio=args.validation_ratio,
            redact_terms=args.redact_term,
        )
        audit = builder.run(args.input.expanduser().resolve(), args.output_dir.expanduser().resolve())
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Build failed: %s", exc)
        return 1

    print("\nBuild completed successfully.")
    print(f"  Training samples:   {audit['samples_train']}")
    print(f"  Validation samples: {audit['samples_validation']}")
    print(f"  Total samples:      {audit['samples_total']}")
    print(f"  Output directory:   {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
