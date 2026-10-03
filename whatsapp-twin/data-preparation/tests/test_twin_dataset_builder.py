#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twin_dataset_builder import DEFAULT_SYSTEM_MESSAGE, WhatsAppDatasetBuilder


class WhatsAppDatasetBuilderTests(unittest.TestCase):
    def build_from_text(self, raw_text: str, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        raw_path = root / "chat.txt"
        output_dir = root / "outputs"
        raw_path.write_text(raw_text, encoding="utf-8")
        builder = WhatsAppDatasetBuilder(self_sender="Ahmed", validation_ratio=0.5, **kwargs)
        audit = builder.run(raw_path, output_dir)
        return output_dir, audit

    def test_arabic_export_becomes_role_based_training_records(self):
        output_dir, audit = self.build_from_text(
            "٢٢‏/٣‏/٢٠٢٣، ١٢:٠٠ ص - ‏صديق: السلام عليكم\n"
            "٢٢‏/٣‏/٢٠٢٣، ١٢:٠١ ص - ‏صديق: ازيك؟\n"
            "٢٢‏/٣‏/٢٠٢٣، ١٢:٠٢ ص - Ahmed: الحمد لله، تمام\n"
        )
        all_samples = json.loads((output_dir / "all_samples.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["samples_total"], 1)
        self.assertEqual(all_samples[0]["response"], "الحمد لله، تمام")
        self.assertIn("Contact: السلام عليكم\nازيك؟", all_samples[0]["instruction"])
        self.assertNotIn("صديق", all_samples[0]["instruction"])

    def test_pii_and_explicit_terms_are_redacted(self):
        output_dir, _ = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact Name: Email me@test.com or call 01012345678. https://example.com secret-word\n"
            "22/3/2023, 10:02 AM - Ahmed: حاضر\n",
            redact_terms=["secret-word"],
        )
        sample_text = (output_dir / "all_samples.json").read_text(encoding="utf-8")
        self.assertIn("[EMAIL]", sample_text)
        self.assertIn("[PHONE]", sample_text)
        self.assertIn("[URL]", sample_text)
        self.assertIn("[REDACTED]", sample_text)
        self.assertNotIn("Contact Name", sample_text)
        self.assertNotIn("secret-word", sample_text)

    def test_system_events_do_not_create_training_content(self):
        output_dir, audit = self.build_from_text(
            "22/3/2023, 10:00 AM - Messages and calls are end-to-end encrypted.\n"
            "22/3/2023, 10:01 AM - Contact: Hello\n"
            "22/3/2023, 10:02 AM - Ahmed: Hi\n"
        )
        self.assertGreaterEqual(audit["system_events_skipped"], 1)
        self.assertEqual(audit["samples_total"], 1)
        self.assertTrue((output_dir / "audit_report.json").is_file())

    def test_long_time_gap_breaks_context(self):
        output_dir, audit = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: first topic\n"
            "22/3/2023, 10:01 AM - Ahmed: first answer\n"
            "22/3/2023, 08:00 PM - Contact: second topic\n"
            "22/3/2023, 08:01 PM - Ahmed: second answer\n",
            max_gap_hours=4,
        )
        samples = json.loads((output_dir / "all_samples.json").read_text(encoding="utf-8"))
        second = next(sample for sample in samples if sample["response"] == "second answer")
        self.assertNotIn("first topic", second["instruction"])
        self.assertGreaterEqual(audit["sessions_total"], 2)

    def test_jsonl_records_are_valid_chat_records(self):
        output_dir, audit = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: hello\n"
            "22/3/2023, 10:01 AM - Ahmed: hi\n"
            "22/3/2023, 10:02 AM - Contact: how are you\n"
            "22/3/2023, 10:03 AM - Ahmed: doing well\n"
        )
        self.assertTrue(audit["output_validation"]["valid"])
        seen_records = 0
        for file_name in ("train.jsonl", "validation.jsonl"):
            for line in (output_dir / file_name).read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                seen_records += 1
                self.assertEqual(record["messages"][0], {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE})
                self.assertEqual(record["messages"][-1]["role"], "assistant")
        self.assertEqual(seen_records, audit["samples_total"])


    def test_custom_system_message_still_validates(self):
        # Regression test: _validate_chat_record used to compare every record
        # against the module-level DEFAULT_SYSTEM_MESSAGE constant instead of
        # the instance's actual system_message, so any custom system message
        # made every record fail validation even though the data was correct.
        output_dir, audit = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: hello\n"
            "22/3/2023, 10:01 AM - Ahmed: hi\n",
            system_message="A custom system prompt, different from the default.",
        )
        self.assertTrue(audit["output_validation"]["valid"])
        first_line = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        self.assertEqual(
            record["messages"][0],
            {"role": "system", "content": "A custom system prompt, different from the default."},
        )

    def test_credential_like_token_is_redacted(self):
        # Regression test: a word glued directly to a long digit run (e.g. a
        # shared login/PIN like "Example12345678") is not a phone number, so
        # PHONE_PATTERN's lookbehind deliberately does not match it. It should
        # still be caught and redacted as a credential-like token.
        output_dir, _ = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: جربي Example12345678 وقوليلي\n"
            "22/3/2023, 10:02 AM - Ahmed: تمام\n"
        )
        sample_text = (output_dir / "all_samples.json").read_text(encoding="utf-8")
        self.assertNotIn("Example12345678", sample_text)
        self.assertIn("[CREDENTIAL]", sample_text)

    def test_duplicate_responses_are_capped(self):
        # Regression test: real chat data has hundreds of exact-duplicate
        # short acknowledgements ("اه", "تمام", ...) with different context
        # each time, so full-sample dedup alone doesn't limit them. Only the
        # first max_duplicate_responses occurrences of the exact same
        # response text should be kept.
        lines = []
        minute = 0
        for i in range(5):
            lines.append(f"22/3/2023, 10:{minute:02d} AM - Contact: question number {i}\n")
            minute += 1
            lines.append(f"22/3/2023, 10:{minute:02d} AM - Ahmed: تمام\n")
            minute += 1
        output_dir, audit = self.build_from_text("".join(lines), max_duplicate_responses=2)
        self.assertEqual(audit["samples_total"], 2)
        self.assertEqual(audit["rejection_reasons"].get("duplicate_response_capped"), 3)

    def test_duplicate_response_cap_disabled_by_zero(self):
        lines = []
        minute = 0
        for i in range(4):
            lines.append(f"22/3/2023, 10:{minute:02d} AM - Contact: question number {i}\n")
            minute += 1
            lines.append(f"22/3/2023, 10:{minute:02d} AM - Ahmed: تمام\n")
            minute += 1
        output_dir, audit = self.build_from_text("".join(lines), max_duplicate_responses=0)
        self.assertEqual(audit["samples_total"], 4)
        self.assertNotIn("duplicate_response_capped", audit["rejection_reasons"])

    def test_edited_message_tag_is_stripped(self):
        # Regression test: WhatsApp appends "<تم تعديل هذه الرسالة>" onto the
        # text of a message the sender edited after sending. Left unstripped,
        # this leaked into ~11% of real training records (3,142 of 27,877).
        output_dir, _ = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: hello\n"
            "22/3/2023, 10:01 AM - Ahmed: تمام هوصلك دلوقتي <تم تعديل هذه الرسالة>\n"
        )
        sample_text = (output_dir / "all_samples.json").read_text(encoding="utf-8")
        self.assertNotIn("تعديل", sample_text)
        self.assertIn("تمام هوصلك دلوقتي", sample_text)

    def test_wider_burst_gap_merges_paused_multipart_reply(self):
        # Regression check for the default burst_gap_seconds: a multi-part
        # Ahmed reply typed with a pause in the middle used to be split into
        # two turns and the second half silently dropped
        # (no_immediate_contact_prompt), losing real training signal.
        output_dir, audit = self.build_from_text(
            "22/3/2023, 10:00 AM - Contact: how do i fix this\n"
            "22/3/2023, 10:01 AM - Ahmed: first part of the answer\n"
            "22/3/2023, 10:04 AM - Ahmed: second part of the answer\n",
            burst_gap_seconds=300,
        )
        self.assertEqual(audit["samples_total"], 1)
        samples = json.loads((output_dir / "all_samples.json").read_text(encoding="utf-8"))
        self.assertIn("first part of the answer", samples[0]["response"])
        self.assertIn("second part of the answer", samples[0]["response"])


if __name__ == "__main__":
    unittest.main()
