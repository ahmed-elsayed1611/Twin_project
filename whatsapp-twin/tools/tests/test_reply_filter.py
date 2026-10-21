"""Tests for tools/reply_filter.py (standard library only).

    python -m unittest discover -s tools/tests -v
"""

import json
import random
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from reply_filter import FALLBACKS, clean_reply, problems, safe_reply  # noqa: E402


def fake_find_names(text):
    return ["زكريا"] if "زكريا" in text else []


class CleanReplyTests(unittest.TestCase):
    def test_placeholders_and_the_vocative_before_them_are_removed(self):
        self.assertEqual(clean_reply("الحمدلله علي السلامه يا [NAME] ❤️"), "الحمدلله علي السلامه ❤️")
        self.assertEqual(clean_reply("[PHONE]\nاه"), "اه")

    def test_a_line_repeated_right_after_itself_is_kept_once(self):
        self.assertEqual(clean_reply("كنت بعمل حاجه\nكنت بعمل حاجه"), "كنت بعمل حاجه")

    def test_a_reply_made_only_of_placeholders_becomes_empty(self):
        self.assertEqual(clean_reply("[NAME] [NAME]\n[NAME]"), "")


class ProblemTests(unittest.TestCase):
    def test_a_normal_reply_is_safe(self):
        self.assertEqual(problems("الساعه 12 إن شاء الله", fake_find_names), [])

    def test_each_problem_is_named(self):
        self.assertEqual(problems(""), ["empty"])
        self.assertIn("private_name", problems("كلم زكريا", fake_find_names))
        self.assertIn("long_number", problems("الرقم 01012345678"))
        self.assertIn("repeated_characters", problems("جداااااااااا"))
        self.assertIn("repeated_text", problems("و بقرا كتب و بقرا كتب\nلا\nلا\nلا"))
        self.assertIn("repeated_text", problems("مش مش مش مش"))

    def test_short_numbers_and_normal_emphasis_are_allowed(self):
        self.assertEqual(problems("هجيلك الساعه 3 ولا 12"), [])
        self.assertEqual(problems("بحب القهوه جدااااا"), [])


class SafeReplyTests(unittest.TestCase):
    def test_returns_the_first_safe_reply(self):
        attempts = iter(["كلم زكريا", "يا [NAME] انا في البيت"])
        reply, info = safe_reply(lambda: next(attempts), fake_find_names)
        self.assertEqual(reply, "انا في البيت")
        self.assertEqual(info, {"attempts": 2, "fallback": False, "rejected": [["private_name"]]})

    def test_falls_back_after_the_tries_and_never_returns_unsafe_text(self):
        reply, info = safe_reply(lambda: "[PHONE]", fake_find_names, tries=3, rng=random.Random(0))
        self.assertIn(reply, FALLBACKS)
        self.assertTrue(info["fallback"])
        self.assertEqual(info["rejected"], [["empty"]] * 3)

    def test_the_info_never_contains_reply_text(self):
        _, info = safe_reply(lambda: "كلم زكريا", fake_find_names, tries=2, rng=random.Random(0))
        self.assertNotIn("زكريا", json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
