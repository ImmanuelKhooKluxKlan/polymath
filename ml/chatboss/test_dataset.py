from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.chatboss.build_dataset import split_bucket, validate_record
from ml.chatboss.create_candidate_v001 import (
    SUPPORT_SCENARIOS,
    SUPPORT_SYSTEM,
    TEACHER_SCENARIOS,
    TEACHER_SYSTEM,
    records,
)
from ml.chatboss.create_candidate_v003 import SUPPORT as SUPPORT_V003
from ml.chatboss.create_candidate_v003 import SUPPORT_SYSTEM as SUPPORT_SYSTEM_V003
from ml.chatboss.create_candidate_v003 import TEACHER as TEACHER_V003
from ml.chatboss.create_candidate_v003 import TEACHER_SYSTEM as TEACHER_SYSTEM_V003
from ml.chatboss.create_candidate_v003 import records as records_v003


ROOT = Path(__file__).resolve().parent


class ChatBossDatasetTests(unittest.TestCase):
    def test_bootstrap_is_balanced_unique_and_large_enough(self) -> None:
        support = records("support", SUPPORT_SYSTEM, SUPPORT_SCENARIOS)
        teacher = records("teacher", TEACHER_SYSTEM, TEACHER_SCENARIOS)
        self.assertGreaterEqual(len(support), 60)
        self.assertGreaterEqual(len(teacher), 60)
        questions = [row["messages"][-2]["content"] for row in [*support, *teacher]]
        self.assertEqual(len(questions), len(set(questions)))

    def test_every_completion_is_short_and_role_grounded(self) -> None:
        rows = [
            *records("support", SUPPORT_SYSTEM, SUPPORT_SCENARIOS),
            *records("teacher", TEACHER_SYSTEM, TEACHER_SCENARIOS),
        ]
        for index, row in enumerate(rows, 1):
            cleaned = validate_record(row, "candidate", index)
            completion = cleaned["completion"][0]["content"]
            self.assertLessEqual(len(completion.split()), 90)
            self.assertIn(cleaned["role"], {"support", "teacher"})
            self.assertEqual(cleaned["chat_template_kwargs"], {"enable_thinking": False})

    def test_v003_supplement_is_balanced_and_valid(self) -> None:
        rows = [
            *records_v003("support", SUPPORT_SYSTEM_V003, SUPPORT_V003),
            *records_v003("teacher", TEACHER_SYSTEM_V003, TEACHER_V003),
        ]
        self.assertGreaterEqual(len(rows), 120)
        roles = {role: sum(row["role"] == role for row in rows) for role in ("support", "teacher")}
        self.assertGreaterEqual(roles["support"], 50)
        self.assertGreaterEqual(roles["teacher"], 50)
        for index, row in enumerate(rows, 1):
            validate_record(row, "v003", index)

    def test_variants_of_one_scenario_cannot_leak_across_the_split(self) -> None:
        rows = records_v003("teacher", TEACHER_SYSTEM_V003, TEACHER_V003)
        grouped: dict[str, set[int]] = {}
        for index, row in enumerate(rows, 1):
            cleaned = validate_record(row, "v003", index)
            grouped.setdefault(row["splitGroup"], set()).add(split_bucket(cleaned))
        self.assertTrue(grouped)
        self.assertTrue(all(len(buckets) == 1 for buckets in grouped.values()))

    def test_holdout_user_prompts_are_not_training_prompts(self) -> None:
        training_prompts = set()
        for filename in (
            ROOT / "data" / "seed_examples.jsonl",
            ROOT / "data" / "candidate_examples_v001.jsonl",
            ROOT / "data" / "candidate_examples_v003.jsonl",
        ):
            for line in filename.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                training_prompts.update(
                    message["content"].strip().lower()
                    for message in row["messages"]
                    if message["role"] == "user"
                )
        holdout_prompts = {
            message["content"].strip().lower()
            for line in (ROOT / "data" / "behavior_holdout_v002.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
            for message in json.loads(line)["messages"]
            if message["role"] == "user"
        }
        self.assertFalse(training_prompts & holdout_prompts)

    def test_sensitive_customer_data_is_rejected(self) -> None:
        unsafe = {
            "role": "support",
            "messages": [
                {"role": "system", "content": SUPPORT_SYSTEM},
                {"role": "user", "content": "My email is private.person@example.com"},
                {"role": "assistant", "content": "I saved it."},
            ],
        }
        with self.assertRaisesRegex(ValueError, "email address"):
            validate_record(unsafe, "unsafe", 1)


if __name__ == "__main__":
    unittest.main()
