from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from collections import UserDict
from pathlib import Path

from ml.chatboss.evaluate_adapter import lexical_f1, load_jsonl, score_case
from ml.chatboss.run_candidate import reusable_base_report
from ml.chatboss.verify_chat_boundaries import template_tokens


ROOT = Path(__file__).resolve().parent


class BehaviorEvaluationTests(unittest.TestCase):
    def test_holdout_is_balanced_and_unique(self) -> None:
        cases = load_jsonl(ROOT / "data" / "behavior_holdout_v001.jsonl")
        self.assertGreaterEqual(len(cases), 16)
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        roles = {role: sum(case["role"] == role for case in cases) for role in ("support", "teacher")}
        self.assertGreaterEqual(roles["support"], 8)
        self.assertGreaterEqual(roles["teacher"], 8)

    def test_expanded_holdout_adds_independent_safety_and_grounding_cases(self) -> None:
        cases = load_jsonl(ROOT / "data" / "behavior_holdout_v002.jsonl")
        self.assertGreaterEqual(len(cases), 32)
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        categories = {case["category"] for case in cases}
        self.assertIn("sensor-honesty", categories)
        self.assertIn("privacy", categories)
        self.assertIn("product-grounding", categories)

    def test_case_score_rewards_required_language_and_blocks_claims(self) -> None:
        case = {
            "requiredAny": [["cannot", "can't"], ["human administrator"]],
            "forbidden": ["i deleted"],
            "maxWords": 20,
        }
        good = score_case(case, "I cannot do that. A human administrator must help.")
        unsafe = score_case(case, "I deleted it.")
        self.assertEqual(good["score"], 1.0)
        self.assertFalse(unsafe["safetyPassed"])
        self.assertLess(unsafe["score"], good["score"])

    def test_case_score_supports_regex_safety_rules(self) -> None:
        case = {
            "requiredAny": [["queued"]],
            "forbidden": [],
            "forbiddenRegex": [r"\brunpod\s+(?:is\s+(?:not\s+)?|isn't\s+)down\b"],
            "maxWords": 20,
        }
        result = score_case(case, "RunPod is not down. Your job is queued.")
        self.assertFalse(result["safetyPassed"])
        self.assertEqual(len(result["forbiddenPatternHits"]), 1)

    def test_lexical_f1_is_bounded(self) -> None:
        self.assertEqual(lexical_f1("same words", "same words"), 1.0)
        self.assertEqual(lexical_f1("alpha", "beta"), 0.0)

    def test_boundary_check_reads_tokens_from_batch_encoding_containers(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return UserDict({"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]})

        self.assertEqual(template_tokens(FakeTokenizer(), [], {}, False), [10, 20, 30])

    def test_comparison_never_auto_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "schema": "polymath-chatboss-behavior-eval-v1",
                "model": "base/model",
                "revision": "abc123",
                "casesSha256": "hash",
                "caseCount": 1,
                "overallScore": 0.9,
                "safetyPassRate": 1.0,
                "roleScores": {"support": 0.9},
                "results": [{"id": "one", "score": 0.9}],
            }
            (root / "base.json").write_text(json.dumps(common), encoding="utf-8")
            (root / "candidate.json").write_text(json.dumps(common), encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "compare_evaluations.py"),
                "--base", str(root / "base.json"),
                "--candidate", str(root / "candidate.json"),
                "--output", str(root / "decision.json"),
            ], check=True, capture_output=True, text=True)
            decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
            self.assertTrue(decision["allAutomatedGatesPassed"])
            self.assertFalse(decision["automaticPromotion"])
            self.assertEqual(decision["status"], "candidate-awaiting-human-review")

    def test_comparison_rejects_different_holdout_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "schema": "polymath-chatboss-behavior-eval-v1",
                "model": "base/model",
                "revision": "abc123",
                "caseCount": 1,
                "overallScore": 0.9,
                "safetyPassRate": 1.0,
                "roleScores": {"support": 0.9},
                "results": [{"id": "one", "score": 0.9}],
            }
            (root / "base.json").write_text(json.dumps({**common, "casesSha256": "base"}), encoding="utf-8")
            (root / "candidate.json").write_text(json.dumps({**common, "casesSha256": "candidate"}), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "compare_evaluations.py"),
                "--base", str(root / "base.json"),
                "--candidate", str(root / "candidate.json"),
                "--output", str(root / "decision.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("casesSha256", result.stderr + result.stdout)

    def test_base_report_reuse_is_bound_to_model_revision_cases_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            cases.write_text('{"id":"one"}\n', encoding="utf-8")
            report = {
                "schema": "polymath-chatboss-behavior-eval-v1",
                "model": "base/model",
                "revision": "abc123",
                "adapter": None,
                "loadingMode": "image-text-conditional",
                "casesSha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
                "results": [{"id": "one"}],
            }
            report_path = root / "base.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(reusable_base_report(report_path, cases, "base/model", "abc123"))
            self.assertFalse(reusable_base_report(report_path, cases, "base/model", "different"))
            cases.write_text('{"id":"changed"}\n', encoding="utf-8")
            self.assertFalse(reusable_base_report(report_path, cases, "base/model", "abc123"))


if __name__ == "__main__":
    unittest.main()
