import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.training.verify_piano_arranger_promotion import verify_promotion


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PianoArrangerPromotionVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile = self.root / "candidate.json"
        self.runtime = self.root / "runtime.py"
        self.profile.write_text('{"id":"candidate-v031"}\n', encoding="utf-8")
        self.runtime.write_text("# frozen runtime\n", encoding="utf-8")
        self.freeze = {
            "schema": "polymath-piano-arranger-candidate-freeze-v1",
            "candidate": {
                "id": "candidate-v031",
                "path": str(self.profile),
                "fileSha256": digest(self.profile),
            },
            "runtime": {"files": {"runtime.py": digest(self.runtime)}},
        }
        self.result = {
            "schema": "polymath-piano-arranger-final-holdout-result-v1",
            "candidate": {
                "id": "candidate-v031",
                "fileSha256": digest(self.profile),
                "profileHashMatchedFreezeAtScoring": True,
                "runtimeHashesMatchedFreezeAtScoring": True,
                "retunedAfterFirstHoldoutScore": False,
            },
            "primaryResult": {
                "decision": "PROMOTE",
                "allAutomatedPromotionGatesPassed": True,
                "baseline": {
                    "exactF1At100ms": 0.6,
                    "physicalDurationMaeSeconds": 0.07,
                    "physicalSevereCutoffRate": 0.01,
                    "rapidRetriggersUnder100ms": 0,
                },
                "candidate": {
                    "exactF1At100ms": 0.9,
                    "physicalDurationMaeSeconds": 0.06,
                    "physicalSevereCutoffRate": 0.0,
                    "rapidRetriggersUnder100ms": 0,
                },
            },
        }
        self.mapping = {
            "schema": "polymath-blind-listening-map-v1",
            "candidateAorB": {
                "supporting": {"A": "baseline v006", "B": "candidate v031"},
                "decisive": {"A": "candidate v031", "B": "baseline v006"},
            },
        }
        self.verdict = {
            "schema": "polymath-piano-arranger-blind-verdict-v1",
            "reviewer": "owner",
            "completedAtUtc": "2026-09-11T15:00:00Z",
            "samePlaybackChain": True,
            "mappingOpenedBeforeVerdict": False,
            "decisiveRecordingIds": ["decisive"],
            "recordings": {
                "supporting": {"winner": "B", "blockerHeard": False, "notes": "clean"},
                "decisive": {"winner": "A", "blockerHeard": False, "notes": "clean"},
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        return verify_promotion(
            freeze=self.freeze,
            result=self.result,
            mapping=self.mapping,
            verdict=self.verdict,
            repo_root=self.root,
        )

    def test_passes_only_when_frozen_automated_and_blind_gates_agree(self):
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["commercialDeploymentAuthorized"])
        self.assertEqual(receipt["scope"], "research-only")

    def test_fails_when_decisive_blind_recording_prefers_baseline(self):
        self.verdict["recordings"]["decisive"]["winner"] = "B"
        receipt = self.verify()
        self.assertEqual(receipt["status"], "FAIL")
        self.assertIn("candidate_won_every_decisive_recording", receipt["failedChecks"])

    def test_fails_when_reviewer_hears_a_blocker(self):
        self.verdict["recordings"]["supporting"]["blockerHeard"] = True
        receipt = self.verify()
        self.assertEqual(receipt["status"], "FAIL")
        self.assertIn("blind_review_has_no_blocker", receipt["failedChecks"])

    def test_fails_after_frozen_profile_is_changed(self):
        self.profile.write_text('{"id":"tampered"}\n', encoding="utf-8")
        receipt = self.verify()
        self.assertEqual(receipt["status"], "FAIL")
        self.assertIn("candidate_hash_matches_freeze", receipt["failedChecks"])

    def test_fails_when_automated_gate_was_not_passed(self):
        self.result["primaryResult"]["allAutomatedPromotionGatesPassed"] = False
        receipt = self.verify()
        self.assertEqual(receipt["status"], "FAIL")
        self.assertIn("all_automated_promotion_gates_passed", receipt["failedChecks"])

    def test_incomplete_rows_fail_every_dependent_blind_gate(self):
        self.verdict["recordings"]["decisive"]["winner"] = ""
        receipt = self.verify()
        self.assertEqual(receipt["status"], "FAIL")
        self.assertFalse(receipt["checks"]["blind_rows_complete"])
        self.assertFalse(receipt["checks"]["blind_review_has_no_blocker"])
        self.assertFalse(receipt["checks"]["candidate_won_every_decisive_recording"])
        self.assertFalse(receipt["checks"]["candidate_did_not_lose_supporting_recordings"])


if __name__ == "__main__":
    unittest.main()
