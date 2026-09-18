from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ml.training.finalize_blind_listening_verdict import canonical_hash, finalize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizeBlindListeningVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pack = self.root / "pack"
        self.pack.mkdir()
        self.profile = self.root / "profile.json"
        self.runtime = self.root / "runtime.py"
        self.baseline = self.root / "baseline.json"
        self.candidate = self.root / "candidate.json"
        self.profile.write_text('{}\n', encoding="utf-8")
        self.runtime.write_text('# frozen\n', encoding="utf-8")
        self.baseline.write_text('{"notes":[]}\n', encoding="utf-8")
        self.candidate.write_text('{"notes":[1]}\n', encoding="utf-8")
        payload = {
            "salt": "fixed-test-salt",
            "mapping": {"A": "candidate", "B": "baseline"},
            "baselineSha256": digest(self.baseline),
            "candidateSha256": digest(self.candidate),
        }
        commitment = canonical_hash(payload)
        (self.pack / "MAPPING-COMMITMENT.sha256").write_text(
            commitment + "\n", encoding="utf-8"
        )
        (self.pack / "SEALED-MAPPING.json").write_text(
            json.dumps(
                {
                    "schema": "polymath-real-video-blind-map-v1",
                    "commitment": commitment,
                    "commitmentPayload": payload,
                    "inputs": {
                        "baseline": {"path": str(self.baseline)},
                        "candidate": {"path": str(self.candidate)},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.freeze = self.root / "freeze.json"
        self.freeze.write_text(
            json.dumps(
                {
                    "schema": "polymath-piano-arranger-candidate-freeze-v1",
                    "candidate": {
                        "id": "candidate-v043-frozen",
                        "path": str(self.profile),
                        "fileSha256": digest(self.profile),
                    },
                    "runtime": {"files": {"runtime.py": digest(self.runtime)}},
                }
            ),
            encoding="utf-8",
        )
        self.result = self.root / "result.json"
        self.result.write_text(
            json.dumps(
                {
                    "schema": "polymath-piano-arranger-final-holdout-result-v1",
                    "candidate": {
                        "id": "candidate-v043-frozen",
                        "fileSha256": digest(self.profile),
                        "profileHashMatchedFreezeAtScoring": True,
                        "runtimeHashesMatchedFreezeAtScoring": True,
                        "retunedAfterFirstHoldoutScore": False,
                    },
                    "primaryResult": {
                        "id": "holdout",
                        "decision": "PROMOTE",
                        "allAutomatedPromotionGatesPassed": True,
                        "baseline": {
                            "exactF1At100ms": 0.8,
                            "physicalDurationMaeSeconds": 0.1,
                            "physicalSevereCutoffRate": 0.01,
                            "rapidRetriggersUnder100ms": 0,
                        },
                        "candidate": {
                            "exactF1At100ms": 0.9,
                            "physicalDurationMaeSeconds": 0.09,
                            "physicalSevereCutoffRate": 0.0,
                            "rapidRetriggersUnder100ms": 0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_finalize(self, choice: str, name: str):
        return finalize(
            freeze_path=self.freeze,
            result_path=self.result,
            pack=self.pack,
            choice=choice,
            reviewer="owner",
            repo_root=self.root,
            output_dir=self.root / name,
        )

    def test_candidate_choice_passes_and_pre_reveal_record_is_written(self) -> None:
        receipt = self.run_finalize("A", "candidate-win")
        self.assertEqual(receipt["status"], "PASS")
        pre_reveal = json.loads(
            (self.root / "candidate-win" / "PRE-REVEAL-CHOICE.json").read_text()
        )
        self.assertFalse(pre_reveal["mappingOpenedAtRecording"])

    def test_baseline_choice_fails_candidate_win_gate(self) -> None:
        receipt = self.run_finalize("B", "baseline-win")
        self.assertEqual(receipt["status"], "FAIL")
        self.assertIn("candidate_won_every_decisive_recording", receipt["failedChecks"])

    def test_tampered_commitment_fails_before_formal_mapping_is_written(self) -> None:
        (self.pack / "MAPPING-COMMITMENT.sha256").write_text("0" * 64 + "\n")
        output = self.root / "tampered"
        with self.assertRaisesRegex(ValueError, "commitment"):
            self.run_finalize("A", "tampered")
        self.assertFalse((output / "FORMAL-MAPPING.json").exists())


if __name__ == "__main__":
    unittest.main()
