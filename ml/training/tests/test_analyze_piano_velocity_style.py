import json
import tempfile
import unittest
from pathlib import Path

from ml.training.analyze_piano_velocity_style import analyze, load_notes, trusted_source_ranges


def note(midi, time, velocity, **extra):
    return {
        "midi": midi,
        "note": "fixture",
        "time": time,
        "duration": 0.3,
        "velocity": velocity,
        "hand": "left" if midi < 60 else "right",
        **extra,
    }


class PianoVelocityStyleAuditTests(unittest.TestCase):
    def test_reports_omissions_extras_and_gesture_coherence(self):
        reference = [note(48, 0, 0.6), note(60, 0, 0.6), note(64, 1, 0.8)]
        candidate = [
            note(48, 0.01, 0.7, velocityBeforeGestureCoherence=0.4, sourceInstrument="bass"),
            note(60, 0.01, 0.7, velocityBeforeGestureCoherence=0.9, sourceInstrument="voice"),
            note(67, 1, 0.5, sourceInstrument="guitar"),
        ]

        result = analyze(reference, candidate, 0.1, 0.035)

        self.assertEqual(result["matching"]["exactPitchMatches"], 2)
        self.assertEqual(result["omissions"]["notes"], 1)
        self.assertEqual(result["extras"]["notes"], 1)
        self.assertAlmostEqual(
            result["velocity"]["matchedExactPitch"]["meanAbsoluteError"],
            0.1,
        )
        self.assertEqual(
            result["gestureCoherence"]["candidate"]["sharedVelocityGroupShare"],
            1.0,
        )
        self.assertEqual(result["velocity"]["candidateChangeFromPreGesture"]["raised"], 1)
        self.assertEqual(result["velocity"]["candidateChangeFromPreGesture"]["lowered"], 1)
        self.assertEqual(
            result["gestureCoherence"]["reference"]["velocityByChordSize"][0]["size"],
            1,
        )
        by_role = {
            row["name"]: row
            for row in result["velocity"]["candidateChangeFromPreGesture"]["byRole"]
        }
        self.assertEqual(by_role["unknown"]["raised"], 1)
        self.assertEqual(by_role["unknown"]["lowered"], 1)

    def test_candidate_filter_uses_the_same_trusted_windows_and_hard_end(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            notes_path = root_path / "candidate.json"
            alignment_path = root_path / "alignment.json"
            notes_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {"midi": 60, "time": 1.0, "duration": 0.2, "velocity": 0.7},
                            {"midi": 61, "time": 2.5, "duration": 0.2, "velocity": 0.7},
                            {"midi": 62, "time": 5.0, "duration": 0.2, "velocity": 0.7},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            alignment_path.write_text(
                json.dumps(
                    {
                        "qualityWindows": [
                            {
                                "sourceStart": 0,
                                "sourceEnd": 2,
                                "status": "trusted",
                                "trainingEligible": True,
                            },
                            {
                                "sourceStart": 2,
                                "sourceEnd": 4,
                                "status": "unsafe",
                                "trainingEligible": False,
                            },
                            {
                                "sourceStart": 4,
                                "sourceEnd": 6,
                                "status": "trusted",
                                "trainingEligible": True,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ranges = trusted_source_ranges(alignment_path)
            loaded = load_notes(
                notes_path,
                include_ranges=ranges,
                end_seconds=5.0,
            )
            self.assertEqual([item["midi"] for item in loaded], [60])


if __name__ == "__main__":
    unittest.main()
