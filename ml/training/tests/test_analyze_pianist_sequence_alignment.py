import json
import tempfile
import unittest
from pathlib import Path

from ml.training.analyze_pianist_sequence_alignment import analyze_song, align_sequences


def group(midi, time):
    return [{"midi": midi, "time": time, "duration": 0.2, "velocity": 0.7}]


class PianistSequenceAlignmentTests(unittest.TestCase):
    def test_shifted_stream_stays_monotonic(self):
        reference = [group(60, 0.02), group(62, 0.31), group(64, 0.61)]
        candidate = [group(60, 0.0), group(62, 0.3), group(64, 0.6)]

        matches, missing, extra, _score = align_sequences(reference, candidate)

        self.assertEqual(matches, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])

    def test_real_insertion_is_left_unmatched(self):
        reference = [group(60, 0.0), group(61, 0.15), group(62, 0.3)]
        candidate = [group(60, 0.0), group(62, 0.3)]

        matches, missing, extra, _score = align_sequences(reference, candidate)

        self.assertEqual(matches, [(0, 0), (2, 1)])
        self.assertEqual(missing, [1])
        self.assertEqual(extra, [])

    def test_already_aligned_reference_is_not_warped_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            alignment = root / "alignment.json"
            reference.write_text(
                json.dumps({"notes": [group(60, 1.0)[0]]}), encoding="utf-8"
            )
            candidate.write_text(
                json.dumps({"notes": [group(60, 1.0)[0]]}), encoding="utf-8"
            )
            alignment.write_text(
                json.dumps(
                    {
                        "anchors": [
                            {"referenceTime": 0.0, "observedTime": 10.0},
                            {"referenceTime": 2.0, "observedTime": 12.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = analyze_song(
                {
                    "id": "already-aligned",
                    "reference": str(reference),
                    "candidate": str(candidate),
                    "alignment": str(alignment),
                    "referenceAlreadyAligned": True,
                },
                onset_window=0.035,
                maximum_time_distance=0.1,
                gap_cost=0.85,
            )

            self.assertEqual(report["alignment"]["matchedGestures"], 1)
            self.assertTrue(report["boundary"]["referenceAlreadyAligned"])
            self.assertEqual(report["pairs"][0]["referenceMidis"], [60])
            self.assertEqual(report["pairs"][0]["missingPitchClasses"], [])


if __name__ == "__main__":
    unittest.main()
