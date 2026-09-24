import unittest

from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    evaluation_view,
    predictions_for_records,
    score_song_timelines,
)


class SongTimelineScoreTests(unittest.TestCase):
    def test_continuation_label_is_not_counted_as_required_restrike(self):
        records = [
            {
                "clipId": "a",
                "songId": "song",
                "sourceStart": 0,
                "instrumentFocus": "acoustic_piano",
                "notes": [{
                    "midi": 60,
                    "time": 4.5,
                    "duration": 0.5,
                    "continuesIntoNextClip": True,
                }],
            },
            {
                "clipId": "b",
                "songId": "song",
                "sourceStart": 5,
                "instrumentFocus": "acoustic_piano",
                "notes": [{
                    "midi": 60,
                    "time": 0,
                    "duration": 0.7,
                    "continuedFromPreviousClip": True,
                }],
            },
        ]
        predictions = [
            [{"midi": 60, "time": 4.5, "duration": 1.2}],
            [],
        ]
        result = score_song_timelines(records, predictions)
        self.assertEqual(result["boundaryAccounting"]["referenceContinuationMerges"], 1)
        self.assertEqual(result["metrics"]["100ms"]["referenceNotes"], 1)
        self.assertEqual(result["metrics"]["100ms"]["predictedNotes"], 1)
        self.assertEqual(result["metrics"]["100ms"]["microF1"], 1.0)
        self.assertEqual(result["diagnostics100ms"]["onsetAndOffset"]["f1"], 1.0)
        self.assertEqual(result["diagnostics100ms"]["frame"]["f1"], 1.0)

    def test_evaluation_pairing_fails_closed_on_timing_mismatch(self):
        records = [{"clipId": "a", "songId": "song", "sourceStart": 5}]
        evaluation = {
            "metrics": {
                "decodedClips": [{
                    "clipId": "a",
                    "songId": "song",
                    "sourceStart": 0,
                    "notes": [],
                }],
            },
        }
        with self.assertRaisesRegex(TimelineScoreError, "Source-start mismatch"):
            predictions_for_records(evaluation, records)

    def test_comparison_side_becomes_a_single_evaluation_view(self):
        comparison = {
            "schema": "polymath-checkpoint-comparison-v1",
            "baseCheckpoint": "/models/control.safetensors",
            "candidateCheckpoint": "/models/candidate.safetensors",
            "validationManifest": "/data/validation.jsonl",
            "clips": 1,
            "instrumentConstraint": ["piano"],
            "baseline": {"decodedClips": []},
            "candidate": {"decodedClips": [{"clipId": "one"}]},
        }
        view = evaluation_view(comparison, "candidate")
        self.assertEqual(view["checkpoint"], "/models/candidate.safetensors")
        self.assertEqual(view["metrics"], comparison["candidate"])

    def test_comparison_requires_an_explicit_side(self):
        comparison = {
            "schema": "polymath-checkpoint-comparison-v1",
            "baseline": {},
            "candidate": {},
        }
        with self.assertRaisesRegex(TimelineScoreError, "require --comparison-side"):
            evaluation_view(comparison)


if __name__ == "__main__":
    unittest.main()
