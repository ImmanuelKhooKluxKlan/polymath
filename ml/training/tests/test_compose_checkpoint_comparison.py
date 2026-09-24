import unittest

from ml.training.compose_checkpoint_comparison import compose_comparison
from ml.training.rescore_song_timelines import TimelineScoreError


def evaluation(checkpoint, f1, source_start=0.0):
    return {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": checkpoint,
        "validationManifest": "/data/validation.jsonl",
        "clips": 1,
        "instrumentConstraint": ["acoustic_piano"],
        "metrics": {
            "50ms": {"microF1": f1 - 0.1},
            "100ms": {"microF1": f1},
            "250ms": {"microF1": f1 + 0.05},
            "decodedClips": [{
                "clipId": "clip-1",
                "songId": "song-1",
                "sourceStart": source_start,
                "notes": [],
            }],
        },
    }


class ComposeCheckpointComparisonTests(unittest.TestCase):
    def test_composes_matching_frozen_evaluations(self):
        result = compose_comparison(evaluation("base", 0.8), evaluation("candidate", 0.81))
        self.assertEqual(result["baseCheckpoint"], "base")
        self.assertEqual(result["candidateCheckpoint"], "candidate")
        self.assertEqual(result["candidateMinusBaselineMicroF1"]["100ms"], 0.01)

    def test_fails_closed_when_clip_timing_differs(self):
        with self.assertRaisesRegex(TimelineScoreError, "identities or timings differ"):
            compose_comparison(
                evaluation("base", 0.8, 0.0),
                evaluation("candidate", 0.81, 5.0),
            )


if __name__ == "__main__":
    unittest.main()
