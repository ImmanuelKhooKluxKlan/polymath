import unittest

from ml.training.extract_checkpoint_evaluation import extract_evaluation
from ml.training.rescore_song_timelines import TimelineScoreError


def comparison(*, include_checkpoint=True, clips=1):
    payload = {
        "schema": "polymath-checkpoint-comparison-v1",
        "validationManifest": "/data/validation.jsonl",
        "clips": clips,
        "instrumentConstraint": ["acoustic_piano"],
        "baseline": {
            "100ms": {"microF1": 0.80},
            "decodedClips": [{
                "clipId": "clip-1",
                "songId": "song-1",
                "sourceStart": 0.0,
                "notes": [],
            }],
        },
        "candidate": {
            "100ms": {"microF1": 0.81},
            "decodedClips": [{
                "clipId": "clip-1",
                "songId": "song-1",
                "sourceStart": 0.0,
                "notes": [{"time": 0.2, "midi": 60, "duration": 0.4}],
            }],
        },
    }
    if include_checkpoint:
        payload.update({
            "baseCheckpoint": "/models/original/model.safetensors",
            "candidateCheckpoint": "/models/candidate/model.safetensors",
        })
    return payload


class ExtractCheckpointEvaluationTests(unittest.TestCase):
    def test_extracts_embedded_candidate_identity_and_predictions(self):
        result = extract_evaluation(comparison(), "candidate")

        self.assertEqual(
            result["checkpoint"], "/models/candidate/model.safetensors"
        )
        self.assertEqual(result["metrics"]["100ms"]["microF1"], 0.81)
        self.assertEqual(result["extraction"]["side"], "candidate")

    def test_requires_explicit_identity_for_legacy_comparison(self):
        with self.assertRaisesRegex(TimelineScoreError, "explicit checkpoint"):
            extract_evaluation(comparison(include_checkpoint=False), "candidate")

        result = extract_evaluation(
            comparison(include_checkpoint=False),
            "candidate",
            checkpoint="phase94-v002-note-on-135:model.safetensors",
            comparison_source="comparison.json",
            comparison_sha256="abc123",
        )
        self.assertEqual(
            result["checkpoint"], "phase94-v002-note-on-135:model.safetensors"
        )
        self.assertEqual(result["extraction"]["sourceSha256"], "abc123")

    def test_rejects_declared_clip_count_mismatch(self):
        with self.assertRaisesRegex(TimelineScoreError, "declares 2 clips"):
            extract_evaluation(comparison(clips=2), "baseline")

    def test_rejects_unknown_schema(self):
        payload = comparison()
        payload["schema"] = "unknown"
        with self.assertRaisesRegex(TimelineScoreError, "not a paired"):
            extract_evaluation(payload, "candidate")


if __name__ == "__main__":
    unittest.main()
