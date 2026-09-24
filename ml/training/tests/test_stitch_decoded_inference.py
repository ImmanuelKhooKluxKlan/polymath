from pathlib import Path

import pytest

from ml.training.rescore_song_timelines import TimelineScoreError
from ml.training.stitch_decoded_inference import stitch_decoded_inference


def test_stitch_decoded_inference_merges_clip_boundary_and_marks_unlabeled() -> None:
    records = [
        {
            "clipId": "p0",
            "songId": "song",
            "sourceStart": 0.0,
            "durationSeconds": 5.0,
            "instrumentFocus": "acoustic_piano",
            "labelsPresent": False,
        },
        {
            "clipId": "p1",
            "songId": "song",
            "sourceStart": 5.0,
            "durationSeconds": 2.0,
            "instrumentFocus": "acoustic_piano",
            "labelsPresent": False,
        },
    ]
    evaluation = {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": "v2",
        "metrics": {
            "decodedClips": [
                {
                    "clipId": "p0",
                    "songId": "song",
                    "sourceStart": 0.0,
                    "notes": [{"midi": 60, "time": 4.8, "duration": 0.3}],
                },
                {
                    "clipId": "p1",
                    "songId": "song",
                    "sourceStart": 5.0,
                    "notes": [{"midi": 60, "time": 0.0, "duration": 0.4}],
                },
            ]
        },
    }
    result = stitch_decoded_inference(evaluation, records)
    assert result["metricsIncluded"] is False
    assert result["accuracyEvidenceAllowed"] is False
    assert result["boundaryAccounting"]["primaryPredictionMerges"] == 1
    assert result["songCount"] == 1
    assert result["songs"][0]["durationSeconds"] == 7.0
    assert result["songs"][0]["noteCount"] == 1
    assert result["songs"][0]["notes"][0]["duration"] == pytest.approx(0.6)


def test_stitch_requires_checkpoint_identity() -> None:
    with pytest.raises(TimelineScoreError, match="checkpoint identity"):
        stitch_decoded_inference(
            {"schema": "polymath-checkpoint-evaluation-v1", "metrics": {"decodedClips": []}},
            [],
        )
