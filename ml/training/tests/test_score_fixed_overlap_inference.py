from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.rescore_song_timelines import TimelineScoreError, sha256_file
from ml.training.score_fixed_overlap_inference import score_files, score_fixed_overlap


def note(midi: int, time: float, duration: float = 0.3) -> dict:
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "velocity": 0.7,
        "instrument": "acoustic_piano",
    }


def prediction() -> dict:
    return {
        "schema": "polymath-fixed-overlap-inference-v1",
        "metricsIncluded": False,
        "songs": [
            {"songId": "song-a", "notes": [note(60, 0.1), note(64, 5.1)]},
            {"songId": "song-b", "notes": [note(67, 0.2)]},
        ],
    }


def references() -> list[dict]:
    return [
        {
            "clipId": "a-0", "songId": "song-a", "split": "test",
            "sourceStart": 0.0, "durationSeconds": 5.0,
            "notes": [note(60, 0.1)],
        },
        {
            "clipId": "a-1", "songId": "song-a", "split": "test",
            "sourceStart": 5.0, "durationSeconds": 5.0,
            "notes": [note(64, 0.1)],
        },
        {
            "clipId": "b-0", "songId": "song-b", "split": "test",
            "sourceStart": 0.0, "durationSeconds": 5.0,
            "notes": [note(67, 0.2)],
        },
    ]


def test_scores_locked_full_song_prediction_without_changing_it() -> None:
    source = prediction()
    before = json.dumps(source, sort_keys=True)
    result = score_fixed_overlap(source, references())
    assert result["songs"] == 2
    assert result["metrics"]["100ms"]["microF1"] == pytest.approx(1.0)
    assert json.dumps(source, sort_keys=True) == before


def test_rejects_song_identity_mismatch() -> None:
    source = prediction()
    source["songs"].pop()
    with pytest.raises(TimelineScoreError, match="different songs"):
        score_fixed_overlap(source, references())


def test_file_score_binds_both_input_hashes(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.json"
    manifest_path = tmp_path / "test.jsonl"
    prediction_path.write_text(json.dumps(prediction()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in references()),
        encoding="utf-8",
    )
    result = score_files(prediction_path, manifest_path)
    assert result["sourcePredictionSha256"] == sha256_file(prediction_path)
    assert result["validationManifestSha256"] == sha256_file(manifest_path)
    assert result["predictionChangedDuringScoring"] is False


def test_rejects_prediction_that_already_contains_metrics() -> None:
    source = prediction()
    source["metricsIncluded"] = True
    with pytest.raises(TimelineScoreError, match="before scoring"):
        score_fixed_overlap(source, references())
