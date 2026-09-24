from pathlib import Path

import pytest

from ml.training.apply_fixed_overlap_policy import (
    apply_fixed_policy,
    write_playable_songs,
)
from ml.training.rescore_song_timelines import TimelineScoreError


def record(clip_id: str, start: float, duration: float, pass_name: str) -> dict:
    return {
        "clipId": clip_id,
        "songId": "song",
        "sourceStart": start,
        "durationSeconds": duration,
        "instrumentFocus": "acoustic_piano",
        "notes": [],
        "labelsPresent": False,
        "inferencePass": {"name": pass_name},
    }


def evaluation(checkpoint: str, rows: list[dict], notes: list[list[dict]]) -> dict:
    return {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": checkpoint,
        "metrics": {
            "decodedClips": [
                {
                    "clipId": row["clipId"],
                    "songId": row["songId"],
                    "sourceStart": row["sourceStart"],
                    "notes": clip_notes,
                }
                for row, clip_notes in zip(rows, notes, strict=True)
            ]
        },
    }


def test_fixed_policy_emits_playable_unlabeled_song(tmp_path: Path) -> None:
    primary = [record("p0", 0.0, 5.0, "primary"), record("p1", 5.0, 5.0, "primary")]
    overlap = [record("o0", 2.5, 5.0, "overlap")]
    primary_eval = evaluation(
        "checkpoint-v2",
        primary,
        [
            [{"midi": 60, "time": 4.9, "duration": 0.6, "instrument": "acoustic_piano"}],
            [],
        ],
    )
    overlap_eval = evaluation(
        "checkpoint-v2",
        overlap,
        [[{"midi": 60, "time": 2.45, "duration": 0.3, "instrument": "acoustic_piano"}]],
    )
    result = apply_fixed_policy(
        primary_eval,
        overlap_eval,
        primary,
        overlap,
        radius_seconds=0.6,
        onset_match_tolerance_seconds=0.1,
    )
    assert result["metricsIncluded"] is False
    assert result["accuracyEvidenceAllowed"] is False
    assert result["songCount"] == 1
    assert result["noteCount"] == 1
    note = result["songs"][0]["notes"][0]
    assert note["midi"] == 60
    assert note["time"] == 4.95
    assert note["duration"] == pytest.approx(0.55)
    output = tmp_path / "playable"
    paths = write_playable_songs(result, output)
    assert paths == [output / "song.json"]
    assert paths[0].is_file()


def test_fixed_policy_rejects_mixed_checkpoints() -> None:
    primary = [record("p0", 0.0, 5.0, "primary")]
    overlap = [record("o0", 2.5, 2.5, "overlap")]
    with pytest.raises(TimelineScoreError, match="same checkpoint"):
        apply_fixed_policy(
            evaluation("a", primary, [[]]),
            evaluation("b", overlap, [[]]),
            primary,
            overlap,
            radius_seconds=0.6,
            onset_match_tolerance_seconds=0.1,
        )


def test_playable_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "playable"
    output.mkdir()
    with pytest.raises(TimelineScoreError, match="Refusing to overwrite"):
        write_playable_songs({"songs": []}, output)
