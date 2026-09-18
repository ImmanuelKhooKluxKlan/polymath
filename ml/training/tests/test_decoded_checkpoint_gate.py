import pytest

from ml.training.decoded_checkpoint_gate import (
    decoded_checkpoint_gate,
    select_decoded_gate_records,
)


def metrics(f1_100=0.8, recall_100=0.79, f1_250=0.9, recall_250=0.89, songs=None):
    songs = songs or {
        "song-a": {
            "100ms": {"microF1": f1_100},
            "250ms": {"microF1": f1_250},
        }
    }
    return {
        "100ms": {"microF1": f1_100, "recall": recall_100},
        "250ms": {"microF1": f1_250, "recall": recall_250},
        "perSongClipScores": songs,
    }


def test_gate_record_panel_is_balanced_spread_and_keeps_reviewed_silence():
    records = []
    for song in ("a", "b"):
        for index in range(10):
            records.append({
                "songId": song,
                "clipId": f"{song}-{index}",
                "sourceStart": index * 5.0,
                "isNegativeExample": song == "a" and index == 4,
            })

    selected = select_decoded_gate_records(records, maximum_per_song=4)

    assert len(selected) == 8
    assert sum(row["songId"] == "a" for row in selected) == 4
    assert sum(row["songId"] == "b" for row in selected) == 4
    assert any(row["clipId"] == "a-4" for row in selected)
    assert {row["clipId"] for row in selected if row["songId"] == "b"} == {
        "b-0", "b-3", "b-6", "b-9",
    }


def test_identical_decoded_candidate_passes():
    result = decoded_checkpoint_gate(metrics(), metrics())
    assert result["passed"] is True
    assert result["failedChecks"] == []


def test_aggregate_f1_regression_rejects_candidate_even_when_recall_rises():
    baseline = metrics()
    candidate = metrics(f1_100=0.793, recall_100=0.795, f1_250=0.899)

    result = decoded_checkpoint_gate(baseline, candidate)

    assert result["passed"] is False
    assert "aggregate_100ms_f1" in result["failedChecks"]


def test_per_song_regression_cannot_hide_behind_aggregate_improvement():
    baseline_songs = {
        "a": {"100ms": {"microF1": 0.8}, "250ms": {"microF1": 0.9}},
        "b": {"100ms": {"microF1": 0.8}, "250ms": {"microF1": 0.9}},
    }
    candidate_songs = {
        "a": {"100ms": {"microF1": 0.85}, "250ms": {"microF1": 0.92}},
        "b": {"100ms": {"microF1": 0.78}, "250ms": {"microF1": 0.89}},
    }
    result = decoded_checkpoint_gate(
        metrics(f1_100=0.82, f1_250=0.91, songs=baseline_songs),
        metrics(f1_100=0.82, f1_250=0.91, songs=candidate_songs),
    )

    assert result["passed"] is False
    assert "song_b_100ms_f1" in result["failedChecks"]


def test_gate_rejects_mismatched_song_sets():
    candidate = metrics(songs={
        "different": {"100ms": {"microF1": 0.8}, "250ms": {"microF1": 0.9}},
    })
    with pytest.raises(ValueError, match="different songs"):
        decoded_checkpoint_gate(metrics(), candidate)
