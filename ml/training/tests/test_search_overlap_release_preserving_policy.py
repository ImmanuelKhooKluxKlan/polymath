from ml.training.search_overlap_release_preserving_policy import (
    merge_song_predictions_release_preserving,
    pair_shifted_to_primary,
    rank_safe_candidates,
)


def window(start, duration=5.0):
    return {"sourceStart": start, "durationSeconds": duration}


def note(time, midi=60, duration=0.5):
    return {
        "time": time,
        "midi": midi,
        "duration": duration,
        "velocity": 0.7,
        "instrument": "acoustic_piano",
    }


def test_shared_shifted_note_preserves_primary_release_time():
    merged = merge_song_predictions_release_preserving(
        [note(4.98, duration=0.50)],
        [note(5.02, duration=2.00)],
        [window(0.0), window(5.0)],
        [window(2.5), window(7.5)],
        0.10,
        onset_match_tolerance_seconds=0.10,
    )

    assert len(merged) == 1
    assert merged[0]["time"] == 5.02
    assert abs(merged[0]["duration"] - 0.46) < 1e-9


def test_unmatched_recovered_note_keeps_shifted_duration():
    merged = merge_song_predictions_release_preserving(
        [],
        [note(5.02, midi=64, duration=0.75)],
        [window(0.0), window(5.0)],
        [window(2.5), window(7.5)],
        0.10,
        onset_match_tolerance_seconds=0.10,
    )

    assert len(merged) == 1
    assert merged[0]["midi"] == 64
    assert merged[0]["duration"] == 0.75


def test_primary_note_outside_boundary_zone_is_unchanged():
    merged = merge_song_predictions_release_preserving(
        [note(3.0, duration=0.4)],
        [note(3.03, duration=1.5)],
        [window(0.0), window(5.0)],
        [window(2.5), window(7.5)],
        0.10,
        onset_match_tolerance_seconds=0.10,
    )

    assert merged == [note(3.0, duration=0.4)]


def test_pairing_is_one_to_one_and_nearest_first():
    assignments = pair_shifted_to_primary(
        [note(5.01), note(5.08)],
        [note(5.00), note(5.10)],
        0.10,
    )

    assert assignments == {0: 0, 1: 1}


def test_only_gate_safe_candidates_can_win():
    rows = [
        {
            "radiusSeconds": 0.6,
            "continuousSong": {"metrics": {
                "100ms": {"microF1": 0.92, "recall": 0.91},
                "250ms": {"microF1": 0.94},
            }},
            "safetyGate": {"researchGatePassed": False},
        },
        {
            "radiusSeconds": 0.25,
            "continuousSong": {"metrics": {
                "100ms": {"microF1": 0.915, "recall": 0.905},
                "250ms": {"microF1": 0.93},
            }},
            "safetyGate": {"researchGatePassed": True},
        },
    ]

    ranked = rank_safe_candidates(rows)

    assert [row["radiusSeconds"] for row in ranked] == [0.25]
