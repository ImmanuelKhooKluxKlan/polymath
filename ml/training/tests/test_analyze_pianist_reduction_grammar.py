from ml.training.analyze_pianist_reduction_grammar import (
    chord_size_counts,
    error_summary,
    occupancy,
    occupancy_counts,
    source_rank_labels,
)


def note(midi, *, hand=None, velocity=0.7, duration=0.2, source_index=0):
    value = {
        "midi": midi,
        "time": 0.0,
        "duration": duration,
        "velocity": velocity,
        "sourceIndex": source_index,
    }
    if hand:
        value["hand"] = hand
    return value


def test_hand_occupancy_prefers_explicit_reference_hands():
    group = [
        note(72, hand="left", source_index=0),
        note(48, hand="right", source_index=1),
    ]
    assert occupancy(group) == "both"
    assert occupancy_counts([group, [note(48, source_index=2)]]) == {
        "left-only": 1,
        "right-only": 0,
        "both": 1,
    }


def test_chord_counts_distinguish_octave_doubling_from_pitch_classes():
    groups = [[note(48), note(60, source_index=1)], [note(64, source_index=2)]]
    assert chord_size_counts(groups, pitch_class=False) == {"1": 1, "2": 1}
    assert chord_size_counts(groups, pitch_class=True) == {"1": 2}


def test_source_ranks_preserve_velocity_ties():
    group = [
        note(48, velocity=0.8, duration=0.4, source_index=0),
        note(55, velocity=0.8, duration=0.2, source_index=1),
        note(60, velocity=0.8, duration=0.1, source_index=2),
    ]
    labels = source_rank_labels(group)
    assert {row["velocityRank"] for row in labels.values()} == {
        "all-velocity-tied"
    }
    assert labels[0]["pitchRank"] == "lowest"
    assert labels[2]["pitchRank"] == "highest"
    assert labels[0]["durationRank"] == "longest"


def test_error_summary_reports_bias_mae_and_correlation():
    result = error_summary([0.2, 0.4, 0.6], [0.1, 0.5, 0.5])
    assert result["count"] == 3
    assert result["bias"] == 0.033333
    assert result["mae"] == 0.1
    assert result["correlation"] is not None
