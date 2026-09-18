from ml.training.apply_raw_support_gesture_swap import (
    best_by_pitch_class,
    onset_groups,
)


def test_onset_groups_keep_near_simultaneous_notes_together():
    notes = [
        {"midi": 60, "time": 1.0},
        {"midi": 64, "time": 1.02},
        {"midi": 67, "time": 1.10},
    ]
    assert onset_groups(notes) == [[0, 1], [2]]


def test_best_by_pitch_class_keeps_highest_probability_source():
    notes = [
        {"midi": 48},
        {"midi": 60},
        {"midi": 49},
    ]
    assert best_by_pitch_class([0, 1, 2], notes, [0.4, 0.8, 0.7]) == {
        0: 1,
        1: 2,
    }
