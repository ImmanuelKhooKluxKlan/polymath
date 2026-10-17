from ml.training.analyze_pianist_register_errors import (
    decorate_gestures,
    duplicate_layer_summary,
    proportion_rows,
    register_training_pairs,
    shift_band,
)


def test_shift_band_names_octave_direction():
    assert shift_band(-12) == "down-1-octave"
    assert shift_band(0) == "exact-octave"
    assert shift_band(12) == "up-1-octave"


def test_gesture_decoration_marks_positions_and_octave_layers():
    notes = [
        {"time": 1.0, "midi": 48, "_payloadIndex": 0},
        {"time": 1.0, "midi": 60, "_payloadIndex": 1},
        {"time": 1.0, "midi": 67, "_payloadIndex": 2},
    ]
    decorated = decorate_gestures(notes)
    assert decorated[0]["isLowest"] is True
    assert decorated[2]["isHighest"] is True
    assert decorated[1]["samePitchClassLayers"] == 2


def test_duplicate_layer_summary_requires_same_source_and_onset():
    notes = [
        {"time": 1.0, "midi": 48, "sourceIndex": 7, "hand": "left"},
        {"time": 1.0, "midi": 60, "sourceIndex": 7, "hand": "right"},
        {"time": 1.2, "midi": 60, "sourceIndex": 7, "hand": "right"},
    ]
    summary = duplicate_layer_summary(notes)
    assert summary["sameSourceSameOnsetGroups"] == 1
    assert summary["crossHandGroups"] == 1
    assert summary["samePitchClassOctaveLayerGroups"] == 1


def test_proportion_rows_counts_exact_octaves():
    rows = [
        {"group": "left", "shiftBand": "exact-octave"},
        {"group": "left", "shiftBand": "up-1-octave"},
    ]
    summary = proportion_rows(rows, lambda item: item["group"])
    assert summary[0]["pairedNotes"] == 2
    assert summary[0]["exactOctaveRate"] == 0.5


def test_exact_preserving_pairing_does_not_cross_correct_octave_layers():
    reference = [{"time": 1.0, "midi": 48}, {"time": 1.04, "midi": 60}]
    candidate = [{"time": 1.0, "midi": 60}, {"time": 1.04, "midi": 48}]
    assert register_training_pairs(reference, candidate, 0.1) == [(0, 0), (1, 1)]
    assert register_training_pairs(reference, candidate, 0.1, pairing_mode="exact-preserving") == [(0, 1), (1, 0)]
    assert candidate[0]["midi"] == 60


def test_exact_preserving_pairing_retains_real_octave_errors_as_training_examples():
    reference = [{"time": 1.0, "midi": 72}]
    candidate = [{"time": 1.02, "midi": 60}]
    assert register_training_pairs(reference, candidate, 0.1, pairing_mode="exact-preserving") == [(0, 0)]
