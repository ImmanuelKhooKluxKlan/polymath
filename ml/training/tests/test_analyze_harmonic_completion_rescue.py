from ml.training.analyze_harmonic_completion_rescue import add_counts


def test_add_counts_accumulates_pitch_class_confusion():
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}

    add_counts(counts, frozenset({0, 4}), frozenset({0, 7}))

    assert counts == {"truePositive": 1, "falsePositive": 1, "falseNegative": 1}
