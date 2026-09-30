import numpy as np

from ml.training.fit_pianist_drop_gate import (
    DROP_FEATURE_NAMES,
    classification_counts,
    drop_feature_vector,
)


def test_drop_features_are_target_blind_and_finite():
    row = {
        "candidatePitchClasses": [0, 4],
        "sourcePitchClasses": [0, 7],
        "candidateChordSize": 2,
        "candidateDuration": 0.4,
        "candidateVelocity": 0.7,
        "audioPitchClassEvidence": [
            {"pitchClass": 0, "meanChroma": 0.6, "attackChroma": 0.7},
            {"pitchClass": 4, "meanChroma": 0.2, "attackChroma": 0.1},
        ],
        "vocalF0Evidence": {
            "available": True,
            "dominantPitchClass": 0,
            "voicedShare": 0.8,
            "meanConfidence": 0.9,
            "pitchClasses": [
                {"pitchClass": 0, "probability": 0.75},
                {"pitchClass": 4, "probability": 0.25},
            ],
        },
        # Deliberately contradictory reference data must not affect features.
        "referencePitchClasses": [[11]],
        "targetGestureCount": 0,
    }

    first = np.asarray(drop_feature_vector(row))
    row["referencePitchClasses"] = [[0, 4, 7]]
    row["targetGestureCount"] = 3
    second = np.asarray(drop_feature_vector(row))

    assert len(first) == len(DROP_FEATURE_NAMES)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


def test_drop_classification_counts_prioritize_false_drop_visibility():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    predicted = np.asarray([True, False, True, False])

    metrics = classification_counts(labels, predicted)

    assert metrics["truePositive"] == 1
    assert metrics["falsePositive"] == 1
    assert metrics["falseNegative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
