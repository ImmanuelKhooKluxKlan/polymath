import numpy as np

from ml.training.fit_pianist_register_router import (
    FEATURE_NAMES,
    policy_metrics,
    record_features,
    route_predictions,
    rounded_shift,
)


def test_record_features_match_frozen_contract():
    values = record_features(
        {
            "candidateMidi": 48,
            "sourceMidi": 36,
            "candidateHand": "left",
            "candidateRole": "bass",
            "sourceFamily": "guitar",
            "candidatePitchBand": "lower-C3-B3",
            "pitchRank": 0.0,
            "gestureSize": 3,
            "samePitchClassLayers": 1,
            "isLowest": True,
            "isHighest": False,
        }
    )
    assert len(values) == len(FEATURE_NAMES)
    assert values[0] == 1.0
    assert all(np.isfinite(values))


def test_route_predictions_abstains_without_confidence_or_margin():
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.8, 0.3, 0.1],
            [0.1, 0.61, 0.6, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.82, 0.1],
        ]
    )
    predictions = route_predictions(probabilities, threshold=0.7, margin=0.2)
    assert predictions.tolist() == [0, 0, 12]


def test_policy_metrics_counts_helpful_and_harmful_moves():
    records = [
        {"label": 0},
        {"label": 12},
        {"label": -12},
    ]
    metrics = policy_metrics(records, np.asarray([12, 12, 0]))
    assert metrics["baselineExactRate"] == 0.333333
    assert metrics["routedExactRate"] == 0.333333
    assert metrics["changes"] == 2
    assert metrics["correctChanges"] == 1
    assert metrics["harmfulChanges"] == 1


def test_rounded_shift_clamps_to_supported_octaves():
    assert rounded_shift(11) == 12
    assert rounded_shift(-38) == -24
    assert rounded_shift(37) == 24
