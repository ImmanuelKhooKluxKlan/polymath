import numpy as np
import pytest

from ml.training.fit_pianist_register_router import (
    FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    policy_metrics,
    record_features,
    route_predictions,
    rounded_shift,
    training_shift_label,
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


def test_training_labels_do_not_claim_a_two_octave_move_fixes_three_octaves():
    assert training_shift_label(36) == 36
    assert training_shift_label(-36) == -36
    assert training_shift_label(36, "legacy-clamped") == 24
    metrics = policy_metrics([{"label": training_shift_label(36)}], np.asarray([24]))
    assert metrics["correctChanges"] == 0
    assert metrics["routedExactRate"] == 0.0


def test_context_features_keep_legacy_prefix_and_add_role_specific_register_axes():
    record = {"candidateMidi": 48, "sourceMidi": 36, "candidateRole": "bass", "sourceFamily": "guitar"}
    legacy = record_features(record)
    contextual = record_features(record, feature_set="role-register-interactions")
    assert contextual[:len(FEATURE_NAMES)] == legacy
    assert len(contextual) == len(CONTEXT_FEATURE_NAMES)
    named = dict(zip(CONTEXT_FEATURE_NAMES, contextual))
    assert named["role_bass__candidate_midi_centered"] == named["candidate_midi_centered"]
    assert named["role_melody__candidate_midi_centered"] == 0.0
    assert named["family_guitar__source_midi_centered"] == named["source_midi_centered"]


def test_exact_training_shift_policy_rejects_non_octaves_and_nonfinite_values():
    with pytest.raises(ValueError, match="exact octave"):
        training_shift_label(11)
    with pytest.raises(ValueError, match="finite"):
        training_shift_label(float("nan"))
