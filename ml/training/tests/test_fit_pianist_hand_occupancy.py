from ml.training.fit_pianist_hand_occupancy import (
    HAND_OCCUPANCY_FEATURE_NAMES,
    evaluate,
    hand_occupancy_feature_map,
    sequence_contexts,
)


def note(midi, *, hand, role="harmony", source="guitar", time=0.0):
    return {
        "midi": midi,
        "time": time,
        "duration": 0.3,
        "scoreDuration": 0.3,
        "velocity": 0.7,
        "sourceVelocityBeforeArrangement": 0.65,
        "selectionProbability": 0.8,
        "hand": hand,
        "arrangementRole": role,
        "sourceInstrument": source,
    }


def test_feature_contract_is_complete_and_finite():
    groups = [
        [note(48, hand="left"), note(72, hand="right", role="melody", source="voice")],
        [note(50, hand="left", time=0.3), note(74, hand="right", time=0.3)],
    ]
    contexts = sequence_contexts(groups)
    features = hand_occupancy_feature_map(groups, 0, contexts)
    assert tuple(features) == HAND_OCCUPANCY_FEATURE_NAMES
    assert all(value == value for value in features.values())
    assert features["right_melody_share"] == 1.0
    assert features["shared_pitch_class_count"] == 1 / 3


def test_evaluate_suppression_improves_a_true_left_only_example():
    group = [note(48, hand="left"), note(61, hand="right")]
    features = [0.0] * len(HAND_OCCUPANCY_FEATURE_NAMES)
    features[0] = 1.0
    example = {
        "songId": "test",
        "candidateGroup": group,
        "targetOccupancy": "left-only",
        "targetPitchClasses": [0],
        "label": 1,
        "features": features,
    }
    model = {
        "weights": [30.0] + [0.0] * (len(features) - 1),
        "means": [0.0] * len(features),
        "scales": [1.0] * len(features),
    }
    baseline = evaluate([example])
    candidate = evaluate([example], model, threshold=0.5)
    assert baseline["pitchClassF1"] < candidate["pitchClassF1"]
    assert candidate["occupancyAccuracy"] == 1.0
    assert candidate["trueSuppressions"] == 1


def test_preserve_melody_blocks_suppression():
    group = [note(48, hand="left"), note(61, hand="right", role="melody")]
    features = [0.0] * len(HAND_OCCUPANCY_FEATURE_NAMES)
    features[0] = 1.0
    example = {
        "songId": "test",
        "candidateGroup": group,
        "targetOccupancy": "left-only",
        "targetPitchClasses": [0],
        "label": 1,
        "features": features,
    }
    model = {
        "weights": [30.0] + [0.0] * (len(features) - 1),
        "means": [0.0] * len(features),
        "scales": [1.0] * len(features),
    }
    result = evaluate([example], model, threshold=0.5, preserve_melody=True)
    assert result["suppressions"] == 0
