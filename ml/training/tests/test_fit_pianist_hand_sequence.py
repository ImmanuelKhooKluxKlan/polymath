import numpy as np

from ml.training.fit_pianist_hand_sequence import (
    STATE_INDEX,
    STATE_NAMES,
    evaluate,
    posterior_probabilities,
    transition_probabilities,
)
from ml.training.fit_pianist_hand_occupancy import HAND_OCCUPANCY_FEATURE_NAMES


def note(midi, hand, role="harmony"):
    return {
        "midi": midi,
        "time": 0.0,
        "duration": 0.3,
        "velocity": 0.7,
        "hand": hand,
        "arrangementRole": role,
    }


def example(target, candidate="both", target_pcs=(0,), candidate_pcs=(0, 1)):
    group = []
    if candidate in {"left-only", "both"}:
        group.append(note(48, "left"))
    if candidate in {"right-only", "both"}:
        group.append(note(60 + candidate_pcs[-1], "right"))
    features = [0.0] * len(HAND_OCCUPANCY_FEATURE_NAMES)
    features[0] = 1.0
    return {
        "songId": "test",
        "targetState": target,
        "candidateState": candidate,
        "targetPitchClasses": list(target_pcs),
        "candidateGroup": group,
        "features": features,
    }


def constant_model(left_probability=0.9):
    probabilities = {
        "left-only": left_probability,
        "right-only": (1.0 - left_probability) / 2.0,
        "both": (1.0 - left_probability) / 2.0,
    }
    states = {}
    for state in STATE_NAMES:
        probability = probabilities[state]
        logit = np.log(probability / (1.0 - probability))
        states[state] = {
            "weights": [float(logit)]
            + [0.0] * (len(HAND_OCCUPANCY_FEATURE_NAMES) - 1),
            "means": [0.0] * len(HAND_OCCUPANCY_FEATURE_NAMES),
            "scales": [1.0] * len(HAND_OCCUPANCY_FEATURE_NAMES),
        }
    return {
        "emissions": {"states": states},
        "initialProbabilities": [1 / 3, 1 / 3, 1 / 3],
        "transitionProbabilities": [[1 / 3] * 3 for _ in range(3)],
    }


def test_transition_rows_are_normalized():
    segments = [
        [example("left-only"), example("both"), example("left-only")],
        [example("right-only")],
    ]
    starts, transitions = transition_probabilities(segments)
    assert abs(sum(starts) - 1.0) < 1e-9
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in transitions)
    assert transitions[STATE_INDEX["left-only"]][STATE_INDEX["both"]] > 0


def test_forward_backward_posteriors_are_normalized():
    segment = [example("left-only"), example("left-only")]
    posterior = posterior_probabilities(
        segment, constant_model(), transition_strength=1.0
    )
    assert posterior.shape == (2, 3)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.all(posterior[:, STATE_INDEX["left-only"]] > 0.5)


def test_high_confidence_left_state_suppresses_only_right_layer():
    segment = [example("left-only")]
    baseline = evaluate([segment])
    candidate = evaluate(
        [segment],
        constant_model(0.98),
        transition_strength=0,
        threshold=0.8,
        margin=0.5,
        preserve_melody=False,
    )
    assert candidate["trueSuppressions"] == 1
    assert candidate["pitchClassF1"] > baseline["pitchClassF1"]
