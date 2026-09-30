import numpy as np
import pytest

pytest.importorskip("sklearn")

from ml.training.fit_pitch_class_state_change_gate import (
    GATE_FEATURE_NAMES,
    fit_gate,
    gate_feature_vector,
    gate_probability,
    select_acceptance_threshold_cross_validated,
)
from ml.training.fit_pitch_class_state_ranker import prepare_song


def raw_song(target=(0, 4, 7)):
    return {
        "id": "fixture",
        "weight": 1.0,
        "cells": [
            {
                "candidatePitchClasses": [0],
                "sourcePitchClasses": [0, 4, 7],
                "referencePitchClasses": [list(target)],
                "candidateRoleSignature": "melody+harmony",
                "candidateDuration": 0.4,
                "candidateVelocity": 0.7,
                "previousGapSeconds": 0.3,
                "nextGapSeconds": 0.3,
                "localCandidateOnsetsPerSecond": 3.0,
            }
        ],
    }


def test_gate_features_are_finite_and_do_not_read_the_target():
    left = prepare_song(raw_song((0, 4, 7)), maximum_states=32)["cells"][0]
    right = prepare_song(raw_song((0, 7)), maximum_states=32)["cells"][0]
    selected = next(index for index, state in enumerate(left["states"]) if len(state) > 1)
    candidate = next(index for index, state in enumerate(left["states"]) if state == left["candidate"])
    scores = np.linspace(0.0, 1.0, len(left["states"]))
    left_vector = gate_feature_vector(left, selected, candidate, scores)
    right_vector = gate_feature_vector(right, selected, candidate, scores)
    assert len(left_vector) == len(GATE_FEATURE_NAMES)
    assert np.all(np.isfinite(left_vector))
    assert np.allclose(left_vector, right_vector)


def test_gate_probability_is_bounded():
    matrix = np.asarray(
        [
            np.zeros(len(GATE_FEATURE_NAMES)),
            np.ones(len(GATE_FEATURE_NAMES)),
            np.full(len(GATE_FEATURE_NAMES), 0.25),
            np.full(len(GATE_FEATURE_NAMES), 0.75),
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    weights = np.ones(4, dtype=float)
    gate = fit_gate(matrix, labels, weights, regularization_c=0.2)
    probability = gate_probability(matrix[0], gate)
    assert 0.0 <= probability <= 1.0


def test_threshold_calibration_holds_out_complete_groups():
    matrix = np.vstack(
        [
            np.full((2, len(GATE_FEATURE_NAMES)), value, dtype=np.float32)
            for value in (0.0, 0.3, 0.7, 1.0)
        ]
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    weights = np.ones(8, dtype=float)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    result = select_acceptance_threshold_cross_validated(
        matrix,
        labels,
        weights,
        groups,
        regularization_c=0.2,
        thresholds=[0.0, 0.5, 1.01],
    )
    assert result["heldoutGroups"] == 4
    assert result["selected"]["threshold"] in {0.0, 0.5, 1.01}
