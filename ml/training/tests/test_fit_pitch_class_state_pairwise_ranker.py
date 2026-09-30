import numpy as np
import pytest

pytest.importorskip("sklearn")

from ml.training.fit_pitch_class_state_pairwise_ranker import (
    pair_features,
    selected_pairs,
    training_arrays,
)


def test_pair_features_reverse_difference_but_keep_context():
    left = np.asarray([1.0, 2.0])
    right = np.asarray([0.5, 4.0])
    forward = pair_features(left, right)
    reverse = pair_features(right, left)
    assert np.allclose(forward[:2], -reverse[:2])
    assert np.allclose(forward[2:], reverse[2:])


def test_pair_selection_compares_best_and_candidate():
    cell = {
        "states": [frozenset([0]), frozenset([0, 4]), frozenset([0, 4, 7])],
        "candidate": frozenset([0]),
        "rewards": np.asarray([0.5, 0.8, 1.0]),
    }
    pairs = selected_pairs(cell, 10)
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_training_arrays_add_both_pair_directions():
    cell = {
        "states": [frozenset([0]), frozenset([0, 4])],
        "candidate": frozenset([0]),
        "rewards": np.asarray([0.5, 1.0]),
        "features": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    }
    song = {"groupId": "song", "weight": 1.0, "cells": [cell]}
    matrix, labels, weights = training_arrays([song], maximum_pairs_per_cell=4)
    assert matrix.shape == (2, 4)
    assert sorted(labels.tolist()) == [0, 1]
    assert weights.sum() == pytest.approx(1.0)
