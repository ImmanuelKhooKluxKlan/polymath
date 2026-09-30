import numpy as np

from ml.training.fit_pitch_class_state_linear_pairwise_ranker import training_arrays


def test_linear_pairwise_examples_are_symmetric_and_gap_weighted():
    cell = {
        "states": [frozenset([0]), frozenset([0, 4]), frozenset([0, 4, 7])],
        "candidate": frozenset([0]),
        "rewards": np.asarray([0.2, 0.6, 1.0]),
        "features": np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
    }
    song = {"groupId": "song", "weight": 1.0, "cells": [cell]}
    matrix, labels, weights = training_arrays([song], maximum_pairs_per_cell=8)
    assert len(matrix) % 2 == 0
    assert np.allclose(matrix[0], -matrix[1])
    assert labels[0] == 1 - labels[1]
    assert np.isclose(weights.sum(), 1.0)
