import numpy as np
import pytest

from ml.training.analyze_state_ranker_topk import top_k_indices


def test_top_k_indices_is_descending_and_deterministic_for_ties():
    scores = np.asarray([0.3, 0.8, 0.8, 0.1])

    assert top_k_indices(scores, 3) == [1, 2, 0]


def test_top_k_indices_clamps_requested_size():
    scores = np.asarray([0.2, 0.1])

    assert top_k_indices(scores, 0) == [0]
    assert top_k_indices(scores, 20) == [0, 1]


def test_top_k_indices_requires_vector():
    with pytest.raises(ValueError, match="one-dimensional"):
        top_k_indices(np.asarray([[0.1, 0.2]]), 1)
