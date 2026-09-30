import numpy as np
import pytest

from ml.training.evaluate_state_ranker_corpus_transfer import (
    blended_argmax,
    effective_evaluation_weights,
    parse_grid,
)


def test_blended_argmax_uses_requested_convex_blend():
    base = np.asarray([0.9, 0.6, 0.1])
    augmented = np.asarray([0.1, 0.7, 1.0])

    assert blended_argmax(base, augmented, 0.0) == 0
    assert blended_argmax(base, augmented, 1.0) == 2
    assert blended_argmax(base, augmented, 0.5) == 1


def test_blended_argmax_rejects_mismatched_state_lattices():
    with pytest.raises(ValueError, match="identical shapes"):
        blended_argmax(np.asarray([1.0]), np.asarray([1.0, 2.0]), 0.5)


def test_group_balanced_weights_do_not_double_count_variants():
    songs = [
        {"id": "alpha", "groupId": "alpha", "weight": 1.0},
        {"id": "alpha-aux", "groupId": "alpha", "weight": 1.0},
        {"id": "beta", "groupId": "beta", "weight": 1.0},
    ]

    weights = effective_evaluation_weights(songs, group_balanced=True)

    assert weights == {"alpha": 0.5, "alpha-aux": 0.5, "beta": 1.0}


def test_parse_grid_clamps_deduplicates_and_sorts():
    assert parse_grid("1, 0.5, 0.5, -1, 2") == [0.0, 0.5, 1.0]
