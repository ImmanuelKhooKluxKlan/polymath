import numpy as np

from ml.training.fit_pitch_class_state_topk_reranker import META_FEATURE_NAMES
from ml.training.fit_pitch_class_cardinality_reranker import (
    cardinality_feature_names,
    cardinality_features,
    select_state_index,
)


def sample_cell():
    return {
        "features": np.zeros((3, len(META_FEATURE_NAMES)), dtype=float),
        "baseScores": np.asarray([0.8, 0.76, 0.6]),
        "states": [frozenset({0, 4, 7}), frozenset({0, 7}), frozenset({0})],
    }


def test_cardinality_features_are_fixed_length_and_target_blind():
    values = cardinality_features(sample_cell(), top_k=5)

    assert len(values) == len(cardinality_feature_names(5))
    assert np.isfinite(values).all()
    assert "target" not in " ".join(cardinality_feature_names(5)).lower()


def test_size_penalty_can_conservatively_override_base_state():
    cell = sample_cell()

    assert select_state_index(cell, predicted_size=2.0, size_penalty=0.0) == 0
    assert select_state_index(cell, predicted_size=2.0, size_penalty=0.1) == 1
