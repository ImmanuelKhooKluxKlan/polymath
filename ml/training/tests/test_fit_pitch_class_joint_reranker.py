import numpy as np

from ml.training.fit_pitch_class_state_topk_reranker import META_FEATURE_NAMES
from ml.training.fit_pitch_class_joint_reranker import joint_scores


class FixedEstimator:
    classes_ = np.asarray([1, 2, 3])

    def predict_proba(self, _matrix):
        return np.asarray([[0.8, 0.15, 0.05]])


def test_joint_scores_add_cardinality_log_probability_by_state_size():
    cell = {
        "features": np.zeros((2, len(META_FEATURE_NAMES))),
        "baseScores": np.asarray([0.5, 0.4]),
        "states": [frozenset({0, 4, 7}), frozenset({0})],
    }
    meta = {
        "family": "ridge-reward",
        "means": np.zeros(len(META_FEATURE_NAMES)),
        "scales": np.ones(len(META_FEATURE_NAMES)),
        "coefficients": np.zeros(len(META_FEATURE_NAMES) + 1),
    }
    cardinality = {
        "family": "multinomial-cardinality",
        "topK": 2,
        "means": np.zeros(len(META_FEATURE_NAMES) + 2 + 2 + 6 + 6 + 3),
        "scales": np.ones(len(META_FEATURE_NAMES) + 2 + 2 + 6 + 6 + 3),
        "estimator": FixedEstimator(),
    }

    scores = joint_scores(cell, meta, cardinality, cardinality_strength=1.0)

    assert scores[1] > scores[0]
