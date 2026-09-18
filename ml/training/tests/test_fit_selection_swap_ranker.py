import sys
from pathlib import Path

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from fit_selection_swap_ranker import (  # noqa: E402
    fit_symmetric_logistic,
    pairwise_metrics,
    scales_for_deltas,
)


def test_pairwise_ranker_learns_direction_without_intercept():
    deltas = np.asarray(
        [
            [0.0, -1.0, 0.5],
            [0.0, -0.8, 0.4],
            [0.0, -1.2, 0.7],
        ],
        dtype=float,
    )
    scales = scales_for_deltas(deltas)
    weights = fit_symmetric_logistic(deltas, ridge=1.0, scales=scales)
    metrics = pairwise_metrics(deltas, weights, scales)
    assert metrics["desiredWinRate"] == 1.0
    assert abs(weights[0]) < 1e-12


def test_zero_variance_features_get_safe_scale():
    deltas = np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=float)
    scales = scales_for_deltas(deltas)
    assert scales[0] == 1.0
    assert scales[1] > 0.0
