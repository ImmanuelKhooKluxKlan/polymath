import numpy as np

from ml.training.fit_pitch_class_state_listwise_ranker import (
    listwise_loss_and_gradient,
    target_distribution,
)


def test_hard_target_distribution_splits_tied_best_states():
    distribution = target_distribution(np.asarray([0.2, 1.0, 1.0, 0.5]), 0.0)
    assert np.allclose(distribution, [0.0, 0.5, 0.5, 0.0])


def test_soft_target_distribution_is_normalized_and_reward_ordered():
    distribution = target_distribution(np.asarray([0.2, 1.0, 0.5]), 0.1)
    assert np.isclose(distribution.sum(), 1.0)
    assert distribution[1] > distribution[2] > distribution[0]


def test_listwise_gradient_matches_finite_difference():
    flattened = {
        "matrix": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        "target": np.asarray([1.0, 0.0, 1.0]),
        "offsets": np.asarray([0, 2]),
        "lengths": np.asarray([2, 1]),
        "cellWeights": np.asarray([0.6, 0.4]),
        "repeatedCellWeights": np.asarray([0.6, 0.6, 0.4]),
    }
    coefficients = np.asarray([0.2, -0.1])
    loss, gradient = listwise_loss_and_gradient(
        coefficients, flattened, l2_regularization=0.03
    )
    assert np.isfinite(loss)
    epsilon = 1e-6
    numerical = np.zeros_like(coefficients)
    for index in range(len(coefficients)):
        step = np.zeros_like(coefficients)
        step[index] = epsilon
        high = listwise_loss_and_gradient(
            coefficients + step, flattened, l2_regularization=0.03
        )[0]
        low = listwise_loss_and_gradient(
            coefficients - step, flattened, l2_regularization=0.03
        )[0]
        numerical[index] = (high - low) / (2.0 * epsilon)
    assert np.allclose(gradient, numerical, atol=1e-6)
