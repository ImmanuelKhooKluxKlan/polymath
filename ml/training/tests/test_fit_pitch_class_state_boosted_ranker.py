import numpy as np
import pytest

pytest.importorskip("sklearn")

from ml.training.fit_pitch_class_state_boosted_ranker import (
    bounded_configurations,
    estimated_training_rows,
    training_arrays,
)


def test_training_arrays_balance_duplicate_song_groups():
    feature = np.zeros((2, 3), dtype=float)
    cell = {"features": feature, "rewards": np.asarray([0.2, 1.0])}
    songs = [
        {"groupId": "same", "weight": 1.0, "cells": [cell]},
        {"groupId": "same", "weight": 1.0, "cells": [cell]},
        {"groupId": "other", "weight": 1.0, "cells": [cell]},
    ]
    matrix, targets, weights = training_arrays(songs, best_state_weight=0.0)
    assert matrix.shape == (6, 3)
    assert targets.shape == (6,)
    # Both historical versions of one song together equal the other song.
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())


def test_boosted_configuration_and_row_guards_are_deterministic():
    selected = bounded_configurations([2], maximum_iterations_cap=80)
    assert selected[0]["configurationIndex"] == 2
    assert selected[0]["maximumIterations"] == 80
    songs = [{"cells": [{"states": [1, 2]}, {"states": [1]}]}]
    assert estimated_training_rows(songs) == 3


def test_invalid_boosted_configuration_is_rejected():
    with pytest.raises(ValueError):
        bounded_configurations([99], maximum_iterations_cap=80)
