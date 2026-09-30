import numpy as np

from ml.training.fit_harmonic_completion_rescue_reranker import (
    make_rescue_meta_song,
)
from ml.training.fit_pitch_class_state_ranker import FEATURE_NAMES


def test_rescue_meta_song_preserves_base_top_one_and_appends_rescue():
    features = np.zeros((2, len(FEATURE_NAMES)), dtype=float)
    features[0, 0] = 1.0
    features[1, 0] = 2.0
    rescue_features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    rescue_features[0, 0] = 3.0
    song = {
        "id": "x",
        "groupId": "x",
        "weight": 1.0,
        "cells": [
            {
                "index": 0,
                "target": frozenset({0, 4, 7}),
                "candidate": frozenset({0, 7}),
                "states": [frozenset({0}), frozenset({0, 7})],
                "features": features,
                "rewards": np.asarray([0.5, 0.8]),
            }
        ],
        "rescues": {
            0: {
                "states": [frozenset({0, 4, 7})],
                "features": rescue_features,
                "rewards": np.asarray([1.0]),
            }
        },
    }
    model = {
        "means": np.zeros(len(FEATURE_NAMES)),
        "scales": np.ones(len(FEATURE_NAMES)),
        "coefficients": np.r_[0.0, np.asarray([1.0] + [0.0] * (len(FEATURE_NAMES) - 1))],
    }

    result = make_rescue_meta_song(song, model, top_k=2, maximum_rescues=1)

    assert result["cells"][0]["states"][0] == frozenset({0, 7})
    assert result["cells"][0]["states"][-1] == frozenset({0, 4, 7})
    assert result["cells"][0]["rescueCount"] == 1
