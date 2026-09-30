import numpy as np

from ml.training.fit_pitch_class_state_ranker import FEATURE_NAMES
from ml.training.fit_pitch_class_state_topk_reranker import META_FEATURE_NAMES
from ml.training.score_pitch_class_state_topk import (
    cardinality_probabilities,
    score_song,
    validate_models,
)
from ml.training.fit_pitch_class_cardinality_reranker import (
    cardinality_feature_names,
)


def serialized_base_model():
    return {
        "schema": "polymath-pitch-class-state-ranker-v1",
        "featureNames": list(FEATURE_NAMES),
        "means": [0.0] * len(FEATURE_NAMES),
        "scales": [1.0] * len(FEATURE_NAMES),
        "coefficients": [0.0] * (len(FEATURE_NAMES) + 1),
        "maximumStates": 8,
    }


def serialized_meta_model():
    return {
        "schema": "polymath-pitch-class-state-topk-reranker-v1",
        "family": "ridge-reward",
        "featureNames": list(META_FEATURE_NAMES),
        "means": [0.0] * len(META_FEATURE_NAMES),
        "scales": [1.0] * len(META_FEATURE_NAMES),
        "coefficients": [0.0] * (len(META_FEATURE_NAMES) + 1),
        "topK": 3,
        "minimumOverrideMargin": 0.02,
        "sha256": "test-meta",
    }


def test_model_contract_is_validated():
    base = serialized_base_model()
    meta = serialized_meta_model()
    assert validate_models(base, meta) == (8, 3, 0.02)

    base["featureNames"] = ["wrong"]
    try:
        validate_models(base, meta)
    except ValueError as error:
        assert "Base model feature contract" in str(error)
    else:
        raise AssertionError("invalid base contract was accepted")


def test_serialized_histogram_contract_requires_runtime_trees():
    base = serialized_base_model()
    meta = serialized_meta_model()
    meta["family"] = "hist-gradient-reward"
    try:
        validate_models(base, meta)
    except ValueError as error:
        assert "runtime trees" in str(error)
    else:
        raise AssertionError("histogram model without trees was accepted")

    meta["baselinePrediction"] = 0.0
    meta["trees"] = [[{"leaf": True, "value": 0.0}]]
    assert validate_models(base, meta) == (8, 3, 0.02)


def test_unanchored_cells_preserve_candidate_pitch_classes():
    song = {
        "id": "tiny",
        "cells": [
            {
                "sourceTime": 1.0,
                "candidatePitchClasses": [0, 7],
                "sourcePitchClasses": [0, 4],
                "referencePitchClasses": [],
            }
        ],
    }
    result = score_song(song, serialized_base_model(), serialized_meta_model())

    assert result["diagnostics"]["anchoredCells"] == 0
    assert result["cells"][0]["selectedPitchClasses"] == [0, 7]
    assert np.asarray(result["cells"][0]["probabilities"])[[0, 7]].tolist() == [
        0.99,
        0.99,
    ]
    assert result["cells"][0]["decision"] == "preserve-unanchored-cell"


def test_serialized_cardinality_model_returns_normalized_size_probabilities():
    top_k = 3
    names = list(cardinality_feature_names(top_k))
    cell = {
        "features": np.zeros((3, len(META_FEATURE_NAMES)), dtype=float),
        "baseScores": np.asarray([0.8, 0.7, 0.6]),
        "states": [frozenset([0]), frozenset([0, 4]), frozenset([0, 4, 7])],
    }
    model = {
        "schema": "polymath-pitch-class-cardinality-reranker-v1",
        "family": "multinomial-cardinality",
        "featureNames": names,
        "means": [0.0] * len(names),
        "scales": [1.0] * len(names),
        "coefficients": [[0.0] * len(names), [0.0] * len(names)],
        "intercepts": [0.0, 1.0],
        "classes": [1, 2],
        "topK": top_k,
    }

    probabilities = cardinality_probabilities(cell, model, top_k=top_k)

    assert np.isclose(float(np.sum(probabilities)), 1.0)
    assert probabilities[1] > probabilities[0]
    assert np.all(probabilities > 0.0)
