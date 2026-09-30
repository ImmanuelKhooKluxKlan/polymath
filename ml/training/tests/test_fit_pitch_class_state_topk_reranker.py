import numpy as np

from ml.training.fit_pitch_class_state_ranker import FEATURE_NAMES
from ml.training.fit_pitch_class_state_topk_reranker import (
    META_FEATURE_NAMES,
    fit_meta_pairwise,
    file_fingerprint,
    meta_feature_matrix,
    meta_predict,
    preparation_cache_spec,
    ranked_state_indices,
    serialize_base_model,
    serialize_model,
    select_meta_index,
    make_meta_song,
)
from sklearn.ensemble import HistGradientBoostingRegressor


def test_preparation_cache_spec_tracks_exact_input_content(tmp_path):
    audit = tmp_path / "audit.json"
    evaluation = tmp_path / "evaluation.json"
    audit.write_text('{"songs": []}', encoding="utf-8")
    evaluation.write_text('{"songs": []}', encoding="utf-8")

    first = preparation_cache_spec(
        audit_path=audit,
        evaluation_path=evaluation,
        meta_extra_paths=[],
        maximum_states=64,
        allow_harmonic_completion=False,
        meta_extra_audit_weight=0.5,
    )
    evaluation.write_text('{"songs": [1]}', encoding="utf-8")
    second = preparation_cache_spec(
        audit_path=audit,
        evaluation_path=evaluation,
        meta_extra_paths=[],
        maximum_states=64,
        allow_harmonic_completion=False,
        meta_extra_audit_weight=0.5,
    )

    assert file_fingerprint(audit)["sha256"] == first["audit"]["sha256"]
    assert first["evaluationAudit"]["sha256"] != second["evaluationAudit"]["sha256"]


def test_meta_features_preserve_state_features_and_encode_rank_margins():
    state_features = np.zeros((4, len(FEATURE_NAMES)), dtype=float)
    state_features[2, 0] = 7.0
    scores = np.asarray([0.1, 0.7, 0.9, 0.6])

    matrix = meta_feature_matrix(state_features, scores, [2, 1, 3])

    assert matrix.shape == (3, len(META_FEATURE_NAMES))
    assert matrix[0, 0] == 7.0
    assert np.allclose(matrix[:, -7], [0.0, -0.2, -0.3])
    assert np.allclose(matrix[:, -6], [0.3, 0.3, 0.3])
    assert np.allclose(matrix[:, -5], [0.0, 0.5, 1.0])
    assert matrix[:, -4].tolist() == [1.0, 0.0, 0.0]
    assert matrix[:, -3].tolist() == [0.0, 1.0, 0.0]
    assert matrix[:, -2].tolist() == [0.0, 0.0, 1.0]


def test_meta_feature_spread_is_safe_when_base_scores_tie():
    state_features = np.zeros((2, len(FEATURE_NAMES)), dtype=float)
    matrix = meta_feature_matrix(
        state_features, np.asarray([0.5, 0.5]), [0, 1]
    )

    assert np.isfinite(matrix).all()
    assert matrix[0, -6] == 1e-9


def test_meta_override_margin_falls_back_to_base_top_one():
    scores = np.asarray([0.60, 0.61, 0.59])

    assert select_meta_index(scores, 0.0) == 1
    assert select_meta_index(scores, 0.02) == 0


def test_candidate_margin_can_abstain_to_incumbent():
    scores = np.asarray([0.61, 0.60, 0.59])

    assert select_meta_index(
        scores,
        0.0,
        incumbent_index=1,
        minimum_candidate_override_margin=0.02,
    ) == 1
    assert select_meta_index(
        scores,
        0.0,
        incumbent_index=1,
        minimum_candidate_override_margin=0.005,
    ) == 0


def test_pairwise_meta_model_learns_consistent_top_state_utility():
    def meta_cell(left_reward, right_reward):
        features = np.zeros((2, len(META_FEATURE_NAMES)), dtype=float)
        features[0, 0] = 1.0
        features[1, 0] = -1.0
        return {
            "features": features,
            "rewards": np.asarray([left_reward, right_reward], dtype=float),
        }

    songs = [
        {
            "id": "alpha",
            "groupId": "alpha",
            "weight": 1.0,
            "cells": [meta_cell(1.0, 0.0), meta_cell(0.8, 0.2)],
        }
    ]

    model = fit_meta_pairwise(songs, regularization_c=1.0)

    assert model["family"] == "pairwise-logistic"
    assert model["coefficients"][0] > 0.0


def test_meta_song_can_limit_cardinality_drift():
    feature_count = len(FEATURE_NAMES)
    states = [
        frozenset([0, 4]),
        frozenset([0]),
        frozenset([0, 4, 7]),
        frozenset([0, 2, 4, 7]),
    ]
    song = {
        "id": "tiny",
        "groupId": "tiny",
        "weight": 1.0,
        "cells": [
            {
                "candidate": frozenset([0, 4]),
                "target": frozenset([0, 4]),
                "states": states,
                "features": np.zeros((len(states), feature_count), dtype=float),
                "rewards": np.asarray([1.0, 0.5, 0.8, 0.6]),
            }
        ],
    }
    base = {
        "means": np.zeros(feature_count),
        "scales": np.ones(feature_count),
        "coefficients": np.zeros(feature_count + 1),
    }

    meta = make_meta_song(
        song, base, top_k=4, maximum_cardinality_delta=1
    )

    assert [len(state) for state in meta["cells"][0]["states"]] == [2, 1, 3]


def test_ranked_states_can_force_incumbent_into_shortlist():
    cell = {
        "candidate": frozenset([11]),
        "states": [
            frozenset([0]),
            frozenset([2]),
            frozenset([4]),
            frozenset([11]),
        ],
    }
    indices = ranked_state_indices(
        cell,
        np.asarray([0.9, 0.8, 0.7, 0.1]),
        top_k=3,
        include_candidate_state=True,
    )

    assert indices == [0, 1, 3]


def test_histogram_model_json_round_trip_matches_sklearn_predictions():
    matrix = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        dtype=float,
    )
    target = np.asarray([0.0, 1.0, 1.0, 2.0, 4.0, 6.0], dtype=float)
    estimator = HistGradientBoostingRegressor(
        max_iter=4,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        early_stopping=False,
        random_state=6701,
    ).fit(matrix, target)
    runtime_model = {
        "family": "hist-gradient-reward",
        "featureNames": ["left", "right"],
        "estimator": estimator,
        "learningRate": 0.1,
        "maximumIterations": 4,
        "maximumLeafNodes": 3,
        "minimumSamplesLeaf": 2,
        "l2Regularization": 0.0,
        "trainingExamples": len(target),
    }
    serialized = serialize_model(
        runtime_model, top_k=3, minimum_override_margin=0.0
    )

    assert serialized["runtimeArtifactAvailable"] is True
    assert np.allclose(
        estimator.predict(matrix), meta_predict(matrix, serialized), atol=1e-9
    )


def test_base_model_serialization_preserves_replay_arrays():
    count = len(FEATURE_NAMES)
    model = {
        "featureNames": list(FEATURE_NAMES),
        "means": np.arange(count, dtype=float),
        "scales": np.ones(count, dtype=float),
        "coefficients": np.arange(count + 1, dtype=float),
        "ridge": 0.01,
        "bestStateWeight": 1.0,
    }

    payload = serialize_base_model(model, maximum_states=64)

    assert payload["maximumStates"] == 64
    assert payload["means"] == list(np.arange(count, dtype=float))
    assert len(payload["sha256"]) == 64
