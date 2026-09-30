"""Cross-fit a focused reranker over a base ranker's top gesture states.

The compact lattice has a much higher top-three oracle than the first-stage
ranker's top-one score.  This module tests whether that headroom is learnable
without reference leakage:

1. an outer fold removes the evaluation composition;
2. every meta-training composition is scored by an inner base model that also
   removed that composition;
3. a small ridge reranker learns rewards only on those out-of-fold top states;
4. the outer composition is scored by a base model and reranked once.

The report remains opened-development research.  It cannot promote a model
until a new frozen song passes end-to-end exact-note and listening gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBRanker
except ImportError:  # XGBoost is an optional research dependency.
    XGBRanker = None

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .analyze_state_ranker_topk import top_k_indices
    from .fit_pitch_class_state_ranker import (
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        normalize_sample_weights,
        predict,
        prepare_song,
        ridge_group_statistics,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1  # type: ignore
    from analyze_state_ranker_topk import top_k_indices  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        normalize_sample_weights,
        predict,
        prepare_song,
        ridge_group_statistics,
    )


META_FEATURE_NAMES = (
    *FEATURE_NAMES,
    "base_score_from_top",
    "base_score_spread",
    "base_rank_fraction",
    "base_is_rank_1",
    "base_is_rank_2",
    "base_is_rank_3",
    "base_is_rank_4_plus",
)


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content fingerprint suitable for a scientific cache key."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def preparation_cache_spec(
    *,
    audit_path: Path,
    evaluation_path: Path | None,
    meta_extra_paths: list[Path],
    maximum_states: int,
    allow_harmonic_completion: bool,
    meta_extra_audit_weight: float,
) -> dict[str, Any]:
    """Describe every input that can change the expensive prepared lattice."""

    return {
        "schema": "polymath-topk-preparation-cache-v1",
        "audit": file_fingerprint(audit_path),
        "evaluationAudit": (
            file_fingerprint(evaluation_path) if evaluation_path is not None else None
        ),
        "metaExtraAudits": [file_fingerprint(path) for path in meta_extra_paths],
        "maximumStates": int(maximum_states),
        "allowHarmonicCompletion": bool(allow_harmonic_completion),
        "metaExtraAuditWeight": float(meta_extra_audit_weight),
        "featureNames": list(FEATURE_NAMES),
    }


def prepared_songs_from_audit(
    path: Path,
    *,
    maximum_states: int,
    allow_harmonic_completion: bool,
    weight_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = (
            max(0.0, finite(raw.get("weight"), 1.0))
            * max(0.0, float(weight_multiplier))
        )
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(
            prepare_song(
                copied,
                maximum_states=maximum_states,
                allow_harmonic_completion=allow_harmonic_completion,
            )
        )
    return songs


def load_or_build_prepared_inputs(
    *,
    cache_path: Path | None,
    spec: dict[str, Any],
    audit_path: Path,
    evaluation_path: Path | None,
    meta_extra_paths: list[Path],
    maximum_states: int,
    allow_harmonic_completion: bool,
    meta_extra_audit_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load an exact prepared-lattice cache, or atomically create one."""

    if cache_path is not None and cache_path.is_file():
        cached = joblib.load(cache_path)
        if isinstance(cached, dict) and cached.get("spec") == spec:
            songs = cached.get("songs")
            evaluation_source = cached.get("evaluationSource")
            meta_extra_songs = cached.get("metaExtraSongs")
            if all(isinstance(value, list) for value in (songs, evaluation_source, meta_extra_songs)):
                return songs, evaluation_source, meta_extra_songs, "hit"

    songs = prepared_songs_from_audit(
        audit_path,
        maximum_states=maximum_states,
        allow_harmonic_completion=allow_harmonic_completion,
    )
    meta_extra_songs = [
        song
        for path in meta_extra_paths
        for song in prepared_songs_from_audit(
            path,
            maximum_states=maximum_states,
            allow_harmonic_completion=allow_harmonic_completion,
            weight_multiplier=meta_extra_audit_weight,
        )
    ]
    evaluation_source = (
        prepared_songs_from_audit(
            evaluation_path,
            maximum_states=maximum_states,
            allow_harmonic_completion=allow_harmonic_completion,
        )
        if evaluation_path is not None
        else songs
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        joblib.dump(
            {
                "spec": spec,
                "songs": songs,
                "evaluationSource": evaluation_source,
                "metaExtraSongs": meta_extra_songs,
            },
            temporary,
            compress=0,
        )
        temporary.replace(cache_path)
    return songs, evaluation_source, meta_extra_songs, "miss"


def meta_feature_matrix(
    state_features: np.ndarray,
    base_scores: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Create target-blind state features for one ranked cell."""

    if not indices:
        raise ValueError("at least one ranked state is required")
    selected_scores = np.asarray([base_scores[index] for index in indices], dtype=float)
    top_score = float(selected_scores[0])
    spread = max(1e-9, top_score - float(selected_scores[-1]))
    count = len(indices)
    rows: list[np.ndarray] = []
    for rank, index in enumerate(indices, start=1):
        rank_fraction = (rank - 1) / max(1, count - 1)
        extras = np.asarray(
            [
                float(base_scores[index]) - top_score,
                spread,
                rank_fraction,
                float(rank == 1),
                float(rank == 2),
                float(rank == 3),
                float(rank >= 4),
            ],
            dtype=float,
        )
        rows.append(np.concatenate([state_features[index], extras]))
    return np.asarray(rows, dtype=float)


def ranked_state_indices(
    cell: dict[str, Any],
    scores: np.ndarray,
    *,
    top_k: int,
    maximum_cardinality_delta: int | None = None,
    include_candidate_state: bool = False,
) -> list[int]:
    """Build a deterministic shortlist and optionally retain the incumbent.

    A reranker that never sees the current arrangement cannot learn to abstain.
    When requested, the candidate state replaces the weakest shortlisted state
    so every decision can be compared directly with the incumbent gesture.
    """

    if maximum_cardinality_delta is None:
        eligible = list(range(len(cell["states"])))
    else:
        candidate_size = len(cell["candidate"])
        eligible = [
            index
            for index, state in enumerate(cell["states"])
            if abs(len(state) - candidate_size) <= maximum_cardinality_delta
        ]
    indices = sorted(
        eligible,
        key=lambda index: (-float(scores[index]), index),
    )[: max(1, int(top_k))]
    if include_candidate_state:
        candidate_indices = [
            index
            for index, state in enumerate(cell["states"])
            if state == cell["candidate"]
        ]
        if len(candidate_indices) != 1:
            raise ValueError("State lattice must contain exactly one candidate state")
        candidate_index = candidate_indices[0]
        if candidate_index not in eligible:
            raise ValueError("Candidate state was excluded by the cardinality policy")
        if candidate_index not in indices:
            if len(indices) >= max(1, int(top_k)):
                indices[-1] = candidate_index
            else:
                indices.append(candidate_index)
    return indices


def make_meta_song(
    song: dict[str, Any],
    base_model: dict[str, Any],
    *,
    top_k: int,
    maximum_cardinality_delta: int | None = None,
    include_candidate_state: bool = False,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell in song["cells"]:
        scores = predict(cell["features"], base_model)
        indices = ranked_state_indices(
            cell,
            scores,
            top_k=top_k,
            maximum_cardinality_delta=maximum_cardinality_delta,
            include_candidate_state=include_candidate_state,
        )
        incumbent_index = indices.index(
            next(
                index
                for index, state in enumerate(cell["states"])
                if state == cell["candidate"]
            )
        ) if include_candidate_state else 0
        cells.append(
            {
                "target": cell["target"],
                "candidate": cell["candidate"],
                "states": [cell["states"][index] for index in indices],
                "features": meta_feature_matrix(cell["features"], scores, indices),
                "rewards": np.asarray(
                    [float(cell["rewards"][index]) for index in indices], dtype=float
                ),
                "baseScores": np.asarray(
                    [float(scores[index]) for index in indices], dtype=float
                ),
                "incumbentIndex": incumbent_index,
            }
        )
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": cells,
    }


def fit_meta_ridge(
    songs: list[dict[str, Any]], *, ridge: float, best_state_weight: float
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        cell_weight = (
            max(1e-9, float(song["weight"]))
            / max(1, group_counts[song["groupId"]])
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            rewards = cell["rewards"]
            weights = np.full(len(rewards), cell_weight / max(1, len(rewards)))
            maximum = float(np.max(rewards))
            weights *= 1.0 + max(0.0, best_state_weight) * np.isclose(
                rewards, maximum, atol=1e-9
            )
            matrices.append(cell["features"])
            targets.append(rewards)
            sample_weights.append(weights)
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    weights = np.concatenate(sample_weights)
    total = max(1e-12, float(np.sum(weights)))
    means = np.sum(matrix * weights[:, None], axis=0) / total
    variances = np.sum(((matrix - means) ** 2) * weights[:, None], axis=0) / total
    scales = np.sqrt(np.maximum(variances, 1e-8))
    design = np.column_stack([np.ones(len(matrix)), (matrix - means) / scales])
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    penalty = np.eye(design.shape[1]) * max(0.0, ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (target * root_weights),
    )
    return {
        "family": "ridge-reward",
        "featureNames": list(META_FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "ridge": float(ridge),
        "bestStateWeight": float(best_state_weight),
    }


def fit_meta_pairwise(
    songs: list[dict[str, Any]],
    *,
    regularization_c: float,
    maximum_iterations: int = 1000,
) -> dict[str, Any]:
    """Fit a scalar utility from reward-ordered pairs within each top-k list."""

    matrices: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        cell_weight = (
            max(1e-9, float(song["weight"]))
            / max(1, group_counts[song["groupId"]])
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            pairs: list[tuple[int, int, float]] = []
            for left in range(len(cell["rewards"])):
                for right in range(left + 1, len(cell["rewards"])):
                    gap = float(cell["rewards"][left] - cell["rewards"][right])
                    if abs(gap) > 1e-9:
                        pairs.append((left, right, gap))
            normalizer = sum(abs(gap) for _, _, gap in pairs)
            for left, right, gap in pairs:
                difference = np.asarray(
                    cell["features"][left] - cell["features"][right], dtype=float
                )
                pair_weight = cell_weight * abs(gap) / max(1e-12, normalizer)
                matrices.extend([difference, -difference])
                labels.extend([int(gap > 0.0), int(gap < 0.0)])
                weights.extend([pair_weight / 2.0, pair_weight / 2.0])
    if not matrices:
        raise ValueError("Top-k pairwise training received no informative pairs")
    matrix = np.asarray(matrices, dtype=float)
    target = np.asarray(labels, dtype=np.int8)
    sample_weights = normalize_sample_weights(np.asarray(weights, dtype=float))
    total = float(np.sum(sample_weights))
    means = np.sum(matrix * sample_weights[:, None], axis=0) / total
    variances = np.sum(
        ((matrix - means) ** 2) * sample_weights[:, None], axis=0
    ) / total
    scales = np.sqrt(np.maximum(variances, 1e-8))
    classifier = LogisticRegression(
        C=max(1e-4, float(regularization_c)),
        fit_intercept=False,
        max_iter=max(1, int(maximum_iterations)),
        solver="lbfgs",
        random_state=6701,
    )
    classifier.fit(
        (matrix - means) / scales,
        target,
        sample_weight=sample_weights,
    )
    return {
        "family": "pairwise-logistic",
        "featureNames": list(META_FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": np.asarray(classifier.coef_[0], dtype=float),
        "regularizationC": float(regularization_c),
        "trainingPairs": int(len(target)),
        "optimizerIterations": int(classifier.n_iter_[0]),
    }


def fit_meta_boosted(
    songs: list[dict[str, Any]],
    *,
    learning_rate: float,
    maximum_iterations: int,
    maximum_leaf_nodes: int,
    minimum_samples_leaf: int,
    l2_regularization: float,
) -> dict[str, Any]:
    """Fit a bounded nonlinear reward model only on cross-fitted top states."""

    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        cell_weight = (
            max(1e-9, float(song["weight"]))
            / max(1, group_counts[song["groupId"]])
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            matrices.append(cell["features"])
            targets.append(cell["rewards"])
            weights.append(
                np.full(
                    len(cell["rewards"]),
                    cell_weight / max(1, len(cell["rewards"])),
                )
            )
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    sample_weights = normalize_sample_weights(np.concatenate(weights))
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=max(1e-4, float(learning_rate)),
        max_iter=max(1, int(maximum_iterations)),
        max_leaf_nodes=max(2, int(maximum_leaf_nodes)),
        min_samples_leaf=max(2, int(minimum_samples_leaf)),
        l2_regularization=max(0.0, float(l2_regularization)),
        early_stopping=False,
        random_state=6701,
    )
    estimator.fit(matrix, target, sample_weight=sample_weights)
    return {
        "family": "hist-gradient-reward",
        "featureNames": list(META_FEATURE_NAMES),
        "estimator": estimator,
        "learningRate": max(1e-4, float(learning_rate)),
        "maximumIterations": max(1, int(maximum_iterations)),
        "maximumLeafNodes": max(2, int(maximum_leaf_nodes)),
        "minimumSamplesLeaf": max(2, int(minimum_samples_leaf)),
        "l2Regularization": max(0.0, float(l2_regularization)),
        "trainingExamples": int(len(target)),
    }


def fit_meta_xgb_ranker(
    songs: list[dict[str, Any]],
    *,
    number_of_trees: int,
    learning_rate: float,
    maximum_depth: int,
    minimum_child_weight: float,
    l2_regularization: float,
) -> dict[str, Any]:
    """Fit LambdaMART with each candidate onset represented as one query."""

    if XGBRanker is None:
        raise RuntimeError(
            "The xgb meta family requires the optional 'xgboost' package."
        )

    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_sizes: list[int] = []
    query_weights: list[float] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        query_weight = (
            max(1e-9, float(song["weight"]))
            / max(1, group_counts[song["groupId"]])
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            matrices.append(cell["features"])
            # NDCG uses non-negative graded relevance.  The lattice reward is
            # set F1, so twelve levels preserve its ordering and common ties.
            targets.append(np.rint(cell["rewards"] * 12.0).astype(np.int16))
            group_sizes.append(len(cell["rewards"]))
            query_weights.append(query_weight)
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    weights = normalize_sample_weights(np.asarray(query_weights, dtype=float))
    estimator = XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=max(2, min(8, max(group_sizes))),
        n_estimators=max(10, int(number_of_trees)),
        learning_rate=max(1e-4, float(learning_rate)),
        max_depth=max(1, int(maximum_depth)),
        min_child_weight=max(0.0, float(minimum_child_weight)),
        reg_lambda=max(0.0, float(l2_regularization)),
        reg_alpha=0.0,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=4,
        random_state=6701,
        verbosity=0,
    )
    estimator.fit(
        matrix,
        target,
        group=np.asarray(group_sizes, dtype=np.int32),
        sample_weight=weights,
        verbose=False,
    )
    return {
        "family": "xgb-lambdamart",
        "featureNames": list(META_FEATURE_NAMES),
        "estimator": estimator,
        "numberOfTrees": max(10, int(number_of_trees)),
        "learningRate": max(1e-4, float(learning_rate)),
        "maximumDepth": max(1, int(maximum_depth)),
        "minimumChildWeight": max(0.0, float(minimum_child_weight)),
        "l2Regularization": max(0.0, float(l2_regularization)),
        "trainingExamples": int(len(target)),
        "trainingQueries": int(len(group_sizes)),
    }


def meta_predict(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    if model.get("family") == "hist-gradient-reward":
        if model.get("estimator") is not None:
            return np.asarray(model["estimator"].predict(matrix), dtype=float)
        return predict_serialized_hist_gradient(matrix, model)
    if model.get("family") == "xgb-lambdamart":
        return np.asarray(model["estimator"].predict(matrix), dtype=float)
    standardized = (matrix - model["means"]) / model["scales"]
    if model.get("family") == "pairwise-logistic":
        return standardized @ model["coefficients"]
    return model["coefficients"][0] + standardized @ model["coefficients"][1:]


def serialize_hist_gradient_estimator(estimator: Any) -> dict[str, Any]:
    """Serialize sklearn's numeric HistGradientBoosting trees to stable JSON.

    The state-reranker features are continuous numeric values, so each tree can
    be represented by its raw threshold nodes.  Leaf values already include
    sklearn's learning-rate shrinkage; inference is therefore the baseline
    prediction plus one selected leaf from every tree.
    """

    baseline = np.asarray(estimator._baseline_prediction, dtype=float).reshape(-1)
    if len(baseline) != 1:
        raise ValueError("Only single-output histogram regressors are supported")
    trees: list[list[dict[str, Any]]] = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("Only one regression tree per boosting stage is supported")
        nodes = stage[0].nodes
        serialized_nodes: list[dict[str, Any]] = []
        for node in nodes:
            if bool(node["is_categorical"]):
                raise ValueError("Categorical histogram splits are not supported")
            serialized_nodes.append(
                {
                    "value": round(float(node["value"]), 12),
                    "feature": int(node["feature_idx"]),
                    "threshold": round(float(node["num_threshold"]), 12),
                    "missingLeft": bool(node["missing_go_to_left"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "leaf": bool(node["is_leaf"]),
                }
            )
        trees.append(serialized_nodes)
    return {
        "baselinePrediction": round(float(baseline[0]), 12),
        "trees": trees,
    }


def predict_serialized_hist_gradient(
    matrix: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    """Run a JSON-serialized numeric HistGradientBoosting regressor."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Serialized histogram inference expects a 2-D matrix")
    trees = model.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("Serialized histogram model has no trees")
    predictions = np.full(
        values.shape[0], float(model.get("baselinePrediction", 0.0)), dtype=float
    )
    for tree in trees:
        if not isinstance(tree, list) or not tree:
            raise ValueError("Serialized histogram model contains an empty tree")
        for row_index, row in enumerate(values):
            node_index = 0
            while True:
                if node_index < 0 or node_index >= len(tree):
                    raise ValueError("Serialized histogram tree has an invalid child index")
                node = tree[node_index]
                if bool(node.get("leaf")):
                    predictions[row_index] += float(node.get("value", 0.0))
                    break
                feature = int(node.get("feature", -1))
                if feature < 0 or feature >= values.shape[1]:
                    raise ValueError("Serialized histogram tree has an invalid feature index")
                feature_value = float(row[feature])
                go_left = (
                    bool(node.get("missingLeft"))
                    if np.isnan(feature_value)
                    else feature_value <= float(node.get("threshold", 0.0))
                )
                node_index = int(node["left"] if go_left else node["right"])
    return predictions


def select_meta_index(
    scores: np.ndarray,
    minimum_override_margin: float,
    *,
    incumbent_index: int = 0,
    minimum_candidate_override_margin: float = 0.0,
) -> int:
    """Keep the base top-one state unless the reranker clears its margin."""

    selected = int(np.argmax(scores))
    if selected != 0:
        margin = float(scores[selected] - scores[0])
        if margin < max(0.0, minimum_override_margin):
            selected = 0
    incumbent_index = max(0, min(len(scores) - 1, int(incumbent_index)))
    if selected != incumbent_index:
        incumbent_margin = float(scores[selected] - scores[incumbent_index])
        if incumbent_margin < max(0.0, minimum_candidate_override_margin):
            selected = incumbent_index
    return selected


def evaluate_meta_song(
    song: dict[str, Any],
    model: dict[str, Any],
    *,
    minimum_override_margin: float = 0.0,
    minimum_candidate_override_margin: float = 0.0,
) -> dict[str, Any]:
    base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    meta_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    oracle_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    changes = improvements = regressions = 0
    for cell in song["cells"]:
        incumbent_index = int(cell.get("incumbentIndex", 0))
        selected_index = select_meta_index(
            meta_predict(cell["features"], model),
            minimum_override_margin,
            incumbent_index=incumbent_index,
            minimum_candidate_override_margin=minimum_candidate_override_margin,
        )
        oracle_index = int(np.argmax(cell["rewards"]))
        target = cell["target"]
        base = cell["states"][incumbent_index]
        selected = cell["states"][selected_index]
        oracle = cell["states"][oracle_index]
        for bucket, state in (
            (base_counts, base),
            (meta_counts, selected),
            (oracle_counts, oracle),
        ):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        base_reward = set_f1(target, base)
        selected_reward = set_f1(target, selected)
        changes += int(selected != base)
        improvements += int(selected_reward > base_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < base_reward)
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": len(song["cells"]),
        "base": micro_metrics(base_counts),
        "reranker": micro_metrics(meta_counts),
        "topKOracle": micro_metrics(oracle_counts),
        "changes": changes,
        "improvements": improvements,
        "regressions": regressions,
        "minimumOverrideMargin": float(minimum_override_margin),
        "minimumCandidateOverrideMargin": float(
            minimum_candidate_override_margin
        ),
    }


def weighted(rows: list[dict[str, Any]], section: str) -> float:
    total = sum(max(0.0, float(row["weight"])) for row in rows)
    return sum(
        max(0.0, float(row["weight"])) * float(row[section]["f1"])
        for row in rows
    ) / max(1e-12, total)


def serialize_model(
    model: dict[str, Any],
    *,
    top_k: int,
    minimum_override_margin: float,
    maximum_cardinality_delta: int | None = None,
    include_candidate_state: bool = False,
    minimum_candidate_override_margin: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "polymath-pitch-class-state-topk-reranker-v1",
        "family": model.get("family", "ridge-reward"),
        "featureNames": model["featureNames"],
        "topK": int(top_k),
        "minimumOverrideMargin": float(minimum_override_margin),
        "maximumCardinalityDelta": maximum_cardinality_delta,
        "includeCandidateState": bool(include_candidate_state),
        "minimumCandidateOverrideMargin": float(
            minimum_candidate_override_margin
        ),
        "trainingBoundary": "whole-composition-cross-fitted-base-scores",
    }
    if model.get("family") == "xgb-lambdamart":
        payload.update(
            {
                "numberOfTrees": model["numberOfTrees"],
                "learningRate": model["learningRate"],
                "maximumDepth": model["maximumDepth"],
                "minimumChildWeight": model["minimumChildWeight"],
                "l2Regularization": model["l2Regularization"],
                "trainingExamples": model["trainingExamples"],
                "trainingQueries": model["trainingQueries"],
                "runtimeArtifactAvailable": False,
                "note": "Research metadata only; booster serialization is deferred until a promotion gate passes.",
            }
        )
    elif model.get("family") == "hist-gradient-reward":
        runtime = serialize_hist_gradient_estimator(model["estimator"])
        payload.update(
            {
                "learningRate": model["learningRate"],
                "maximumIterations": model["maximumIterations"],
                "maximumLeafNodes": model["maximumLeafNodes"],
                "minimumSamplesLeaf": model["minimumSamplesLeaf"],
                "l2Regularization": model["l2Regularization"],
                "trainingExamples": model["trainingExamples"],
                **runtime,
                "runtimeArtifactAvailable": True,
                "note": "Numeric histogram trees are serialized for deterministic research inference.",
            }
        )
    elif model.get("family") == "pairwise-logistic":
        payload.update(
            {
                "means": [round(float(value), 10) for value in model["means"]],
                "scales": [round(float(value), 10) for value in model["scales"]],
                "coefficients": [
                    round(float(value), 10) for value in model["coefficients"]
                ],
                "regularizationC": model["regularizationC"],
                "trainingPairs": model["trainingPairs"],
                "optimizerIterations": model["optimizerIterations"],
            }
        )
    else:
        payload.update(
            {
                "means": [round(float(value), 10) for value in model["means"]],
                "scales": [round(float(value), 10) for value in model["scales"]],
                "coefficients": [
                    round(float(value), 10) for value in model["coefficients"]
                ],
                "ridge": model["ridge"],
                "bestStateWeight": model["bestStateWeight"],
            }
        )
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def serialize_base_model(
    model: dict[str, Any], *, maximum_states: int
) -> dict[str, Any]:
    """Serialize the composition-excluded first-stage ranker for replay."""

    payload = {
        "schema": "polymath-pitch-class-state-ranker-v1",
        "featureNames": list(model["featureNames"]),
        "means": [round(float(value), 10) for value in model["means"]],
        "scales": [round(float(value), 10) for value in model["scales"]],
        "coefficients": [
            round(float(value), 10) for value in model["coefficients"]
        ],
        "ridge": float(model["ridge"]),
        "bestStateWeight": float(model["bestStateWeight"]),
        "minimumMargin": 0.0,
        "maximumStates": int(maximum_states),
        "evidenceBoundary": "evaluation composition excluded from base fitting",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument(
        "--evaluation-audit",
        help="Optional separate audit whose songs are evaluated but never trained on.",
    )
    parser.add_argument(
        "--meta-extra-audit",
        nargs="+",
        help="Optional paired songs used only to teach the top-k reranker.",
    )
    parser.add_argument("--meta-extra-audit-weight", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--maximum-cardinality-delta",
        type=int,
        help=(
            "Optional swap-first constraint: shortlist only states whose pitch-class "
            "count differs from the incumbent gesture by at most this amount."
        ),
    )
    parser.add_argument(
        "--include-candidate-state",
        action="store_true",
        help="Force the incumbent arrangement into every top-k shortlist.",
    )
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument(
        "--allow-harmonic-completion",
        action="store_true",
        help="Research-only: add complete implied triads when two tones are supported.",
    )
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--meta-family",
        choices=("ridge", "pairwise", "boosted", "xgb"),
        default="ridge",
    )
    parser.add_argument("--meta-ridge-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--meta-best-state-weight-grid", type=float, nargs="+", default=[0.0, 1.0, 3.0, 8.0])
    parser.add_argument(
        "--meta-regularization-c-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--boosted-learning-rate", type=float, default=0.05)
    parser.add_argument("--boosted-maximum-iterations", type=int, default=160)
    parser.add_argument(
        "--boosted-maximum-leaf-nodes-grid",
        type=int,
        nargs="+",
        default=[3, 7, 15],
    )
    parser.add_argument(
        "--boosted-minimum-samples-leaf-grid",
        type=int,
        nargs="+",
        default=[20, 50],
    )
    parser.add_argument(
        "--boosted-l2-grid", type=float, nargs="+", default=[1.0, 10.0]
    )
    parser.add_argument("--xgb-number-of-trees", type=int, default=160)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--xgb-maximum-depth-grid", type=int, nargs="+", default=[2, 3]
    )
    parser.add_argument(
        "--xgb-minimum-child-weight-grid",
        type=float,
        nargs="+",
        default=[5.0, 20.0],
    )
    parser.add_argument(
        "--xgb-l2-grid", type=float, nargs="+", default=[10.0, 50.0]
    )
    parser.add_argument(
        "--override-margin-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12],
    )
    parser.add_argument(
        "--candidate-override-margin-grid",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Minimum reranker advantage over the incumbent candidate before "
            "a gesture may change. Requires --include-candidate-state."
        ),
    )
    parser.add_argument(
        "--prepared-cache",
        help=(
            "Optional joblib cache for the expensive prepared state lattice. "
            "The cache is reused only when every input fingerprint and preparation "
            "setting matches exactly."
        ),
    )
    parser.add_argument(
        "--heldout-model-dir",
        help=(
            "Optional directory for replayable composition-excluded base and meta "
            "models for each evaluation fold."
        ),
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    maximum_states = max(4, min(256, int(args.maximum_states)))
    top_k = max(2, min(maximum_states, int(args.top_k)))
    maximum_cardinality_delta = (
        None
        if args.maximum_cardinality_delta is None
        else max(0, int(args.maximum_cardinality_delta))
    )
    meta_extra_paths = [
        Path(value).resolve() for value in (args.meta_extra_audit or [])
    ]
    evaluation_path = (
        Path(args.evaluation_audit).resolve() if args.evaluation_audit else None
    )
    prepared_cache_path = (
        Path(args.prepared_cache).resolve() if args.prepared_cache else None
    )
    preparation_spec = preparation_cache_spec(
        audit_path=audit_path,
        evaluation_path=evaluation_path,
        meta_extra_paths=meta_extra_paths,
        maximum_states=maximum_states,
        allow_harmonic_completion=bool(args.allow_harmonic_completion),
        meta_extra_audit_weight=max(0.0, args.meta_extra_audit_weight),
    )
    songs, evaluation_source, meta_extra_songs, prepared_cache_status = (
        load_or_build_prepared_inputs(
            cache_path=prepared_cache_path,
            spec=preparation_spec,
            audit_path=audit_path,
            evaluation_path=evaluation_path,
            meta_extra_paths=meta_extra_paths,
            maximum_states=maximum_states,
            allow_harmonic_completion=bool(args.allow_harmonic_completion),
            meta_extra_audit_weight=max(0.0, args.meta_extra_audit_weight),
        )
    )
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in evaluation_source if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    # Cache base models by excluded composition set.  Every cached fit is
    # reference-clean for every composition it later scores.
    base_statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    all_base_groups = set(base_statistics)
    model_cache: dict[frozenset[str], dict[str, Any]] = {}

    def base_model_excluding(groups: set[str]) -> dict[str, Any]:
        key = frozenset(groups)
        if key not in model_cache:
            model_cache[key] = fit_ridge_from_group_statistics(
                base_statistics,
                included_groups=sorted(all_base_groups - set(key)),
                ridge=max(0.0, args.base_ridge),
                best_state_weight=max(0.0, args.base_best_state_weight),
            )
        return model_cache[key]

    fold_material: list[dict[str, Any]] = []
    for heldout in evaluation:
        outer_group = heldout["groupId"]
        meta_training: list[dict[str, Any]] = []
        training_groups = sorted(
            {song["groupId"] for song in songs if song["groupId"] != outer_group}
        )
        for inner_group in training_groups:
            inner_model = base_model_excluding({outer_group, inner_group})
            meta_training.extend(
                make_meta_song(
                    song,
                    inner_model,
                    top_k=top_k,
                    maximum_cardinality_delta=maximum_cardinality_delta,
                    include_candidate_state=bool(args.include_candidate_state),
                )
                for song in songs
                if song["groupId"] == inner_group
            )
        outer_model = base_model_excluding({outer_group})
        for extra_song in meta_extra_songs:
            if extra_song["groupId"] == outer_group:
                continue
            excluded = {outer_group}
            if extra_song["groupId"] in all_base_groups:
                excluded.add(extra_song["groupId"])
            extra_model = base_model_excluding(excluded)
            meta_training.append(
                make_meta_song(
                    extra_song,
                    extra_model,
                    top_k=top_k,
                    maximum_cardinality_delta=maximum_cardinality_delta,
                    include_candidate_state=bool(args.include_candidate_state),
                )
            )
        meta_heldout = make_meta_song(
            heldout,
            outer_model,
            top_k=top_k,
            maximum_cardinality_delta=maximum_cardinality_delta,
            include_candidate_state=bool(args.include_candidate_state),
        )
        fold_material.append(
            {
                "heldout": heldout,
                "outerBaseModel": outer_model,
                "metaTraining": meta_training,
                "metaHeldout": meta_heldout,
            }
        )

    trials: list[dict[str, Any]] = []
    if args.meta_family == "xgb":
        configurations = [
            {
                "family": "xgb",
                "maximumDepth": max(1, int(depth)),
                "minimumChildWeight": max(0.0, float(child_weight)),
                "l2Regularization": max(0.0, float(l2)),
                "bestStateWeight": 0.0,
            }
            for depth in args.xgb_maximum_depth_grid
            for child_weight in args.xgb_minimum_child_weight_grid
            for l2 in args.xgb_l2_grid
        ]
    elif args.meta_family == "boosted":
        configurations = [
            {
                "family": "boosted",
                "maximumLeafNodes": max(2, int(leaves)),
                "minimumSamplesLeaf": max(2, int(minimum_leaf)),
                "l2Regularization": max(0.0, float(l2)),
                "bestStateWeight": 0.0,
            }
            for leaves in args.boosted_maximum_leaf_nodes_grid
            for minimum_leaf in args.boosted_minimum_samples_leaf_grid
            for l2 in args.boosted_l2_grid
        ]
    elif args.meta_family == "pairwise":
        configurations = [
            {
                "family": "pairwise",
                "regularizationC": max(1e-4, float(value)),
                "bestStateWeight": 0.0,
            }
            for value in args.meta_regularization_c_grid
        ]
    else:
        configurations = [
            {
                "family": "ridge",
                "ridge": max(0.0, float(ridge)),
                "bestStateWeight": max(0.0, float(best_weight)),
            }
            for ridge in args.meta_ridge_grid
            for best_weight in args.meta_best_state_weight_grid
        ]

    def fit_configuration(
        meta_training: list[dict[str, Any]], configuration: dict[str, Any]
    ) -> dict[str, Any]:
        if configuration["family"] == "xgb":
            return fit_meta_xgb_ranker(
                meta_training,
                number_of_trees=args.xgb_number_of_trees,
                learning_rate=args.xgb_learning_rate,
                maximum_depth=int(configuration["maximumDepth"]),
                minimum_child_weight=float(configuration["minimumChildWeight"]),
                l2_regularization=float(configuration["l2Regularization"]),
            )
        if configuration["family"] == "boosted":
            return fit_meta_boosted(
                meta_training,
                learning_rate=args.boosted_learning_rate,
                maximum_iterations=args.boosted_maximum_iterations,
                maximum_leaf_nodes=int(configuration["maximumLeafNodes"]),
                minimum_samples_leaf=int(configuration["minimumSamplesLeaf"]),
                l2_regularization=float(configuration["l2Regularization"]),
            )
        if configuration["family"] == "pairwise":
            return fit_meta_pairwise(
                meta_training,
                regularization_c=float(configuration["regularizationC"]),
            )
        return fit_meta_ridge(
            meta_training,
            ridge=float(configuration["ridge"]),
            best_state_weight=float(configuration["bestStateWeight"]),
        )

    for configuration in configurations:
        fitted_folds: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for fold in fold_material:
            model = fit_configuration(fold["metaTraining"], configuration)
            fitted_folds.append((fold["metaHeldout"], model))
        for margin in sorted(
            {max(0.0, float(value)) for value in args.override_margin_grid}
        ):
            for candidate_margin in sorted(
                {
                    max(0.0, float(value))
                    for value in args.candidate_override_margin_grid
                }
            ):
                rows = [
                    evaluate_meta_song(
                        heldout,
                        model,
                        minimum_override_margin=margin,
                        minimum_candidate_override_margin=candidate_margin,
                    )
                    for heldout, model in fitted_folds
                ]
                trials.append(
                    {
                        **configuration,
                        "minimumOverrideMargin": margin,
                        "minimumCandidateOverrideMargin": candidate_margin,
                        "baseWeightedPitchClassF1": round(weighted(rows, "base"), 6),
                        "weightedPitchClassF1": round(weighted(rows, "reranker"), 6),
                        "worstSongPitchClassF1": round(
                            min(float(row["reranker"]["f1"]) for row in rows), 6
                        ),
                        "topKOracleWeightedPitchClassF1": round(
                            weighted(rows, "topKOracle"), 6
                        ),
                        "folds": rows,
                    }
                )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            -float(row.get("bestStateWeight", 0.0)),
            -float(
                row.get(
                    "ridge",
                    row.get(
                        "regularizationC",
                        row.get("maximumLeafNodes", row.get("maximumDepth", 0.0)),
                    ),
                )
            ),
            -float(row["minimumOverrideMargin"]),
            -float(row.get("minimumCandidateOverrideMargin", 0.0)),
        ),
    )

    # Final research artifact: all compositions receive cross-fitted base
    # scores before the meta model is trained.
    final_meta: list[dict[str, Any]] = []
    all_groups = sorted({song["groupId"] for song in songs})
    for group in all_groups:
        inner_model = base_model_excluding({group})
        final_meta.extend(
            make_meta_song(
                song,
                inner_model,
                top_k=top_k,
                maximum_cardinality_delta=maximum_cardinality_delta,
                include_candidate_state=bool(args.include_candidate_state),
            )
            for song in songs
            if song["groupId"] == group
        )
    for extra_song in meta_extra_songs:
        excluded = (
            {extra_song["groupId"]}
            if extra_song["groupId"] in all_base_groups
            else set()
        )
        extra_model = base_model_excluding(excluded)
        final_meta.append(
            make_meta_song(
                extra_song,
                extra_model,
                top_k=top_k,
                maximum_cardinality_delta=maximum_cardinality_delta,
                include_candidate_state=bool(args.include_candidate_state),
            )
        )
    final_model = fit_configuration(final_meta, best)
    model_payload = serialize_model(
        final_model,
        top_k=top_k,
        minimum_override_margin=float(best["minimumOverrideMargin"]),
        maximum_cardinality_delta=maximum_cardinality_delta,
        include_candidate_state=bool(args.include_candidate_state),
        minimum_candidate_override_margin=float(
            best.get("minimumCandidateOverrideMargin", 0.0)
        ),
    )
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    heldout_models: list[dict[str, Any]] = []
    if args.heldout_model_dir:
        heldout_model_dir = Path(args.heldout_model_dir).resolve()
        heldout_model_dir.mkdir(parents=True, exist_ok=True)
        for fold_index, fold in enumerate(fold_material, start=1):
            heldout_id = str(fold["heldout"]["id"])
            suffix = f"-{fold_index}" if len(fold_material) > 1 else ""
            base_path = heldout_model_dir / f"{heldout_id}{suffix}-base.json"
            meta_path = heldout_model_dir / f"{heldout_id}{suffix}-meta.json"
            heldout_base = serialize_base_model(
                fold["outerBaseModel"], maximum_states=maximum_states
            )
            heldout_meta = serialize_model(
                fit_configuration(fold["metaTraining"], best),
                top_k=top_k,
                minimum_override_margin=float(best["minimumOverrideMargin"]),
                maximum_cardinality_delta=maximum_cardinality_delta,
                include_candidate_state=bool(args.include_candidate_state),
                minimum_candidate_override_margin=float(
                    best.get("minimumCandidateOverrideMargin", 0.0)
                ),
            )
            base_path.write_text(json.dumps(heldout_base, indent=2), encoding="utf-8")
            meta_path.write_text(json.dumps(heldout_meta, indent=2), encoding="utf-8")
            heldout_models.append(
                {
                    "id": heldout_id,
                    "base": str(base_path),
                    "baseSha256": heldout_base["sha256"],
                    "meta": str(meta_path),
                    "metaSha256": heldout_meta["sha256"],
                }
            )

    report = {
        "schema": "polymath-pitch-class-state-topk-reranker-training-v1",
        "status": "opened-development-only",
        "audit": str(audit_path),
        "evaluationAudit": str(evaluation_path) if evaluation_path else None,
        "metaExtraAudits": [str(path) for path in meta_extra_paths],
        "evaluationSongIds": [song["id"] for song in evaluation],
        "settings": {
            "maximumStates": maximum_states,
            "topK": top_k,
            "maximumCardinalityDelta": maximum_cardinality_delta,
            "includeCandidateState": bool(args.include_candidate_state),
            "allowHarmonicCompletion": bool(args.allow_harmonic_completion),
            "metaFamily": args.meta_family,
            "baseRidge": max(0.0, args.base_ridge),
            "baseBestStateWeight": max(0.0, args.base_best_state_weight),
            "metaTrainingBoundary": "outer-and-inner-composition-excluded",
            "cachedBaseFits": len(model_cache),
            "metaExtraAuditWeight": max(0.0, args.meta_extra_audit_weight),
            "preparedCache": (
                str(prepared_cache_path) if prepared_cache_path is not None else None
            ),
            "preparedCacheStatus": prepared_cache_status,
            "overrideMarginGrid": sorted(
                {max(0.0, float(value)) for value in args.override_margin_grid}
            ),
            "candidateOverrideMarginGrid": sorted(
                {
                    max(0.0, float(value))
                    for value in args.candidate_override_margin_grid
                }
            ),
        },
        "best": best,
        "trials": trials,
        "model": {"path": str(model_path), "sha256": model_payload["sha256"]},
        "heldoutModels": heldout_models,
        "decision": {
            "promotable": False,
            "reason": "Meta hyperparameters were compared on opened songs; a new frozen song is required.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "model": str(model_path),
                "topK": top_k,
                "baseWeightedPitchClassF1": best["baseWeightedPitchClassF1"],
                "weightedPitchClassF1": best["weightedPitchClassF1"],
                "worstSongPitchClassF1": best["worstSongPitchClassF1"],
                "topKOracleWeightedPitchClassF1": best[
                    "topKOracleWeightedPitchClassF1"
                ],
                "metaFamily": best["family"],
                "ridge": best.get("ridge"),
                "regularizationC": best.get("regularizationC"),
                "maximumLeafNodes": best.get("maximumLeafNodes"),
                "maximumDepth": best.get("maximumDepth"),
                "minimumChildWeight": best.get("minimumChildWeight"),
                "minimumSamplesLeaf": best.get("minimumSamplesLeaf"),
                "l2Regularization": best.get("l2Regularization"),
                "bestStateWeight": best.get("bestStateWeight", 0.0),
                "minimumOverrideMargin": best["minimumOverrideMargin"],
                "minimumCandidateOverrideMargin": best.get(
                    "minimumCandidateOverrideMargin", 0.0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
