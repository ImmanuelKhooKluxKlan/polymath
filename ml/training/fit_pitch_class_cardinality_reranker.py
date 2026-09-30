"""Cross-fit a chord-cardinality prior over a state ranker's top-k list.

The base state lattice already contains much better answers, but it commonly
keeps too many pitch classes.  This module predicts target set size from
target-blind query evidence, then softly penalizes top-k states whose size is
far from that prediction.  Every training query receives base scores from a
model that excluded its entire composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics
    from .fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        prepare_song,
        ridge_group_statistics,
    )
    from .fit_pitch_class_state_topk_reranker import (
        META_FEATURE_NAMES,
        make_meta_song,
    )
except ImportError:  # pragma: no cover - direct script fallback.
    from analyze_pitch_class_state_lattice import micro_metrics
    from fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        prepare_song,
        ridge_group_statistics,
    )
    from fit_pitch_class_state_topk_reranker import (
        META_FEATURE_NAMES,
        make_meta_song,
    )


def cardinality_feature_names(top_k: int) -> tuple[str, ...]:
    return (
        *(f"top1_{name}" for name in META_FEATURE_NAMES),
        *(f"rank_{rank + 1}_score_margin" for rank in range(top_k)),
        *(f"rank_{rank + 1}_state_size" for rank in range(top_k)),
        *(f"size_{size}_best_score_margin" for size in range(1, 7)),
        *(f"size_{size}_available" for size in range(1, 7)),
        "topk_state_size_mean",
        "topk_state_size_std",
        "topk_unique_state_sizes",
    )


def cardinality_features(cell: dict[str, Any], *, top_k: int) -> np.ndarray:
    """Create fixed-length, target-blind features for one top-k query."""

    matrix = np.asarray(cell["features"], dtype=float)
    scores = np.asarray(cell["baseScores"], dtype=float)
    states = cell["states"]
    actual = min(top_k, len(states))
    if not actual:
        raise ValueError("Cardinality features require at least one state.")
    margins = np.full(top_k, 2.0, dtype=float)
    sizes = np.zeros(top_k, dtype=float)
    margins[:actual] = scores[0] - scores[:actual]
    sizes[:actual] = [len(state) for state in states[:actual]]
    best_by_size = np.full(6, 2.0, dtype=float)
    available = np.zeros(6, dtype=float)
    for state, score in zip(states[:actual], scores[:actual], strict=True):
        size = min(6, max(1, len(state)))
        best_by_size[size - 1] = min(best_by_size[size - 1], scores[0] - score)
        available[size - 1] = 1.0
    active_sizes = sizes[:actual]
    values = np.concatenate(
        [
            matrix[0],
            margins,
            sizes / 6.0,
            best_by_size,
            available,
            np.asarray(
                [
                    float(np.mean(active_sizes)) / 6.0,
                    float(np.std(active_sizes)) / 6.0,
                    len(set(int(value) for value in active_sizes)) / 6.0,
                ]
            ),
        ]
    )
    return np.nan_to_num(values, nan=0.0, posinf=2.0, neginf=-2.0)


def fit_cardinality_ridge(
    songs: list[dict[str, Any]], *, top_k: int, ridge: float
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    targets: list[float] = []
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
            matrices.append(cardinality_features(cell, top_k=top_k))
            targets.append(float(min(6, max(1, len(cell["target"])))))
            weights.append(cell_weight)
    matrix = np.vstack(matrices)
    target = np.asarray(targets, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    total = max(1e-12, float(np.sum(sample_weights)))
    means = np.sum(matrix * sample_weights[:, None], axis=0) / total
    variances = (
        np.sum(((matrix - means) ** 2) * sample_weights[:, None], axis=0) / total
    )
    scales = np.sqrt(np.maximum(variances, 1e-8))
    design = np.column_stack([np.ones(len(matrix)), (matrix - means) / scales])
    roots = np.sqrt(sample_weights)
    weighted_design = design * roots[:, None]
    penalty = np.eye(design.shape[1]) * max(0.0, float(ridge))
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (target * roots),
    )
    return {
        "family": "ridge-cardinality",
        "featureNames": list(cardinality_feature_names(top_k)),
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "ridge": float(ridge),
        "topK": int(top_k),
        "trainingExamples": len(target),
    }


def fit_cardinality_logistic(
    songs: list[dict[str, Any]], *, top_k: int, regularization_c: float
) -> dict[str, Any]:
    """Fit a discrete target-size distribution instead of averaging modes."""

    matrices: list[np.ndarray] = []
    targets: list[int] = []
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
            matrices.append(cardinality_features(cell, top_k=top_k))
            targets.append(min(6, max(1, len(cell["target"]))))
            weights.append(cell_weight)
    matrix = np.vstack(matrices)
    target = np.asarray(targets, dtype=np.int16)
    sample_weights = np.asarray(weights, dtype=float)
    total = max(1e-12, float(np.sum(sample_weights)))
    means = np.sum(matrix * sample_weights[:, None], axis=0) / total
    variances = (
        np.sum(((matrix - means) ** 2) * sample_weights[:, None], axis=0) / total
    )
    scales = np.sqrt(np.maximum(variances, 1e-8))
    standardized = (matrix - means) / scales
    sample_weights *= len(sample_weights) / max(1e-12, float(np.sum(sample_weights)))
    estimator = LogisticRegression(
        C=max(1e-5, float(regularization_c)),
        solver="lbfgs",
        max_iter=2000,
        random_state=6701,
    )
    estimator.fit(standardized, target, sample_weight=sample_weights)
    return {
        "family": "multinomial-cardinality",
        "featureNames": list(cardinality_feature_names(top_k)),
        "means": means,
        "scales": scales,
        "estimator": estimator,
        "regularizationC": float(regularization_c),
        "topK": int(top_k),
        "trainingExamples": len(target),
    }


def predict_cardinality(cell: dict[str, Any], model: dict[str, Any]) -> float:
    values = cardinality_features(cell, top_k=int(model["topK"]))
    standardized = (values - model["means"]) / model["scales"]
    if model["family"] == "multinomial-cardinality":
        probabilities = model["estimator"].predict_proba(standardized[None, :])[0]
        classes = np.asarray(model["estimator"].classes_, dtype=float)
        return float(probabilities @ classes)
    prediction = float(model["coefficients"][0] + standardized @ model["coefficients"][1:])
    return min(6.0, max(1.0, prediction))


def predict_cardinality_probabilities(
    cell: dict[str, Any], model: dict[str, Any]
) -> np.ndarray | None:
    if model["family"] != "multinomial-cardinality":
        return None
    values = cardinality_features(cell, top_k=int(model["topK"]))
    standardized = (values - model["means"]) / model["scales"]
    learned = model["estimator"].predict_proba(standardized[None, :])[0]
    probabilities = np.full(6, 1e-6, dtype=float)
    for target_class, probability in zip(
        model["estimator"].classes_, learned, strict=True
    ):
        probabilities[min(6, max(1, int(target_class))) - 1] = float(probability)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def select_state_index(
    cell: dict[str, Any],
    *,
    predicted_size: float,
    size_penalty: float,
    size_probabilities: np.ndarray | None = None,
) -> int:
    scores = np.asarray(cell["baseScores"], dtype=float)
    if size_probabilities is None:
        adjusted = np.asarray(
            [
                score - max(0.0, size_penalty) * abs(len(state) - predicted_size)
                for state, score in zip(cell["states"], scores, strict=True)
            ],
            dtype=float,
        )
    else:
        adjusted = np.asarray(
            [
                score
                + max(0.0, size_penalty)
                * np.log(max(1e-6, float(size_probabilities[min(6, max(1, len(state))) - 1])))
                for state, score in zip(cell["states"], scores, strict=True)
            ],
            dtype=float,
        )
    return int(np.argmax(adjusted))


def evaluate_song(
    song: dict[str, Any], model: dict[str, Any], *, size_penalty: float
) -> dict[str, Any]:
    base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    selected_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    predictions: list[float] = []
    targets: list[int] = []
    changes = 0
    for cell in song["cells"]:
        prediction = predict_cardinality(cell, model)
        probabilities = predict_cardinality_probabilities(cell, model)
        selected_index = select_state_index(
            cell,
            predicted_size=prediction,
            size_penalty=size_penalty,
            size_probabilities=probabilities,
        )
        target = cell["target"]
        base = cell["states"][0]
        selected = cell["states"][selected_index]
        for bucket, state in ((base_counts, base), (selected_counts, selected)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        changes += int(selected_index != 0)
        predictions.append(prediction)
        targets.append(len(target))
    target_values = np.asarray(targets, dtype=float)
    predicted_values = np.asarray(predictions, dtype=float)
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": len(song["cells"]),
        "base": micro_metrics(base_counts),
        "cardinalityReranker": micro_metrics(selected_counts),
        "changes": changes,
        "cardinalityMeanAbsoluteError": round(
            float(np.mean(np.abs(predicted_values - target_values))), 6
        ),
        "roundedCardinalityExactShare": round(
            float(np.mean(np.rint(predicted_values) == target_values)), 6
        ),
        "meanPredictedCardinality": round(float(np.mean(predicted_values)), 6),
        "meanTargetCardinality": round(float(np.mean(target_values)), 6),
    }


def weighted(rows: list[dict[str, Any]], section: str) -> float:
    total = sum(max(0.0, float(row["weight"])) for row in rows)
    return sum(
        max(0.0, float(row["weight"])) * float(row[section]["f1"])
        for row in rows
    ) / max(1e-12, total)


def serialize_model(model: dict[str, Any], *, size_penalty: float) -> dict[str, Any]:
    payload = {
        "schema": "polymath-pitch-class-cardinality-reranker-v1",
        "family": model["family"],
        "featureNames": model["featureNames"],
        "means": [round(float(value), 10) for value in model["means"]],
        "scales": [round(float(value), 10) for value in model["scales"]],
        "topK": int(model["topK"]),
        "sizePenalty": float(size_penalty),
        "trainingExamples": int(model["trainingExamples"]),
        "trainingBoundary": "whole-composition-cross-fitted-base-scores",
    }
    if model["family"] == "multinomial-cardinality":
        payload.update(
            {
                "classes": [int(value) for value in model["estimator"].classes_],
                "coefficients": [
                    [round(float(value), 10) for value in row]
                    for row in model["estimator"].coef_
                ],
                "intercepts": [
                    round(float(value), 10) for value in model["estimator"].intercept_
                ],
                "regularizationC": float(model["regularizationC"]),
            }
        )
    else:
        payload.update(
            {
                "coefficients": [
                    round(float(value), 10) for value in model["coefficients"]
                ],
                "ridge": float(model["ridge"]),
            }
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def load_songs(path: Path, maximum_states: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(prepare_song(copied, maximum_states=maximum_states))
    return songs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--evaluation-audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--cardinality-family", choices=("ridge", "logistic"), default="ridge"
    )
    parser.add_argument("--cardinality-ridge-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument(
        "--cardinality-c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0]
    )
    parser.add_argument(
        "--size-penalty-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.4, 0.8],
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    maximum_states = max(4, min(256, int(args.maximum_states)))
    top_k = max(2, min(maximum_states, int(args.top_k)))
    songs = load_songs(audit_path, maximum_states)
    evaluation_path = Path(args.evaluation_audit).resolve() if args.evaluation_audit else None
    evaluation_source = (
        load_songs(evaluation_path, maximum_states) if evaluation_path else songs
    )
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in evaluation_source if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    all_groups = set(statistics)
    model_cache: dict[frozenset[str], dict[str, Any]] = {}

    def base_model_excluding(groups: set[str]) -> dict[str, Any]:
        key = frozenset(groups)
        if key not in model_cache:
            model_cache[key] = fit_ridge_from_group_statistics(
                statistics,
                included_groups=sorted(all_groups - groups),
                ridge=max(0.0, args.base_ridge),
                best_state_weight=max(0.0, args.base_best_state_weight),
            )
        return model_cache[key]

    fold_material: list[dict[str, Any]] = []
    for heldout in evaluation:
        outer_group = heldout["groupId"]
        training: list[dict[str, Any]] = []
        for inner_group in sorted(all_groups - {outer_group}):
            inner_model = base_model_excluding({outer_group, inner_group})
            training.extend(
                make_meta_song(song, inner_model, top_k=top_k)
                for song in songs
                if song["groupId"] == inner_group
            )
        outer_model = base_model_excluding({outer_group})
        fold_material.append(
            {
                "training": training,
                "heldout": make_meta_song(heldout, outer_model, top_k=top_k),
            }
        )

    trials: list[dict[str, Any]] = []
    if args.cardinality_family == "logistic":
        configurations = [
            {"family": "logistic", "regularizationC": max(1e-5, float(value))}
            for value in sorted(set(args.cardinality_c_grid))
        ]
    else:
        configurations = [
            {"family": "ridge", "ridge": max(0.0, float(value))}
            for value in sorted(set(args.cardinality_ridge_grid))
        ]
    for configuration in configurations:
        fitted = [
            (
                fold["heldout"],
                (
                    fit_cardinality_logistic(
                        fold["training"],
                        top_k=top_k,
                        regularization_c=float(configuration["regularizationC"]),
                    )
                    if configuration["family"] == "logistic"
                    else fit_cardinality_ridge(
                        fold["training"],
                        top_k=top_k,
                        ridge=float(configuration["ridge"]),
                    )
                ),
            )
            for fold in fold_material
        ]
        for penalty in sorted({max(0.0, float(value)) for value in args.size_penalty_grid}):
            rows = [
                evaluate_song(song, model, size_penalty=penalty)
                for song, model in fitted
            ]
            trials.append(
                {
                    **configuration,
                    "sizePenalty": penalty,
                    "baseWeightedPitchClassF1": round(weighted(rows, "base"), 6),
                    "weightedPitchClassF1": round(
                        weighted(rows, "cardinalityReranker"), 6
                    ),
                    "worstSongPitchClassF1": round(
                        min(float(row["cardinalityReranker"]["f1"]) for row in rows), 6
                    ),
                    "meanCardinalityAbsoluteError": round(
                        float(np.mean([row["cardinalityMeanAbsoluteError"] for row in rows])), 6
                    ),
                    "folds": rows,
                }
            )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            -float(row.get("ridge", row.get("regularizationC", 0.0))),
            -float(row["sizePenalty"]),
        ),
    )

    final_training: list[dict[str, Any]] = []
    for group in sorted(all_groups):
        inner_model = base_model_excluding({group})
        final_training.extend(
            make_meta_song(song, inner_model, top_k=top_k)
            for song in songs
            if song["groupId"] == group
        )
    final_model = (
        fit_cardinality_logistic(
            final_training,
            top_k=top_k,
            regularization_c=float(best["regularizationC"]),
        )
        if best["family"] == "logistic"
        else fit_cardinality_ridge(
            final_training, top_k=top_k, ridge=float(best["ridge"])
        )
    )
    model_payload = serialize_model(final_model, size_penalty=float(best["sizePenalty"]))
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    report = {
        "schema": "polymath-pitch-class-cardinality-reranker-training-v1",
        "status": "opened-development-only",
        "audit": str(audit_path),
        "evaluationAudit": str(evaluation_path) if evaluation_path else None,
        "evaluationSongIds": args.evaluation_song_ids,
        "settings": {
            "topK": top_k,
            "maximumStates": maximum_states,
            "cardinalityFamily": args.cardinality_family,
            "validationBoundary": "outer-and-inner-composition-excluded",
            "cachedBaseFits": len(model_cache),
        },
        "best": best,
        "trials": trials,
        "model": {"path": str(model_path), "sha256": model_payload["sha256"]},
        "decision": {
            "promotable": False,
            "reason": "Opened-song strategy selection requires independent transfer validation.",
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
                **{key: value for key, value in best.items() if key != "folds"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
