"""Fit a globally consistent linear pairwise piano-state ranker.

Training examples are differences between two states from the same onset.
Their weights are proportional to the reference F1 gap.  Unlike the boosted
pairwise tournament, inference uses one scalar utility per state, so pairwise
cycles are impossible and no absolute cross-onset reward calibration is
required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .fit_pitch_class_state_pairwise_ranker import selected_pairs
    from .fit_pitch_class_state_ranker import (
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        normalize_sample_weights,
        prepare_song,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1  # type: ignore
    from fit_pitch_class_state_pairwise_ranker import selected_pairs  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        normalize_sample_weights,
        prepare_song,
    )


def training_arrays(
    songs: list[dict[str, Any]], *, maximum_pairs_per_cell: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        song_weight = max(1e-9, song["weight"]) / max(
            1, group_counts[song["groupId"]]
        )
        cell_weight = song_weight / max(1, len(song["cells"]))
        for cell in song["cells"]:
            pairs = selected_pairs(cell, maximum_pairs=max(1, maximum_pairs_per_cell))
            if not pairs:
                continue
            normalizer = sum(
                abs(float(cell["rewards"][left]) - float(cell["rewards"][right]))
                for left, right in pairs
            )
            for left, right in pairs:
                reward_gap = float(cell["rewards"][left] - cell["rewards"][right])
                if abs(reward_gap) <= 1e-9:
                    continue
                difference = np.asarray(
                    cell["features"][left] - cell["features"][right],
                    dtype=np.float32,
                )
                pair_weight = cell_weight * abs(reward_gap) / max(1e-12, normalizer)
                matrices.extend([difference, -difference])
                labels.extend([int(reward_gap > 0.0), int(reward_gap < 0.0)])
                weights.extend([pair_weight / 2.0, pair_weight / 2.0])
    if not matrices:
        raise ValueError("Linear pairwise training received no informative pairs")
    return (
        np.asarray(matrices, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
    )


def fit_model(
    songs: list[dict[str, Any]],
    *,
    maximum_pairs_per_cell: int,
    regularization_c: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    matrix, labels, weights = training_arrays(
        songs, maximum_pairs_per_cell=maximum_pairs_per_cell
    )
    normalized_weights = normalize_sample_weights(weights)
    total_weight = float(np.sum(normalized_weights))
    means = np.sum(matrix * normalized_weights[:, None], axis=0) / total_weight
    variances = np.sum(
        ((matrix - means) ** 2) * normalized_weights[:, None], axis=0
    ) / total_weight
    scales = np.sqrt(np.maximum(variances, 1e-8))
    standardized = (matrix - means) / scales
    classifier = LogisticRegression(
        C=max(1e-4, float(regularization_c)),
        fit_intercept=False,
        max_iter=max(1, int(maximum_iterations)),
        solver="lbfgs",
        random_state=6701,
    )
    classifier.fit(standardized, labels, sample_weight=normalized_weights)
    return {
        "featureNames": list(FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": np.asarray(classifier.coef_[0], dtype=float),
        "regularizationC": float(regularization_c),
        "maximumPairsPerCell": int(maximum_pairs_per_cell),
        "trainingPairs": int(len(labels)),
        "optimizerIterations": int(classifier.n_iter_[0]),
    }


def predict(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    standardized = (matrix - model["means"]) / model["scales"]
    return standardized @ model["coefficients"]


def evaluate_song(song: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    baseline = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    ranker = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    exact = baseline_exact = changes = improvements = regressions = 0
    for cell in song["cells"]:
        selected = cell["states"][int(np.argmax(predict(cell["features"], model)))]
        candidate = cell["candidate"]
        target = cell["target"]
        for bucket, state in ((baseline, candidate), (ranker, selected)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        selected_reward = set_f1(target, selected)
        baseline_reward = set_f1(target, candidate)
        exact += int(selected == target)
        baseline_exact += int(candidate == target)
        changes += int(selected != candidate)
        improvements += int(selected_reward > baseline_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < baseline_reward)
    cells = len(song["cells"])
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": cells,
        "baseline": {
            **micro_metrics(baseline),
            "exactSetRate": round(baseline_exact / max(1, cells), 6),
        },
        "ranker": {
            **micro_metrics(ranker),
            "exactSetRate": round(exact / max(1, cells), 6),
        },
        "changes": changes,
        "improvements": improvements,
        "regressions": regressions,
    }


def weighted(rows: list[dict[str, Any]], weights: dict[str, float], field: str) -> float:
    total = sum(max(0.0, weights.get(row["id"], 1.0)) for row in rows)
    return sum(
        max(0.0, weights.get(row["id"], 1.0)) * float(row["ranker"][field])
        for row in rows
    ) / max(1e-12, total)


def serialize(model: dict[str, Any], maximum_states: int) -> dict[str, Any]:
    payload = {
        "schema": "polymath-pitch-class-state-linear-pairwise-ranker-v1",
        "featureNames": model["featureNames"],
        "means": [round(float(value), 10) for value in model["means"]],
        "scales": [round(float(value), 10) for value in model["scales"]],
        "coefficients": [round(float(value), 10) for value in model["coefficients"]],
        "regularizationC": model["regularizationC"],
        "maximumPairsPerCell": model["maximumPairsPerCell"],
        "maximumStates": maximum_states,
        "trainingPairs": model["trainingPairs"],
        "optimizerIterations": model["optimizerIterations"],
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, nargs="+")
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--maximum-pairs-per-cell", type=int, default=24)
    parser.add_argument("--regularization-c", type=float, default=0.2)
    parser.add_argument("--maximum-iterations", type=int, default=1000)
    parser.add_argument("--maximum-training-pairs", type=int, default=400_000)
    parser.add_argument("--maximum-fits", type=int, default=5)
    parser.add_argument("--extra-audit-weight", type=float, default=0.5)
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in args.audit]
    raw_songs: list[dict[str, Any]] = []
    for audit_index, path in enumerate(paths):
        audit = json.loads(path.read_text(encoding="utf-8-sig"))
        multiplier = 1.0 if audit_index == 0 else max(0.0, args.extra_audit_weight)
        for raw in audit.get("songs") or []:
            copied = dict(raw)
            copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0)) * multiplier
            copied["groupId"] = canonical_song_id(raw.get("id"))
            raw_songs.append(copied)
    maximum_states = max(4, min(256, int(args.maximum_states)))
    maximum_pairs = max(1, int(args.maximum_pairs_per_cell))
    songs = [prepare_song(song, maximum_states=maximum_states) for song in raw_songs]
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation songs: {sorted(missing)}")
    required_fits = len(evaluation) + 1
    if required_fits > max(1, int(args.maximum_fits)):
        raise ValueError(
            f"Requested {required_fits} fits exceeds --maximum-fits {args.maximum_fits}"
        )
    estimated_pairs = 2 * maximum_pairs * sum(len(song["cells"]) for song in songs)
    if estimated_pairs > max(1, int(args.maximum_training_pairs)):
        raise ValueError(
            f"Estimated {estimated_pairs} directed pairs exceeds "
            f"--maximum-training-pairs {args.maximum_training_pairs}"
        )

    folds: list[dict[str, Any]] = []
    optimizers: dict[str, Any] = {}
    started = time.perf_counter()
    for fit_number, held_out in enumerate(evaluation, start=1):
        print(
            json.dumps(
                {
                    "event": "linear_pairwise_fit_started",
                    "fit": fit_number,
                    "totalFits": required_fits,
                    "heldOut": held_out["id"],
                }
            ),
            flush=True,
        )
        training = [song for song in songs if song["groupId"] != held_out["groupId"]]
        model = fit_model(
            training,
            maximum_pairs_per_cell=maximum_pairs,
            regularization_c=max(1e-4, args.regularization_c),
            maximum_iterations=max(1, args.maximum_iterations),
        )
        optimizers[held_out["id"]] = {
            "trainingPairs": model["trainingPairs"],
            "iterations": model["optimizerIterations"],
        }
        folds.append(evaluate_song(held_out, model))

    print(
        json.dumps(
            {
                "event": "linear_pairwise_final_fit_started",
                "fit": required_fits,
                "totalFits": required_fits,
            }
        ),
        flush=True,
    )
    final = fit_model(
        songs,
        maximum_pairs_per_cell=maximum_pairs,
        regularization_c=max(1e-4, args.regularization_c),
        maximum_iterations=max(1, args.maximum_iterations),
    )
    payload = serialize(final, maximum_states)
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    temporary_model.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_model.replace(model_path)

    song_weights = {song["id"]: song["weight"] for song in evaluation}
    aggregate = {
        "weightedPitchClassF1": round(weighted(folds, song_weights, "f1"), 6),
        "worstSongPitchClassF1": round(
            min(float(row["ranker"]["f1"]) for row in folds), 6
        ),
        "weightedExactSetRate": round(
            weighted(folds, song_weights, "exactSetRate"), 6
        ),
    }
    report = {
        "schema": "polymath-pitch-class-state-linear-pairwise-training-v1",
        "status": "opened-development-only",
        "audits": [str(path) for path in paths],
        "evaluationSongIds": [song["id"] for song in evaluation],
        "settings": {
            "maximumStates": maximum_states,
            "maximumPairsPerCell": maximum_pairs,
            "regularizationC": max(1e-4, args.regularization_c),
            "maximumIterations": max(1, args.maximum_iterations),
        },
        "resourceBounds": {
            "requiredFits": required_fits,
            "maximumFits": int(args.maximum_fits),
            "estimatedDirectedPairs": estimated_pairs,
            "maximumTrainingPairs": int(args.maximum_training_pairs),
            "elapsedSeconds": round(time.perf_counter() - started, 3),
        },
        "aggregate": aggregate,
        "foldOptimizers": optimizers,
        "folds": folds,
        "model": {"path": str(model_path), "sha256": payload["sha256"]},
        "decision": {
            "promotable": False,
            "reason": "Opened development songs; exact-note, render, and sealed-song gates remain.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_output.replace(output)
    print(json.dumps({"output": str(output), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
