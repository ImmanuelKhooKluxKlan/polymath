"""Fit a composition-held-out listwise ranker over complete piano states.

Reward regression predicts an absolute F1 value for states from unrelated
onsets.  Inference only needs an ordering *within one onset*.  This module
therefore optimizes a grouped softmax objective: probability mass should land
on the best reference-scored state(s) in each frozen, target-blind lattice.

The model is deliberately linear and convex.  That makes this an objective
test rather than an unbounded capacity sweep, while preserving the same
target-blind state features and whole-song validation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .fit_pitch_class_state_ranker import (
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        prepare_song,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        prepare_song,
    )


def target_distribution(rewards: np.ndarray, temperature: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or not len(rewards):
        raise ValueError("Each listwise cell needs at least one reward")
    if temperature <= 0.0:
        best = np.isclose(rewards, float(np.max(rewards)), atol=1e-9)
        return best.astype(float) / max(1, int(np.sum(best)))
    shifted = (rewards - float(np.max(rewards))) / max(1e-6, temperature)
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    return probabilities / max(1e-12, float(np.sum(probabilities)))


def flatten_lists(
    songs: list[dict[str, Any]], *, target_temperature: float
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lengths: list[int] = []
    cell_weights: list[float] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        song_weight = max(1e-9, song["weight"]) / max(
            1, group_counts[song["groupId"]]
        )
        per_cell = song_weight / max(1, len(song["cells"]))
        for cell in song["cells"]:
            matrix = np.asarray(cell["features"], dtype=np.float64)
            matrices.append(matrix)
            targets.append(
                target_distribution(cell["rewards"], target_temperature)
            )
            lengths.append(len(matrix))
            cell_weights.append(per_cell)
    if not matrices:
        raise ValueError("Listwise training received no cells")
    weights = np.asarray(cell_weights, dtype=np.float64)
    weights /= max(1e-12, float(np.sum(weights)))
    lengths_array = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths_array)[:-1]]
    )
    repeated_cell_weights = np.repeat(weights, lengths_array)
    state_weights = repeated_cell_weights / np.repeat(lengths_array, lengths_array)
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    means = np.sum(matrix * state_weights[:, None], axis=0) / max(
        1e-12, float(np.sum(state_weights))
    )
    variances = np.sum(
        ((matrix - means) ** 2) * state_weights[:, None], axis=0
    ) / max(1e-12, float(np.sum(state_weights)))
    scales = np.sqrt(np.maximum(variances, 1e-8))
    standardized = (matrix - means) / scales
    return {
        "matrix": standardized,
        "target": target,
        "lengths": lengths_array,
        "offsets": offsets,
        "cellWeights": weights,
        "repeatedCellWeights": repeated_cell_weights,
        "means": means,
        "scales": scales,
    }


def listwise_loss_and_gradient(
    coefficients: np.ndarray,
    flattened: dict[str, Any],
    *,
    l2_regularization: float,
) -> tuple[float, np.ndarray]:
    matrix = flattened["matrix"]
    target = flattened["target"]
    offsets = flattened["offsets"]
    lengths = flattened["lengths"]
    cell_weights = flattened["cellWeights"]
    repeated_cell_weights = flattened["repeatedCellWeights"]
    scores = matrix @ coefficients
    cell_maxima = np.maximum.reduceat(scores, offsets)
    shifted = scores - np.repeat(cell_maxima, lengths)
    exponentials = np.exp(np.clip(shifted, -60.0, 0.0))
    denominators = np.add.reduceat(exponentials, offsets)
    probabilities = exponentials / np.repeat(denominators, lengths)
    expected_score = np.add.reduceat(target * scores, offsets)
    log_partition = cell_maxima + np.log(np.maximum(1e-12, denominators))
    ridge = max(0.0, float(l2_regularization))
    loss = float(np.sum(cell_weights * (log_partition - expected_score)))
    loss += 0.5 * ridge * float(coefficients @ coefficients)
    errors = (probabilities - target) * repeated_cell_weights
    gradient = matrix.T @ errors + ridge * coefficients
    return loss, np.asarray(gradient, dtype=np.float64)


def fit_model(
    songs: list[dict[str, Any]],
    *,
    target_temperature: float,
    l2_regularization: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    flattened = flatten_lists(songs, target_temperature=target_temperature)
    initial = np.zeros(flattened["matrix"].shape[1], dtype=np.float64)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return listwise_loss_and_gradient(
            values,
            flattened,
            l2_regularization=l2_regularization,
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max(1, int(maximum_iterations)), "ftol": 1e-10, "gtol": 1e-7},
    )
    return {
        "featureNames": list(FEATURE_NAMES),
        "means": flattened["means"],
        "scales": flattened["scales"],
        "coefficients": np.asarray(result.x, dtype=np.float64),
        "targetTemperature": float(target_temperature),
        "l2Regularization": float(l2_regularization),
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "functionEvaluations": int(result.nfev),
            "finalLoss": float(result.fun),
        },
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
        candidate_reward = set_f1(target, candidate)
        baseline_exact += int(candidate == target)
        exact += int(selected == target)
        changes += int(selected != candidate)
        improvements += int(selected_reward > candidate_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < candidate_reward)
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
        "schema": "polymath-pitch-class-state-listwise-ranker-v1",
        "featureNames": model["featureNames"],
        "means": [round(float(value), 10) for value in model["means"]],
        "scales": [round(float(value), 10) for value in model["scales"]],
        "coefficients": [round(float(value), 10) for value in model["coefficients"]],
        "targetTemperature": model["targetTemperature"],
        "l2Regularization": model["l2Regularization"],
        "optimizer": model["optimizer"],
        "maximumStates": maximum_states,
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
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--maximum-iterations", type=int, default=160)
    parser.add_argument("--maximum-training-rows", type=int, default=150_000)
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
    songs = [prepare_song(song, maximum_states=maximum_states) for song in raw_songs]
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation songs: {sorted(missing)}")
    required_fits = len(evaluation) + 1
    if required_fits > max(1, int(args.maximum_fits)):
        raise ValueError(
            f"Requested {required_fits} listwise fits exceeds --maximum-fits {args.maximum_fits}"
        )
    state_rows = sum(len(cell["states"]) for song in songs for cell in song["cells"])
    if state_rows > max(1, int(args.maximum_training_rows)):
        raise ValueError(
            f"Prepared {state_rows} state rows exceeds --maximum-training-rows "
            f"{args.maximum_training_rows}"
        )

    started = time.perf_counter()
    folds: list[dict[str, Any]] = []
    fold_optimizers: dict[str, Any] = {}
    for fit_number, held_out in enumerate(evaluation, start=1):
        print(
            json.dumps(
                {
                    "event": "listwise_fit_started",
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
            target_temperature=max(0.0, args.target_temperature),
            l2_regularization=max(0.0, args.l2_regularization),
            maximum_iterations=max(1, args.maximum_iterations),
        )
        fold_optimizers[held_out["id"]] = model["optimizer"]
        folds.append(evaluate_song(held_out, model))

    print(
        json.dumps(
            {"event": "listwise_final_fit_started", "fit": required_fits, "totalFits": required_fits}
        ),
        flush=True,
    )
    final = fit_model(
        songs,
        target_temperature=max(0.0, args.target_temperature),
        l2_regularization=max(0.0, args.l2_regularization),
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
        "schema": "polymath-pitch-class-state-listwise-training-v1",
        "status": "opened-development-only",
        "audits": [str(path) for path in paths],
        "evaluationSongIds": [song["id"] for song in evaluation],
        "settings": {
            "maximumStates": maximum_states,
            "targetTemperature": max(0.0, args.target_temperature),
            "l2Regularization": max(0.0, args.l2_regularization),
            "maximumIterations": max(1, args.maximum_iterations),
        },
        "resourceBounds": {
            "requiredFits": required_fits,
            "maximumFits": int(args.maximum_fits),
            "preparedStateRows": state_rows,
            "maximumTrainingRows": int(args.maximum_training_rows),
            "elapsedSeconds": round(time.perf_counter() - started, 3),
        },
        "aggregate": aggregate,
        "foldOptimizers": fold_optimizers,
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
