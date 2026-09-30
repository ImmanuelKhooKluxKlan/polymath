"""Fit a nonlinear whole-gesture ranker with song-group holdouts.

This is a research-only companion to ``fit_pitch_class_state_ranker``.  It
uses shallow histogram gradient-boosted trees to model interactions between
source evidence, texture, and complete chord states.  Models are stored with
joblib and must not be promoted until an unopened song and end-to-end exact
note gates pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
from .fit_pitch_class_state_ranker import (
    canonical_song_id,
    finite,
    normalize_sample_weights,
    prepare_song,
)


def training_arrays(
    songs: list[dict[str, Any]], *, best_state_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    group_totals: dict[str, int] = {}
    for song in songs:
        group_totals[song["groupId"]] = group_totals.get(song["groupId"], 0) + 1
    for song in songs:
        # Every unique song group contributes the requested total weight even
        # when two historical candidates exist for that same composition.
        group_divisor = max(1, group_totals[song["groupId"]])
        cell_weight = (
            max(1e-9, song["weight"])
            / group_divisor
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            rewards = cell["rewards"]
            maximum = float(np.max(rewards))
            weights = np.full(len(rewards), cell_weight / max(1, len(rewards)))
            weights *= 1.0 + max(0.0, best_state_weight) * np.isclose(
                rewards, maximum, atol=1e-9
            )
            matrices.append(cell["features"])
            targets.append(rewards)
            sample_weights.append(weights)
    return np.vstack(matrices), np.concatenate(targets), np.concatenate(sample_weights)


def fit_model(
    songs: list[dict[str, Any]],
    *,
    maximum_iterations: int,
    learning_rate: float,
    maximum_leaf_nodes: int,
    minimum_samples_leaf: int,
    l2_regularization: float,
    best_state_weight: float,
) -> HistGradientBoostingRegressor:
    matrix, targets, weights = training_arrays(
        songs, best_state_weight=best_state_weight
    )
    weights = normalize_sample_weights(weights)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=maximum_iterations,
        max_leaf_nodes=maximum_leaf_nodes,
        min_samples_leaf=minimum_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=6701,
    )
    model.fit(matrix, targets, sample_weight=weights)
    return model


def evaluate_song(
    song: dict[str, Any], model: HistGradientBoostingRegressor
) -> dict[str, Any]:
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    baseline_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    exact = baseline_exact = changes = improvements = regressions = 0
    reward = baseline_reward = 0.0
    for cell in song["cells"]:
        scores = model.predict(cell["features"])
        selected = cell["states"][int(np.argmax(scores))]
        target = cell["target"]
        candidate = cell["candidate"]
        for bucket, state in ((counts, selected), (baseline_counts, candidate)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        selected_reward = set_f1(target, selected)
        current_reward = set_f1(target, candidate)
        reward += selected_reward
        baseline_reward += current_reward
        exact += int(selected == target)
        baseline_exact += int(candidate == target)
        changes += int(selected != candidate)
        improvements += int(selected_reward > current_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < current_reward)
    cells = len(song["cells"])
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": cells,
        "baseline": {
            **micro_metrics(baseline_counts),
            "macroF1": round(baseline_reward / max(1, cells), 6),
            "exactSetRate": round(baseline_exact / max(1, cells), 6),
        },
        "ranker": {
            **micro_metrics(counts),
            "macroF1": round(reward / max(1, cells), 6),
            "exactSetRate": round(exact / max(1, cells), 6),
        },
        "changes": changes,
        "improvements": improvements,
        "regressions": regressions,
    }


def weighted_metric(
    rows: list[dict[str, Any]], weights: dict[str, float], field: str
) -> float:
    total = sum(max(0.0, weights.get(row["id"], 1.0)) for row in rows)
    return sum(
        max(0.0, weights.get(row["id"], 1.0))
        * float(row["ranker"][field])
        for row in rows
    ) / max(1e-12, total)


def configurations() -> list[dict[str, Any]]:
    return [
        {
            "maximumIterations": 120,
            "learningRate": 0.04,
            "maximumLeafNodes": 7,
            "minimumSamplesLeaf": 120,
            "l2Regularization": 2.0,
            "bestStateWeight": 1.0,
        },
        {
            "maximumIterations": 160,
            "learningRate": 0.035,
            "maximumLeafNodes": 15,
            "minimumSamplesLeaf": 150,
            "l2Regularization": 4.0,
            "bestStateWeight": 1.0,
        },
        {
            "maximumIterations": 140,
            "learningRate": 0.04,
            "maximumLeafNodes": 15,
            "minimumSamplesLeaf": 250,
            "l2Regularization": 10.0,
            "bestStateWeight": 1.0,
        },
        {
            "maximumIterations": 180,
            "learningRate": 0.03,
            "maximumLeafNodes": 23,
            "minimumSamplesLeaf": 250,
            "l2Regularization": 12.0,
            "bestStateWeight": 2.0,
        },
    ]


def bounded_configurations(indices: list[int], maximum_iterations_cap: int) -> list[dict[str, Any]]:
    available = configurations()
    selected: list[dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(available):
            raise ValueError(
                f"Configuration index {index} is outside 0..{len(available) - 1}"
            )
        config = dict(available[index])
        config["configurationIndex"] = index
        config["maximumIterations"] = min(
            max(1, int(maximum_iterations_cap)), int(config["maximumIterations"])
        )
        selected.append(config)
    if not selected:
        raise ValueError("At least one boosted configuration is required")
    return selected


def estimated_training_rows(songs: list[dict[str, Any]]) -> int:
    return sum(len(cell["states"]) for song in songs for cell in song["cells"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, nargs="+")
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--extra-audit-weight", type=float, default=0.5)
    parser.add_argument(
        "--configuration-index",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Explicit bounded subset of the four predeclared configurations.",
    )
    parser.add_argument("--maximum-iterations-cap", type=int, default=180)
    parser.add_argument("--maximum-fits", type=int, default=20)
    parser.add_argument("--maximum-training-rows", type=int, default=250_000)
    args = parser.parse_args()

    audit_paths = [Path(value).resolve() for value in args.audit]
    raw_songs: list[dict[str, Any]] = []
    for audit_index, audit_path in enumerate(audit_paths):
        audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
        multiplier = 1.0 if audit_index == 0 else max(0.0, args.extra_audit_weight)
        for song in audit.get("songs") or []:
            copied = dict(song)
            copied["weight"] = max(0.0, finite(song.get("weight"), 1.0)) * multiplier
            copied["groupId"] = canonical_song_id(song.get("id"))
            raw_songs.append(copied)
    maximum_states = max(4, min(256, int(args.maximum_states)))
    songs = [prepare_song(song, maximum_states=maximum_states) for song in raw_songs]
    evaluation_ids = set(args.evaluation_song_ids)
    evaluation_songs = [song for song in songs if song["id"] in evaluation_ids]
    missing = evaluation_ids - {song["id"] for song in evaluation_songs}
    if missing:
        raise ValueError(f"Unknown evaluation songs: {sorted(missing)}")
    weights = {song["id"]: song["weight"] for song in evaluation_songs}

    selected_configurations = bounded_configurations(
        args.configuration_index, args.maximum_iterations_cap
    )
    required_fits = len(selected_configurations) * len(evaluation_songs) + 1
    if required_fits > max(1, int(args.maximum_fits)):
        raise ValueError(
            f"Requested {required_fits} boosted fits exceeds --maximum-fits "
            f"{args.maximum_fits}"
        )
    training_rows = estimated_training_rows(songs)
    if training_rows > max(1, int(args.maximum_training_rows)):
        raise ValueError(
            f"Prepared {training_rows} state rows exceeds --maximum-training-rows "
            f"{args.maximum_training_rows}"
        )

    trials: list[dict[str, Any]] = []
    fit_number = 0
    started = time.perf_counter()
    for config in selected_configurations:
        folds: list[dict[str, Any]] = []
        for held_out in evaluation_songs:
            fit_number += 1
            print(
                json.dumps(
                    {
                        "event": "boosted_fit_started",
                        "fit": fit_number,
                        "totalFits": required_fits,
                        "configurationIndex": config["configurationIndex"],
                        "heldOut": held_out["id"],
                    }
                ),
                flush=True,
            )
            training = [
                song for song in songs if song["groupId"] != held_out["groupId"]
            ]
            model = fit_model(
                training,
                maximum_iterations=config["maximumIterations"],
                learning_rate=config["learningRate"],
                maximum_leaf_nodes=config["maximumLeafNodes"],
                minimum_samples_leaf=config["minimumSamplesLeaf"],
                l2_regularization=config["l2Regularization"],
                best_state_weight=config["bestStateWeight"],
            )
            folds.append(evaluate_song(held_out, model))
        trials.append(
            {
                "config": config,
                "weightedPitchClassF1": round(
                    weighted_metric(folds, weights, "f1"), 6
                ),
                "weightedExactSetRate": round(
                    weighted_metric(folds, weights, "exactSetRate"), 6
                ),
                "worstSongPitchClassF1": round(
                    min(float(row["ranker"]["f1"]) for row in folds), 6
                ),
                "folds": folds,
            }
        )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            float(row["weightedExactSetRate"]),
        ),
    )
    config = best["config"]
    fit_number += 1
    print(
        json.dumps(
            {
                "event": "boosted_final_fit_started",
                "fit": fit_number,
                "totalFits": required_fits,
                "configurationIndex": config["configurationIndex"],
            }
        ),
        flush=True,
    )
    final = fit_model(
        songs,
        maximum_iterations=config["maximumIterations"],
        learning_rate=config["learningRate"],
        maximum_leaf_nodes=config["maximumLeafNodes"],
        minimum_samples_leaf=config["minimumSamplesLeaf"],
        l2_regularization=config["l2Regularization"],
        best_state_weight=config["bestStateWeight"],
    )
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema": "polymath-pitch-class-state-boosted-ranker-v1",
            "maximumStates": maximum_states,
            "configuration": config,
            "model": final,
        },
        temporary_model_path,
    )
    temporary_model_path.replace(model_path)
    report = {
        "schema": "polymath-pitch-class-state-boosted-training-v1",
        "status": "opened-development-only",
        "audits": [str(path) for path in audit_paths],
        "evaluationSongIds": args.evaluation_song_ids,
        "resourceBounds": {
            "configurationIndices": args.configuration_index,
            "maximumIterationsCap": int(args.maximum_iterations_cap),
            "requiredFits": required_fits,
            "maximumFits": int(args.maximum_fits),
            "preparedStateRows": training_rows,
            "maximumTrainingRows": int(args.maximum_training_rows),
            "elapsedSeconds": round(time.perf_counter() - started, 3),
        },
        "foldRule": "exclude every candidate sharing the held-out canonical song id",
        "best": best,
        "trials": trials,
        "model": str(model_path),
        "decision": {
            "promotable": False,
            "reason": "Opened development folds; end-to-end exact-note and sealed-song gates remain.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_output.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "model": str(model_path),
                "bestConfig": config,
                "weightedPitchClassF1": best["weightedPitchClassF1"],
                "worstSongPitchClassF1": best["worstSongPitchClassF1"],
                "weightedExactSetRate": best["weightedExactSetRate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
