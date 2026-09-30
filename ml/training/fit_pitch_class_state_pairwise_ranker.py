"""Fit a pairwise whole-gesture ranker with duplicate-safe song holdouts.

Absolute reward regression wastes capacity comparing unrelated cells.  This
research model compares two complete pitch-class states from the *same* onset
and learns which is closer to the authored gesture.  At inference, a Borda
tournament selects the state with the strongest pairwise support.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
from .fit_pitch_class_state_ranker import (
    canonical_song_id,
    finite,
    normalize_sample_weights,
    prepare_song,
)


def candidate_state_index(cell: dict[str, Any]) -> int:
    for index, state in enumerate(cell["states"]):
        if state == cell["candidate"]:
            return index
    raise ValueError("Frozen candidate is missing from its state lattice")


def selected_pairs(cell: dict[str, Any], maximum_pairs: int) -> list[tuple[int, int]]:
    rewards = cell["rewards"]
    count = len(rewards)
    if count < 2:
        return []
    order = sorted(range(count), key=lambda index: (-float(rewards[index]), index))
    best = order[0]
    candidate = candidate_state_index(cell)
    pairs: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if left == right or abs(float(rewards[left]) - float(rewards[right])) < 0.025:
            return
        pairs.add((min(left, right), max(left, right)))

    for other in order[1:]:
        add(best, other)
    for other in order:
        add(candidate, other)
    for left in order[:5]:
        for right in order[-5:]:
            add(left, right)
    ranked = sorted(
        pairs,
        key=lambda pair: (
            -abs(float(rewards[pair[0]]) - float(rewards[pair[1]])),
            pair,
        ),
    )
    return ranked[: max(1, maximum_pairs)]


def pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left - right, (left + right) / 2.0]).astype(np.float32)


def training_arrays(
    songs: list[dict[str, Any]], *, maximum_pairs_per_cell: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    group_totals: dict[str, int] = {}
    for song in songs:
        group_totals[song["groupId"]] = group_totals.get(song["groupId"], 0) + 1
    for song in songs:
        song_weight = max(1e-9, song["weight"]) / max(
            1, group_totals[song["groupId"]]
        )
        cell_weight = song_weight / max(1, len(song["cells"]))
        for cell in song["cells"]:
            pairs = selected_pairs(cell, maximum_pairs_per_cell)
            if not pairs:
                continue
            example_weight = cell_weight / (2.0 * len(pairs))
            for left_index, right_index in pairs:
                left = cell["features"][left_index]
                right = cell["features"][right_index]
                left_better = float(
                    cell["rewards"][left_index] > cell["rewards"][right_index]
                )
                matrices.append(pair_features(left, right))
                labels.append(left_better)
                weights.append(example_weight)
                matrices.append(pair_features(right, left))
                labels.append(1.0 - left_better)
                weights.append(example_weight)
    return (
        np.asarray(matrices, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
    )


def fit_model(
    songs: list[dict[str, Any]], *, maximum_pairs_per_cell: int, maximum_iterations: int
) -> HistGradientBoostingClassifier:
    matrix, labels, weights = training_arrays(
        songs, maximum_pairs_per_cell=maximum_pairs_per_cell
    )
    weights = normalize_sample_weights(weights)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=max(1, int(maximum_iterations)),
        max_leaf_nodes=15,
        min_samples_leaf=220,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=6701,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def tournament_scores(
    cell: dict[str, Any], model: HistGradientBoostingClassifier
) -> np.ndarray:
    count = len(cell["states"])
    if count == 1:
        return np.ones(1, dtype=float)
    pairs: list[tuple[int, int]] = []
    features: list[np.ndarray] = []
    for left in range(count):
        for right in range(left + 1, count):
            pairs.append((left, right))
            features.append(pair_features(cell["features"][left], cell["features"][right]))
    probabilities = model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]
    scores = np.zeros(count, dtype=float)
    for (left, right), probability in zip(pairs, probabilities):
        scores[left] += float(probability)
        scores[right] += 1.0 - float(probability)
    return scores / max(1, count - 1)


def evaluate_song(
    song: dict[str, Any],
    model: HistGradientBoostingClassifier,
    *,
    candidate_bonus: float,
) -> dict[str, Any]:
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    baseline_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    exact = baseline_exact = changes = improvements = regressions = 0
    reward = baseline_reward = 0.0
    for cell in song["cells"]:
        scores = tournament_scores(cell, model)
        scores[candidate_state_index(cell)] += max(0.0, candidate_bonus)
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


def weighted(rows: list[dict[str, Any]], weights: dict[str, float], field: str) -> float:
    total = sum(max(0.0, weights.get(row["id"], 1.0)) for row in rows)
    return sum(
        max(0.0, weights.get(row["id"], 1.0)) * float(row["ranker"][field])
        for row in rows
    ) / max(1e-12, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, nargs="+")
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--maximum-pairs-per-cell", type=int, default=24)
    parser.add_argument("--maximum-iterations", type=int, default=140)
    parser.add_argument("--maximum-training-pairs", type=int, default=400_000)
    parser.add_argument("--maximum-fits", type=int, default=10)
    parser.add_argument("--extra-audit-weight", type=float, default=0.5)
    parser.add_argument(
        "--candidate-bonus-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.025, 0.05, 0.1, 0.2],
    )
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.audit]
    raw_songs: list[dict[str, Any]] = []
    for audit_index, path in enumerate(paths):
        audit = json.loads(path.read_text(encoding="utf-8-sig"))
        multiplier = 1.0 if audit_index == 0 else max(0.0, args.extra_audit_weight)
        for song in audit.get("songs") or []:
            copied = dict(song)
            copied["weight"] = max(0.0, finite(song.get("weight"), 1.0)) * multiplier
            copied["groupId"] = canonical_song_id(song.get("id"))
            raw_songs.append(copied)
    maximum_states = max(4, min(256, int(args.maximum_states)))
    songs = [prepare_song(song, maximum_states=maximum_states) for song in raw_songs]
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation songs: {sorted(missing)}")
    song_weights = {song["id"]: song["weight"] for song in evaluation}

    required_fits = len(evaluation) + 1
    if required_fits > max(1, int(args.maximum_fits)):
        raise ValueError(
            f"Requested {required_fits} pairwise fits exceeds --maximum-fits "
            f"{args.maximum_fits}"
        )
    maximum_pairs_per_cell = max(4, args.maximum_pairs_per_cell)
    estimated_pairs = 2 * maximum_pairs_per_cell * sum(
        len(song["cells"]) for song in songs
    )
    if estimated_pairs > max(1, int(args.maximum_training_pairs)):
        raise ValueError(
            f"Estimated {estimated_pairs} directed training pairs exceeds "
            f"--maximum-training-pairs {args.maximum_training_pairs}"
        )

    fold_models: dict[str, HistGradientBoostingClassifier] = {}
    started = time.perf_counter()
    for fit_number, held_out in enumerate(evaluation, start=1):
        print(
            json.dumps(
                {
                    "event": "pairwise_fit_started",
                    "fit": fit_number,
                    "totalFits": required_fits,
                    "heldOut": held_out["id"],
                }
            ),
            flush=True,
        )
        training = [song for song in songs if song["groupId"] != held_out["groupId"]]
        fold_models[held_out["id"]] = fit_model(
            training,
            maximum_pairs_per_cell=maximum_pairs_per_cell,
            maximum_iterations=max(1, args.maximum_iterations),
        )
    trials: list[dict[str, Any]] = []
    for bonus in args.candidate_bonus_grid:
        folds = [
            evaluate_song(
                song,
                fold_models[song["id"]],
                candidate_bonus=max(0.0, float(bonus)),
            )
            for song in evaluation
        ]
        trials.append(
            {
                "candidateBonus": float(bonus),
                "weightedPitchClassF1": round(weighted(folds, song_weights, "f1"), 6),
                "worstSongPitchClassF1": round(
                    min(float(row["ranker"]["f1"]) for row in folds), 6
                ),
                "weightedExactSetRate": round(
                    weighted(folds, song_weights, "exactSetRate"), 6
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
    print(
        json.dumps(
            {
                "event": "pairwise_final_fit_started",
                "fit": required_fits,
                "totalFits": required_fits,
            }
        ),
        flush=True,
    )
    final = fit_model(
        songs,
        maximum_pairs_per_cell=maximum_pairs_per_cell,
        maximum_iterations=max(1, args.maximum_iterations),
    )
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema": "polymath-pitch-class-state-pairwise-ranker-v1",
            "maximumStates": maximum_states,
            "maximumPairsPerCell": maximum_pairs_per_cell,
            "maximumIterations": max(1, args.maximum_iterations),
            "candidateBonus": best["candidateBonus"],
            "model": final,
        },
        temporary_model,
    )
    temporary_model.replace(model_path)
    report = {
        "schema": "polymath-pitch-class-state-pairwise-training-v1",
        "status": "opened-development-only",
        "audits": [str(path) for path in paths],
        "evaluationSongIds": args.evaluation_song_ids,
        "foldRule": "exclude all historical candidates sharing the held-out canonical song id",
        "resourceBounds": {
            "maximumPairsPerCell": maximum_pairs_per_cell,
            "maximumIterations": max(1, args.maximum_iterations),
            "requiredFits": required_fits,
            "maximumFits": int(args.maximum_fits),
            "estimatedDirectedPairs": estimated_pairs,
            "maximumTrainingPairs": int(args.maximum_training_pairs),
            "elapsedSeconds": round(time.perf_counter() - started, 3),
        },
        "best": best,
        "trials": trials,
        "model": str(model_path),
        "decision": {
            "promotable": False,
            "reason": "Opened development and no end-to-end exact-note validation yet.",
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
                "candidateBonus": best["candidateBonus"],
                "weightedPitchClassF1": best["weightedPitchClassF1"],
                "worstSongPitchClassF1": best["worstSongPitchClassF1"],
                "weightedExactSetRate": best["weightedExactSetRate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
