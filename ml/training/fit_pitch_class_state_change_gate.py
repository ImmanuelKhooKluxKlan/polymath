"""Cross-fit a conservative gate around the whole-state piano selector.

The state ranker improves recall but sometimes accepts a larger chord whose
extra pitch classes reduce precision.  This module does not invent another
state lattice.  It learns one bounded decision instead: when the ranker wants
to replace the unchanged arranger state, is that replacement likely to help?

Every gate-training example is produced by an emission model that did not see
that example's composition.  The outer evaluation also holds out complete
canonical song groups, so historical exports of the same song cannot leak
across either layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .fit_pitch_class_state_ranker import (
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        fit_ridge,
        normalize_sample_weights,
        predict,
        prepare_song,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        FEATURE_NAMES,
        canonical_song_id,
        finite,
        fit_ridge,
        normalize_sample_weights,
        predict,
        prepare_song,
    )


SCALAR_FEATURE_NAMES = (
    "emission_margin_over_candidate",
    "emission_margin_over_runner_up",
    "selected_emission",
    "candidate_emission",
    "selected_size",
    "candidate_size",
    "size_delta",
    "state_overlap_f1",
    "added_pitch_classes",
    "removed_pitch_classes",
)
GATE_FEATURE_NAMES = (
    *SCALAR_FEATURE_NAMES,
    *(f"candidate__{name}" for name in FEATURE_NAMES),
    *(f"selected_minus_candidate__{name}" for name in FEATURE_NAMES),
)


def candidate_state_index(cell: dict[str, Any]) -> int:
    for index, state in enumerate(cell["states"]):
        if state == cell["candidate"]:
            return index
    raise ValueError("Frozen candidate is missing from its state lattice")


def proposed_change(
    cell: dict[str, Any], emission_model: dict[str, Any]
) -> tuple[int, int, np.ndarray]:
    scores = predict(cell["features"], emission_model)
    return int(np.argmax(scores)), candidate_state_index(cell), scores


def gate_feature_vector(
    cell: dict[str, Any],
    selected_index: int,
    candidate_index: int,
    scores: np.ndarray,
) -> np.ndarray:
    """Create target-blind features describing one proposed replacement."""

    selected = cell["states"][selected_index]
    candidate = cell["states"][candidate_index]
    runner_up = max(
        (float(score) for index, score in enumerate(scores) if index != selected_index),
        default=float(scores[selected_index]),
    )
    selected_features = np.asarray(cell["features"][selected_index], dtype=float)
    candidate_features = np.asarray(cell["features"][candidate_index], dtype=float)
    scalar = np.asarray(
        [
            float(scores[selected_index] - scores[candidate_index]),
            float(scores[selected_index] - runner_up),
            float(scores[selected_index]),
            float(scores[candidate_index]),
            len(selected) / 6.0,
            len(candidate) / 6.0,
            (len(selected) - len(candidate)) / 6.0,
            set_f1(selected, candidate),
            len(selected - candidate) / 6.0,
            len(candidate - selected) / 6.0,
        ],
        dtype=float,
    )
    vector = np.concatenate(
        [scalar, candidate_features, selected_features - candidate_features]
    )
    if len(vector) != len(GATE_FEATURE_NAMES):
        raise AssertionError(
            f"Gate feature contract drifted: {len(vector)} != {len(GATE_FEATURE_NAMES)}"
        )
    return vector


def cross_fitted_gate_examples(
    songs: list[dict[str, Any]],
    *,
    ridge: float,
    best_state_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build gate labels only from composition-held-out ranker decisions."""

    groups = sorted({song["groupId"] for song in songs})
    if len(groups) < 3:
        raise ValueError("Cross-fitted change gating requires at least three song groups")
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1

    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    example_groups: list[str] = []
    by_group: dict[str, dict[str, int]] = {}
    for held_group in groups:
        fit_songs = [song for song in songs if song["groupId"] != held_group]
        validation_songs = [song for song in songs if song["groupId"] == held_group]
        emission_model = fit_ridge(
            fit_songs, ridge=ridge, best_state_weight=best_state_weight
        )
        group_stats = {"examples": 0, "improvements": 0, "regressions": 0}
        for song in validation_songs:
            base_weight = (
                max(1e-9, song["weight"])
                / max(1, group_counts[song["groupId"]])
                / max(1, len(song["cells"]))
            )
            for cell in song["cells"]:
                selected_index, candidate_index, scores = proposed_change(
                    cell, emission_model
                )
                if selected_index == candidate_index:
                    continue
                selected_reward = set_f1(
                    cell["target"], cell["states"][selected_index]
                )
                candidate_reward = set_f1(cell["target"], cell["candidate"])
                delta = selected_reward - candidate_reward
                if abs(delta) <= 1e-9:
                    continue
                improved = int(delta > 0.0)
                features.append(
                    gate_feature_vector(
                        cell, selected_index, candidate_index, scores
                    )
                )
                labels.append(improved)
                # Larger musical mistakes matter more, but retain a floor so
                # small recurring regressions are not silently ignored.
                weights.append(base_weight * max(0.05, abs(delta)))
                example_groups.append(held_group)
                group_stats["examples"] += 1
                group_stats["improvements" if improved else "regressions"] += 1
        by_group[held_group] = group_stats
    if not features or len(set(labels)) != 2:
        raise ValueError("Gate training needs both helpful and harmful proposed changes")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
        np.asarray(example_groups, dtype=object),
        {"heldoutGroups": len(groups), "byGroup": by_group},
    )


def fit_gate(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization_c: float,
) -> dict[str, Any]:
    total_weight = max(1e-12, float(np.sum(weights)))
    means = np.sum(matrix * weights[:, None], axis=0) / total_weight
    variances = np.sum(((matrix - means) ** 2) * weights[:, None], axis=0) / total_weight
    scales = np.sqrt(np.maximum(variances, 1e-8))
    standardized = (matrix - means) / scales
    classifier = LogisticRegression(
        C=max(1e-4, float(regularization_c)),
        max_iter=2000,
        solver="lbfgs",
        random_state=6701,
    )
    classifier.fit(
        standardized,
        labels,
        sample_weight=normalize_sample_weights(weights),
    )
    return {
        "featureNames": list(GATE_FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "classifier": classifier,
        "regularizationC": float(regularization_c),
    }


def gate_probability(vector: np.ndarray, gate: dict[str, Any]) -> float:
    standardized = (vector - gate["means"]) / gate["scales"]
    return float(gate["classifier"].predict_proba(standardized[None, :])[0, 1])


def select_acceptance_threshold_cross_validated(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    *,
    regularization_c: float,
    thresholds: list[float],
) -> dict[str, Any]:
    """Calibrate the gate on composition-held-out gate predictions."""

    probabilities = np.zeros(len(labels), dtype=float)
    unique_groups = sorted(set(str(value) for value in groups))
    for held_group in unique_groups:
        validation = np.asarray([str(value) == held_group for value in groups])
        training = ~validation
        if len(set(labels[training].tolist())) != 2:
            raise ValueError(
                f"Gate calibration fold {held_group!r} lacks both decision classes"
            )
        gate = fit_gate(
            matrix[training],
            labels[training],
            weights[training],
            regularization_c=regularization_c,
        )
        standardized = (matrix[validation] - gate["means"]) / gate["scales"]
        probabilities[validation] = gate["classifier"].predict_proba(standardized)[:, 1]

    total_weight = max(1e-12, float(np.sum(weights)))
    trials: list[dict[str, Any]] = []
    for threshold in sorted(set(min(1.01, max(0.0, float(x))) for x in thresholds)):
        accepted = probabilities >= threshold
        correct = accepted == labels.astype(bool)
        accepted_weight = float(np.sum(weights[accepted]))
        accepted_positive_weight = float(np.sum(weights[accepted & (labels > 0)]))
        trials.append(
            {
                "threshold": threshold,
                "weightedDecisionAccuracy": round(
                    float(np.sum(weights[correct])) / total_weight, 6
                ),
                "acceptedExamples": int(np.sum(accepted)),
                "acceptedWeightShare": round(accepted_weight / total_weight, 6),
                "acceptedPositiveWeightShare": round(
                    accepted_positive_weight / max(1e-12, accepted_weight), 6
                ),
            }
        )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedDecisionAccuracy"]),
            float(row["acceptedPositiveWeightShare"]),
            float(row["threshold"]),
        ),
    )
    return {
        "method": "nested-composition-held-out-decision-calibration",
        "heldoutGroups": len(unique_groups),
        "selected": best,
        "trials": trials,
    }


def _add_counts(
    bucket: dict[str, int], target: frozenset[int], state: frozenset[int]
) -> None:
    bucket["truePositive"] += len(target & state)
    bucket["falsePositive"] += len(state - target)
    bucket["falseNegative"] += len(target - state)


def evaluate_song(
    song: dict[str, Any],
    emission_model: dict[str, Any],
    gate: dict[str, Any],
    *,
    acceptance_threshold: float = 0.5,
) -> dict[str, Any]:
    baseline = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    ungated = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    gated = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    stats = {
        "proposedChanges": 0,
        "acceptedChanges": 0,
        "rejectedChanges": 0,
        "acceptedImprovements": 0,
        "acceptedRegressions": 0,
        "rejectedImprovements": 0,
        "rejectedRegressions": 0,
    }
    for cell in song["cells"]:
        selected_index, candidate_index, scores = proposed_change(cell, emission_model)
        selected = cell["states"][selected_index]
        candidate = cell["candidate"]
        target = cell["target"]
        accepted = selected_index == candidate_index
        if selected_index != candidate_index:
            stats["proposedChanges"] += 1
            probability = gate_probability(
                gate_feature_vector(cell, selected_index, candidate_index, scores), gate
            )
            accepted = probability >= acceptance_threshold
            stats["acceptedChanges" if accepted else "rejectedChanges"] += 1
            delta = set_f1(target, selected) - set_f1(target, candidate)
            if delta > 1e-9:
                stats[
                    "acceptedImprovements" if accepted else "rejectedImprovements"
                ] += 1
            elif delta < -1e-9:
                stats[
                    "acceptedRegressions" if accepted else "rejectedRegressions"
                ] += 1
        chosen = selected if accepted else candidate
        _add_counts(baseline, target, candidate)
        _add_counts(ungated, target, selected)
        _add_counts(gated, target, chosen)
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": len(song["cells"]),
        "baseline": micro_metrics(baseline),
        "ungated": micro_metrics(ungated),
        "gated": micro_metrics(gated),
        **stats,
    }


def weighted_metric(
    rows: list[dict[str, Any]], weights: dict[str, float], branch: str
) -> float:
    total = sum(max(0.0, weights.get(row["id"], 1.0)) for row in rows)
    return sum(
        max(0.0, weights.get(row["id"], 1.0)) * float(row[branch]["f1"])
        for row in rows
    ) / max(1e-12, total)


def model_digest(payload: dict[str, Any]) -> str:
    stable = {
        "schema": payload["schema"],
        "featureNames": payload["featureNames"],
        "ridge": payload["ridge"],
        "bestStateWeight": payload["bestStateWeight"],
        "acceptanceThreshold": payload["acceptanceThreshold"],
        "regularizationC": payload["regularizationC"],
        "maximumStates": payload["maximumStates"],
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, nargs="+")
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--best-state-weight", type=float, default=1.0)
    parser.add_argument("--regularization-c", type=float, default=0.2)
    parser.add_argument(
        "--acceptance-threshold-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01],
    )
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

    folds: list[dict[str, Any]] = []
    for held_out in evaluation:
        training = [song for song in songs if song["groupId"] != held_out["groupId"]]
        emission = fit_ridge(
            training,
            ridge=max(0.0, args.ridge),
            best_state_weight=max(0.0, args.best_state_weight),
        )
        matrix, labels, weights, example_groups, crossfit = cross_fitted_gate_examples(
            training,
            ridge=max(0.0, args.ridge),
            best_state_weight=max(0.0, args.best_state_weight),
        )
        gate = fit_gate(
            matrix,
            labels,
            weights,
            regularization_c=max(1e-4, args.regularization_c),
        )
        threshold_selection = select_acceptance_threshold_cross_validated(
            matrix,
            labels,
            weights,
            example_groups,
            regularization_c=max(1e-4, args.regularization_c),
            thresholds=args.acceptance_threshold_grid,
        )
        selected_threshold = float(threshold_selection["selected"]["threshold"])
        row = evaluate_song(
            held_out,
            emission,
            gate,
            acceptance_threshold=selected_threshold,
        )
        row["gateTraining"] = {
            "examples": int(len(labels)),
            "positiveShare": round(float(np.mean(labels)), 6),
            **crossfit,
        }
        row["thresholdSelection"] = threshold_selection
        folds.append(row)

    song_weights = {song["id"]: song["weight"] for song in evaluation}
    final_emission = fit_ridge(
        songs,
        ridge=max(0.0, args.ridge),
        best_state_weight=max(0.0, args.best_state_weight),
    )
    matrix, labels, weights, example_groups, final_crossfit = cross_fitted_gate_examples(
        songs,
        ridge=max(0.0, args.ridge),
        best_state_weight=max(0.0, args.best_state_weight),
    )
    final_gate = fit_gate(
        matrix,
        labels,
        weights,
        regularization_c=max(1e-4, args.regularization_c),
    )
    final_threshold_selection = select_acceptance_threshold_cross_validated(
        matrix,
        labels,
        weights,
        example_groups,
        regularization_c=max(1e-4, args.regularization_c),
        thresholds=args.acceptance_threshold_grid,
    )
    final_threshold = float(final_threshold_selection["selected"]["threshold"])
    model_payload = {
        "schema": "polymath-pitch-class-state-change-gate-v1",
        "featureNames": list(GATE_FEATURE_NAMES),
        "maximumStates": maximum_states,
        "ridge": float(args.ridge),
        "bestStateWeight": float(args.best_state_weight),
        "acceptanceThreshold": final_threshold,
        "regularizationC": float(args.regularization_c),
        "emissionModel": final_emission,
        "gateModel": final_gate,
        "evidenceBoundary": (
            "Gate inputs contain target-blind proposed/candidate state evidence. "
            "Every gate label was generated by a composition-held-out emission model."
        ),
    }
    model_payload["sha256"] = model_digest(model_payload)
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(model_payload, temporary_model)
    temporary_model.replace(model_path)

    report = {
        "schema": "polymath-pitch-class-state-change-gate-training-v1",
        "status": "opened-development-only",
        "audits": [str(path) for path in paths],
        "evaluationSongIds": [song["id"] for song in evaluation],
        "settings": {
            "maximumStates": maximum_states,
            "ridge": float(args.ridge),
            "bestStateWeight": float(args.best_state_weight),
            "regularizationC": float(args.regularization_c),
            "acceptanceThresholdGrid": [
                float(value) for value in args.acceptance_threshold_grid
            ],
        },
        "aggregate": {
            "baselinePitchClassF1": round(
                weighted_metric(folds, song_weights, "baseline"), 6
            ),
            "ungatedPitchClassF1": round(
                weighted_metric(folds, song_weights, "ungated"), 6
            ),
            "gatedPitchClassF1": round(
                weighted_metric(folds, song_weights, "gated"), 6
            ),
            "worstSongGatedPitchClassF1": round(
                min(float(row["gated"]["f1"]) for row in folds), 6
            ),
        },
        "folds": folds,
        "finalGateTraining": {
            "examples": int(len(labels)),
            "positiveShare": round(float(np.mean(labels)), 6),
            **final_crossfit,
        },
        "finalThresholdSelection": final_threshold_selection,
        "model": {"path": str(model_path), "sha256": model_payload["sha256"]},
        "decision": {
            "promotable": False,
            "reason": (
                "Opened development songs only; exact-note, audio-render, and sealed-song "
                "gates remain mandatory."
            ),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_output.replace(output)
    print(json.dumps({"output": str(output), **report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
