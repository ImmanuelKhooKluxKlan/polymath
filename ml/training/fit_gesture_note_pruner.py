"""Fit a conservative within-gesture chord-tone pruner.

This trainer never changes onset timing and never invents a note.  It learns
from whole-song-separated aligned references which pitches inside a selected
simultaneous gesture are likely authored chord tones.  The output is a small
JSON logistic model that can be evaluated without NumPy at inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_gesture_structure import (  # noqa: E402
    GESTURE_NOTE_FEATURE_NAMES,
    gesture_note_feature_rows,
    selected_note_indices,
)

from ml.training.analyze_piano_velocity_style import (  # noqa: E402
    load_notes,
    onset_groups,
    trusted_source_ranges,
)
from ml.training.fit_paired_gesture_velocity import (  # noqa: E402
    candidate_gesture_groups,
    match_gesture_groups,
)
from ml.training.analyze_pianist_sequence_alignment import align_sequences  # noqa: E402
from ml.training.train_piano_arranger_adapter import train_logistic_model  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def build_song_examples(
    row: dict[str, Any],
    onset_window: float,
    tolerance: float,
    matching_mode: str = "greedy",
    sequence_gap_cost: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every manifest song needs an id")
    reference_path = Path(str(row["reference"])).resolve()
    candidate_path = Path(str(row["candidate"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    end_seconds = (
        finite(row.get("candidateEndSeconds"), math.nan)
        if row.get("candidateEndSeconds") is not None
        else None
    )
    if end_seconds is not None and not math.isfinite(end_seconds):
        end_seconds = None
    ranges = trusted_source_ranges(alignment_path)
    reference_notes = load_notes(reference_path, training_only=True)
    candidate_notes = load_notes(
        candidate_path, include_ranges=ranges, end_seconds=end_seconds
    )
    reference_groups = onset_groups(reference_notes, onset_window)
    candidate_groups = candidate_gesture_groups(candidate_notes, onset_window)
    feature_rows = gesture_note_feature_rows(candidate_groups)
    feature_by_group = {
        id(group): rows for group, rows in zip(candidate_groups, feature_rows)
    }
    if matching_mode == "sequence":
        matched_indices, _missing, _extra, _score = align_sequences(
            reference_groups,
            candidate_groups,
            maximum_time_distance=max(0.25, tolerance),
            gap_cost=sequence_gap_cost,
        )
        matches = [
            (
                reference_groups[reference_index],
                candidate_groups[candidate_index],
                len(
                    {int(note["midi"]) % 12 for note in reference_groups[reference_index]}
                    & {int(note["midi"]) % 12 for note in candidate_groups[candidate_index]}
                ),
            )
            for reference_index, candidate_index in matched_indices
            if {
                int(note["midi"]) % 12
                for note in reference_groups[reference_index]
            }
            & {
                int(note["midi"]) % 12
                for note in candidate_groups[candidate_index]
            }
        ]
    else:
        matches = match_gesture_groups(
            reference_groups,
            candidate_groups,
            tolerance,
            minimum_pitch_class_overlap=1,
        )
    examples: list[dict[str, Any]] = []
    positives = 0
    negatives = 0
    for reference_group, candidate_group, _overlap in matches:
        reference_pcs = {int(note["midi"]) % 12 for note in reference_group}
        labels = [
            int(note["midi"]) % 12 in reference_pcs for note in candidate_group
        ]
        positives += sum(labels)
        negatives += len(labels) - sum(labels)
        examples.append(
            {
                "songId": song_id,
                "sourceTime": round(float(candidate_group[0]["time"]), 6),
                "referenceTime": round(float(reference_group[0]["time"]), 6),
                "candidateGroup": candidate_group,
                "referencePitchClasses": sorted(reference_pcs),
                "features": [
                    [row[name] for name in GESTURE_NOTE_FEATURE_NAMES]
                    for row in feature_by_group[id(candidate_group)]
                ],
                "labels": labels,
            }
        )
    return examples, {
        "id": song_id,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "alignment": str(alignment_path),
        "candidateEndSeconds": end_seconds,
        "trustedSourceRanges": ranges,
        "referenceGestures": len(reference_groups),
        "candidateGestures": len(candidate_groups),
        "matchedGestures": len(examples),
        "positiveCandidateNotes": positives,
        "negativeCandidateNotes": negatives,
        "negativeShare": round(negatives / max(1, positives + negatives), 6),
        "matchingMode": matching_mode,
        "sequenceGapCost": sequence_gap_cost if matching_mode == "sequence" else None,
    }


def flatten_training(
    groups: list[dict[str, Any]], positive_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        counts[str(group["songId"])] += len(group["labels"])
    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    song_count = max(1, len(counts))
    for group in groups:
        song = str(group["songId"])
        for row, label in zip(group["features"], group["labels"]):
            features.append(row)
            labels.append(float(label))
            weights.append(
                (positive_weight if label else 1.0)
                / (song_count * counts[song])
            )
    weight_array = np.asarray(weights, dtype=float)
    weight_array *= len(weights) / max(1e-12, float(weight_array.sum()))
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=float),
        weight_array,
    )


def fit_model(
    groups: list[dict[str, Any]],
    *,
    positive_weight: float,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    features, labels, sample_weights = flatten_training(groups, positive_weight)
    if not len(features) or labels.min() == labels.max():
        raise ValueError("Gesture-note training requires both kept and extra notes")
    weights, means, scales, history = train_logistic_model(
        features,
        labels,
        sample_weights,
        np.ones(len(labels), dtype=bool),
        seed=seed,
        iterations=iterations,
        learning_rate=0.014,
        l2=0.025,
    )
    return {
        "type": "standardized-logistic-gesture-note-pruner-v1",
        "featureNames": list(GESTURE_NOTE_FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "positiveTrainingWeight": round(float(positive_weight), 6),
    }, history


def probabilities_for_group(
    group: dict[str, Any], model: dict[str, Any]
) -> list[float]:
    weights = np.asarray(model["weights"], dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    matrix = np.asarray(group["features"], dtype=float)
    logits = ((matrix - means) / np.maximum(1e-9, np.abs(scales))) @ weights
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).tolist()


def metrics(
    groups: list[dict[str, Any]],
    model: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    exact = removals = correct_removals = wrong_removals = 0
    chord_size_error = 0.0
    changed_groups = 0
    for example in groups:
        candidate = example["candidateGroup"]
        if model is None:
            selected = set(range(len(candidate)))
        else:
            probabilities = probabilities_for_group(example, model)
            selected = selected_note_indices(candidate, probabilities, policy or {})
        if len(selected) < len(candidate):
            changed_groups += 1
        labels = example["labels"]
        for index in range(len(candidate)):
            if index in selected:
                if labels[index]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                removals += 1
                if labels[index]:
                    wrong_removals += 1
                else:
                    correct_removals += 1
        selected_pcs = {int(candidate[index]["midi"]) % 12 for index in selected}
        reference_pcs = set(example["referencePitchClasses"])
        false_negative += len(reference_pcs - selected_pcs)
        exact += int(selected_pcs == reference_pcs)
        chord_size_error += abs(len(selected_pcs) - len(reference_pcs))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "gestures": len(groups),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exactPitchClassSets": exact,
        "exactPitchClassSetRate": round(exact / max(1, len(groups)), 6),
        "meanAbsoluteChordSizeError": round(chord_size_error / max(1, len(groups)), 6),
        "changedGroups": changed_groups,
        "removedNotes": removals,
        "correctRemovals": correct_removals,
        "wrongRemovals": wrong_removals,
    }


def evaluate_policy_grid(
    groups: list[dict[str, Any]],
    *,
    positive_weight_grid: list[float],
    threshold_grid: list[float],
    gap_grid: list[float],
    maximum_removals_grid: list[int],
    minimum_group_size_grid: list[int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    songs = sorted({str(group["songId"]) for group in groups})
    if len(songs) < 3:
        raise ValueError("At least three songs are required for whole-song validation")
    baseline_by_song = {
        song: metrics([group for group in groups if group["songId"] == song])
        for song in songs
    }
    trials: list[dict[str, Any]] = []
    for positive_weight in positive_weight_grid:
        fold_models = {
            song: fit_model(
                [group for group in groups if group["songId"] != song],
                positive_weight=positive_weight,
                iterations=iterations,
                seed=seed,
            )[0]
            for song in songs
        }
        for threshold in threshold_grid:
            for gap in gap_grid:
                for maximum_removals in maximum_removals_grid:
                    for minimum_group_size in minimum_group_size_grid:
                        policy = {
                            "threshold": threshold,
                            "minimumProbabilityGap": gap,
                            "maximumRemovalsPerGesture": maximum_removals,
                            "minimumGroupSize": minimum_group_size,
                            "minimumRemainingNotes": 1,
                            "preserveMelody": True,
                        }
                        folds = []
                        combined_tp_weighted_f1 = 0.0
                        total_gestures = 0
                        for song in songs:
                            validation = [
                                group for group in groups if group["songId"] == song
                            ]
                            candidate = metrics(validation, fold_models[song], policy)
                            baseline = baseline_by_song[song]
                            folds.append(
                            {
                                "heldOutSong": song,
                                "baseline": baseline,
                                "candidate": candidate,
                                "f1Delta": round(candidate["f1"] - baseline["f1"], 6),
                                "recallDelta": round(candidate["recall"] - baseline["recall"], 6),
                                "exactSetRateDelta": round(
                                    candidate["exactPitchClassSetRate"]
                                    - baseline["exactPitchClassSetRate"],
                                    6,
                                ),
                            }
                            )
                            combined_tp_weighted_f1 += candidate["f1"] * len(validation)
                            total_gestures += len(validation)
                        aggregate_f1 = combined_tp_weighted_f1 / max(1, total_gestures)
                        worst_f1 = min(fold["f1Delta"] for fold in folds)
                        worst_recall = min(fold["recallDelta"] for fold in folds)
                        worst_exact = min(fold["exactSetRateDelta"] for fold in folds)
                        changed_groups = sum(
                            int(fold["candidate"]["changedGroups"]) for fold in folds
                        )
                        removed_notes = sum(
                            int(fold["candidate"]["removedNotes"]) for fold in folds
                        )
                        score = (
                            aggregate_f1
                            + min(0.0, worst_f1) * 5.0
                            + min(0.0, worst_recall) * 8.0
                            + min(0.0, worst_exact) * 2.0
                        )
                        trials.append(
                        {
                            "positiveTrainingWeight": positive_weight,
                            **policy,
                            "weightedCandidateF1": round(aggregate_f1, 6),
                            "changedGroups": changed_groups,
                            "removedNotes": removed_notes,
                            "worstSongF1Delta": round(worst_f1, 6),
                            "worstSongRecallDelta": round(worst_recall, 6),
                            "worstSongExactSetRateDelta": round(worst_exact, 6),
                            "selectionScore": round(score, 6),
                            "folds": folds,
                        }
                        )
    trials.sort(
        key=lambda row: (
            -row["selectionScore"],
            -row["weightedCandidateF1"],
            row["maximumRemovalsPerGesture"],
            row["threshold"],
        )
    )
    changed_trials = [row for row in trials if int(row["changedGroups"]) > 0]
    top_by_f1 = sorted(
        changed_trials,
        key=lambda row: (
            -row["weightedCandidateF1"],
            -row["worstSongF1Delta"],
            -row["worstSongRecallDelta"],
            row["removedNotes"],
        ),
    )
    return {
        "songs": songs,
        "best": trials[0],
        "topTrials": trials[:30],
        "topChangedByF1": top_by_f1[:30],
    }


def profile_hash(profile: dict[str, Any]) -> str:
    payload = copy.deepcopy(profile)
    payload.pop("profileSha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_float_grid(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected finite non-negative values")
    return values


def parse_int_grid(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected non-negative integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--matching-mode", choices=("greedy", "sequence"), default="greedy")
    parser.add_argument("--sequence-gap-cost", type=float, default=0.85)
    parser.add_argument("--positive-weight-grid", type=parse_float_grid, default=parse_float_grid("1.5,2,3,4"))
    parser.add_argument("--threshold-grid", type=parse_float_grid, default=parse_float_grid("0.1,0.15,0.2,0.25,0.3,0.4,0.5"))
    parser.add_argument("--probability-gap-grid", type=parse_float_grid, default=parse_float_grid("0.05,0.1,0.15,0.2"))
    parser.add_argument("--maximum-removals-grid", type=parse_int_grid, default=parse_int_grid("1,2"))
    parser.add_argument("--minimum-group-size-grid", type=parse_int_grid, default=parse_int_grid("2,3,4,5"))
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0x47535452)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    groups: list[dict[str, Any]] = []
    dataset = []
    for row in manifest.get("songs") or []:
        examples, description = build_song_examples(
            row,
            min(0.08, max(0.005, args.onset_window_seconds)),
            (
                min(0.75, max(0.05, args.tolerance_seconds))
                if args.matching_mode == "sequence"
                else min(0.25, max(0.02, args.tolerance_seconds))
            ),
            args.matching_mode,
            max(0.05, args.sequence_gap_cost),
        )
        if len(examples) < 20:
            raise ValueError(f"{description['id']} has too few paired gestures")
        groups.extend(examples)
        dataset.append(description)

    validation = evaluate_policy_grid(
        groups,
        positive_weight_grid=args.positive_weight_grid,
        threshold_grid=[min(1.0, value) for value in args.threshold_grid],
        gap_grid=[min(1.0, value) for value in args.probability_gap_grid],
        maximum_removals_grid=args.maximum_removals_grid,
        minimum_group_size_grid=args.minimum_group_size_grid,
        iterations=max(100, args.iterations),
        seed=args.seed,
    )
    best = validation["best"]
    model, history = fit_model(
        groups,
        positive_weight=best["positiveTrainingWeight"],
        iterations=max(100, args.iterations),
        seed=args.seed,
    )
    profile = {
        "schema": "polymath-gesture-note-pruner-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
        "onsetWindowSeconds": round(args.onset_window_seconds, 6),
        "selectionModel": model,
        "threshold": best["threshold"],
        "minimumProbabilityGap": best["minimumProbabilityGap"],
        "maximumRemovalsPerGesture": best["maximumRemovalsPerGesture"],
        "minimumGroupSize": best["minimumGroupSize"],
        "minimumRemainingNotes": best["minimumRemainingNotes"],
        "preserveMelody": best["preserveMelody"],
        "training": {
            "manifest": str(manifest_path),
            "method": "whole-song leave-one-out candidate-only chord-tone classification",
            "commercialUseAllowed": False,
            "decision": "EXPERIMENTAL_ONLY",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "polymath-gesture-note-pruner-report-v1",
        "manifest": str(manifest_path),
        "outputProfile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "dataset": dataset,
        "baseline": metrics(groups),
        "crossValidation": validation,
        "finalInSample": metrics(groups, model, profile),
        "optimizationHistory": history,
        "evidenceBoundary": "Private development references; newly frozen songs are required before promotion.",
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputProfile": str(output_path),
                "dataset": dataset,
                "bestCrossValidation": best,
                "baseline": report["baseline"],
                "finalInSample": report["finalInSample"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
