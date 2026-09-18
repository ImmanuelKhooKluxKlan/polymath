"""Fit a conservative, song-independent broken-chord texture gate.

The model predicts whether one current arranger onset should be unfolded into
multiple pianist gestures.  It uses candidate/source structure only.  Song id,
absolute time, reference pitches, and reference timing are intentionally absent
from the feature contract.  Validation holds out one complete song at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES = (
    "bias",
    "candidate_chord_size",
    "candidate_chord_size_squared",
    "candidate_pitch_class_count",
    "candidate_duration_log",
    "candidate_velocity",
    "candidate_source_velocity",
    "candidate_mean_midi_centered",
    "candidate_pitch_span",
    "candidate_left_share",
    "candidate_melody_share",
    "candidate_bass_share",
    "previous_gap_log",
    "next_gap_log",
    "next_gap_squared",
    "previous_pitch_class_similarity",
    "next_pitch_class_similarity",
    "previous_chord_size",
    "next_chord_size",
    "previous_chord_size_delta",
    "next_chord_size_delta",
    "local_onsets_per_second",
    "source_note_count_log",
    "source_attack_note_count_log",
    "source_pitch_class_count",
    "source_attack_pitch_class_count",
    "source_duration_log",
    "source_attack_duration_log",
    "source_voice_share",
    "source_guitar_share",
    "source_bass_share",
    "interior_source_onset_count",
    "interior_source_note_count_log",
    "interior_source_pitch_class_count",
    "interior_source_first_offset_ratio",
    "interior_source_last_offset_ratio",
    "interior_source_velocity",
    "interior_source_duration_log",
    "interior_source_voice_share",
    "interior_source_guitar_share",
    "interior_source_bass_share",
    "interior_source_pitch_class_similarity",
    "role_has_harmony",
    "role_has_melody",
    "role_has_bass",
    "gap_x_chord_size",
    "gap_x_source_duration",
    "gap_x_voice_share",
    "gap_x_guitar_share",
)


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def feature_map(row: dict[str, Any]) -> dict[str, float]:
    chord_size = max(1.0, finite(row.get("candidateChordSize"), 1.0))
    next_gap = min(2.0, max(0.03, finite(row.get("nextGapSeconds"), 0.3)))
    previous_gap = min(2.0, max(0.03, finite(row.get("previousGapSeconds"), next_gap)))
    source_duration = min(4.0, max(0.0, finite(row.get("sourceMedianDuration"))))
    source_attack_duration = min(
        4.0, max(0.0, finite(row.get("sourceAttackMedianDuration")))
    )
    role = str(row.get("candidateRoleSignature") or "")
    source_voice = min(1.0, max(0.0, finite(row.get("sourceVoiceShare"))))
    source_guitar = min(1.0, max(0.0, finite(row.get("sourceGuitarShare"))))
    source_bass = min(1.0, max(0.0, finite(row.get("sourceBassShare"))))
    previous_size = max(1.0, finite(row.get("previousCandidateChordSize"), chord_size))
    next_size = max(1.0, finite(row.get("nextCandidateChordSize"), chord_size))
    interior_onsets = min(8.0, max(0.0, finite(row.get("interiorSourceOnsetCount"))))
    interior_notes = max(0.0, finite(row.get("interiorSourceNoteCount")))
    interior_duration = min(4.0, max(0.0, finite(row.get("interiorSourceDuration"))))
    return {
        "bias": 1.0,
        "candidate_chord_size": chord_size / 6.0,
        "candidate_chord_size_squared": (chord_size / 6.0) ** 2,
        "candidate_pitch_class_count": finite(row.get("candidatePitchClassCount"), chord_size) / 6.0,
        "candidate_duration_log": math.log1p(max(0.03, finite(row.get("candidateDuration"), 0.3))),
        "candidate_velocity": min(1.0, max(0.01, finite(row.get("candidateVelocity"), 0.72))),
        "candidate_source_velocity": min(1.0, max(0.01, finite(row.get("candidateSourceVelocity"), 0.72))),
        "candidate_mean_midi_centered": min(1.5, max(-1.5, finite(row.get("candidateMeanMidiCentered")))),
        "candidate_pitch_span": min(2.0, max(0.0, finite(row.get("candidatePitchSpan"))) / 24.0),
        "candidate_left_share": min(1.0, max(0.0, finite(row.get("candidateLeftShare")))),
        "candidate_melody_share": min(1.0, max(0.0, finite(row.get("candidateMelodyShare")))),
        "candidate_bass_share": min(1.0, max(0.0, finite(row.get("candidateBassShare")))),
        "previous_gap_log": math.log1p(previous_gap),
        "next_gap_log": math.log1p(next_gap),
        "next_gap_squared": next_gap * next_gap,
        "previous_pitch_class_similarity": min(1.0, max(0.0, finite(row.get("previousPitchClassSimilarity")))),
        "next_pitch_class_similarity": min(1.0, max(0.0, finite(row.get("nextPitchClassSimilarity")))),
        "previous_chord_size": previous_size / 6.0,
        "next_chord_size": next_size / 6.0,
        "previous_chord_size_delta": (chord_size - previous_size) / 6.0,
        "next_chord_size_delta": (next_size - chord_size) / 6.0,
        "local_onsets_per_second": min(2.0, max(0.0, finite(row.get("localCandidateOnsetsPerSecond"))) / 10.0),
        "source_note_count_log": math.log1p(max(0.0, finite(row.get("sourceNoteCount")))) / math.log(25.0),
        "source_attack_note_count_log": math.log1p(max(0.0, finite(row.get("sourceAttackNoteCount")))) / math.log(17.0),
        "source_pitch_class_count": min(2.0, max(0.0, finite(row.get("sourcePitchClassCount"))) / 6.0),
        "source_attack_pitch_class_count": min(2.0, max(0.0, finite(row.get("sourceAttackPitchClassCount"))) / 6.0),
        "source_duration_log": math.log1p(source_duration),
        "source_attack_duration_log": math.log1p(source_attack_duration),
        "source_voice_share": source_voice,
        "source_guitar_share": source_guitar,
        "source_bass_share": source_bass,
        "interior_source_onset_count": interior_onsets / 4.0,
        "interior_source_note_count_log": math.log1p(interior_notes) / math.log(25.0),
        "interior_source_pitch_class_count": min(2.0, max(0.0, finite(row.get("interiorSourcePitchClassCount"))) / 6.0),
        "interior_source_first_offset_ratio": min(2.0, max(0.0, finite(row.get("interiorSourceFirstOffsetSeconds"))) / next_gap),
        "interior_source_last_offset_ratio": min(2.0, max(0.0, finite(row.get("interiorSourceLastOffsetSeconds"))) / next_gap),
        "interior_source_velocity": min(1.0, max(0.0, finite(row.get("interiorSourceVelocity")))),
        "interior_source_duration_log": math.log1p(interior_duration),
        "interior_source_voice_share": min(1.0, max(0.0, finite(row.get("interiorSourceVoiceShare")))),
        "interior_source_guitar_share": min(1.0, max(0.0, finite(row.get("interiorSourceGuitarShare")))),
        "interior_source_bass_share": min(1.0, max(0.0, finite(row.get("interiorSourceBassShare")))),
        "interior_source_pitch_class_similarity": min(1.0, max(0.0, finite(row.get("interiorSourcePitchClassSimilarity")))),
        "role_has_harmony": float("harmony" in role),
        "role_has_melody": float("melody" in role),
        "role_has_bass": float("bass" in role),
        "gap_x_chord_size": next_gap * chord_size / 6.0,
        "gap_x_source_duration": next_gap * source_duration,
        "gap_x_voice_share": next_gap * source_voice,
        "gap_x_guitar_share": next_gap * source_guitar,
    }


def feature_vector(row: dict[str, Any]) -> list[float]:
    values = feature_map(row)
    return [values[name] for name in FEATURE_NAMES]


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    means[0] = 0.0
    scales[0] = 1.0
    scales[scales < 1e-8] = 1.0
    return (matrix - means) / scales, means, scales


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
    positive_weight: float,
    iterations: int = 80,
) -> np.ndarray:
    weights = np.zeros(matrix.shape[1], dtype=float)
    sample_weights = np.where(labels > 0.5, positive_weight, 1.0)
    penalty = np.eye(matrix.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    for _iteration in range(iterations):
        probabilities = sigmoid(matrix @ weights)
        gradient = matrix.T @ (sample_weights * (probabilities - labels)) + penalty @ weights
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = matrix.T @ (matrix * curvature[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels > 0.5))
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    positive_rank_sum = float(np.sum(ranks[labels > 0.5]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    positive = labels > 0.5
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    beta_squared = 0.25
    f05 = (1.0 + beta_squared) * precision * recall / max(
        1e-12, beta_squared * precision + recall
    )
    return {
        "examples": len(labels),
        "positives": int(np.sum(positive)),
        "prevalence": round(float(np.mean(positive)), 6),
        "predictedPositive": int(np.sum(predicted)),
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "trueNegative": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "f0_5": round(f05, 6),
        "accuracy": round((tp + tn) / max(1, len(labels)), 6),
        "rocAuc": round(float(auc(labels, probabilities)), 6)
        if auc(labels, probabilities) is not None
        else None,
        "brier": round(float(np.mean((probabilities - labels) ** 2)), 6),
    }


def load_examples(audit: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for song in audit.get("songs") or []:
        song_id = str(song.get("id") or "")
        for row in song.get("cells") or []:
            examples.append(
                {
                    "songId": song_id,
                    "label": float(int(row.get("targetGestureCount", 0)) >= 2),
                    "features": feature_vector(row),
                }
            )
    return examples


def parameter_grid() -> Iterable[tuple[float, float]]:
    for ridge in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        for positive_weight in (1.0, 1.5, 2.0, 3.0):
            yield ridge, positive_weight


def cross_validate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    songs = sorted({str(item["songId"]) for item in examples})
    if len(songs) < 2:
        raise ValueError("Texture fitting requires at least two complete songs")
    trials: list[dict[str, Any]] = []
    thresholds = np.arange(0.30, 0.851, 0.025)
    for ridge, positive_weight in parameter_grid():
        heldout_predictions: list[dict[str, Any]] = []
        for heldout in songs:
            training = [item for item in examples if item["songId"] != heldout]
            validation = [item for item in examples if item["songId"] == heldout]
            train_matrix = np.asarray([item["features"] for item in training], dtype=float)
            train_labels = np.asarray([item["label"] for item in training], dtype=float)
            standardized, means, scales = standardize(train_matrix)
            weights = fit_logistic(
                standardized,
                train_labels,
                ridge=ridge,
                positive_weight=positive_weight,
            )
            validation_matrix = np.asarray([item["features"] for item in validation], dtype=float)
            probabilities = sigmoid(((validation_matrix - means) / scales) @ weights)
            heldout_predictions.append(
                {
                    "songId": heldout,
                    "labels": np.asarray([item["label"] for item in validation], dtype=float),
                    "probabilities": probabilities,
                }
            )
        all_labels = np.concatenate([item["labels"] for item in heldout_predictions])
        all_probabilities = np.concatenate([item["probabilities"] for item in heldout_predictions])
        for threshold in thresholds:
            aggregate = classification_metrics(all_labels, all_probabilities, float(threshold))
            folds = [
                {
                    "heldOutSong": item["songId"],
                    **classification_metrics(item["labels"], item["probabilities"], float(threshold)),
                }
                for item in heldout_predictions
            ]
            active = [fold for fold in folds if int(fold["predictedPositive"]) > 0]
            worst_precision = min((float(fold["precision"]) for fold in active), default=0.0)
            minimum_predictions = min((int(fold["predictedPositive"]) for fold in folds), default=0)
            # False expansions are more audible than missed optional ornaments,
            # so F0.5 and worst-song precision deliberately dominate recall.
            score = (
                float(aggregate["f0_5"])
                + 0.20 * worst_precision
                + 0.05 * float(aggregate["recall"])
                - 0.05 * max(0.0, 0.45 - float(aggregate["precision"]))
                - 0.02 * float(minimum_predictions == 0)
            )
            trials.append(
                {
                    "ridge": ridge,
                    "positiveWeight": positive_weight,
                    "threshold": round(float(threshold), 3),
                    "selectionScore": round(score, 6),
                    "worstSongPrecision": round(worst_precision, 6),
                    "minimumSongPredictions": minimum_predictions,
                    "aggregate": aggregate,
                    "folds": folds,
                }
            )
    trials.sort(
        key=lambda item: (
            -float(item["selectionScore"]),
            -float(item["aggregate"]["precision"]),
            -float(item["aggregate"]["recall"]),
            float(item["ridge"]),
        )
    )
    return {"songs": songs, "best": trials[0], "topTrials": trials[:30]}


def sha256_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--exclude-song",
        action="append",
        default=[],
        help="Exclude a complete song from fitting and model selection.",
    )
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    examples = load_examples(audit)
    excluded_songs = {str(value) for value in args.exclude_song if str(value)}
    examples = [
        example
        for example in examples
        if str(example["songId"]) not in excluded_songs
    ]
    if not examples:
        raise ValueError("No texture examples remain after excluding songs")
    cv = cross_validate(examples)
    best = cv["best"]
    matrix = np.asarray([item["features"] for item in examples], dtype=float)
    labels = np.asarray([item["label"] for item in examples], dtype=float)
    standardized, means, scales = standardize(matrix)
    weights = fit_logistic(
        standardized,
        labels,
        ridge=float(best["ridge"]),
        positive_weight=float(best["positiveWeight"]),
    )
    model = {
        "schema": "polymath-pianist-texture-gate-v1",
        "id": "pianella-broken-chord-gate-v154",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "trainingSongIds": sorted({str(item["songId"]) for item in examples}),
        "excludedSongIds": sorted(excluded_songs),
        "featureNames": list(FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "threshold": float(best["threshold"]),
        "ridge": float(best["ridge"]),
        "positiveWeight": float(best["positiveWeight"]),
        "inferenceBoundary": {
            "usesSongIdentity": False,
            "usesAbsoluteSongTime": False,
            "usesReferenceAtInference": False,
            "changesPitchWithoutSourceSupport": False,
            "status": "research-only",
        },
    }
    model["modelSha256"] = sha256_payload(model)
    output_model = Path(args.output_model).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    importance = sorted(
        (
            {"feature": name, "weight": round(float(weight), 6), "absoluteWeight": round(abs(float(weight)), 6)}
            for name, weight in zip(FEATURE_NAMES, weights)
        ),
        key=lambda item: -float(item["absoluteWeight"]),
    )
    report = {
        "schema": "polymath-pianist-texture-training-report-v1",
        "evidenceBoundary": (
            "Complete-song leave-one-out validation; Kiss Me reference is hard-cut before 02:30. "
            "The gate predicts expansion only and does not yet prove generated pitches or timing."
        ),
        "audit": str(audit_path),
        "excludedSongs": sorted(excluded_songs),
        "examples": len(examples),
        "positiveExamples": int(np.sum(labels)),
        "crossValidation": cv,
        "finalModel": str(output_model),
        "modelSha256": model["modelSha256"],
        "featureImportance": importance,
        "decision": "RESEARCH_ONLY",
    }
    output_report = Path(args.output_report).resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(output_model),
                "report": str(output_report),
                "examples": len(examples),
                "best": best,
                "topFeatures": importance[:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
