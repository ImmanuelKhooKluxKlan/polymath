"""Fit a conservative gate for candidate onsets the pianist omits entirely.

The pitch-class state ranker intentionally handles non-empty gestures.  It
therefore cannot learn from texture-audit cells whose authored reference plays
nothing.  This module models that missing decision separately and prioritizes
drop precision: a false drop destroys a real musical event, while a missed
drop merely preserves the existing candidate.

All features are available at inference.  Whole compositions, including known
variants, are excluded together in validation folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, pitch_classes, target_set
    from .fit_pianist_texture_decoder import FEATURE_NAMES as TEXTURE_FEATURE_NAMES
    from .fit_pianist_texture_decoder import feature_vector as texture_feature_vector
    from .fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        normalize_sample_weights,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, pitch_classes, target_set  # type: ignore
    from fit_pianist_texture_decoder import FEATURE_NAMES as TEXTURE_FEATURE_NAMES  # type: ignore
    from fit_pianist_texture_decoder import feature_vector as texture_feature_vector  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        canonical_song_id,
        finite,
        normalize_sample_weights,
    )


DROP_FEATURE_NAMES = (
    *TEXTURE_FEATURE_NAMES,
    "candidate_source_pitch_precision",
    "candidate_source_pitch_recall",
    "candidate_audio_mean_chroma_mean",
    "candidate_audio_attack_chroma_mean",
    "candidate_audio_attack_chroma_maximum",
    "candidate_view_agreement_mean",
    "candidate_attack_agreement_mean",
    "candidate_fullmix_presence",
    "candidate_vocal_presence",
    "candidate_backing_presence",
    "candidate_fullmix_attack_presence",
    "candidate_vocal_attack_presence",
    "candidate_backing_attack_presence",
    "vocal_f0_available",
    "candidate_vocal_f0_probability_maximum",
    "candidate_vocal_f0_probability_sum",
    "candidate_contains_vocal_f0_dominant",
    "vocal_f0_voiced_share",
    "vocal_f0_mean_confidence",
)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def drop_feature_vector(row: dict[str, Any]) -> list[float]:
    candidate = pitch_classes(row.get("candidatePitchClasses") or [])
    source = pitch_classes(row.get("sourcePitchClasses") or [])
    audio = {
        int(item.get("pitchClass", -1)) % 12: item
        for item in row.get("audioPitchClassEvidence") or []
        if isinstance(item, dict) and isinstance(item.get("pitchClass"), (int, float))
    }
    multiview = {
        int(item.get("pitchClass", -1)) % 12: item
        for item in row.get("multiViewPitchClassEvidence") or []
        if isinstance(item, dict) and isinstance(item.get("pitchClass"), (int, float))
    }

    def audio_values(field: str) -> list[float]:
        return [
            min(1.0, max(0.0, finite(audio.get(pc, {}).get(field))))
            for pc in candidate
        ]

    def agreement(field: str) -> list[float]:
        available_views = max(1.0, finite(row.get("multiViewCount"), 0.0))
        return [
            min(
                1.0,
                max(0.0, finite(multiview.get(pc, {}).get(field))) / available_views,
            )
            for pc in candidate
        ]

    def view_presence(name: str, *, attack: bool = False) -> float:
        values: list[float] = []
        field = "attackCount" if attack else "nearCount"
        for pc in candidate:
            view = (multiview.get(pc, {}).get("views") or {}).get(name) or {}
            values.append(
                float(bool(view.get("available")) and finite(view.get(field)) > 0.0)
            )
        return _mean(values)

    f0 = row.get("vocalF0Evidence") or {}
    f0_rows = {
        int(item.get("pitchClass", -1)) % 12: item
        for item in f0.get("pitchClasses") or []
        if isinstance(item, dict) and isinstance(item.get("pitchClass"), (int, float))
    }
    probabilities = [
        min(1.0, max(0.0, finite(f0_rows.get(pc, {}).get("probability"))))
        for pc in candidate
    ]
    dominant = f0.get("dominantPitchClass")
    dominant_pc = int(dominant) % 12 if isinstance(dominant, (int, float)) else None
    attack_chroma = audio_values("attackChroma")
    values = [
        *texture_feature_vector(row),
        len(candidate & source) / max(1, len(candidate)),
        len(candidate & source) / max(1, len(source)),
        _mean(audio_values("meanChroma")),
        _mean(attack_chroma),
        max(attack_chroma, default=0.0),
        _mean(agreement("viewAgreementCount")),
        _mean(agreement("attackAgreementCount")),
        view_presence("fullMix"),
        view_presence("vocals"),
        view_presence("backing"),
        view_presence("fullMix", attack=True),
        view_presence("vocals", attack=True),
        view_presence("backing", attack=True),
        float(bool(f0.get("available"))),
        max(probabilities, default=0.0),
        sum(probabilities),
        float(dominant_pc is not None and dominant_pc in candidate),
        min(1.0, max(0.0, finite(f0.get("voicedShare")))),
        min(1.0, max(0.0, finite(f0.get("meanConfidence")))),
    ]
    if len(values) != len(DROP_FEATURE_NAMES):
        raise AssertionError(
            f"Drop feature contract drifted: {len(values)} != {len(DROP_FEATURE_NAMES)}"
        )
    return values


def prepare_songs(audit: dict[str, Any]) -> list[dict[str, Any]]:
    songs: list[dict[str, Any]] = []
    for raw in audit.get("songs") or []:
        song_id = str(raw.get("id") or "")
        rows = list(raw.get("cells") or [])
        if not song_id or not rows:
            continue
        songs.append(
            {
                "id": song_id,
                "groupId": canonical_song_id(song_id),
                "weight": max(0.0, finite(raw.get("weight"), 1.0)),
                "rows": rows,
                "features": np.asarray(
                    [drop_feature_vector(row) for row in rows], dtype=float
                ),
                "labels": np.asarray(
                    [float(int(row.get("targetGestureCount", 0)) == 0) for row in rows],
                    dtype=np.int8,
                ),
            }
        )
    return songs


def training_arrays(
    songs: list[dict[str, Any]], *, positive_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    for song in songs:
        cell_weight = (
            max(1e-9, song["weight"])
            / max(1, group_counts[song["groupId"]])
            / max(1, len(song["rows"]))
        )
        sample_weights = np.full(len(song["rows"]), cell_weight, dtype=float)
        sample_weights *= np.where(
            song["labels"] > 0,
            max(0.01, float(positive_weight)),
            1.0,
        )
        matrices.append(song["features"])
        labels.append(song["labels"])
        weights.append(sample_weights)
    return np.vstack(matrices), np.concatenate(labels), np.concatenate(weights)


def fit_model(
    songs: list[dict[str, Any]], *, regularization_c: float, positive_weight: float
) -> dict[str, Any]:
    matrix, labels, weights = training_arrays(
        songs, positive_weight=positive_weight
    )
    normalized_weights = normalize_sample_weights(weights)
    total = float(np.sum(normalized_weights))
    means = np.sum(matrix * normalized_weights[:, None], axis=0) / total
    variances = np.sum(
        ((matrix - means) ** 2) * normalized_weights[:, None], axis=0
    ) / total
    scales = np.sqrt(np.maximum(variances, 1e-8))
    # Preserve the explicit bias feature used by the established texture
    # contract instead of centering it to zero.
    means[0] = 0.0
    scales[0] = 1.0
    classifier = LogisticRegression(
        C=max(1e-4, float(regularization_c)),
        fit_intercept=False,
        max_iter=1500,
        solver="lbfgs",
        random_state=6701,
    )
    classifier.fit(
        (matrix - means) / scales,
        labels,
        sample_weight=normalized_weights,
    )
    return {
        "featureNames": list(DROP_FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": np.asarray(classifier.coef_[0], dtype=float),
        "regularizationC": float(regularization_c),
        "positiveWeight": float(positive_weight),
        "optimizerIterations": int(classifier.n_iter_[0]),
    }


def probabilities(song: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    logits = ((song["features"] - model["means"]) / model["scales"]) @ model[
        "coefficients"
    ]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))


def classification_counts(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    positive = labels > 0
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f0_5": round(
            1.25 * precision * recall / max(1e-12, 0.25 * precision + recall), 6
        ),
    }


def evaluate_song(
    song: dict[str, Any], model: dict[str, Any], *, threshold: float
) -> dict[str, Any]:
    scores = probabilities(song, model)
    predicted_drop = scores >= threshold
    baseline = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    gated = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    baseline_exact = gated_exact = 0
    for row, drop in zip(song["rows"], predicted_drop):
        target = target_set(row)
        candidate = pitch_classes(row.get("candidatePitchClasses") or [])
        selected = frozenset() if bool(drop) else candidate
        for bucket, state in ((baseline, candidate), (gated, selected)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        baseline_exact += int(candidate == target)
        gated_exact += int(selected == target)
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": len(song["rows"]),
        "drop": classification_counts(song["labels"], predicted_drop),
        "baseline": {
            **micro_metrics(baseline),
            "exactSetRate": round(baseline_exact / max(1, len(song["rows"])), 6),
        },
        "gated": {
            **micro_metrics(gated),
            "exactSetRate": round(gated_exact / max(1, len(song["rows"])), 6),
        },
        "threshold": float(threshold),
        "predictedDrops": int(np.sum(predicted_drop)),
    }


def weighted(rows: list[dict[str, Any]], section: str, field: str = "f1") -> float:
    total = sum(max(0.0, float(row["weight"])) for row in rows)
    return sum(
        max(0.0, float(row["weight"])) * float(row[section][field])
        for row in rows
    ) / max(1e-12, total)


def serialize(model: dict[str, Any], threshold: float) -> dict[str, Any]:
    payload = {
        "schema": "polymath-pianist-drop-gate-v1",
        "featureNames": model["featureNames"],
        "means": [round(float(value), 10) for value in model["means"]],
        "scales": [round(float(value), 10) for value in model["scales"]],
        "coefficients": [round(float(value), 10) for value in model["coefficients"]],
        "regularizationC": model["regularizationC"],
        "positiveWeight": model["positiveWeight"],
        "threshold": float(threshold),
        "evidenceBoundary": "candidate/source/audio features only; no reference fields at inference",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--evaluation-song-ids", nargs="+")
    parser.add_argument("--transfer-audit")
    parser.add_argument("--regularization-c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--positive-weight-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    songs = prepare_songs(audit)
    wanted = set(args.evaluation_song_ids or [])
    evaluation = [song for song in songs if not wanted or song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")
    trials: list[dict[str, Any]] = []
    for regularization_c in args.regularization_c_grid:
        for positive_weight in args.positive_weight_grid:
            fitted: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for heldout in evaluation:
                training = [
                    song for song in songs if song["groupId"] != heldout["groupId"]
                ]
                model = fit_model(
                    training,
                    regularization_c=regularization_c,
                    positive_weight=positive_weight,
                )
                fitted.append((heldout, model))
            for threshold in sorted(
                {max(0.0, min(1.0, float(value))) for value in args.threshold_grid}
            ):
                rows = [
                    evaluate_song(song, model, threshold=threshold)
                    for song, model in fitted
                ]
                deltas = [
                    float(row["gated"]["f1"]) - float(row["baseline"]["f1"])
                    for row in rows
                ]
                drop_precision = weighted(rows, "drop", "precision")
                trials.append(
                    {
                        "regularizationC": float(regularization_c),
                        "positiveWeight": float(positive_weight),
                        "threshold": threshold,
                        "baselineWeightedPitchClassF1": round(
                            weighted(rows, "baseline"), 6
                        ),
                        "weightedPitchClassF1": round(weighted(rows, "gated"), 6),
                        "worstSongPitchClassF1Delta": round(min(deltas), 6),
                        "dropPrecision": round(drop_precision, 6),
                        "dropRecall": round(weighted(rows, "drop", "recall"), 6),
                        "safe": drop_precision >= 0.80 and min(deltas) >= -0.002,
                        "folds": rows,
                    }
                )
    safe = [trial for trial in trials if trial["safe"]]
    candidates = safe or trials
    best = max(
        candidates,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1Delta"]),
            float(row["dropPrecision"]),
            -float(row["threshold"]),
        ),
    )
    final_model = fit_model(
        songs,
        regularization_c=float(best["regularizationC"]),
        positive_weight=float(best["positiveWeight"]),
    )
    model_payload = serialize(final_model, float(best["threshold"]))
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    transfer = None
    if args.transfer_audit:
        transfer_path = Path(args.transfer_audit).resolve()
        transfer_payload = json.loads(transfer_path.read_text(encoding="utf-8-sig"))
        transfer_rows = [
            evaluate_song(song, final_model, threshold=float(best["threshold"]))
            for song in prepare_songs(transfer_payload)
        ]
        transfer = {
            "audit": str(transfer_path),
            "baselineWeightedPitchClassF1": round(
                weighted(transfer_rows, "baseline"), 6
            ),
            "weightedPitchClassF1": round(weighted(transfer_rows, "gated"), 6),
            "dropPrecision": round(weighted(transfer_rows, "drop", "precision"), 6),
            "dropRecall": round(weighted(transfer_rows, "drop", "recall"), 6),
            "songs": transfer_rows,
        }
    report = {
        "schema": "polymath-pianist-drop-gate-training-v1",
        "status": "opened-development-only",
        "audit": str(audit_path),
        "evaluationSongIds": [song["id"] for song in evaluation],
        "best": best,
        "trials": trials,
        "transfer": transfer,
        "model": {"path": str(model_path), "sha256": model_payload["sha256"]},
        "decision": {
            "promotable": False,
            "reason": "A listening gate and exact-note end-to-end validation remain required.",
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
                "baselineWeightedPitchClassF1": best[
                    "baselineWeightedPitchClassF1"
                ],
                "weightedPitchClassF1": best["weightedPitchClassF1"],
                "worstSongPitchClassF1Delta": best[
                    "worstSongPitchClassF1Delta"
                ],
                "dropPrecision": best["dropPrecision"],
                "dropRecall": best["dropRecall"],
                "threshold": best["threshold"],
                "transfer": (
                    {
                        "baselineWeightedPitchClassF1": transfer[
                            "baselineWeightedPitchClassF1"
                        ],
                        "weightedPitchClassF1": transfer["weightedPitchClassF1"],
                        "dropPrecision": transfer["dropPrecision"],
                        "dropRecall": transfer["dropRecall"],
                    }
                    if transfer
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
