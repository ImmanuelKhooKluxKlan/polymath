"""Fit a transposition-invariant Pianella chord-membership decoder.

The production arranger already finds useful onsets, but it often chooses the
wrong subset of the much denser full-mix transcription.  This research model
scores each of the twelve pitch classes at every arranger gesture.  Its feature
layout is expressed relative to the pitch class being scored, so the model
cannot memorize a song key or an absolute chord name.

Whole songs are held out when fitting coefficients.  Auxiliary songs stay in
the training side of every primary-song fold.  The operating thresholds are
selected on out-of-fold development predictions and therefore remain
development evidence; a different sealed song is still required for promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .fit_pianist_texture_decoder import fit_logistic, sigmoid, standardize


CONTEXT_NAMES = (
    "candidate",
    "source",
    "previous_candidate",
    "next_candidate",
    "previous_source",
    "next_source",
)
SCALAR_FEATURE_NAMES = (
    "bias",
    "candidate_chord_size",
    "source_pitch_class_count",
    "previous_gap_log",
    "next_gap_log",
    "candidate_duration",
    "candidate_velocity",
    "source_note_count",
    "source_attack_note_count",
    "source_voice_share",
    "source_guitar_share",
    "source_bass_share",
    "candidate_left_share",
    "candidate_melody_share",
    "candidate_bass_share",
    "role_has_melody",
    "role_has_harmony",
    "role_has_bass",
    "local_onset_density",
    "candidate_and_source_support",
    "source_context_persistence",
)
FEATURE_NAMES = tuple(
    f"{context}_interval_{interval:+d}"
    for context in CONTEXT_NAMES
    for interval in range(12)
) + SCALAR_FEATURE_NAMES


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def pitch_bits(values: Iterable[Any]) -> np.ndarray:
    selected = {
        int(value) % 12
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    return np.asarray([float(index in selected) for index in range(12)], dtype=float)


def target_pitch_classes(row: dict[str, Any]) -> set[int]:
    return {
        int(value) % 12
        for gesture in row.get("referencePitchClasses") or []
        for value in gesture
    }


def scalar_features(
    row: dict[str, Any],
    *,
    pitch_class: int,
    contexts: list[np.ndarray],
) -> list[float]:
    role = str(row.get("candidateRoleSignature") or "")
    current_candidate, current_source, _, _, previous_source, next_source = contexts
    return [
        1.0,
        min(1.0, max(0.0, finite(row.get("candidateChordSize"))) / 6.0),
        min(1.0, max(0.0, finite(row.get("sourcePitchClassCount"))) / 12.0),
        math.log1p(max(0.03, finite(row.get("previousGapSeconds"), 0.3))),
        math.log1p(max(0.03, finite(row.get("nextGapSeconds"), 0.3))),
        min(1.0, max(0.0, finite(row.get("candidateDuration"), 0.3)) / 2.0),
        min(1.0, max(0.0, finite(row.get("candidateVelocity"), 0.5))),
        min(1.0, max(0.0, finite(row.get("sourceNoteCount"))) / 20.0),
        min(1.0, max(0.0, finite(row.get("sourceAttackNoteCount"))) / 12.0),
        min(1.0, max(0.0, finite(row.get("sourceVoiceShare")))),
        min(1.0, max(0.0, finite(row.get("sourceGuitarShare")))),
        min(1.0, max(0.0, finite(row.get("sourceBassShare")))),
        min(1.0, max(0.0, finite(row.get("candidateLeftShare")))),
        min(1.0, max(0.0, finite(row.get("candidateMelodyShare")))),
        min(1.0, max(0.0, finite(row.get("candidateBassShare")))),
        float("melody" in role),
        float("harmony" in role),
        float("bass" in role),
        min(
            1.0,
            max(0.0, finite(row.get("localCandidateOnsetsPerSecond"))) / 8.0,
        ),
        float(current_candidate[pitch_class] * current_source[pitch_class]),
        float(
            current_source[pitch_class]
            * (previous_source[pitch_class] + next_source[pitch_class])
            / 2.0
        ),
    ]


def feature_vector(
    row: dict[str, Any],
    *,
    pitch_class: int,
    contexts: list[np.ndarray],
) -> list[float]:
    relative = [
        float(value)
        for context in contexts
        for value in np.roll(context, -pitch_class)
    ]
    return relative + scalar_features(
        row, pitch_class=pitch_class, contexts=contexts
    )


def examples_from_song(song: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(song.get("cells") or [])
    candidate = [pitch_bits(row.get("candidatePitchClasses") or []) for row in rows]
    source = [pitch_bits(row.get("sourcePitchClasses") or []) for row in rows]
    zero = np.zeros(12, dtype=float)
    examples: list[dict[str, Any]] = []
    song_id = str(song.get("id") or "")
    for index, row in enumerate(rows):
        target = target_pitch_classes(row)
        contexts = [
            candidate[index],
            source[index],
            candidate[index - 1] if index else zero,
            candidate[index + 1] if index + 1 < len(rows) else zero,
            source[index - 1] if index else zero,
            source[index + 1] if index + 1 < len(rows) else zero,
        ]
        for pitch_class in range(12):
            examples.append(
                {
                    "songId": song_id,
                    "cellIndex": index,
                    "pitchClass": pitch_class,
                    "features": feature_vector(
                        row, pitch_class=pitch_class, contexts=contexts
                    ),
                    "label": float(pitch_class in target),
                    "candidate": float(candidate[index][pitch_class]),
                    "source": float(source[index][pitch_class]),
                    "anchored": bool(target),
                }
            )
    return examples


def fit_model(
    examples: list[dict[str, Any]],
    *,
    ridge: float,
    positive_weight: float,
    iterations: int = 50,
) -> dict[str, Any]:
    matrix = np.asarray([row["features"] for row in examples], dtype=float)
    labels = np.asarray([row["label"] for row in examples], dtype=float)
    standardized, means, scales = standardize(matrix)
    weights = fit_logistic(
        standardized,
        labels,
        ridge=ridge,
        positive_weight=positive_weight,
        iterations=iterations,
    )
    return {
        "featureNames": list(FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "ridge": ridge,
        "positiveWeight": positive_weight,
    }


def predict(examples: list[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    if list(model.get("featureNames") or []) != list(FEATURE_NAMES):
        raise ValueError("Pitch-class decoder feature contract does not match")
    matrix = np.asarray([row["features"] for row in examples], dtype=float)
    means = np.asarray(model.get("means") or [], dtype=float)
    scales = np.asarray(model.get("scales") or [], dtype=float)
    weights = np.asarray(model.get("weights") or [], dtype=float)
    if not (
        matrix.shape[1] == len(means) == len(scales) == len(weights)
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(np.isfinite(weights))
        and np.all(scales > 1e-9)
    ):
        raise ValueError("Pitch-class decoder contains invalid coefficient arrays")
    return sigmoid(((matrix - means) / scales) @ weights)


def decoded_set_metrics(
    examples: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    add_threshold: float,
    remove_threshold: float,
    source_supported_additions_only: bool = True,
) -> dict[str, Any]:
    cells: dict[int, list[tuple[dict[str, Any], float]]] = {}
    for row, probability in zip(examples, probabilities):
        cells.setdefault(int(row["cellIndex"]), []).append(
            (row, float(probability))
        )
    true_positive = false_positive = false_negative = exact = additions = removals = 0
    anchored_exact = anchored_cells = 0
    for values in cells.values():
        target = {
            int(row["pitchClass"])
            for row, _probability in values
            if float(row["label"]) > 0.5
        }
        decoded = {
            int(row["pitchClass"])
            for row, _probability in values
            if float(row["candidate"]) > 0.5
        }
        for row, probability in values:
            pitch_class = int(row["pitchClass"])
            if (
                float(row["candidate"]) <= 0.5
                and probability >= add_threshold
                and (
                    not source_supported_additions_only
                    or float(row["source"]) > 0.5
                )
            ):
                decoded.add(pitch_class)
                additions += 1
            if (
                float(row["candidate"]) > 0.5
                and probability <= remove_threshold
            ):
                decoded.discard(pitch_class)
                removals += 1
        true_positive += len(decoded & target)
        false_positive += len(decoded - target)
        false_negative += len(target - decoded)
        exact += int(decoded == target)
        if target:
            anchored_cells += 1
            anchored_exact += int(decoded == target)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "cells": len(cells),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exactSetRate": round(exact / max(1, len(cells)), 6),
        "anchoredExactSetRate": round(
            anchored_exact / max(1, anchored_cells), 6
        ),
        "additions": additions,
        "removals": removals,
    }


def metric_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 6)
        for key in (
            "precision",
            "recall",
            "f1",
            "exactSetRate",
            "anchoredExactSetRate",
        )
    }


def parameter_grid() -> Iterable[tuple[float, float]]:
    for ridge in (1.0, 3.0, 10.0):
        for positive_weight in (1.0, 1.5, 2.0):
            yield ridge, positive_weight


def operating_points() -> Iterable[tuple[float, float]]:
    for add_threshold in np.arange(0.40, 0.751, 0.025):
        for remove_threshold in np.arange(0.025, 0.251, 0.025):
            yield round(float(add_threshold), 3), round(float(remove_threshold), 3)


def model_hash(model: dict[str, Any]) -> str:
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train(
    audit: dict[str, Any],
    primary_song_ids: set[str],
    *,
    output_folds_root: Path,
    profile_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    song_examples = {
        str(song.get("id")): examples_from_song(song)
        for song in audit.get("songs") or []
    }
    missing = primary_song_ids - set(song_examples)
    if missing:
        raise ValueError(f"Primary songs missing from audit: {sorted(missing)}")
    if len(primary_song_ids) < 2:
        raise ValueError("At least two primary songs are required")
    all_ids = set(song_examples)
    cached: dict[tuple[float, float], list[dict[str, Any]]] = {}
    models: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    for ridge, positive_weight in parameter_grid():
        folds: list[dict[str, Any]] = []
        fold_models: dict[str, dict[str, Any]] = {}
        for held_out in sorted(primary_song_ids):
            training_ids = sorted(all_ids - {held_out})
            training = [
                example
                for song_id in training_ids
                for example in song_examples[song_id]
            ]
            validation = song_examples[held_out]
            model = fit_model(
                training,
                ridge=ridge,
                positive_weight=positive_weight,
            )
            fold_models[held_out] = model
            folds.append(
                {
                    "heldOutSong": held_out,
                    "trainingSongs": training_ids,
                    "examples": validation,
                    "probabilities": predict(validation, model),
                }
            )
        cached[(ridge, positive_weight)] = folds
        models[(ridge, positive_weight)] = fold_models

    trials: list[dict[str, Any]] = []
    for (ridge, positive_weight), folds in cached.items():
        for add_threshold, remove_threshold in operating_points():
            evaluated: list[dict[str, Any]] = []
            for fold in folds:
                baseline = decoded_set_metrics(
                    fold["examples"],
                    fold["probabilities"],
                    add_threshold=2.0,
                    remove_threshold=-1.0,
                )
                candidate = decoded_set_metrics(
                    fold["examples"],
                    fold["probabilities"],
                    add_threshold=add_threshold,
                    remove_threshold=remove_threshold,
                )
                evaluated.append(
                    {
                        "heldOutSong": fold["heldOutSong"],
                        "trainingSongs": fold["trainingSongs"],
                        "baseline": baseline,
                        "candidate": candidate,
                        "deltas": metric_deltas(baseline, candidate),
                    }
                )
            average_f1_delta = sum(
                float(row["deltas"]["f1"]) for row in evaluated
            ) / len(evaluated)
            worst_f1_delta = min(
                float(row["deltas"]["f1"]) for row in evaluated
            )
            average_exact_delta = sum(
                float(row["deltas"]["exactSetRate"]) for row in evaluated
            ) / len(evaluated)
            worst_exact_delta = min(
                float(row["deltas"]["exactSetRate"]) for row in evaluated
            )
            all_song_f1_non_regression = worst_f1_delta >= -1e-9
            exact_set_safety = worst_exact_delta >= -0.005
            selection_score = (
                average_f1_delta
                + 0.75 * worst_f1_delta
                + 0.20 * average_exact_delta
                + 0.10 * worst_exact_delta
                - (0.08 if not all_song_f1_non_regression else 0.0)
                - (0.03 if not exact_set_safety else 0.0)
            )
            trials.append(
                {
                    "ridge": ridge,
                    "positiveWeight": positive_weight,
                    "addThreshold": add_threshold,
                    "removeThreshold": remove_threshold,
                    "selectionScore": round(selection_score, 8),
                    "averageF1Delta": round(average_f1_delta, 6),
                    "worstSongF1Delta": round(worst_f1_delta, 6),
                    "averageExactSetRateDelta": round(average_exact_delta, 6),
                    "worstSongExactSetRateDelta": round(worst_exact_delta, 6),
                    "gates": {
                        "allPrimarySongsF1NonRegression": all_song_f1_non_regression,
                        "worstSongExactSetRateWithinHalfPoint": exact_set_safety,
                    },
                    "folds": evaluated,
                }
            )
    trials.sort(
        key=lambda row: (
            not all(bool(value) for value in row["gates"].values()),
            -float(row["selectionScore"]),
            -float(row["worstSongF1Delta"]),
            -float(row["averageF1Delta"]),
            float(row["ridge"]),
        )
    )
    best = trials[0]
    key = (float(best["ridge"]), float(best["positiveWeight"]))
    output_folds_root.mkdir(parents=True, exist_ok=True)
    for held_out, fold_model in models[key].items():
        payload = {
            "schema": "polymath-pianist-pitch-class-decoder-v1",
            "id": f"{profile_id}-without-{held_out}",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "trainingSongIds": sorted(all_ids - {held_out}),
            "excludedSongIds": [held_out],
            **fold_model,
            "addThreshold": float(best["addThreshold"]),
            "removeThreshold": float(best["removeThreshold"]),
            "sourceSupportedAdditionsOnly": True,
            "inferenceBoundary": {
                "usesSongIdentity": False,
                "usesAbsoluteSongTime": False,
                "usesReferenceAtInference": False,
                "usesOnlyCandidateAndSourcePitchStructure": True,
                "status": "research-only"
            },
        }
        payload["modelSha256"] = model_hash(payload)
        path = output_folds_root / held_out / "model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    all_examples = [
        example for song_id in sorted(all_ids) for example in song_examples[song_id]
    ]
    final_model = fit_model(
        all_examples,
        ridge=float(best["ridge"]),
        positive_weight=float(best["positiveWeight"]),
    )
    profile = {
        "schema": "polymath-pianist-pitch-class-decoder-v1",
        "id": profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "trainingSongIds": sorted(all_ids),
        "primaryDevelopmentSongIds": sorted(primary_song_ids),
        **final_model,
        "addThreshold": float(best["addThreshold"]),
        "removeThreshold": float(best["removeThreshold"]),
        "sourceSupportedAdditionsOnly": True,
        "inferenceBoundary": {
            "usesSongIdentity": False,
            "usesAbsoluteSongTime": False,
            "usesReferenceAtInference": False,
            "usesOnlyCandidateAndSourcePitchStructure": True,
            "status": "research-only"
        },
    }
    profile["modelSha256"] = model_hash(profile)
    report = {
        "schema": "polymath-pianist-pitch-class-decoder-training-v1",
        "evidenceBoundary": (
            "Whole-song coefficient holdouts with a development-selected global "
            "operating point. All songs are opened development evidence; this "
            "does not replace an untouched final holdout or listening test."
        ),
        "dataset": {
            "songs": {
                song_id: len(examples) // 12
                for song_id, examples in song_examples.items()
            },
            "primarySongs": sorted(primary_song_ids),
            "auxiliarySongs": sorted(all_ids - primary_song_ids),
        },
        "best": best,
        "topTrials": trials[:30],
        "decision": (
            "RESEARCH_CANDIDATE"
            if all(bool(value) for value in best["gates"].values())
            and float(best["averageF1Delta"]) > 0
            else "REJECT"
        ),
    }
    return profile, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-folds-root", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--primary-song", action="append", default=[])
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    primary = {str(value) for value in args.primary_song if str(value)}
    if not primary:
        raise ValueError("Pass every primary development song with --primary-song")
    profile, report = train(
        audit,
        primary,
        output_folds_root=Path(args.output_folds_root).resolve(),
        profile_id=str(args.profile_id),
    )
    profile_path = Path(args.output_profile).resolve()
    report_path = Path(args.output_report).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    report["audit"] = str(audit_path)
    report["outputProfile"] = str(profile_path)
    report["outputFoldsRoot"] = str(Path(args.output_folds_root).resolve())
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": str(profile_path),
                "report": str(report_path),
                "decision": report["decision"],
                "best": {
                    key: report["best"][key]
                    for key in (
                        "ridge",
                        "positiveWeight",
                        "addThreshold",
                        "removeThreshold",
                        "averageF1Delta",
                        "worstSongF1Delta",
                        "averageExactSetRateDelta",
                        "worstSongExactSetRateDelta",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
