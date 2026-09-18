"""Fit a conservative pianist-touch correction from aligned trusted gestures.

This stage does not copy notes or timestamps from a reference performance.  It
learns how source loudness, chord size, duration, register, hand/role mix, and
the existing arranger velocity relate to one human hammer velocity.  Training
and validation are split by song whenever more than one song is supplied.
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
from statistics import median
from typing import Any

import numpy as np

from ml.training.analyze_piano_velocity_style import (
    load_notes,
    onset_groups,
    trusted_source_ranges,
)


SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_gesture_calibration import (  # noqa: E402
    GESTURE_CALIBRATION_FEATURE_NAMES,
    gesture_calibration_features,
    gesture_sequence_contexts,
)


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


def velocity_band(value: float) -> str:
    if value < 0.45:
        return "quiet-<0.45"
    if value < 0.55:
        return "quiet-0.45-0.54"
    if value < 0.65:
        return "soft-0.55-0.64"
    if value < 0.75:
        return "medium-0.65-0.74"
    if value < 0.85:
        return "strong-0.75-0.84"
    return "accent->=0.85"


def filter_candidate_notes(
    path: Path,
    ranges: list[tuple[float, float]],
    end_seconds: float | None,
) -> list[dict[str, Any]]:
    return load_notes(
        path,
        include_ranges=ranges,
        end_seconds=end_seconds,
    )


def candidate_gesture_groups(
    notes: list[dict[str, Any]], onset_window: float
) -> list[list[dict[str, Any]]]:
    """Prefer the renderer's immutable gesture id, with an onset fallback."""

    if notes and all(note.get("gestureDynamicGroup") is not None for note in notes):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for note in notes:
            grouped[int(note["gestureDynamicGroup"])].append(note)
        return sorted(
            (
                sorted(group, key=lambda note: (note["time"], note["midi"]))
                for group in grouped.values()
            ),
            key=lambda group: (group[0]["time"], group[0]["midi"]),
        )
    return onset_groups(notes, onset_window)


def group_pitch_classes(group: list[dict[str, Any]]) -> set[int]:
    return {int(note["midi"]) % 12 for note in group}


def match_gesture_groups(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    tolerance: float,
    minimum_pitch_class_overlap: int = 1,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], int]]:
    """Greedily match aligned onsets while rejecting unrelated nearby noise."""

    available = set(range(len(candidate)))
    output: list[tuple[list[dict[str, Any]], list[dict[str, Any]], int]] = []
    for expected in reference:
        expected_time = float(expected[0]["time"])
        expected_pitch_classes = group_pitch_classes(expected)
        choices: list[tuple[float, int, int]] = []
        for index in available:
            observed = candidate[index]
            difference = abs(float(observed[0]["time"]) - expected_time)
            if difference > tolerance:
                continue
            overlap = len(expected_pitch_classes & group_pitch_classes(observed))
            if overlap < minimum_pitch_class_overlap:
                continue
            choices.append((difference, -overlap, index))
        if not choices:
            continue
        _difference, negative_overlap, selected = min(choices)
        available.remove(selected)
        output.append((expected, candidate[selected], -negative_overlap))
    return output


def build_song_examples(
    row: dict[str, Any],
    onset_window: float,
    tolerance: float,
    minimum_pitch_class_overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every paired-gesture manifest row requires an id")
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
    candidate_notes = filter_candidate_notes(candidate_path, ranges, end_seconds)
    reference_groups = onset_groups(reference_notes, onset_window)
    candidate_groups = candidate_gesture_groups(candidate_notes, onset_window)
    candidate_base_velocities = [
        median(float(note["velocity"]) for note in group)
        for group in candidate_groups
    ]
    sequence_contexts = gesture_sequence_contexts(
        candidate_groups, candidate_base_velocities
    )
    context_by_group = {
        id(group): context
        for group, context in zip(candidate_groups, sequence_contexts)
    }
    matched = match_gesture_groups(
        reference_groups,
        candidate_groups,
        tolerance,
        minimum_pitch_class_overlap,
    )
    examples: list[dict[str, Any]] = []
    for reference_group, candidate_group, overlap in matched:
        base_velocity = median(float(note["velocity"]) for note in candidate_group)
        target_velocity = median(float(note["velocity"]) for note in reference_group)
        feature_map = gesture_calibration_features(
            candidate_group,
            base_velocity,
            context_by_group[id(candidate_group)],
        )
        examples.append(
            {
                "songId": song_id,
                "sourceTime": round(float(candidate_group[0]["time"]), 6),
                "referenceTime": round(float(reference_group[0]["time"]), 6),
                "baseVelocity": base_velocity,
                "targetVelocity": target_velocity,
                "targetBand": velocity_band(target_velocity),
                "pitchClassOverlap": overlap,
                "features": [
                    feature_map[name] for name in GESTURE_CALIBRATION_FEATURE_NAMES
                ],
            }
        )
    return examples, {
        "id": song_id,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "alignment": str(alignment_path),
        "candidateEndSeconds": end_seconds,
        "trustedSourceRanges": ranges,
        "referenceNotes": len(reference_notes),
        "candidateNotes": len(candidate_notes),
        "referenceGestures": len(reference_groups),
        "candidateGestures": len(candidate_groups),
        "matchedGestures": len(examples),
        "matchedGestureRate": round(
            len(examples) / max(1, len(reference_groups)), 6
        ),
        "sequenceContextFeatures": True,
    }


def song_equal_weights(
    examples: list[dict[str, Any]], band_balance: float = 0.0
) -> np.ndarray:
    """Give every song equal influence, optionally balancing its touch bands.

    ``band_balance=0`` preserves the original song-equal objective.  At ``1``
    every represented target-velocity band receives an equal share *inside*
    its song, so a long run of ordinary mezzo gestures cannot drown out the
    comparatively rare quiet and accent examples.  Intermediate values blend
    the two objectives without duplicating examples.
    """

    band_balance = min(1.0, max(0.0, float(band_balance)))
    counts: dict[str, int] = defaultdict(int)
    band_counts: dict[tuple[str, str], int] = defaultdict(int)
    bands_by_song: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        song = str(example["songId"])
        band = str(example["targetBand"])
        counts[song] += 1
        band_counts[(song, band)] += 1
        bands_by_song[song].add(band)
    song_count = max(1, len(counts))
    values_list: list[float] = []
    for example in examples:
        song = str(example["songId"])
        band = str(example["targetBand"])
        song_equal = 1.0 / (song_count * counts[song])
        band_equal = 1.0 / (
            song_count
            * max(1, len(bands_by_song[song]))
            * band_counts[(song, band)]
        )
        values_list.append(
            (1.0 - band_balance) * song_equal + band_balance * band_equal
        )
    values = np.asarray(values_list, dtype=float)
    return values * len(examples) / max(1e-12, float(values.sum()))


def standardize(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = max(1e-12, float(weights.sum()))
    means = (matrix * weights[:, None]).sum(axis=0) / total
    variance = (
        ((matrix - means) ** 2) * weights[:, None]
    ).sum(axis=0) / total
    scales = np.sqrt(np.maximum(variance, 0.0))
    means[0] = 0.0  # The bias feature stays exactly one.
    scales[0] = 1.0
    scales[scales < 1e-8] = 1.0
    return (matrix - means) / scales, means, scales


def fit_ridge_correction(
    examples: list[dict[str, Any]],
    ridge: float,
    *,
    band_balance: float = 0.0,
    huber_delta: float = 0.12,
    iterations: int = 6,
) -> dict[str, Any]:
    if len(examples) < 12:
        raise ValueError("At least 12 matched gestures are required")
    matrix = np.asarray([example["features"] for example in examples], dtype=float)
    target = np.asarray(
        [example["targetVelocity"] - example["baseVelocity"] for example in examples],
        dtype=float,
    )
    base_weights = song_equal_weights(examples, band_balance)
    standardized, means, scales = standardize(matrix, base_weights)
    robust = np.ones(len(examples), dtype=float)
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    penalty = np.eye(matrix.shape[1], dtype=float) * max(0.0, float(ridge))
    penalty[0, 0] = 0.0
    for _ in range(max(1, iterations)):
        weights = base_weights * robust
        weighted = standardized * np.sqrt(weights[:, None])
        weighted_target = target * np.sqrt(weights)
        system = weighted.T @ weighted + penalty
        right = weighted.T @ weighted_target
        try:
            coefficients = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(system, right, rcond=None)[0]
        residual = target - standardized @ coefficients
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        mask = absolute > huber_delta
        robust[mask] = huber_delta / absolute[mask]
    return {
        "featureNames": list(GESTURE_CALIBRATION_FEATURE_NAMES),
        "weights": coefficients.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "ridge": float(ridge),
        "huberDelta": float(huber_delta),
        "trainingBandBalance": round(
            min(1.0, max(0.0, float(band_balance))), 6
        ),
    }


def model_config(
    fitted: dict[str, Any],
    *,
    profile_id: str,
    blend: float,
    maximum_correction: float,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "id": profile_id,
        "type": "standardized-huber-ridge-gesture-correction-v1",
        "featureNames": fitted["featureNames"],
        "weights": [round(float(value), 10) for value in fitted["weights"]],
        "means": [round(float(value), 10) for value in fitted["means"]],
        "scales": [round(float(value), 10) for value in fitted["scales"]],
        "blend": round(float(blend), 6),
        "maximumCorrection": round(float(maximum_correction), 6),
        "ridge": round(float(fitted["ridge"]), 6),
        "huberDelta": round(float(fitted["huberDelta"]), 6),
        "trainingBandBalance": round(
            float(fitted.get("trainingBandBalance", 0.0)), 6
        ),
    }


def predict_examples(
    examples: list[dict[str, Any]], config: dict[str, Any]
) -> list[float]:
    output: list[float] = []
    for example in examples:
        feature_values = dict(
            zip(GESTURE_CALIBRATION_FEATURE_NAMES, example["features"])
        )
        # Recreate a tiny group whose feature extraction is bypassed by using
        # the exact standardized contract below.  This mirrors the runtime
        # formula and avoids inventing note fields in training evaluation.
        names = config["featureNames"]
        correction = sum(
            float(weight)
            * (feature_values[name] - float(mean))
            / max(1e-9, abs(float(scale)))
            for name, weight, mean, scale in zip(
                names, config["weights"], config["means"], config["scales"]
            )
        )
        maximum = float(config["maximumCorrection"])
        correction = max(-maximum, min(maximum, correction))
        value = float(example["baseVelocity"]) + float(config["blend"]) * correction
        output.append(max(0.38, min(0.94, value)))
    return output


def metric_summary(
    examples: list[dict[str, Any]], predicted: list[float] | None = None
) -> dict[str, Any]:
    values = (
        predicted
        if predicted is not None
        else [float(example["baseVelocity"]) for example in examples]
    )
    errors = [
        value - float(example["targetVelocity"])
        for value, example in zip(values, examples)
    ]
    by_band: list[dict[str, Any]] = []
    for band in sorted({str(example["targetBand"]) for example in examples}):
        indices = [
            index
            for index, example in enumerate(examples)
            if example["targetBand"] == band
        ]
        band_errors = [errors[index] for index in indices]
        by_band.append(
            {
                "name": band,
                "gestures": len(indices),
                "meanSignedError": round(sum(band_errors) / len(indices), 6),
                "meanAbsoluteError": round(
                    sum(abs(value) for value in band_errors) / len(indices), 6
                ),
            }
        )
    return {
        "gestures": len(examples),
        "meanSignedError": round(sum(errors) / len(errors), 6),
        "meanAbsoluteError": round(
            sum(abs(value) for value in errors) / len(errors), 6
        ),
        "within005": sum(abs(value) <= 0.05 for value in errors),
        "within010": sum(abs(value) <= 0.10 for value in errors),
        "byTargetVelocityBand": by_band,
    }


def cross_validate(
    examples: list[dict[str, Any]],
    ridge_grid: list[float],
    blend_grid: list[float],
    maximum_correction_grid: list[float],
    profile_id: str,
    band_balance_grid: list[float] | None = None,
) -> dict[str, Any]:
    songs = sorted({str(example["songId"]) for example in examples})
    if len(songs) < 2:
        raise ValueError("Song-level validation requires at least two songs")
    trials: list[dict[str, Any]] = []
    balances = band_balance_grid or [0.0]
    for band_balance in balances:
        for ridge in ridge_grid:
            fold_models: dict[str, dict[str, Any]] = {}
            for song in songs:
                training = [example for example in examples if example["songId"] != song]
                fold_models[song] = fit_ridge_correction(
                    training, ridge, band_balance=band_balance
                )
            for blend in blend_grid:
                for maximum_correction in maximum_correction_grid:
                    folds: list[dict[str, Any]] = []
                    all_validation: list[dict[str, Any]] = []
                    all_predictions: list[float] = []
                    for song in songs:
                        validation = [
                            example for example in examples if example["songId"] == song
                        ]
                        config = model_config(
                            fold_models[song],
                            profile_id=profile_id,
                            blend=blend,
                            maximum_correction=maximum_correction,
                        )
                        predicted = predict_examples(validation, config)
                        baseline = metric_summary(validation)
                        candidate = metric_summary(validation, predicted)
                        baseline_bands = {
                            str(row["name"]): row
                            for row in baseline["byTargetVelocityBand"]
                        }
                        candidate_bands = {
                            str(row["name"]): row
                            for row in candidate["byTargetVelocityBand"]
                        }
                        band_deltas = []
                        for band in sorted(baseline_bands):
                            baseline_band = baseline_bands[band]
                            candidate_band = candidate_bands[band]
                            band_deltas.append(
                                {
                                    "name": band,
                                    "gestures": int(baseline_band["gestures"]),
                                    "baselineMae": baseline_band["meanAbsoluteError"],
                                    "candidateMae": candidate_band["meanAbsoluteError"],
                                    "candidateMinusBaselineMae": round(
                                        float(candidate_band["meanAbsoluteError"])
                                        - float(baseline_band["meanAbsoluteError"]),
                                        6,
                                    ),
                                }
                            )
                        supported_band_deltas = [
                            row for row in band_deltas if int(row["gestures"]) >= 10
                        ]
                        worst_band_regression = max(
                            (
                                float(row["candidateMinusBaselineMae"])
                                for row in supported_band_deltas
                            ),
                            default=0.0,
                        )
                        folds.append(
                            {
                                "heldOutSong": song,
                                "baseline": baseline,
                                "candidate": candidate,
                                "candidateMinusBaselineMae": round(
                                    candidate["meanAbsoluteError"]
                                    - baseline["meanAbsoluteError"],
                                    6,
                                ),
                                "velocityBandDeltas": band_deltas,
                                "worstSupportedVelocityBandRegression": round(
                                    worst_band_regression, 6
                                ),
                            }
                        )
                        all_validation.extend(validation)
                        all_predictions.extend(predicted)
                    baseline = metric_summary(all_validation)
                    candidate = metric_summary(all_validation, all_predictions)
                    worst_regression = max(
                        fold["candidateMinusBaselineMae"] for fold in folds
                    )
                    worst_band_regression = max(
                        float(fold["worstSupportedVelocityBandRegression"])
                        for fold in folds
                    )
                    trials.append(
                        {
                            "bandBalance": round(float(band_balance), 6),
                            "ridge": ridge,
                            "blend": blend,
                            "maximumCorrection": maximum_correction,
                            "baselineMae": baseline["meanAbsoluteError"],
                            "candidateMae": candidate["meanAbsoluteError"],
                            "candidateMinusBaselineMae": round(
                                candidate["meanAbsoluteError"]
                                - baseline["meanAbsoluteError"],
                                6,
                            ),
                            "worstSongRegression": worst_regression,
                            "worstVelocityBandRegression": round(
                                worst_band_regression, 6
                            ),
                            "selectionScore": round(
                                candidate["meanAbsoluteError"]
                                + max(0.0, worst_regression) * 2.0
                                + max(0.0, worst_band_regression) * 2.0,
                                6,
                            ),
                            "folds": folds,
                        }
                    )
    trials.sort(
        key=lambda row: (
            row["selectionScore"],
            row["candidateMae"],
            row["maximumCorrection"],
            row["blend"],
            row["bandBalance"],
            row["ridge"],
        )
    )
    return {"songs": songs, "best": trials[0], "topTrials": trials[:20]}


def profile_hash(profile: dict[str, Any]) -> str:
    value = copy.deepcopy(profile)
    value.pop("profileSha256", None)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_grid(value: str) -> list[float]:
    output = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not output or any(not math.isfinite(item) or item < 0 for item in output):
        raise argparse.ArgumentTypeError("Grid values must be finite and non-negative")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-pitch-class-overlap", type=int, default=1)
    parser.add_argument("--ridge-grid", type=parse_grid, default=parse_grid("0.03,0.1,0.3,1,3,10"))
    parser.add_argument("--blend-grid", type=parse_grid, default=parse_grid("0.25,0.5,0.75,1"))
    parser.add_argument("--maximum-correction-grid", type=parse_grid, default=parse_grid("0.08,0.12,0.18"))
    parser.add_argument(
        "--band-balance-grid",
        type=parse_grid,
        default=parse_grid("0,0.25,0.5,0.75,1"),
        help=(
            "Blend between song-equal and song-plus-velocity-band-equal "
            "training weights. One prevents common mezzo gestures from "
            "overwhelming rare quiet/accent examples."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    rows = manifest.get("songs") or []
    if len(rows) < 2:
        raise ValueError("The manifest needs at least two songs for song-level validation")
    onset_window = min(0.08, max(0.005, args.onset_window_seconds))
    tolerance = min(0.25, max(0.02, args.tolerance_seconds))
    examples: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    for row in rows:
        song_examples, song_report = build_song_examples(
            row,
            onset_window,
            tolerance,
            max(0, args.minimum_pitch_class_overlap),
        )
        if len(song_examples) < 12:
            raise ValueError(
                f"{song_report['id']} has too few reliable matched gestures "
                f"({len(song_examples)})"
            )
        examples.extend(song_examples)
        dataset.append(song_report)

    validation = cross_validate(
        examples,
        args.ridge_grid,
        [min(1.0, value) for value in args.blend_grid],
        [min(0.35, value) for value in args.maximum_correction_grid],
        args.profile_id,
        [min(1.0, value) for value in args.band_balance_grid],
    )
    best = validation["best"]
    fitted = fit_ridge_correction(
        examples,
        best["ridge"],
        band_balance=best["bandBalance"],
    )
    config = model_config(
        fitted,
        profile_id=args.profile_id,
        blend=best["blend"],
        maximum_correction=best["maximumCorrection"],
    )
    in_sample_predictions = predict_examples(examples, config)

    base_profile_path = Path(args.base_profile).resolve()
    profile = copy.deepcopy(load_json(base_profile_path))
    profile.pop("profileSha256", None)
    profile["id"] = args.profile_id
    profile["createdAt"] = datetime.now(timezone.utc).isoformat()
    gesture = profile.setdefault("decoder", {}).setdefault("gestureDynamics", {})
    if not gesture.get("enabled"):
        raise ValueError("The base profile must enable gesture dynamics")
    gesture["pairedCalibration"] = config
    training = profile.setdefault("training", {})
    training["pairedGestureVelocityCalibration"] = {
        "schema": "polymath-paired-gesture-velocity-training-v1",
        "method": "song-equal-huber-ridge-delta-with-song-level-validation",
        "manifest": str(manifest_path),
        "baseProfile": str(base_profile_path),
        "baseProfileId": load_json(base_profile_path).get("id"),
        "onsetWindowSeconds": round(onset_window, 6),
        "toleranceSeconds": round(tolerance, 6),
        "minimumPitchClassOverlap": max(0, args.minimum_pitch_class_overlap),
        "songs": dataset,
        "crossValidation": best,
        "commercialUseAllowed": False,
        "decision": "EXPERIMENTAL_ONLY",
    }
    profile["profileSha256"] = profile_hash(profile)

    output_profile = Path(args.output_profile).resolve()
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_profile.with_name(output_profile.name + ".tmp")
    temporary.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_profile)

    report = {
        "schema": "polymath-paired-gesture-velocity-report-v1",
        "evidenceBoundary": (
            "Development-song calibration only. Song-level validation does not "
            "replace a newly frozen untouched-song listening test."
        ),
        "manifest": str(manifest_path),
        "baseProfile": str(base_profile_path),
        "outputProfile": str(output_profile),
        "profileSha256": profile["profileSha256"],
        "dataset": dataset,
        "crossValidation": validation,
        "finalCalibrationFit": {
            "baseline": metric_summary(examples),
            "candidate": metric_summary(examples, in_sample_predictions),
        },
        "model": config,
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputProfile": str(output_profile),
                "matchedGestures": len(examples),
                "bestCrossValidation": best,
                "finalCalibrationFit": report["finalCalibrationFit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
