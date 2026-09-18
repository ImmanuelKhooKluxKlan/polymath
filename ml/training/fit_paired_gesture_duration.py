"""Fit a conservative Pianella-like gesture-duration correction.

The source transcription already gives useful timing, but its note lengths are
compressed toward a small set of values.  This trainer learns a bounded change
to the median written hold of an onset gesture.  It never changes pitches,
onsets, chord sizes, velocities, or song length.  Validation is leave-one-song-
out so a memorized timing map cannot win.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ml.training.analyze_piano_velocity_style import load_notes, onset_groups, trusted_source_ranges
from ml.training.fit_paired_gesture_velocity import (
    candidate_gesture_groups,
    finite,
    load_json,
    match_gesture_groups,
    parse_grid,
    profile_hash,
    standardize,
)


SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_gesture_calibration import (  # noqa: E402
    GESTURE_DURATION_FEATURE_NAMES,
    gesture_duration_features,
    gesture_sequence_contexts,
)


def duration_band(value: float) -> str:
    if value < 0.18:
        return "short-<0.18"
    if value < 0.55:
        return "medium-0.18-0.54"
    return "long->=0.55"


def gesture_duration(group: list[dict[str, Any]]) -> float:
    return float(
        median(
            float(note.get("scoreDuration", note.get("duration", 0.2)))
            for note in group
        )
    )


def build_song_examples(
    row: dict[str, Any],
    onset_window: float,
    tolerance: float,
    minimum_pitch_class_overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every paired-duration manifest row requires an id")
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
        candidate_path,
        include_ranges=ranges,
        end_seconds=end_seconds,
    )
    reference_groups = onset_groups(reference_notes, onset_window)
    candidate_groups = candidate_gesture_groups(candidate_notes, onset_window)
    reference_next_gap = {
        id(group): max(
            0.03,
            float(reference_groups[index + 1][0]["time"])
            - float(group[0]["time"]),
        )
        for index, group in enumerate(reference_groups[:-1])
    }
    base_velocities = [
        median(float(note.get("velocity", 0.72)) for note in group)
        for group in candidate_groups
    ]
    contexts = gesture_sequence_contexts(candidate_groups, base_velocities)
    context_by_group = {
        id(group): context for group, context in zip(candidate_groups, contexts)
    }
    matched = match_gesture_groups(
        reference_groups,
        candidate_groups,
        tolerance,
        minimum_pitch_class_overlap,
    )
    examples: list[dict[str, Any]] = []
    for reference_group, candidate_group, overlap in matched:
        base_duration = gesture_duration(candidate_group)
        target_duration = gesture_duration(reference_group)
        reference_gap = reference_next_gap.get(
            id(reference_group), target_duration
        )
        candidate_gap = max(
            0.03,
            float(
                context_by_group[id(candidate_group)].get(
                    "next_gap_seconds", base_duration
                )
            ),
        )
        feature_map = gesture_duration_features(
            candidate_group,
            base_duration,
            context_by_group[id(candidate_group)],
        )
        examples.append(
            {
                "songId": song_id,
                "sourceTime": round(float(candidate_group[0]["time"]), 6),
                "referenceTime": round(float(reference_group[0]["time"]), 6),
                "baseDuration": base_duration,
                "targetDuration": target_duration,
                "referenceNextGapSeconds": reference_gap,
                "candidateNextGapSeconds": candidate_gap,
                "targetArticulationRatio": target_duration
                / max(0.03, reference_gap),
                "baseArticulationRatio": base_duration
                / max(0.03, candidate_gap),
                "targetBand": duration_band(target_duration),
                "pitchClassOverlap": overlap,
                "features": [
                    feature_map[name] for name in GESTURE_DURATION_FEATURE_NAMES
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
        "referenceGestures": len(reference_groups),
        "candidateGestures": len(candidate_groups),
        "matchedGestures": len(examples),
        "matchedGestureRate": round(
            len(examples) / max(1, len(reference_groups)), 6
        ),
    }


def balanced_weights(
    examples: list[dict[str, Any]], band_balance: float
) -> np.ndarray:
    band_balance = min(1.0, max(0.0, float(band_balance)))
    song_counts: dict[str, int] = defaultdict(int)
    band_counts: dict[tuple[str, str], int] = defaultdict(int)
    bands: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        song = str(example["songId"])
        band = str(example["targetBand"])
        song_counts[song] += 1
        band_counts[(song, band)] += 1
        bands[song].add(band)
    song_total = max(1, len(song_counts))
    values = []
    for example in examples:
        song = str(example["songId"])
        band = str(example["targetBand"])
        equal_song = 1.0 / (song_total * song_counts[song])
        equal_band = 1.0 / (
            song_total * max(1, len(bands[song])) * band_counts[(song, band)]
        )
        values.append(
            (1.0 - band_balance) * equal_song + band_balance * equal_band
        )
    weights = np.asarray(values, dtype=float)
    return weights * len(examples) / max(1e-12, float(weights.sum()))


def fit_model(
    examples: list[dict[str, Any]],
    ridge: float,
    *,
    band_balance: float,
    huber_delta: float = 0.35,
    iterations: int = 7,
    target_mode: str = "correction",
) -> dict[str, Any]:
    if len(examples) < 12:
        raise ValueError("At least 12 matched gestures are required")
    matrix = np.asarray([example["features"] for example in examples], dtype=float)
    if target_mode == "articulation-ratio":
        target = np.asarray(
            [
                math.log(
                    max(0.15, min(5.0, float(example["targetArticulationRatio"])))
                )
                for example in examples
            ],
            dtype=float,
        )
    else:
        target = np.asarray(
            [
                math.log1p(float(example["targetDuration"]))
                - math.log1p(float(example["baseDuration"]))
                for example in examples
            ],
            dtype=float,
        )
    base_weights = balanced_weights(examples, band_balance)
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
        "featureNames": list(GESTURE_DURATION_FEATURE_NAMES),
        "weights": coefficients.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "ridge": float(ridge),
        "huberDelta": float(huber_delta),
        "trainingBandBalance": round(band_balance, 6),
        "targetMode": target_mode,
    }


def model_config(
    fitted: dict[str, Any],
    *,
    profile_id: str,
    blend: float,
    maximum_log_correction: float,
    target_mode: str,
    gate_policy: str = "all",
) -> dict[str, Any]:
    gates = {
        "all": (0.0, 8.0),
        "melody": (1e-6, 0.0),
        "melody-or-base-lt-0.15": (1e-6, 0.15),
        "melody-or-base-lt-0.18": (1e-6, 0.18),
        "melody-or-base-lt-0.22": (1e-6, 0.22),
    }
    if gate_policy not in gates:
        raise ValueError(f"Unknown duration gate policy: {gate_policy}")
    minimum_melody_share, maximum_non_melody_duration = gates[gate_policy]
    return {
        "enabled": True,
        "id": profile_id,
        "type": "standardized-huber-ridge-gesture-log-duration-correction-v1",
        "featureNames": fitted["featureNames"],
        "weights": [round(float(value), 10) for value in fitted["weights"]],
        "means": [round(float(value), 10) for value in fitted["means"]],
        "scales": [round(float(value), 10) for value in fitted["scales"]],
        "blend": round(float(blend), 6),
        "maximumLogCorrection": round(float(maximum_log_correction), 6),
        "predictionMode": target_mode,
        "gatePolicy": gate_policy,
        "minimumMelodyShare": minimum_melody_share,
        "maximumNonMelodyBaseDurationSeconds": maximum_non_melody_duration,
        "minimumDurationSeconds": 0.05,
        "maximumDurationSeconds": 4.0,
        "ridge": round(float(fitted["ridge"]), 6),
        "huberDelta": round(float(fitted["huberDelta"]), 6),
        "trainingBandBalance": round(
            float(fitted.get("trainingBandBalance", 0.0)), 6
        ),
    }


def predict(
    examples: list[dict[str, Any]], config: dict[str, Any]
) -> list[float]:
    output = []
    for example in examples:
        values = dict(zip(GESTURE_DURATION_FEATURE_NAMES, example["features"]))
        melody_share = float(values.get("melody_share", 0.0))
        if (
            float(config.get("minimumMelodyShare", 0.0)) > 0.0
            and melody_share < float(config["minimumMelodyShare"])
            and float(example["baseDuration"])
            > float(config.get("maximumNonMelodyBaseDurationSeconds", 8.0))
        ):
            output.append(float(example["baseDuration"]))
            continue
        raw_prediction = sum(
            float(weight) * (values[name] - float(mean)) / max(1e-9, abs(float(scale)))
            for name, weight, mean, scale in zip(
                config["featureNames"],
                config["weights"],
                config["means"],
                config["scales"],
            )
        )
        maximum = float(config["maximumLogCorrection"])
        if config.get("predictionMode") == "articulation-ratio":
            predicted_ratio = math.exp(max(math.log(0.15), min(math.log(5.0), raw_prediction)))
            predicted_duration = predicted_ratio * float(
                example["candidateNextGapSeconds"]
            )
            raw_correction = (
                math.log1p(max(0.05, predicted_duration))
                - math.log1p(float(example["baseDuration"]))
            )
        else:
            raw_correction = raw_prediction
        correction = max(-maximum, min(maximum, raw_correction))
        value = math.expm1(
            math.log1p(float(example["baseDuration"]))
            + float(config["blend"]) * correction
        )
        output.append(max(0.05, min(4.0, value)))
    return output


def metrics(
    examples: list[dict[str, Any]], predicted: list[float] | None = None
) -> dict[str, Any]:
    values = predicted or [float(example["baseDuration"]) for example in examples]
    errors = [
        value - float(example["targetDuration"])
        for value, example in zip(values, examples)
    ]
    bands = []
    for band in sorted({str(example["targetBand"]) for example in examples}):
        indices = [
            index
            for index, example in enumerate(examples)
            if example["targetBand"] == band
        ]
        band_errors = [errors[index] for index in indices]
        bands.append(
            {
                "name": band,
                "gestures": len(indices),
                "meanSignedErrorSeconds": round(
                    sum(band_errors) / len(band_errors), 6
                ),
                "meanAbsoluteErrorSeconds": round(
                    sum(abs(value) for value in band_errors) / len(band_errors),
                    6,
                ),
            }
        )
    return {
        "gestures": len(examples),
        "meanSignedErrorSeconds": round(sum(errors) / len(errors), 6),
        "meanAbsoluteErrorSeconds": round(
            sum(abs(value) for value in errors) / len(errors), 6
        ),
        "within005Seconds": sum(abs(value) <= 0.05 for value in errors),
        "within010Seconds": sum(abs(value) <= 0.10 for value in errors),
        "byTargetDurationBand": bands,
    }


def cross_validate(
    examples: list[dict[str, Any]],
    ridge_grid: list[float],
    blend_grid: list[float],
    maximum_grid: list[float],
    balance_grid: list[float],
    profile_id: str,
    target_mode: str,
    gate_policies: list[str],
) -> dict[str, Any]:
    songs = sorted({str(example["songId"]) for example in examples})
    if len(songs) < 2:
        raise ValueError("Song-level validation requires at least two songs")
    trials = []
    for balance in balance_grid:
        for ridge in ridge_grid:
            fold_models = {
                song: fit_model(
                    [example for example in examples if example["songId"] != song],
                    ridge,
                    band_balance=balance,
                    target_mode=target_mode,
                )
                for song in songs
            }
            for blend in blend_grid:
                for maximum in maximum_grid:
                    for gate_policy in gate_policies:
                        folds = []
                        all_validation = []
                        all_predictions = []
                        for song in songs:
                            validation = [
                                example for example in examples if example["songId"] == song
                            ]
                            config = model_config(
                                fold_models[song],
                                profile_id=profile_id,
                                blend=blend,
                                maximum_log_correction=maximum,
                                target_mode=target_mode,
                                gate_policy=gate_policy,
                            )
                            predicted = predict(validation, config)
                            baseline = metrics(validation)
                            candidate = metrics(validation, predicted)
                            base_bands = {
                            row["name"]: row for row in baseline["byTargetDurationBand"]
                        }
                            cand_bands = {
                            row["name"]: row for row in candidate["byTargetDurationBand"]
                        }
                            band_deltas = []
                            for band in sorted(base_bands):
                                count = int(base_bands[band]["gestures"])
                                delta = (
                                float(cand_bands[band]["meanAbsoluteErrorSeconds"])
                                - float(base_bands[band]["meanAbsoluteErrorSeconds"])
                            )
                                band_deltas.append(
                                {
                                    "name": band,
                                    "gestures": count,
                                    "candidateMinusBaselineMaeSeconds": round(delta, 6),
                                }
                            )
                            supported = [row for row in band_deltas if row["gestures"] >= 10]
                            folds.append(
                            {
                                "heldOutSong": song,
                                "baseline": baseline,
                                "candidate": candidate,
                                "candidateMinusBaselineMaeSeconds": round(
                                    candidate["meanAbsoluteErrorSeconds"]
                                    - baseline["meanAbsoluteErrorSeconds"],
                                    6,
                                ),
                                "durationBandDeltas": band_deltas,
                                "worstSupportedDurationBandRegression": round(
                                    max(
                                        (
                                            float(row["candidateMinusBaselineMaeSeconds"])
                                            for row in supported
                                        ),
                                        default=0.0,
                                    ),
                                    6,
                                ),
                            }
                        )
                            all_validation.extend(validation)
                            all_predictions.extend(predicted)
                        baseline = metrics(all_validation)
                        candidate = metrics(all_validation, all_predictions)
                        worst_song = max(
                        float(fold["candidateMinusBaselineMaeSeconds"])
                        for fold in folds
                    )
                        worst_band = max(
                        float(fold["worstSupportedDurationBandRegression"])
                        for fold in folds
                    )
                        trials.append(
                        {
                            "bandBalance": round(balance, 6),
                            "ridge": ridge,
                            "blend": blend,
                            "maximumLogCorrection": maximum,
                            "gatePolicy": gate_policy,
                            "baselineMaeSeconds": baseline["meanAbsoluteErrorSeconds"],
                            "candidateMaeSeconds": candidate["meanAbsoluteErrorSeconds"],
                            "candidateMinusBaselineMaeSeconds": round(
                                candidate["meanAbsoluteErrorSeconds"]
                                - baseline["meanAbsoluteErrorSeconds"],
                                6,
                            ),
                            "worstSongRegression": round(worst_song, 6),
                            "worstDurationBandRegression": round(worst_band, 6),
                            "selectionScore": round(
                                candidate["meanAbsoluteErrorSeconds"]
                                + 2.0 * max(0.0, worst_song)
                                + 2.0 * max(0.0, worst_band),
                                6,
                            ),
                            "folds": folds,
                        }
                    )
    trials.sort(
        key=lambda row: (
            row["selectionScore"],
            row["candidateMaeSeconds"],
            row["maximumLogCorrection"],
            row["blend"],
            row["gatePolicy"],
            row["bandBalance"],
            row["ridge"],
        )
    )
    non_regressing = [
        row
        for row in trials
        if float(row["worstSongRegression"]) <= 0.0
        and float(row["worstDurationBandRegression"]) <= 0.0
    ]
    return {
        "songs": songs,
        "best": trials[0],
        "bestNonRegressing": non_regressing[0] if non_regressing else None,
        "nonRegressingTrialCount": len(non_regressing),
        "topTrials": trials[:20],
    }


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
    parser.add_argument("--ridge-grid", type=parse_grid, default=parse_grid("0.1,0.3,1,3,10"))
    parser.add_argument("--blend-grid", type=parse_grid, default=parse_grid("0.25,0.5,0.75,1"))
    parser.add_argument("--maximum-log-correction-grid", type=parse_grid, default=parse_grid("0.25,0.5,0.75,1"))
    parser.add_argument("--band-balance-grid", type=parse_grid, default=parse_grid("0,0.25,0.5,0.75,1"))
    parser.add_argument(
        "--target-mode",
        choices=("correction", "articulation-ratio"),
        default="correction",
    )
    parser.add_argument(
        "--gate-policies",
        default="all,melody,melody-or-base-lt-0.15,melody-or-base-lt-0.18,melody-or-base-lt-0.22",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    rows = manifest.get("songs") or []
    if len(rows) < 2:
        raise ValueError("The manifest needs at least two songs")
    onset_window = min(0.08, max(0.005, args.onset_window_seconds))
    tolerance = min(0.25, max(0.02, args.tolerance_seconds))
    examples = []
    dataset = []
    for row in rows:
        song_examples, song_report = build_song_examples(
            row,
            onset_window,
            tolerance,
            max(0, args.minimum_pitch_class_overlap),
        )
        if len(song_examples) < 12:
            raise ValueError(
                f"{song_report['id']} has too few matched gestures ({len(song_examples)})"
            )
        examples.extend(song_examples)
        dataset.append(song_report)

    validation = cross_validate(
        examples,
        args.ridge_grid,
        [min(1.0, value) for value in args.blend_grid],
        [min(2.0, value) for value in args.maximum_log_correction_grid],
        [min(1.0, value) for value in args.band_balance_grid],
        args.profile_id,
        args.target_mode,
        [
            value.strip()
            for value in args.gate_policies.split(",")
            if value.strip()
        ],
    )
    best = validation["best"]
    fitted = fit_model(
        examples,
        best["ridge"],
        band_balance=best["bandBalance"],
        target_mode=args.target_mode,
    )
    config = model_config(
        fitted,
        profile_id=args.profile_id,
        blend=best["blend"],
        maximum_log_correction=best["maximumLogCorrection"],
        target_mode=args.target_mode,
        gate_policy=best["gatePolicy"],
    )
    in_sample = predict(examples, config)
    base_profile_path = Path(args.base_profile).resolve()
    profile = copy.deepcopy(load_json(base_profile_path))
    profile.pop("profileSha256", None)
    profile["id"] = args.profile_id
    profile["createdAt"] = datetime.now(timezone.utc).isoformat()
    gesture = profile.setdefault("decoder", {}).setdefault("gestureDynamics", {})
    if not gesture.get("enabled"):
        raise ValueError("The base profile must enable gesture dynamics")
    gesture["pairedDurationCalibration"] = config
    profile.setdefault("training", {})["pairedGestureDurationCalibration"] = {
        "schema": "polymath-paired-gesture-duration-training-v1",
        "method": "song-equal-huber-ridge-log-duration-with-song-level-validation",
        "manifest": str(manifest_path),
        "baseProfile": str(base_profile_path),
        "onsetWindowSeconds": round(onset_window, 6),
        "toleranceSeconds": round(tolerance, 6),
        "minimumPitchClassOverlap": max(0, args.minimum_pitch_class_overlap),
        "targetMode": args.target_mode,
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
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_profile)
    report = {
        "schema": "polymath-paired-gesture-duration-report-v1",
        "evidenceBoundary": (
            "Development-song calibration only; a new untouched song and "
            "sealed listening test remain mandatory."
        ),
        "manifest": str(manifest_path),
        "baseProfile": str(base_profile_path),
        "outputProfile": str(output_profile),
        "profileSha256": profile["profileSha256"],
        "dataset": dataset,
        "crossValidation": validation,
        "finalCalibrationFit": {
            "baseline": metrics(examples),
            "candidate": metrics(examples, in_sample),
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
