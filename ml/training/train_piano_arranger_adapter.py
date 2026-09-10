"""Train the small route-specific Polymath piano arranger profile.

This does *not* fine-tune MuScriptor. MuScriptor is the instrument-aware
transcriber shared by Piano and Band. The learned profile produced here is the
next stage used only by Piano: it learns which detected events resemble an
approved piano reduction, how to revoice them, and the target duration/density
style.

The trainer uses NumPy, but its JSON output is decoded by the standard-library
only ``server/piano_arranger_adapter.py`` module in production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    DURATION_FEATURE_NAMES,
    FEATURE_NAMES,
    arrangement_role,
    duration_feature_rows,
    normalize_source_notes,
    raw_feature_rows,
)


SCHEMA = "polymath-piano-arranger-profile-v1"
DEFAULT_BLOCK_SECONDS = 20.0
DEFAULT_SEED = 0x504F4C59


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def weighted_quantile(values: Iterable[float], weights: Iterable[float], fraction: float) -> float:
    pairs = sorted(
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and float(weight) > 0
    )
    if not pairs:
        return 0.0
    total = sum(weight for _value, weight in pairs)
    target = clamp(fraction, 0.0, 1.0) * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def normalized_target_notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("notes", [])):
        try:
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
            velocity = clamp(float(item.get("velocity", 0.72)), 0.0, 1.0)
        except (TypeError, ValueError):
            continue
        if not (21 <= midi <= 108) or time < 0:
            continue
        notes.append(
            {
                "sourceIndex": index,
                "midi": midi,
                "time": time,
                "duration": duration,
                "velocity": velocity,
                "instrument": "acoustic_piano",
            }
        )
    return sorted(notes, key=lambda note: (note["time"], note["midi"]))


def target_role(note: dict[str, Any]) -> str:
    midi = int(note["midi"])
    if midi <= 52:
        return "bass"
    if midi >= 72:
        return "melody"
    return "harmony"


def alignment_match_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    strongest: dict[int, dict[str, Any]] = {}
    for match in report.get("matches", []):
        observed = match.get("observed") or {}
        reference = match.get("reference") or {}
        try:
            source_index = int(observed["sourceIndex"])
        except (KeyError, TypeError, ValueError):
            continue
        candidate = {
            "exactPitch": bool(match.get("exactPitch")),
            "observed": observed,
            "reference": reference,
        }
        previous = strongest.get(source_index)
        if previous is None or candidate["exactPitch"] and not previous["exactPitch"]:
            strongest[source_index] = candidate
    return strongest


def alignment_training_ranges(report: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Return trusted source-time ranges when a section-aware report provides them.

    ``None`` preserves compatibility with older whole-song alignment reports. An
    empty list means the report explicitly found no trustworthy training region.
    """
    windows = report.get("qualityWindows")
    if not isinstance(windows, list):
        return None
    ranges: list[tuple[float, float]] = []
    for window in windows:
        if not isinstance(window, dict) or window.get("status") != "trusted":
            continue
        try:
            start = float(window["sourceStartSeconds"])
            end = float(window["sourceEndSeconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    return ranges


def note_is_in_training_range(
    note: dict[str, Any],
    ranges: list[tuple[float, float]] | None,
) -> bool:
    if ranges is None:
        return True
    time = float(note["time"])
    return any(start <= time < end for start, end in ranges)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    truth = labels >= 0.5
    true_positive = float(weights[predicted & truth].sum())
    false_positive = float(weights[predicted & ~truth].sum())
    false_negative = float(weights[~predicted & truth].sum())
    true_negative = float(weights[~predicted & ~truth].sum())
    precision = true_positive / max(1e-9, true_positive + false_positive)
    recall = true_positive / max(1e-9, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (true_positive + true_negative) / max(
        1e-9, true_positive + true_negative + false_positive + false_negative
    )
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "weightedTruePositive": round(true_positive, 3),
        "weightedFalsePositive": round(false_positive, 3),
        "weightedFalseNegative": round(false_negative, 3),
        "weightedTrueNegative": round(true_negative, 3),
    }


def train_logistic_model(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    training_mask: np.ndarray,
    *,
    seed: int,
    iterations: int = 1200,
    learning_rate: float = 0.025,
    l2: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    means = features[training_mask].mean(axis=0)
    scales = features[training_mask].std(axis=0)
    means[0] = 0.0
    scales[0] = 1.0
    scales[scales < 1e-8] = 1.0
    standardized = (features - means) / scales
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=features.shape[1])
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta_one = 0.9
    beta_two = 0.999
    epsilon = 1e-8
    history: list[dict[str, float]] = []
    x_train = standardized[training_mask]
    y_train = labels[training_mask]
    w_train = sample_weights[training_mask]
    weight_total = max(1e-9, float(w_train.sum()))

    for iteration in range(1, iterations + 1):
        logits = np.clip(x_train @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x_train.T @ ((probabilities - y_train) * w_train)) / weight_total
        regularizer = l2 * weights
        regularizer[0] = 0.0
        gradient += regularizer
        first_moment = beta_one * first_moment + (1.0 - beta_one) * gradient
        second_moment = beta_two * second_moment + (1.0 - beta_two) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta_one**iteration)
        corrected_second = second_moment / (1.0 - beta_two**iteration)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            loss = -np.sum(
                w_train
                * (
                    y_train * np.log(np.maximum(probabilities, 1e-9))
                    + (1.0 - y_train) * np.log(np.maximum(1.0 - probabilities, 1e-9))
                )
            ) / weight_total
            history.append(
                {
                    "iteration": iteration,
                    "weightedBinaryCrossEntropy": round(float(loss), 8),
                    "weightL2Norm": round(float(np.linalg.norm(weights)), 8),
                }
            )
    return weights, means, scales, history


def duration_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    if not len(targets):
        return {
            "meanAbsoluteErrorSeconds": 0.0,
            "weightedRootMeanSquaredErrorSeconds": 0.0,
            "within100ms": 0.0,
            "within250ms": 0.0,
        }
    normalized_weights = weights / max(1e-9, float(weights.sum()))
    absolute = np.abs(predictions - targets)
    squared = np.square(predictions - targets)
    return {
        "meanAbsoluteErrorSeconds": round(float(np.sum(absolute * normalized_weights)), 6),
        "weightedRootMeanSquaredErrorSeconds": round(
            float(math.sqrt(np.sum(squared * normalized_weights))), 6
        ),
        "within100ms": round(float(np.sum(normalized_weights[absolute <= 0.10])), 6),
        "within250ms": round(float(np.sum(normalized_weights[absolute <= 0.25])), 6),
    }


def train_duration_model(
    features: np.ndarray,
    target_seconds: np.ndarray,
    sample_weights: np.ndarray,
    training_mask: np.ndarray,
    *,
    ridge: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a compact weighted ridge model to log key-hold duration."""
    means = features[training_mask].mean(axis=0)
    scales = features[training_mask].std(axis=0)
    means[0] = 0.0
    scales[0] = 1.0
    scales[scales < 1e-8] = 1.0
    standardized = (features - means) / scales
    x_train = standardized[training_mask]
    y_train = np.log1p(target_seconds[training_mask])
    w_train = sample_weights[training_mask]
    weight_total = max(1e-9, float(w_train.sum()))
    weighted_x = x_train * w_train[:, None]
    gram = (x_train.T @ weighted_x) / weight_total
    regularizer = np.eye(features.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    right_hand_side = (x_train.T @ (w_train * y_train)) / weight_total
    try:
        learned_weights = np.linalg.solve(gram + regularizer, right_hand_side)
    except np.linalg.LinAlgError:
        learned_weights = np.linalg.pinv(gram + regularizer) @ right_hand_side
    predicted_seconds = np.expm1(
        np.clip(standardized @ learned_weights, 0.0, math.log1p(6.0))
    )
    predicted_seconds = np.clip(predicted_seconds, 0.05, 4.0)
    validation_mask = ~training_mask
    diagnostics = {
        "ridge": ridge,
        "training": duration_regression_metrics(
            target_seconds[training_mask],
            predicted_seconds[training_mask],
            sample_weights[training_mask],
        ),
        "heldOutBlocks": duration_regression_metrics(
            target_seconds[validation_mask],
            predicted_seconds[validation_mask],
            sample_weights[validation_mask],
        ),
    }
    return learned_weights, means, scales, predicted_seconds, diagnostics


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in np.linspace(0.08, 0.92, 169):
        metrics = classification_metrics(labels, probabilities, weights, float(threshold))
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def onset_cluster_sizes(notes: list[dict[str, Any]], tolerance: float = 0.04) -> list[int]:
    if not notes:
        return []
    clusters: list[list[dict[str, Any]]] = []
    for note in notes:
        if not clusters or note["time"] - clusters[-1][0]["time"] > tolerance:
            clusters.append([note])
        else:
            clusters[-1].append(note)
    return [len(cluster) for cluster in clusters]


def same_pitch_gaps(notes: list[dict[str, Any]]) -> list[float]:
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(float(note["time"]))
    gaps: list[float] = []
    for times in by_pitch.values():
        times.sort()
        gaps.extend(current - previous for previous, current in zip(times, times[1:]) if current > previous)
    return gaps


def build_style_profile(
    pairs: list[dict[str, Any]],
    match_maps: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_densities: list[float] = []
    pair_weights: list[float] = []
    cluster_values: list[float] = []
    cluster_weights: list[float] = []
    retrigger_values: list[float] = []
    retrigger_weights: list[float] = []
    role_values: dict[str, dict[str, list[float]]] = {
        role: defaultdict(list) for role in ("melody", "bass", "harmony")
    }
    role_weights: dict[str, dict[str, list[float]]] = {
        role: defaultdict(list) for role in ("melody", "bass", "harmony")
    }
    matched_role_values: dict[str, dict[str, list[float]]] = {
        role: defaultdict(list) for role in ("melody", "bass", "harmony")
    }
    matched_role_weights: dict[str, dict[str, list[float]]] = {
        role: defaultdict(list) for role in ("melody", "bass", "harmony")
    }

    for pair in pairs:
        pair_weight = float(pair["weight"])
        target = normalized_target_notes(load_json(Path(pair["target"])))
        if not target:
            continue
        target_duration = max(note["time"] + note["duration"] for note in target)
        target_densities.append(len(target) / max(1.0, target_duration))
        pair_weights.append(pair_weight)
        for value in onset_cluster_sizes(target):
            cluster_values.append(float(value))
            cluster_weights.append(pair_weight)
        for value in same_pitch_gaps(target):
            if value <= 1.0:
                retrigger_values.append(value)
                retrigger_weights.append(pair_weight)
        for note in target:
            role = target_role(note)
            for field in ("midi", "duration", "velocity"):
                role_values[role][field].append(float(note[field]))
                role_weights[role][field].append(pair_weight)

        source_lookup = {
            note["sourceIndex"]: note
            for note in normalize_source_notes(load_json(Path(pair["source"])).get("notes", []))
        }
        for source_index, match in match_maps[pair["id"]].items():
            source = source_lookup.get(source_index)
            reference = match.get("reference") or {}
            if source is None:
                continue
            try:
                target_midi = int(round(float(reference["midi"])))
                target_duration = max(0.01, float(reference.get("duration", source["duration"])))
                target_velocity = clamp(float(reference.get("velocity", source["velocity"])), 0.0, 1.0)
            except (KeyError, TypeError, ValueError):
                continue
            role = arrangement_role(source)
            evidence_weight = pair_weight * (1.0 if match["exactPitch"] else 0.65)
            matched_role_values[role]["octaveShift"].append(float(target_midi - source["midi"]))
            matched_role_weights[role]["octaveShift"].append(evidence_weight)
            matched_role_values[role]["targetMidi"].append(float(target_midi))
            matched_role_weights[role]["targetMidi"].append(evidence_weight)
            matched_role_values[role]["targetDuration"].append(target_duration)
            matched_role_weights[role]["targetDuration"].append(evidence_weight)
            matched_role_values[role]["durationScale"].append(target_duration / max(0.05, source["duration"]))
            matched_role_weights[role]["durationScale"].append(evidence_weight)
            matched_role_values[role]["velocity"].append(target_velocity)
            matched_role_weights[role]["velocity"].append(evidence_weight)

    target_nps = weighted_quantile(target_densities, pair_weights, 0.5)
    maximum_cluster = round(weighted_quantile(cluster_values, cluster_weights, 0.95))
    minimum_retrigger = weighted_quantile(retrigger_values, retrigger_weights, 0.02)
    roles: dict[str, dict[str, Any]] = {}
    role_defaults = {
        "melody": (55, 88, 0.35, 0.86, 1.25, 0.08, 0.05),
        "bass": (28, 55, 0.55, 0.68, 1.80, 0.12, 0.07),
        "harmony": (45, 79, 0.45, 0.62, 1.10, 0.12, 0.06),
    }
    for role, defaults in role_defaults.items():
        minimum, maximum, duration, velocity, bridge, overlap, maximum_extension = defaults
        midi_values = matched_role_values[role]["targetMidi"] or role_values[role]["midi"]
        midi_weights = matched_role_weights[role]["targetMidi"] or role_weights[role]["midi"]
        if midi_values:
            minimum = int(round(weighted_quantile(midi_values, midi_weights, 0.04)))
            maximum = int(round(weighted_quantile(midi_values, midi_weights, 0.96)))
            preferred_center = weighted_quantile(midi_values, midi_weights, 0.5)
        else:
            preferred_center = (minimum + maximum) / 2.0
        duration_values = matched_role_values[role]["targetDuration"] or role_values[role]["duration"]
        duration_value_weights = (
            matched_role_weights[role]["targetDuration"]
            or role_weights[role]["duration"]
        )
        if duration_values:
            duration = weighted_quantile(duration_values, duration_value_weights, 0.5)
        velocity_values = matched_role_values[role]["velocity"] or role_values[role]["velocity"]
        velocity_weights = matched_role_weights[role]["velocity"] or role_weights[role]["velocity"]
        if velocity_values:
            velocity = weighted_quantile(velocity_values, velocity_weights, 0.5)
        octave_values = matched_role_values[role]["octaveShift"]
        octave_weights = matched_role_weights[role]["octaveShift"]
        octave_shift = 0
        if octave_values:
            octave_shift = round(weighted_quantile(octave_values, octave_weights, 0.5) / 12.0) * 12
        ratio_values = matched_role_values[role]["durationScale"]
        ratio_weights = matched_role_weights[role]["durationScale"]
        duration_scale = 1.0
        if ratio_values:
            duration_scale = clamp(weighted_quantile(ratio_values, ratio_weights, 0.5), 0.35, 3.0)
        roles[role] = {
            "minimumMidi": int(clamp(minimum, 21, 108)),
            "maximumMidi": int(clamp(maximum, 21, 108)),
            "preferredCenterMidi": round(float(clamp(preferred_center, 21, 108)), 3),
            "octaveShift": int(octave_shift),
            "medianDuration": round(float(clamp(duration, 0.05, 3.0)), 4),
            "durationScale": round(float(duration_scale), 4),
            "velocity": round(float(clamp(velocity, 0.05, 1.0)), 3),
            "maximumBridgeSeconds": bridge,
            "legatoOverlapSeconds": overlap,
            "maximumLegatoExtensionSeconds": maximum_extension,
        }
        if roles[role]["maximumMidi"] < roles[role]["minimumMidi"]:
            roles[role]["minimumMidi"], roles[role]["maximumMidi"] = (
                roles[role]["maximumMidi"],
                roles[role]["minimumMidi"],
            )

    style = {
        "targetNotesPerSecond": round(clamp(target_nps, 3.5, 9.5), 4),
        "maximumOnsetCluster": int(clamp(maximum_cluster, 3, 8)),
        "minimumRetriggerSeconds": round(clamp(minimum_retrigger, 0.08, 0.14), 4),
        "duplicateOnsetSeconds": 0.065,
        "defaultAutoplayReleaseSeconds": 0.62,
    }
    diagnostics = {
        "targetDensityByPair": {
            pair["id"]: round(density, 4)
            for pair, density in zip(pairs, target_densities)
        },
        "weightedMedianTargetNotesPerSecond": style["targetNotesPerSecond"],
        "weightedP95OnsetCluster": style["maximumOnsetCluster"],
        "weightedP08SamePitchGapSeconds": style["minimumRetriggerSeconds"],
    }
    return {"style": style, "roles": roles}, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--duration-prediction-weight", type=float, default=0.22)
    parser.add_argument("--density-multiplier", type=float, default=1.50)
    parser.add_argument("--expand-sparse-harmony", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    pairs = manifest.get("pairs") or []
    if len(pairs) < 2:
        raise ValueError("At least two aligned source/target pairs are required.")

    all_features: list[list[float]] = []
    all_labels: list[float] = []
    all_weights: list[float] = []
    all_training: list[bool] = []
    all_pair_ids: list[str] = []
    duration_features: list[list[float]] = []
    duration_targets: list[float] = []
    duration_weights: list[float] = []
    duration_training: list[bool] = []
    duration_pair_ids: list[str] = []
    match_maps: dict[str, dict[int, dict[str, Any]]] = {}
    pair_reports: list[dict[str, Any]] = []

    for pair in pairs:
        pair_id = str(pair["id"])
        pair_weight = float(pair.get("weight", 1.0))
        source_payload = load_json(Path(pair["source"]))
        source_notes = normalize_source_notes(source_payload.get("notes", []))
        features = raw_feature_rows(source_notes)
        duration_rows = duration_feature_rows(source_notes)
        alignment = load_json(Path(pair["alignmentReport"]))
        matches = alignment_match_map(alignment)
        training_ranges = alignment_training_ranges(alignment)
        match_maps[pair_id] = matches
        positive_count = 0
        exact_positive_count = 0
        validation_count = 0
        duration_count = 0
        ignored_count = 0
        for note, row, duration_row in zip(source_notes, features, duration_rows):
            if not note_is_in_training_range(note, training_ranges):
                ignored_count += 1
                continue
            match = matches.get(note["sourceIndex"])
            positive = match is not None
            exact = positive and match["exactPitch"]
            block = int(float(note["time"]) // DEFAULT_BLOCK_SECONDS)
            is_training = block % 5 != 4
            evidence_weight = pair_weight * (
                1.5 if exact else 0.95 if positive else 0.16
            )
            all_features.append(row)
            all_labels.append(1.0 if positive else 0.0)
            all_weights.append(evidence_weight)
            all_training.append(is_training)
            all_pair_ids.append(pair_id)
            if positive:
                try:
                    target_duration = clamp(
                        float(match["reference"].get("duration", note["duration"])),
                        0.05,
                        6.0,
                    )
                except (AttributeError, TypeError, ValueError):
                    target_duration = float(note["duration"])
                duration_features.append(duration_row)
                duration_targets.append(target_duration)
                duration_weights.append(pair_weight * (1.0 if exact else 0.65))
                duration_training.append(is_training)
                duration_pair_ids.append(pair_id)
                duration_count += 1
            positive_count += int(positive)
            exact_positive_count += int(exact)
            validation_count += int(not is_training)
        pair_reports.append(
            {
                "id": pair_id,
                "weight": pair_weight,
                "sourceNonPercussiveNotes": len(source_notes),
                "positiveMatchedSourceNotes": positive_count,
                "exactPitchPositiveSourceNotes": exact_positive_count,
                "validationNotes": validation_count,
                "durationExamples": duration_count,
                "ignoredOutsideTrustedSections": ignored_count,
                "trustedSourceRanges": len(training_ranges) if training_ranges is not None else None,
                "alignmentConfidence": alignment.get("metrics", {}).get("confidence"),
                "alignmentVerdict": alignment.get("metrics", {}).get("verdict"),
            }
        )

    feature_matrix = np.asarray(all_features, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.float64)
    sample_weights = np.asarray(all_weights, dtype=np.float64)
    training_mask = np.asarray(all_training, dtype=bool)
    validation_mask = ~training_mask
    learned_weights, means, scales, history = train_logistic_model(
        feature_matrix,
        labels,
        sample_weights,
        training_mask,
        seed=args.seed,
        iterations=args.iterations,
    )
    standardized = (feature_matrix - means) / scales
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(standardized @ learned_weights, -30.0, 30.0)))
    threshold, validation_metrics = choose_threshold(
        labels[validation_mask], probabilities[validation_mask], sample_weights[validation_mask]
    )
    training_metrics = classification_metrics(
        labels[training_mask], probabilities[training_mask], sample_weights[training_mask], threshold
    )
    duration_feature_matrix = np.asarray(duration_features, dtype=np.float64)
    duration_target_array = np.asarray(duration_targets, dtype=np.float64)
    duration_weight_array = np.asarray(duration_weights, dtype=np.float64)
    duration_training_mask = np.asarray(duration_training, dtype=bool)
    if not len(duration_feature_matrix) or not duration_training_mask.any() or duration_training_mask.all():
        raise ValueError("Duration training requires matched notes in both training and validation blocks.")
    (
        duration_learned_weights,
        duration_means,
        duration_scales,
        duration_predicted_seconds,
        duration_diagnostics,
    ) = train_duration_model(
        duration_feature_matrix,
        duration_target_array,
        duration_weight_array,
        duration_training_mask,
    )
    per_pair_metrics: dict[str, dict[str, float]] = {}
    pair_id_array = np.asarray(all_pair_ids)
    for pair in pairs:
        pair_id = str(pair["id"])
        mask = (pair_id_array == pair_id) & validation_mask
        if mask.any():
            per_pair_metrics[pair_id] = classification_metrics(
                labels[mask], probabilities[mask], sample_weights[mask], threshold
            )
    duration_pair_metrics: dict[str, dict[str, float]] = {}
    duration_pair_id_array = np.asarray(duration_pair_ids)
    duration_validation_mask = ~duration_training_mask
    for pair in pairs:
        pair_id = str(pair["id"])
        mask = (duration_pair_id_array == pair_id) & duration_validation_mask
        if mask.any():
            duration_pair_metrics[pair_id] = duration_regression_metrics(
                duration_target_array[mask],
                duration_predicted_seconds[mask],
                duration_weight_array[mask],
            )

    learned_style, style_diagnostics = build_style_profile(pairs, match_maps)
    profile: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "id": args.profile_id or manifest.get("profileId", "pianella-supervised-v001"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "routing": {
            "upstream": "MuScriptor instrument-aware transcription",
            "enabledFor": ["piano"],
            "bypassFor": ["band"],
            "fullModeUsesVoiceAsPianoMelody": True,
            "instrumentalModeExcludesVoice": True,
        },
        "selectionModel": {
            "type": "standardized-logistic-note-ranker-v1",
            "featureNames": list(FEATURE_NAMES),
            "weights": [round(float(value), 10) for value in learned_weights],
            "means": [round(float(value), 10) for value in means],
            "scales": [round(float(value), 10) for value in scales],
            "threshold": round(float(threshold), 6),
        },
        "durationModel": {
            "type": "standardized-log-duration-ridge-v1",
            "featureNames": list(DURATION_FEATURE_NAMES),
            "weights": [round(float(value), 10) for value in duration_learned_weights],
            "means": [round(float(value), 10) for value in duration_means],
            "scales": [round(float(value), 10) for value in duration_scales],
            "predictionWeight": round(clamp(args.duration_prediction_weight, 0.0, 1.0), 4),
            "minimumSeconds": 0.05,
            "maximumSeconds": 4.0,
        },
        **learned_style,
        "decoder": {
            "windowSeconds": 0.5,
            "expandSparseHarmony": bool(args.expand_sparse_harmony),
            "compactHarmonyVoicing": False,
            "preCleanupDensityMultiplier": round(clamp(args.density_multiplier, 0.6, 3.0), 4),
            "sourceDurationWeight": 1.0,
        },
        "training": {
            "manifest": str(manifest_path),
            "seed": args.seed,
            "iterations": args.iterations,
            "trainingExamples": int(training_mask.sum()),
            "validationExamples": int(validation_mask.sum()),
            "durationTrainingExamples": int(duration_training_mask.sum()),
            "durationValidationExamples": int((~duration_training_mask).sum()),
            "sourcePairs": pair_reports,
            "commercialUseAllowed": False,
            "purpose": "private research evaluation",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "polymath-piano-arranger-training-report-v1",
        "profile": str(output_path),
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "featureNames": list(FEATURE_NAMES),
        "trainingMetrics": training_metrics,
        "heldOutBlockMetrics": validation_metrics,
        "heldOutBlockMetricsByPair": per_pair_metrics,
        "durationMetrics": duration_diagnostics,
        "durationHeldOutBlockMetricsByPair": duration_pair_metrics,
        "styleDiagnostics": style_diagnostics,
        "optimizationHistory": history,
        "sourcePairs": pair_reports,
        "warning": (
            "Held-out blocks are not the same as held-out songs. Full-song route-level "
            "promotion evaluation is still required before any website integration."
        ),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": str(output_path),
                "trainingExamples": int(training_mask.sum()),
                "validationExamples": int(validation_mask.sum()),
                "trainingF1": training_metrics["f1"],
                "heldOutBlockF1": validation_metrics["f1"],
                "heldOutDurationMaeSeconds": duration_diagnostics["heldOutBlocks"]["meanAbsoluteErrorSeconds"],
                "threshold": round(threshold, 6),
                "targetNotesPerSecond": profile["style"]["targetNotesPerSecond"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
