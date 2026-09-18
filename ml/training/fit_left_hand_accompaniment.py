"""Fit a low-capacity, left-hand-only piano accompaniment adapter.

This trainer deliberately leaves the accepted main piano arranger untouched.
It learns a ranking model from trusted note-coordinate matches, then stores
reference-derived density, duration, onset-cluster, and retrigger constraints
under ``decoder.leftHandAccompaniment`` in a copied base profile.

The result is song-calibration evidence, not a claim of held-out accuracy. A
candidate must still win a sealed listening test and later transfer to other
songs before promotion.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    normalize_source_notes,
    selection_feature_rows,
)
try:
    from .train_piano_arranger_adapter import (
        classification_metrics,
        train_logistic_model,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from train_piano_arranger_adapter import (  # type: ignore
        classification_metrics,
        train_logistic_model,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def trusted_source_ranges(report: dict[str, Any]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for window in report.get("qualityWindows") or []:
        if not isinstance(window, dict) or window.get("status") != "trusted":
            continue
        try:
            start = float(window["sourceStart"])
            end = float(window["sourceEnd"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    return ranges


def in_ranges(time: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= time < end for start, end in ranges)


def notes_in_ranges(
    notes: list[dict[str, Any]], ranges: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Keep decoder statistics inside the same trusted clock as supervision."""

    selected: list[dict[str, Any]] = []
    for note in notes:
        try:
            time = float(note.get("time"))
        except (TypeError, ValueError):
            continue
        if in_ranges(time, ranges):
            selected.append(note)
    return selected


def nearest_time_distance(times: list[float], time: float) -> float:
    if not times:
        return math.inf
    position = bisect.bisect_left(times, time)
    candidates = []
    if position < len(times):
        candidates.append(abs(times[position] - time))
    if position > 0:
        candidates.append(abs(times[position - 1] - time))
    return min(candidates, default=math.inf)


def target_onset_groups(
    notes: list[dict[str, Any]], tolerance: float = 0.035
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for note in sorted(notes, key=lambda item: (float(item["time"]), int(item["midi"]))):
        if not groups or float(note["time"]) - float(groups[-1][0]["time"]) > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    return [
        {
            "time": float(group[0]["time"]),
            "size": len(group),
            "pitchClasses": {int(note["midi"]) % 12 for note in group},
            "pitches": {int(note["midi"]) for note in group},
        }
        for group in groups
    ]


def nearest_target_onset(
    groups: list[dict[str, Any]], group_times: list[float], time: float
) -> tuple[dict[str, Any] | None, float]:
    if not groups:
        return None, math.inf
    position = bisect.bisect_left(group_times, time)
    candidates = [
        index
        for index in (position - 1, position)
        if 0 <= index < len(groups)
    ]
    index = min(candidates, key=lambda item: abs(group_times[item] - time))
    return groups[index], abs(group_times[index] - time)


def strongest_left_matches(
    report: dict[str, Any],
    ranges: list[tuple[float, float]],
    reference_hand_split: int,
) -> dict[int, dict[str, Any]]:
    matches: dict[int, dict[str, Any]] = {}
    for match in report.get("matches") or []:
        observed = match.get("observed") or {}
        reference = match.get("reference") or {}
        try:
            source_index = int(observed["sourceIndex"])
            observed_time = float(observed["time"])
            reference_midi = int(round(float(reference["midi"])))
        except (KeyError, TypeError, ValueError):
            continue
        if reference_midi >= reference_hand_split or not in_ranges(observed_time, ranges):
            continue
        previous = matches.get(source_index)
        quality = (
            int(bool(match.get("exactPitch"))),
            -abs(float(match.get("coarseResidual", 99.0))),
        )
        if previous is None or quality > previous["quality"]:
            matches[source_index] = {"quality": quality, "match": match}
    return {source_index: item["match"] for source_index, item in matches.items()}


def fast_retrigger_count(notes: list[dict[str, Any]]) -> int:
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(float(note["time"]))
    count = 0
    for times in by_pitch.values():
        times.sort()
        for previous, current in zip(times, times[1:]):
            gap = current - previous
            if 0.065 <= gap < 0.18:
                count += 1
    return count


def model_payload(
    feature_names: tuple[str, ...],
    weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return {
        "type": "standardized-logistic-left-hand-ranker-v2",
        "featureNames": list(feature_names),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "threshold": round(float(threshold), 4),
    }


def build_pairwise_ranking_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    training_mask: np.ndarray,
    times: np.ndarray,
    *,
    maximum_negatives_per_positive: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn note labels into local positive-vs-negative ranking examples.

    The decoder keeps a quota of notes from each short window, so global
    classification accuracy is not its real task.  These symmetric pairs ask
    the model the question the decoder actually needs answered: *which of two
    nearby candidate notes better resembles the pianist's chosen note?*
    """

    if len(features) != len(labels) or len(labels) != len(times):
        raise ValueError("Pairwise ranking inputs must have equal lengths.")
    pairs: list[np.ndarray] = []
    pair_labels: list[float] = []
    pair_weights: list[float] = []
    pair_training: list[bool] = []
    positive_indices = np.flatnonzero(labels >= 0.5)
    negative_indices = np.flatnonzero(labels < 0.5)
    for positive_index in positive_indices:
        positive_time = float(times[positive_index])
        same_split = training_mask[negative_indices] == training_mask[positive_index]
        local_negatives = negative_indices[
            same_split
            & (np.abs(times[negative_indices] - positive_time) <= 0.08)
        ]
        ranked_negatives = sorted(
            (int(index) for index in local_negatives),
            key=lambda index: (abs(float(times[index]) - positive_time), index),
        )
        if len(ranked_negatives) < maximum_negatives_per_positive:
            window = int(positive_time // 4.0)
            window_negatives = negative_indices[
                same_split
                & ((times[negative_indices] // 4.0).astype(np.int64) == window)
            ]
            for index in sorted(
                (int(item) for item in window_negatives),
                key=lambda item: (abs(float(times[item]) - positive_time), item),
            ):
                if index not in ranked_negatives:
                    ranked_negatives.append(index)
                if len(ranked_negatives) >= maximum_negatives_per_positive:
                    break
        for negative_index in ranked_negatives[:maximum_negatives_per_positive]:
            difference = features[positive_index] - features[negative_index]
            weight = max(1.0, float(sample_weights[positive_index]))
            is_training = bool(training_mask[positive_index])
            pairs.extend((difference, -difference))
            pair_labels.extend((1.0, 0.0))
            pair_weights.extend((weight, weight))
            pair_training.extend((is_training, is_training))
    if not pairs:
        raise ValueError("No local positive-negative ranking pairs were available.")
    return (
        np.asarray(pairs, dtype=np.float64),
        np.asarray(pair_labels, dtype=np.float64),
        np.asarray(pair_weights, dtype=np.float64),
        np.asarray(pair_training, dtype=bool),
    )


def train_mlp_model(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    training_mask: np.ndarray,
    *,
    seed: int,
    iterations: int,
    hidden_size: int = 16,
    learning_rate: float = 0.004,
    l2: float = 0.0015,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, float]]]:
    """Train a tiny deterministic nonlinear ranker with Adam.

    This is deliberately small enough to execute with Python's standard
    library at inference time.  NumPy is required only for offline fitting.
    """

    means = features[training_mask].mean(axis=0)
    scales = features[training_mask].std(axis=0)
    scales = np.where(scales < 1e-7, 1.0, scales)
    standardized = np.clip((features - means) / scales, -8.0, 8.0)
    train_indices = np.flatnonzero(training_mask)
    validation_mask = ~training_mask
    rng = np.random.default_rng(seed)
    input_size = standardized.shape[1]
    hidden_weights = rng.normal(
        0.0, math.sqrt(2.0 / max(1, input_size + hidden_size)),
        size=(input_size, hidden_size),
    )
    hidden_biases = np.zeros(hidden_size, dtype=np.float64)
    output_weights = rng.normal(0.0, 0.12, size=hidden_size)
    output_bias = np.zeros(1, dtype=np.float64)
    parameters = [hidden_weights, hidden_biases, output_weights, output_bias]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    batch_size = min(384, len(train_indices))
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_parameters = [parameter.copy() for parameter in parameters]

    def probabilities(indices: np.ndarray) -> np.ndarray:
        hidden = np.tanh(
            standardized[indices] @ hidden_weights + hidden_biases
        )
        logits = hidden @ output_weights + float(output_bias[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def weighted_loss(mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        predicted = np.clip(probabilities(indices), 1e-7, 1.0 - 1e-7)
        weights = sample_weights[indices]
        losses = -(
            labels[indices] * np.log(predicted)
            + (1.0 - labels[indices]) * np.log(1.0 - predicted)
        )
        return float(np.sum(weights * losses) / max(1e-9, np.sum(weights)))

    for iteration in range(1, iterations + 1):
        batch = rng.choice(train_indices, size=batch_size, replace=False)
        x = standardized[batch]
        y = labels[batch]
        weights = sample_weights[batch]
        hidden = np.tanh(x @ hidden_weights + hidden_biases)
        logits = hidden @ output_weights + float(output_bias[0])
        predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (predicted - y) * weights / max(1e-9, float(weights.sum()))
        gradient_output_weights = hidden.T @ error + l2 * output_weights
        gradient_output_bias = np.asarray([error.sum()], dtype=np.float64)
        hidden_error = (
            error[:, None] * output_weights[None, :] * (1.0 - hidden * hidden)
        )
        gradient_hidden_weights = x.T @ hidden_error + l2 * hidden_weights
        gradient_hidden_biases = hidden_error.sum(axis=0)
        gradients = [
            gradient_hidden_weights,
            gradient_hidden_biases,
            gradient_output_weights,
            gradient_output_bias,
        ]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * gradient
            second_moments[index] = beta2 * second_moments[index] + (1.0 - beta2) * (gradient * gradient)
            first_hat = first_moments[index] / (1.0 - beta1 ** iteration)
            second_hat = second_moments[index] / (1.0 - beta2 ** iteration)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            training_loss = weighted_loss(training_mask)
            validation_loss = weighted_loss(validation_mask)
            history.append(
                {
                    "iteration": float(iteration),
                    "trainingWeightedBinaryCrossEntropy": round(training_loss, 8),
                    "validationWeightedBinaryCrossEntropy": round(validation_loss, 8),
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_parameters = [parameter.copy() for parameter in parameters]

    hidden_weights[:], hidden_biases[:], output_weights[:], output_bias[:] = best_parameters
    all_probabilities = probabilities(np.arange(len(features)))
    payload = {
        "type": "standardized-mlp-left-hand-ranker-v1",
        "featureNames": list(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES),
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "hiddenWeights": [
            [round(float(value), 10) for value in row]
            for row in hidden_weights
        ],
        "hiddenBiases": [round(float(value), 10) for value in hidden_biases],
        "outputWeights": [round(float(value), 10) for value in output_weights],
        "outputBias": round(float(output_bias[0]), 10),
        "hiddenActivation": "tanh",
        "hiddenSize": hidden_size,
    }
    return payload, all_probabilities, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--aligned-target", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument(
        "--duration-calibration",
        default="",
        help="Optional provisional raw-rebuild output used to calibrate post-selection source-duration quantiles.",
    )
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--output-hand-split", type=int, default=72)
    parser.add_argument("--register-shift", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0x4C454654)
    parser.add_argument(
        "--training-scope",
        choices=("baseline", "raw", "supplemental"),
        default="baseline",
        help="Fit only the frozen baseline's candidates or rebuild accompaniment from raw events.",
    )
    parser.add_argument(
        "--objective",
        choices=("classification", "pairwise"),
        default="classification",
        help="Pairwise directly trains the local ranking used by the windowed decoder.",
    )
    parser.add_argument(
        "--model-type",
        choices=("linear", "mlp"),
        default="linear",
        help="MLP captures nonlinear chord-context interactions; linear remains the conservative default.",
    )
    parser.add_argument(
        "--label-policy",
        choices=(
            "alignment-match",
            "pitch-class-proximity",
            "nearest-chord-proximity",
        ),
        default="alignment-match",
        help="Proximity rewards all plausible notes near the aligned ideal, not only the aligner's single chosen source event.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("frozen-baseline", "raw-rebuild", "hybrid"),
        default="",
        help="Optional decoder mode override; hybrid fits on the baseline but lets strong raw events compete for its slots.",
    )
    parser.add_argument(
        "--supplemental-profile",
        default="",
        help="Optional profile whose selection model ranks raw events absent from the frozen baseline.",
    )
    parser.add_argument(
        "--onset-ranking-profile",
        default="",
        help="Optional accepted profile whose model remains responsible for onset timing while this fit ranks notes inside each chord.",
    )
    parser.add_argument(
        "--chord-completion-profile",
        default="",
        help="Optional raw-event ranker used to rebuild chord tones at already-selected accompaniment onsets.",
    )
    parser.add_argument(
        "--target-onset-keep-ratio",
        type=float,
        default=None,
        help="Optional calibrated onset-retention override embedded in the output profile.",
    )
    parser.add_argument(
        "--enforce-size-distribution",
        action="store_true",
        help="Embed the aligned ideal's one/two/three-note chord-size distribution.",
    )
    parser.add_argument(
        "--fast-retrigger-keep-share",
        type=float,
        default=None,
        help="Optional calibrated share of intentional fast accompaniment repeats.",
    )
    parser.add_argument(
        "--chord-interval-prior-strength",
        type=float,
        default=0.0,
        help="Optional weight for the aligned pianist's transposition-invariant chord-interval preferences.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    alignment_path = Path(args.alignment).resolve()
    aligned_target_path = Path(args.aligned_target).resolve()
    baseline_path = Path(args.baseline).resolve()
    base_profile_path = Path(args.base_profile).resolve()
    source_payload = load_json(source_path)
    alignment = load_json(alignment_path)
    aligned_target = load_json(aligned_target_path)
    baseline = load_json(baseline_path)
    profile = load_json(base_profile_path)

    source_notes = normalize_source_notes(source_payload.get("notes", []))
    feature_names = tuple(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
    rows = selection_feature_rows(source_notes, feature_names)
    ranges = trusted_source_ranges(alignment)
    if not ranges:
        raise ValueError("The alignment contains no trusted source-time windows.")
    matches = strongest_left_matches(
        alignment, ranges, args.reference_hand_split
    )
    target_left = [
        note
        for note in aligned_target.get("notes", [])
        if bool(note.get("trainingEligible"))
        and int(note.get("midi", 999)) < args.reference_hand_split
    ]
    target_times_by_pitch_class: dict[int, list[float]] = defaultdict(list)
    target_times_by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in target_left:
        target_times_by_pitch_class[int(note["midi"]) % 12].append(float(note["time"]))
        target_times_by_pitch[int(note["midi"])].append(float(note["time"]))
    for times in target_times_by_pitch_class.values():
        times.sort()
    for times in target_times_by_pitch.values():
        times.sort()
    target_groups = target_onset_groups(target_left)
    target_group_times = [float(group["time"]) for group in target_groups]
    baseline_left_source_indices = {
        int(note["sourceIndex"])
        for note in baseline.get("notes", [])
        if int(note.get("midi", 999)) < args.output_hand_split
        and note.get("arrangementRole") != "melody"
        and note.get("sourceIndex") is not None
        and in_ranges(float(note.get("time", -1.0)), ranges)
    }

    maximum_source_midi = args.output_hand_split - args.register_shift
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    sample_weights: list[float] = []
    training_flags: list[bool] = []
    source_indices: list[int] = []
    example_times: list[float] = []
    for note, row in zip(source_notes, rows):
        time = float(note["time"])
        if args.training_scope == "baseline":
            outside_scope = (
                int(note["midi"]) >= maximum_source_midi
                or int(note["sourceIndex"]) not in baseline_left_source_indices
            )
        elif args.training_scope == "supplemental":
            outside_scope = int(note["sourceIndex"]) in baseline_left_source_indices
        else:
            outside_scope = False
        if outside_scope or not in_ranges(time, ranges):
            continue
        match = matches.get(int(note["sourceIndex"]))
        if args.label_policy == "nearest-chord-proximity":
            target_group, onset_distance = nearest_target_onset(
                target_groups, target_group_times, time
            )
            positive = bool(
                target_group
                and onset_distance <= 0.25
                and int(note["midi"]) % 12 in target_group["pitchClasses"]
            )
            exact = bool(
                positive
                and int(note["midi"]) in target_group["pitches"]
                and onset_distance <= 0.10
            )
            positive_weight = (
                1.35
                if exact
                else 1.18
                if onset_distance <= 0.10
                else 1.0
            )
        elif args.label_policy == "pitch-class-proximity":
            pitch_class_distance = nearest_time_distance(
                target_times_by_pitch_class[int(note["midi"]) % 12], time
            )
            exact_distance = nearest_time_distance(
                target_times_by_pitch[int(note["midi"])], time
            )
            positive = pitch_class_distance <= 0.25
            exact = exact_distance <= 0.10
            positive_weight = (
                1.35
                if exact
                else 1.18
                if pitch_class_distance <= 0.10
                else 1.0
            )
        else:
            positive = match is not None
            exact = bool(positive and match.get("exactPitch"))
            positive_weight = 1.35 if exact else 1.0
        feature_rows.append(row)
        labels.append(1.0 if positive else 0.0)
        sample_weights.append(positive_weight if positive else 0.28)
        # Every fifth 20-second block is held out. This tests temporal transfer
        # inside the song instead of scoring only the examples used to fit.
        training_flags.append(int(time // 20.0) % 5 != 4)
        source_indices.append(int(note["sourceIndex"]))
        example_times.append(time)

    features = np.asarray(feature_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    weights_array = np.asarray(sample_weights, dtype=np.float64)
    training_mask = np.asarray(training_flags, dtype=bool)
    validation_mask = ~training_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("Both training and held-out temporal blocks are required.")
    if label_array[training_mask].sum() < 10 or label_array[validation_mask].sum() < 5:
        raise ValueError("Insufficient positive left-hand examples for a safe fit.")

    optimization_features = features
    optimization_labels = label_array
    optimization_weights = weights_array
    optimization_training_mask = training_mask
    if args.objective == "pairwise":
        if args.model_type != "linear":
            raise ValueError("Pairwise difference training currently requires --model-type linear.")
        (
            optimization_features,
            optimization_labels,
            optimization_weights,
            optimization_training_mask,
        ) = build_pairwise_ranking_dataset(
            features,
            label_array,
            weights_array,
            training_mask,
            np.asarray(example_times, dtype=np.float64),
        )
    optimization_validation_mask = ~optimization_training_mask

    learned_model: dict[str, Any]
    if args.model_type == "mlp":
        learned_model, probabilities, history = train_mlp_model(
            optimization_features,
            optimization_labels,
            optimization_weights,
            optimization_training_mask,
            seed=args.seed,
            iterations=args.iterations,
        )
    else:
        learned_weights, means, scales, history = train_logistic_model(
            optimization_features,
            optimization_labels,
            optimization_weights,
            optimization_training_mask,
            seed=args.seed,
            iterations=args.iterations,
            learning_rate=0.018,
            l2=0.008,
        )
        standardized = (optimization_features - means) / scales
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(standardized @ learned_weights, -30.0, 30.0))
        )
        learned_model = model_payload(
            feature_names, learned_weights, means, scales, 0.5
        )
    threshold_results = [
        classification_metrics(
            optimization_labels[optimization_validation_mask],
            probabilities[optimization_validation_mask],
            optimization_weights[optimization_validation_mask],
            float(threshold),
        )
        for threshold in np.linspace(0.10, 0.90, 161)
    ]
    best_threshold_metrics = max(
        threshold_results,
        key=lambda item: (item["f1"], item["precision"], item["recall"]),
    )
    threshold = float(best_threshold_metrics["threshold"])
    learned_model["threshold"] = round(threshold, 4)
    training_metrics = classification_metrics(
        optimization_labels[optimization_training_mask],
        probabilities[optimization_training_mask],
        optimization_weights[optimization_training_mask],
        threshold,
    )

    baseline_left = notes_in_ranges([
        note
        for note in baseline.get("notes", [])
        if int(note.get("midi", 999)) < args.output_hand_split
    ], ranges)
    baseline_left_non_voice = [
        note
        for note in baseline_left
        if str(note.get("sourceInstrument") or "").lower() != "voice"
    ]
    if not target_left or not baseline_left_non_voice:
        raise ValueError("Aligned target and baseline must both contain left-hand notes.")

    if args.training_scope in {"raw", "supplemental"}:
        source_style_notes = notes_in_ranges([
            note
            for note in source_notes
            if str(note.get("instrument") or "").lower() != "voice"
        ], ranges)
    else:
        source_style_notes = baseline_left_non_voice

    duration_source_values = [
        float(note["duration"]) for note in source_style_notes
    ]
    duration_calibration_path: Path | None = None
    if args.duration_calibration:
        duration_calibration_path = Path(args.duration_calibration).resolve()
        calibration = load_json(duration_calibration_path)
        calibrated_values = [
            float(note["sourceDurationBeforeLeftHandStyle"])
            for note in calibration.get("notes", [])
            if note.get("sourceDurationBeforeLeftHandStyle") is not None
            and note.get("arrangementRole") != "melody"
            and int(note.get("midi", 999)) < args.output_hand_split
            and in_ranges(float(note.get("time", -1.0)), ranges)
        ]
        if calibrated_values:
            duration_source_values = calibrated_values

    source_duration_quantiles = [
        quantile(duration_source_values, fraction)
        for fraction in (0.10, 0.50, 0.90)
    ]
    target_duration_quantiles = [
        quantile([float(note["duration"]) for note in target_left], fraction)
        for fraction in (0.10, 0.50, 0.90)
    ]
    baseline_fast = fast_retrigger_count(baseline_left_non_voice)
    target_fast = fast_retrigger_count(target_left)
    target_raw_keep_ratio = len(target_left) / max(1, len(source_style_notes))
    # The decoder revoices most redundant repeats into alternate chord tones,
    # so only a small compensation is needed for onset caps and the few events
    # that have no harmonically supported replacement.
    minimum_keep_ratio = 0.10 if args.training_scope != "baseline" else 0.50
    density_compensation = 0.018 if args.training_scope != "baseline" else 0.10
    compensated_keep_ratio = min(
        0.96, max(minimum_keep_ratio, target_raw_keep_ratio + density_compensation)
    )
    retrigger_keep_share = min(1.0, target_fast / max(1, baseline_fast))

    source_mode = args.source_mode or (
        "raw-rebuild" if args.training_scope != "baseline" else "frozen-baseline"
    )
    supplemental_profile_path: Path | None = None
    supplemental_selection_model: dict[str, Any] | None = None
    if args.supplemental_profile:
        supplemental_profile_path = Path(args.supplemental_profile).resolve()
        supplemental_profile = load_json(supplemental_profile_path)
        supplemental_selection_model = supplemental_profile.get("selectionModel") or (
            supplemental_profile.get("decoder", {})
            .get("leftHandAccompaniment", {})
            .get("selectionModel")
        )
        if not supplemental_selection_model:
            raise ValueError("Supplemental profile contains no selection model.")
    onset_profile_path: Path | None = None
    onset_selection_model: dict[str, Any] | None = None
    if args.onset_ranking_profile:
        onset_profile_path = Path(args.onset_ranking_profile).resolve()
        onset_profile = load_json(onset_profile_path)
        onset_selection_model = onset_profile.get("selectionModel") or (
            onset_profile.get("decoder", {})
            .get("leftHandAccompaniment", {})
            .get("selectionModel")
        )
        if not onset_selection_model:
            raise ValueError("Onset-ranking profile contains no selection model.")
    chord_completion_profile_path: Path | None = None
    chord_completion_model: dict[str, Any] | None = None
    if args.chord_completion_profile:
        chord_completion_profile_path = Path(args.chord_completion_profile).resolve()
        chord_completion_profile = load_json(chord_completion_profile_path)
        chord_completion_model = chord_completion_profile.get("selectionModel") or (
            chord_completion_profile.get("decoder", {})
            .get("leftHandAccompaniment", {})
            .get("selectionModel")
        )
        if not chord_completion_model:
            raise ValueError("Chord-completion profile contains no selection model.")
    target_onsets = len(
        {int(round(float(note["time"]) * 1000)) for note in target_left}
    )
    baseline_onsets = len(
        {
            int(round(float(note["time"]) * 1000))
            for note in baseline_left_non_voice
        }
    )
    target_onset_keep_ratio = min(
        0.98, target_onsets / max(1, baseline_onsets) + 0.02
    )
    if args.target_onset_keep_ratio is not None:
        target_onset_keep_ratio = min(
            1.0, max(0.1, float(args.target_onset_keep_ratio))
        )
    if args.fast_retrigger_keep_share is not None:
        retrigger_keep_share = min(
            1.0, max(0.0, float(args.fast_retrigger_keep_share))
        )
    target_notes_per_onset = len(target_left) / max(1, target_onsets)
    target_size_counts = Counter(
        min(3, int(group["size"])) for group in target_groups
    )
    target_interval_counts: Counter[int] = Counter()
    for group in target_groups:
        pitches = sorted(int(value) for value in group["pitches"])
        if len(pitches) < 2:
            continue
        bass = pitches[0]
        for pitch in pitches[1:]:
            target_interval_counts[(pitch - bass) % 12] += 1
    maximum_interval_count = max(
        (count for interval, count in target_interval_counts.items() if interval),
        default=1,
    )
    target_interval_preferences = {
        str(interval): round(target_interval_counts.get(interval, 0) / maximum_interval_count, 6)
        for interval in range(1, 12)
    }
    left_config = {
        "enabled": True,
        "profile": args.profile_id,
        "sourceMode": source_mode,
        "rightHandFrozen": True,
        "handSplitMidi": args.output_hand_split,
        "referenceHandSplitMidi": args.reference_hand_split,
        "registerShiftSemitones": args.register_shift,
        "rebuildMinimumMidi": args.output_hand_split - 26,
        "rebuildMaximumMidi": args.output_hand_split - 2,
        "rebuildVelocityRange": [0.38, 0.50],
        "supplementalMinimumProbability": round(threshold, 4),
        "windowSeconds": 4.0,
        # A dedicated gesture model makes raw rebuilding safe to rank by
        # complete attacks.  Without one, raw mode retains the legacy
        # note-by-note quota behavior for reproducibility.
        "onsetFirstSelection": (
            source_mode in {"frozen-baseline", "hybrid"}
            or onset_selection_model is not None
        ),
        "targetOnsetKeepRatio": round(target_onset_keep_ratio, 4),
        "targetNotesPerOnset": round(target_notes_per_onset, 4),
        "targetOnsetSizeShares": {
            str(size): round(
                target_size_counts.get(size, 0) / max(1, len(target_groups)), 6
            )
            for size in (1, 2, 3)
        },
        "enforceTargetNotesPerOnset": (
            args.enforce_size_distribution or source_mode == "hybrid"
        ),
        "targetKeepRatio": round(compensated_keep_ratio, 4),
        "uncompensatedReferenceKeepRatio": round(target_raw_keep_ratio, 4),
        "minimumNotesPerWindow": 1,
        "maximumNotesPerOnset": 3,
        "minimumSamePitchGapSeconds": 0.18,
        "fastRetriggerKeepShare": round(retrigger_keep_share, 4),
        "revoiceRapidRetriggers": True,
        "rapidRevoiceSupportRadiusSeconds": 0.35,
        "removeVoiceBelowSplit": False,
        "preserveMelodyBelowSplit": True,
        "bassAnchorBonus": 0.05,
        "sourceDurationQuantiles": [round(value, 6) for value in source_duration_quantiles],
        "targetDurationQuantiles": [round(value, 6) for value in target_duration_quantiles],
        "selectionModel": learned_model,
    }
    if onset_selection_model is not None:
        left_config["onsetSelectionModel"] = onset_selection_model
    if chord_completion_model is not None:
        left_config.update(
            {
                "chordCompletionModel": chord_completion_model,
                "chordCompletionRadiusSeconds": 0.08,
                "chordCompletionExistingPitchBonus": 0.035,
                "chordCompletionCrossFamilyBonus": 0.025,
                "chordCompletionMinimumGain": 0.16,
                "chordCompletionIntervalPreferences": target_interval_preferences,
                "chordCompletionIntervalPriorStrength": min(
                    0.50, max(0.0, float(args.chord_interval_prior_strength))
                ),
            }
        )
    if supplemental_selection_model is not None:
        left_config["supplementalSelectionModel"] = supplemental_selection_model
        left_config["supplementalMinimumProbability"] = round(
            float(supplemental_selection_model.get("threshold", 0.5)), 4
        )
    profile["id"] = args.profile_id
    profile["createdAt"] = datetime.now(timezone.utc).isoformat()
    profile.setdefault("decoder", {})["leftHandAccompaniment"] = left_config
    profile.setdefault("training", {})["leftHandAccompaniment"] = {
        "schema": "polymath-left-hand-accompaniment-training-v1",
        "method": "trusted-coordinate-left-hand-ranking-plus-performance-statistics",
        "trainingScope": args.training_scope,
        "objective": args.objective,
        "labelPolicy": args.label_policy,
        "modelType": args.model_type,
        "source": str(source_path),
        "alignment": str(alignment_path),
        "alignedTarget": str(aligned_target_path),
        "baseline": str(baseline_path),
        "durationCalibration": (
            str(duration_calibration_path) if duration_calibration_path else None
        ),
        "supplementalProfile": (
            str(supplemental_profile_path) if supplemental_profile_path else None
        ),
        "onsetRankingProfile": str(onset_profile_path) if onset_profile_path else None,
        "chordCompletionProfile": (
            str(chord_completion_profile_path)
            if chord_completion_profile_path
            else None
        ),
        "baseProfile": str(base_profile_path),
        "trainingExamples": int(training_mask.sum()),
        "validationExamples": int(validation_mask.sum()),
        "trustedWindowCount": len(ranges),
        "warning": "Kiss Me is calibration data; transfer must be tested on untouched songs before production promotion.",
        "commercialUseAllowed": False,
        "purpose": "private research evaluation",
    }
    profile.pop("profileSha256", None)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()

    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_payload = {
        "schema": "polymath-left-hand-accompaniment-training-report-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "profile": str(output_path),
        "profileId": args.profile_id,
        "profileSha256": profile["profileSha256"],
        "data": {
            "trustedWindows": len(ranges),
            "eligibleSourceExamples": len(labels),
            "positiveExamples": int(label_array.sum()),
            "baselineEligibleSourceIndices": len(baseline_left_source_indices),
            "trainingExamples": int(training_mask.sum()),
            "heldOutExamples": int(validation_mask.sum()),
            "optimizationExamples": len(optimization_labels),
            "optimizationTrainingExamples": int(optimization_training_mask.sum()),
            "optimizationHeldOutExamples": int(optimization_validation_mask.sum()),
            "targetLeftNotes": len(target_left),
            "baselineLeftNotes": len(baseline_left),
            "baselineNonVoiceLeftNotes": len(baseline_left_non_voice),
            "sourceStyleNotes": len(source_style_notes),
            "trainingScope": args.training_scope,
            "objective": args.objective,
            "labelPolicy": args.label_policy,
            "modelType": args.model_type,
        },
        "model": {
            "trainingMetrics": training_metrics,
            "heldOutTemporalBlockMetrics": best_threshold_metrics,
            "optimizationHistory": history,
            "featureNames": list(feature_names),
        },
        "learnedPerformance": {
            "sourceDurationQuantiles": left_config["sourceDurationQuantiles"],
            "targetDurationQuantiles": left_config["targetDurationQuantiles"],
            "baselineFastRetriggers": baseline_fast,
            "targetFastRetriggers": target_fast,
            "fastRetriggerKeepShare": left_config["fastRetriggerKeepShare"],
            "targetKeepRatio": left_config["targetKeepRatio"],
            "targetOnsetKeepRatio": left_config["targetOnsetKeepRatio"],
            "targetOnsets": target_onsets,
            "baselineOnsets": baseline_onsets,
            "targetNotesPerOnset": left_config["targetNotesPerOnset"],
            "chordCompletionIntervalPreferences": target_interval_preferences,
            "chordCompletionIntervalPriorStrength": left_config.get(
                "chordCompletionIntervalPriorStrength", 0.0
            ),
            "uncompensatedReferenceKeepRatio": left_config[
                "uncompensatedReferenceKeepRatio"
            ],
        },
        "warning": "These are song-calibration and temporal-block metrics, not cross-song accuracy.",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
