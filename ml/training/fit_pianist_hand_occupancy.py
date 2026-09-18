"""Fit a conservative model for excessive two-hand layering.

The current piano reducer often gets the number of physical strikes right but
plays both hands where an approved pianist uses the left hand alone.  This
trainer learns that *hand occupancy* decision from complete-song-separated
examples.  It can only suppress an existing right-hand layer; it never changes
onsets, moves pitches, invents notes, or edits a single-hand melody.

The output is an experimental JSON profile.  Whole-song leave-one-out results
must pass before the profile is considered for runtime integration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:  # Support module and direct-script execution.
    from .analyze_pianist_gesture_patterns import (
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        pitch_set,
        set_f1,
        trusted_source_ranges,
    )
    from .analyze_pianist_reduction_grammar import explicit_hand, family, occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .train_piano_arranger_adapter import train_logistic_model
except ImportError:  # pragma: no cover - direct CLI path.
    from analyze_pianist_gesture_patterns import (
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        pitch_set,
        set_f1,
        trusted_source_ranges,
    )
    from analyze_pianist_reduction_grammar import explicit_hand, family, occupancy
    from analyze_pianist_sequence_alignment import align_sequences

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ml.training.train_piano_arranger_adapter import train_logistic_model


HAND_OCCUPANCY_FEATURE_NAMES = (
    "bias",
    "current_left_only",
    "current_right_only",
    "current_both",
    "left_note_count",
    "right_note_count",
    "left_pitch_class_count",
    "right_pitch_class_count",
    "shared_pitch_class_count",
    "right_melody_share",
    "right_harmony_share",
    "left_bass_share",
    "left_harmony_share",
    "right_voice_source_share",
    "right_guitar_source_share",
    "right_bass_source_share",
    "left_voice_source_share",
    "left_guitar_source_share",
    "left_bass_source_share",
    "right_velocity",
    "left_velocity",
    "right_source_velocity",
    "left_source_velocity",
    "right_duration_expression",
    "left_duration_expression",
    "right_selection_probability",
    "left_selection_probability",
    "right_mean_midi",
    "left_mean_midi",
    "right_pitch_span",
    "left_pitch_span",
    "previous_gap_expression",
    "next_gap_expression",
    "previous_left_only",
    "previous_right_only",
    "previous_both",
    "next_left_only",
    "next_right_only",
    "next_both",
    "previous_right_pc_similarity",
    "next_right_pc_similarity",
    "previous_left_pc_similarity",
    "next_left_pc_similarity",
    "local_both_share",
    "local_left_only_share",
    "local_right_only_share",
    "local_mean_notes_per_gesture",
    "song_progress",
    "ordinal_parity",
    "ordinal_mod4_sin",
    "ordinal_mod4_cos",
    "run_mod4_sin",
    "run_mod4_cos",
    "run_mod8_sin",
    "run_mod8_cos",
)


def rounded(value: float) -> float:
    return round(float(value), 6)


def parse_float_grid(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("Expected finite non-negative values")
    return values


def hand_notes(group: list[dict[str, Any]], hand: str) -> list[dict[str, Any]]:
    return [note for note in group if explicit_hand(note) == hand]


def share(notes: list[dict[str, Any]], predicate) -> float:
    return sum(bool(predicate(note)) for note in notes) / max(1, len(notes))


def note_median(notes: list[dict[str, Any]], key: str, fallback: float) -> float:
    values = [finite(note.get(key), fallback) for note in notes]
    return float(median(values)) if values else fallback


def source_velocity(note: dict[str, Any]) -> float:
    return finite(note.get("sourceVelocityBeforeArrangement"), finite(note.get("velocity"), 0.72))


def selection_probability(note: dict[str, Any]) -> float:
    return finite(
        note.get("selectionProbability", note.get("leftHandSelectionProbability")),
        0.5,
    )


def duration_expression(notes: list[dict[str, Any]]) -> float:
    value = note_median(notes, "scoreDuration", note_median(notes, "duration", 0.2))
    return max(0.0, min(1.0, math.log1p(max(0.03, value)) / math.log1p(1.5)))


def hand_pitch_classes(group: list[dict[str, Any]], hand: str) -> set[int]:
    return {int(note["midi"]) % 12 for note in hand_notes(group, hand)}


def set_similarity(left: set[int], right: set[int]) -> float:
    return set_f1(left, right) if left or right else 1.0


def sequence_contexts(groups: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    contexts: list[dict[str, float]] = []
    run_index = 0
    for index, group in enumerate(groups):
        time = float(group[0]["time"])
        previous_gap = (
            time - float(groups[index - 1][0]["time"]) if index else 9.0
        )
        next_gap = (
            float(groups[index + 1][0]["time"]) - time
            if index + 1 < len(groups)
            else 9.0
        )
        if index == 0 or previous_gap >= 0.75:
            run_index = 0
        else:
            run_index += 1
        local = [
            observed
            for observed in groups
            if abs(float(observed[0]["time"]) - time) <= 2.0
        ]
        local_counts = Counter(occupancy(observed) for observed in local)
        contexts.append(
            {
                "previousGap": previous_gap,
                "nextGap": next_gap,
                "runIndex": float(run_index),
                "localBothShare": local_counts["both"] / max(1, len(local)),
                "localLeftOnlyShare": local_counts["left-only"] / max(1, len(local)),
                "localRightOnlyShare": local_counts["right-only"] / max(1, len(local)),
                "localMeanNotes": sum(len(observed) for observed in local)
                / max(1, len(local)),
            }
        )
    return contexts


def hand_occupancy_feature_map(
    groups: list[list[dict[str, Any]]],
    index: int,
    contexts: list[dict[str, float]],
) -> dict[str, float]:
    group = groups[index]
    left = hand_notes(group, "left")
    right = hand_notes(group, "right")
    left_pcs = hand_pitch_classes(group, "left")
    right_pcs = hand_pitch_classes(group, "right")
    previous = groups[index - 1] if index else []
    following = groups[index + 1] if index + 1 < len(groups) else []
    previous_occupancy = occupancy(previous) if previous else ""
    next_occupancy = occupancy(following) if following else ""
    current_occupancy = occupancy(group)
    context = contexts[index]

    def mean_midi(notes: list[dict[str, Any]], fallback: float) -> float:
        return sum(int(note["midi"]) for note in notes) / max(1, len(notes)) if notes else fallback

    def pitch_span(notes: list[dict[str, Any]]) -> float:
        midis = [int(note["midi"]) for note in notes]
        return (max(midis) - min(midis)) / 24.0 if len(midis) > 1 else 0.0

    ordinal = float(index)
    run_index = context["runIndex"]
    return {
        "bias": 1.0,
        "current_left_only": float(current_occupancy == "left-only"),
        "current_right_only": float(current_occupancy == "right-only"),
        "current_both": float(current_occupancy == "both"),
        "left_note_count": len(left) / 4.0,
        "right_note_count": len(right) / 4.0,
        "left_pitch_class_count": len(left_pcs) / 4.0,
        "right_pitch_class_count": len(right_pcs) / 4.0,
        "shared_pitch_class_count": len(left_pcs & right_pcs) / 3.0,
        "right_melody_share": share(right, lambda note: note.get("arrangementRole") == "melody"),
        "right_harmony_share": share(right, lambda note: note.get("arrangementRole") == "harmony"),
        "left_bass_share": share(left, lambda note: note.get("arrangementRole") == "bass"),
        "left_harmony_share": share(left, lambda note: note.get("arrangementRole") == "harmony"),
        "right_voice_source_share": share(right, lambda note: family(note.get("sourceInstrument")) == "voice"),
        "right_guitar_source_share": share(right, lambda note: family(note.get("sourceInstrument")) == "guitar"),
        "right_bass_source_share": share(right, lambda note: family(note.get("sourceInstrument")) == "bass"),
        "left_voice_source_share": share(left, lambda note: family(note.get("sourceInstrument")) == "voice"),
        "left_guitar_source_share": share(left, lambda note: family(note.get("sourceInstrument")) == "guitar"),
        "left_bass_source_share": share(left, lambda note: family(note.get("sourceInstrument")) == "bass"),
        "right_velocity": note_median(right, "velocity", 0.72),
        "left_velocity": note_median(left, "velocity", 0.72),
        "right_source_velocity": float(median([source_velocity(note) for note in right])) if right else 0.72,
        "left_source_velocity": float(median([source_velocity(note) for note in left])) if left else 0.72,
        "right_duration_expression": duration_expression(right),
        "left_duration_expression": duration_expression(left),
        "right_selection_probability": float(median([selection_probability(note) for note in right])) if right else 0.5,
        "left_selection_probability": float(median([selection_probability(note) for note in left])) if left else 0.5,
        "right_mean_midi": (mean_midi(right, 72.0) - 72.0) / 24.0,
        "left_mean_midi": (mean_midi(left, 48.0) - 48.0) / 24.0,
        "right_pitch_span": pitch_span(right),
        "left_pitch_span": pitch_span(left),
        "previous_gap_expression": min(1.0, context["previousGap"] / 0.75),
        "next_gap_expression": min(1.0, context["nextGap"] / 0.75),
        "previous_left_only": float(previous_occupancy == "left-only"),
        "previous_right_only": float(previous_occupancy == "right-only"),
        "previous_both": float(previous_occupancy == "both"),
        "next_left_only": float(next_occupancy == "left-only"),
        "next_right_only": float(next_occupancy == "right-only"),
        "next_both": float(next_occupancy == "both"),
        "previous_right_pc_similarity": set_similarity(
            right_pcs, hand_pitch_classes(previous, "right") if previous else set()
        ),
        "next_right_pc_similarity": set_similarity(
            right_pcs, hand_pitch_classes(following, "right") if following else set()
        ),
        "previous_left_pc_similarity": set_similarity(
            left_pcs, hand_pitch_classes(previous, "left") if previous else set()
        ),
        "next_left_pc_similarity": set_similarity(
            left_pcs, hand_pitch_classes(following, "left") if following else set()
        ),
        "local_both_share": context["localBothShare"],
        "local_left_only_share": context["localLeftOnlyShare"],
        "local_right_only_share": context["localRightOnlyShare"],
        "local_mean_notes_per_gesture": context["localMeanNotes"] / 6.0,
        "song_progress": index / max(1, len(groups) - 1),
        "ordinal_parity": 1.0 if index % 2 else -1.0,
        "ordinal_mod4_sin": math.sin(2.0 * math.pi * ordinal / 4.0),
        "ordinal_mod4_cos": math.cos(2.0 * math.pi * ordinal / 4.0),
        "run_mod4_sin": math.sin(2.0 * math.pi * run_index / 4.0),
        "run_mod4_cos": math.cos(2.0 * math.pi * run_index / 4.0),
        "run_mod8_sin": math.sin(2.0 * math.pi * run_index / 8.0),
        "run_mod8_cos": math.cos(2.0 * math.pi * run_index / 8.0),
    }


def load_song_sequences(
    row: dict[str, Any], onset_window: float
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, Any]]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every manifest song requires an id")
    reference_path = Path(str(row["reference"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = Path(str(row["candidate"])).resolve()
    alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    ranges = trusted_source_ranges(alignment)
    reference_end_value = finite(row.get("referenceEndSeconds"), math.nan)
    reference_end = reference_end_value if math.isfinite(reference_end_value) else None
    candidate_end_value = finite(row.get("candidateEndSeconds"), math.nan)
    candidate_end = candidate_end_value if math.isfinite(candidate_end_value) else None
    transpose = int(row.get("referenceTransposeSemitones") or 0)
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    reference = normalize_notes(
        reference_payload, transpose=transpose, hard_end=reference_end
    )
    if not bool(row.get("referenceAlreadyAligned")):
        reference = map_reference(reference, monotonic_anchors(alignment))
    reference = [
        note
        for note in reference
        if note.get("trainingEligible") is not False
        and inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    candidate = [
        note
        for note in normalize_notes(candidate_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    return group_onsets(reference, onset_window), group_onsets(candidate, onset_window), {
        "id": song_id,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "alignment": str(alignment_path),
        "referenceEndSeconds": reference_end,
        "candidateEndSeconds": candidate_end,
    }


def build_song_examples(
    row: dict[str, Any],
    *,
    onset_window: float,
    maximum_time_distance: float,
    gap_cost: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference, candidate, description = load_song_sequences(row, onset_window)
    matches, missing, extra, score = align_sequences(
        reference,
        candidate,
        maximum_time_distance=maximum_time_distance,
        gap_cost=gap_cost,
    )
    contexts = sequence_contexts(candidate)
    examples: list[dict[str, Any]] = []
    rejected_no_pitch_overlap = 0
    for reference_index, candidate_index in matches:
        target = reference[reference_index]
        observed = candidate[candidate_index]
        if occupancy(observed) != "both":
            continue
        pc_overlap = pitch_set(target, True) & pitch_set(observed, True)
        if not pc_overlap:
            rejected_no_pitch_overlap += 1
            continue
        feature_map = hand_occupancy_feature_map(candidate, candidate_index, contexts)
        examples.append(
            {
                "songId": description["id"],
                "candidateIndex": candidate_index,
                "referenceIndex": reference_index,
                "sourceTime": rounded(float(observed[0]["time"])),
                "targetOccupancy": occupancy(target),
                "label": int(occupancy(target) == "left-only"),
                "targetPitchClasses": sorted(pitch_set(target, True)),
                "candidateGroup": observed,
                "features": [
                    feature_map[name] for name in HAND_OCCUPANCY_FEATURE_NAMES
                ],
            }
        )
    description.update(
        {
            "referenceGestures": len(reference),
            "candidateGestures": len(candidate),
            "sequenceMatches": len(matches),
            "missingReferenceGestures": len(missing),
            "extraCandidateGestures": len(extra),
            "sequenceScore": rounded(score),
            "candidateBothExamples": len(examples),
            "leftOnlyTargets": sum(example["label"] for example in examples),
            "rejectedNoPitchOverlap": rejected_no_pitch_overlap,
        }
    )
    return examples, description


def flatten_training(
    examples: list[dict[str, Any]], positive_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    song_counts = Counter(str(example["songId"]) for example in examples)
    features = np.asarray([example["features"] for example in examples], dtype=float)
    labels = np.asarray([example["label"] for example in examples], dtype=float)
    song_count = max(1, len(song_counts))
    sample_weights = np.asarray(
        [
            (positive_weight if example["label"] else 1.0)
            / (song_count * song_counts[str(example["songId"])])
            for example in examples
        ],
        dtype=float,
    )
    sample_weights *= len(sample_weights) / max(1e-12, float(sample_weights.sum()))
    return features, labels, sample_weights


def fit_model(
    examples: list[dict[str, Any]],
    *,
    positive_weight: float,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    features, labels, sample_weights = flatten_training(examples, positive_weight)
    if len(examples) < 20 or labels.min() == labels.max():
        raise ValueError("Hand-occupancy training needs at least 20 mixed examples")
    weights, means, scales, history = train_logistic_model(
        features,
        labels,
        sample_weights,
        np.ones(len(labels), dtype=bool),
        seed=seed,
        iterations=iterations,
        learning_rate=0.012,
        l2=0.035,
    )
    return {
        "type": "standardized-logistic-right-layer-suppressor-v1",
        "featureNames": list(HAND_OCCUPANCY_FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "positiveTrainingWeight": rounded(positive_weight),
    }, history


def probability(example: dict[str, Any], model: dict[str, Any]) -> float:
    features = np.asarray(example["features"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    logit = float(((features - means) / np.maximum(1e-9, np.abs(scales))) @ weights)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def evaluate(
    examples: list[dict[str, Any]],
    model: dict[str, Any] | None = None,
    *,
    threshold: float = 1.1,
    preserve_melody: bool = False,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    pitch_true_positive = pitch_false_positive = pitch_false_negative = 0
    exact = occupancy_correct = suppressions = 0
    for example in examples:
        candidate = example["candidateGroup"]
        right = hand_notes(candidate, "right")
        has_melody = any(note.get("arrangementRole") == "melody" for note in right)
        suppress = bool(
            model is not None
            and probability(example, model) >= threshold
            and not (preserve_melody and has_melody)
        )
        label = bool(example["label"])
        true_positive += int(suppress and label)
        false_positive += int(suppress and not label)
        false_negative += int(not suppress and label)
        true_negative += int(not suppress and not label)
        suppressions += int(suppress)
        selected = hand_notes(candidate, "left") if suppress else candidate
        selected_pcs = pitch_set(selected, True)
        target_pcs = set(int(value) for value in example["targetPitchClasses"])
        pitch_true_positive += len(selected_pcs & target_pcs)
        pitch_false_positive += len(selected_pcs - target_pcs)
        pitch_false_negative += len(target_pcs - selected_pcs)
        exact += int(selected_pcs == target_pcs)
        predicted_occupancy = "left-only" if suppress else "both"
        occupancy_correct += int(predicted_occupancy == example["targetOccupancy"])
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    pitch_precision = pitch_true_positive / max(
        1, pitch_true_positive + pitch_false_positive
    )
    pitch_recall = pitch_true_positive / max(
        1, pitch_true_positive + pitch_false_negative
    )
    pitch_f1 = 2.0 * pitch_precision * pitch_recall / max(
        1e-9, pitch_precision + pitch_recall
    )
    return {
        "examples": len(examples),
        "positiveTargets": sum(int(example["label"]) for example in examples),
        "suppressions": suppressions,
        "trueSuppressions": true_positive,
        "wrongSuppressions": false_positive,
        "missedSuppressions": false_negative,
        "classificationPrecision": rounded(precision),
        "classificationRecall": rounded(recall),
        "classificationF1": rounded(f1),
        "occupancyAccuracy": rounded(occupancy_correct / max(1, len(examples))),
        "pitchClassPrecision": rounded(pitch_precision),
        "pitchClassRecall": rounded(pitch_recall),
        "pitchClassF1": rounded(pitch_f1),
        "exactPitchClassSets": exact,
        "exactPitchClassSetRate": rounded(exact / max(1, len(examples))),
    }


def search_cross_validation(
    examples: list[dict[str, Any]],
    *,
    positive_weight_grid: list[float],
    threshold_grid: list[float],
    preserve_melody_grid: list[bool],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    songs = sorted({str(example["songId"]) for example in examples})
    if len(songs) < 3:
        raise ValueError("At least three complete songs are required")
    baseline = {
        song: evaluate([example for example in examples if example["songId"] == song])
        for song in songs
    }
    trials: list[dict[str, Any]] = []
    for positive_weight in positive_weight_grid:
        fold_models = {
            song: fit_model(
                [example for example in examples if example["songId"] != song],
                positive_weight=positive_weight,
                iterations=iterations,
                seed=seed,
            )[0]
            for song in songs
        }
        for threshold in threshold_grid:
            for preserve_melody in preserve_melody_grid:
                folds = []
                weighted_pitch_f1 = 0.0
                weighted_occupancy = 0.0
                total = 0
                for song in songs:
                    validation = [
                        example for example in examples if example["songId"] == song
                    ]
                    result = evaluate(
                        validation,
                        fold_models[song],
                        threshold=threshold,
                        preserve_melody=preserve_melody,
                    )
                    control = baseline[song]
                    folds.append(
                        {
                            "heldOutSong": song,
                            "baseline": control,
                            "candidate": result,
                            "pitchClassF1Delta": rounded(
                                result["pitchClassF1"] - control["pitchClassF1"]
                            ),
                            "exactSetRateDelta": rounded(
                                result["exactPitchClassSetRate"]
                                - control["exactPitchClassSetRate"]
                            ),
                            "occupancyAccuracyDelta": rounded(
                                result["occupancyAccuracy"]
                                - control["occupancyAccuracy"]
                            ),
                        }
                    )
                    weighted_pitch_f1 += result["pitchClassF1"] * len(validation)
                    weighted_occupancy += result["occupancyAccuracy"] * len(validation)
                    total += len(validation)
                worst_pitch = min(row["pitchClassF1Delta"] for row in folds)
                worst_exact = min(row["exactSetRateDelta"] for row in folds)
                worst_occupancy = min(row["occupancyAccuracyDelta"] for row in folds)
                suppressions = sum(row["candidate"]["suppressions"] for row in folds)
                wrong = sum(row["candidate"]["wrongSuppressions"] for row in folds)
                precision = (suppressions - wrong) / max(1, suppressions)
                aggregate_pitch = weighted_pitch_f1 / max(1, total)
                aggregate_occupancy = weighted_occupancy / max(1, total)
                score = (
                    aggregate_pitch
                    + 0.20 * aggregate_occupancy
                    + min(0.0, worst_pitch) * 8.0
                    + min(0.0, worst_exact) * 3.0
                    + min(0.0, worst_occupancy) * 1.5
                    + min(0.0, precision - 0.70) * 0.5
                )
                trials.append(
                    {
                        "positiveTrainingWeight": positive_weight,
                        "threshold": threshold,
                        "preserveMelody": preserve_melody,
                        "suppressions": suppressions,
                        "wrongSuppressions": wrong,
                        "suppressionPrecision": rounded(precision),
                        "weightedPitchClassF1": rounded(aggregate_pitch),
                        "weightedOccupancyAccuracy": rounded(aggregate_occupancy),
                        "worstSongPitchClassF1Delta": rounded(worst_pitch),
                        "worstSongExactSetRateDelta": rounded(worst_exact),
                        "worstSongOccupancyAccuracyDelta": rounded(worst_occupancy),
                        "selectionScore": rounded(score),
                        "folds": folds,
                    }
                )
    trials.sort(
        key=lambda row: (
            -float(row["selectionScore"]),
            -float(row["weightedPitchClassF1"]),
            -float(row["suppressionPrecision"]),
            int(row["suppressions"]),
        )
    )
    changed = [row for row in trials if int(row["suppressions"]) > 0]
    return {
        "songs": songs,
        "baseline": baseline,
        "best": trials[0],
        "topTrials": trials[:25],
        "topChangedByPitchF1": sorted(
            changed,
            key=lambda row: (
                -float(row["weightedPitchClassF1"]),
                -float(row["worstSongPitchClassF1Delta"]),
                -float(row["suppressionPrecision"]),
            ),
        )[:25],
    }


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--maximum-time-distance-seconds", type=float, default=0.55)
    parser.add_argument("--gap-cost", type=float, default=0.85)
    parser.add_argument(
        "--positive-weight-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.75,1,1.5,2,3"),
    )
    parser.add_argument(
        "--threshold-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.55,0.65,0.72,0.78,0.84,0.90,0.94,0.97"),
    )
    parser.add_argument(
        "--preserve-melody-grid", default="true,false", choices=("true", "false", "true,false")
    )
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0x48414E44)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    examples: list[dict[str, Any]] = []
    dataset = []
    for row in manifest.get("songs") or []:
        song_examples, description = build_song_examples(
            row,
            onset_window=max(0.005, float(args.onset_window_seconds)),
            maximum_time_distance=max(
                0.05, float(args.maximum_time_distance_seconds)
            ),
            gap_cost=max(0.05, float(args.gap_cost)),
        )
        examples.extend(song_examples)
        dataset.append(description)
    preserve_grid = (
        [True, False]
        if args.preserve_melody_grid == "true,false"
        else [args.preserve_melody_grid == "true"]
    )
    cross_validation = search_cross_validation(
        examples,
        positive_weight_grid=args.positive_weight_grid,
        threshold_grid=args.threshold_grid,
        preserve_melody_grid=preserve_grid,
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    best = cross_validation["best"]
    final_model, history = fit_model(
        examples,
        positive_weight=float(best["positiveTrainingWeight"]),
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    profile = {
        "schema": "polymath-pianist-hand-occupancy-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
        "onsetWindowSeconds": rounded(args.onset_window_seconds),
        "rightLayerSuppressor": final_model,
        "threshold": rounded(float(best["threshold"])),
        "preserveMelody": bool(best["preserveMelody"]),
        "training": {
            "manifest": str(manifest_path),
            "method": "whole-song leave-one-out hand-occupancy classification",
            "commercialUseAllowed": False,
            "decision": "EXPERIMENTAL_ONLY",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_profile = Path(args.output_profile).resolve()
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    weights = list(zip(HAND_OCCUPANCY_FEATURE_NAMES, final_model["weights"]))
    report = {
        "schema": "polymath-pianist-hand-occupancy-training-v1",
        "evidenceBoundary": (
            "Private research only. Whole-song folds prevent coordinate leakage; "
            "Kiss Me evidence at/after 02:30 is excluded."
        ),
        "manifest": str(manifest_path),
        "outputProfile": str(output_profile),
        "dataset": dataset,
        "examples": len(examples),
        "positiveExamples": sum(int(example["label"]) for example in examples),
        "crossValidation": cross_validation,
        "finalInSample": evaluate(
            examples,
            final_model,
            threshold=float(best["threshold"]),
            preserve_melody=bool(best["preserveMelody"]),
        ),
        "featureImportance": [
            {"name": name, "weight": rounded(weight)}
            for name, weight in sorted(weights, key=lambda item: -abs(item[1]))
        ],
        "optimizationHistory": history,
        "decision": (
            "CANDIDATE_FOR_APPLICATION_TEST"
            if int(best["suppressions"]) >= 5
            and float(best["suppressionPrecision"]) >= 0.80
            and float(best["worstSongPitchClassF1Delta"]) >= 0
            and float(best["worstSongExactSetRateDelta"]) >= 0
            and float(best["worstSongOccupancyAccuracyDelta"]) >= 0
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputProfile": str(output_profile),
                "report": str(report_path),
                "examples": len(examples),
                "positiveExamples": report["positiveExamples"],
                "best": best,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
