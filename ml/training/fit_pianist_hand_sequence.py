"""Fit a sequence-aware pianist hand-occupancy model.

Static note rules cannot explain a repeated left/right/both-hand accompaniment
figure.  This experiment combines three learned emission models with a small
Markov sequence prior, then uses forward/backward posteriors to identify only
high-confidence cases where a two-hand arranger gesture should lose its right
layer.  Onsets and pitches are otherwise frozen.

Training and validation are separated by complete song.  A profile remains
research-only unless every held-out regression gate passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support module and direct-script execution.
    from .analyze_pianist_gesture_patterns import pitch_set
    from .analyze_pianist_reduction_grammar import occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import (
        HAND_OCCUPANCY_FEATURE_NAMES,
        hand_notes,
        hand_occupancy_feature_map,
        load_song_sequences,
        parse_float_grid,
        rounded,
        sequence_contexts,
    )
    from .train_piano_arranger_adapter import train_logistic_model
except ImportError:  # pragma: no cover
    from analyze_pianist_gesture_patterns import pitch_set
    from analyze_pianist_reduction_grammar import occupancy
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import (
        HAND_OCCUPANCY_FEATURE_NAMES,
        hand_notes,
        hand_occupancy_feature_map,
        load_song_sequences,
        parse_float_grid,
        rounded,
        sequence_contexts,
    )
    from train_piano_arranger_adapter import train_logistic_model


STATE_NAMES = ("left-only", "right-only", "both")
STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}


def build_song_sequence(
    row: dict[str, Any],
    *,
    onset_window: float,
    maximum_time_distance: float,
    gap_cost: float,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    reference, candidate, description = load_song_sequences(row, onset_window)
    matches, missing, extra, score = align_sequences(
        reference,
        candidate,
        maximum_time_distance=maximum_time_distance,
        gap_cost=gap_cost,
    )
    contexts = sequence_contexts(candidate)
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    rejected = 0
    previous_pair: tuple[int, int] | None = None
    for reference_index, candidate_index in matches:
        target = reference[reference_index]
        observed = candidate[candidate_index]
        if not pitch_set(target, True) & pitch_set(observed, True):
            rejected += 1
            if current:
                segments.append(current)
                current = []
            previous_pair = None
            continue
        if previous_pair is not None:
            reference_gap = reference_index - previous_pair[0]
            candidate_gap = candidate_index - previous_pair[1]
            time_gap = float(observed[0]["time"]) - float(
                current[-1]["candidateGroup"][0]["time"]
            )
            if reference_gap != 1 or candidate_gap != 1 or time_gap > 1.20:
                if current:
                    segments.append(current)
                current = []
        feature_map = hand_occupancy_feature_map(candidate, candidate_index, contexts)
        current.append(
            {
                "songId": description["id"],
                "referenceIndex": reference_index,
                "candidateIndex": candidate_index,
                "targetState": occupancy(target),
                "candidateState": occupancy(observed),
                "targetPitchClasses": sorted(pitch_set(target, True)),
                "candidateGroup": observed,
                "features": [
                    feature_map[name] for name in HAND_OCCUPANCY_FEATURE_NAMES
                ],
            }
        )
        previous_pair = (reference_index, candidate_index)
    if current:
        segments.append(current)
    description.update(
        {
            "referenceGestures": len(reference),
            "candidateGestures": len(candidate),
            "sequenceMatches": len(matches),
            "missingReferenceGestures": len(missing),
            "extraCandidateGestures": len(extra),
            "sequenceScore": rounded(score),
            "usableExamples": sum(len(segment) for segment in segments),
            "segments": len(segments),
            "rejectedNoPitchOverlap": rejected,
        }
    )
    return segments, description


def all_examples(segments: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [example for segment in segments for example in segment]


def emission_sample_weights(
    examples: list[dict[str, Any]], state: str, class_balance: float
) -> np.ndarray:
    class_balance = max(0.0, min(1.0, float(class_balance)))
    song_counts = Counter(str(example["songId"]) for example in examples)
    state_counts: Counter[tuple[str, bool]] = Counter(
        (str(example["songId"]), example["targetState"] == state)
        for example in examples
    )
    song_states: dict[str, set[bool]] = defaultdict(set)
    for song, label in state_counts:
        song_states[song].add(label)
    song_count = max(1, len(song_counts))
    values = []
    for example in examples:
        song = str(example["songId"])
        label = example["targetState"] == state
        song_equal = 1.0 / (song_count * song_counts[song])
        class_equal = 1.0 / (
            song_count
            * max(1, len(song_states[song]))
            * state_counts[(song, label)]
        )
        values.append((1.0 - class_balance) * song_equal + class_balance * class_equal)
    weights = np.asarray(values, dtype=float)
    return weights * len(weights) / max(1e-12, float(weights.sum()))


def fit_emissions(
    examples: list[dict[str, Any]],
    *,
    class_balance: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    features = np.asarray([example["features"] for example in examples], dtype=float)
    emissions: dict[str, Any] = {}
    for offset, state in enumerate(STATE_NAMES):
        labels = np.asarray(
            [float(example["targetState"] == state) for example in examples],
            dtype=float,
        )
        weights = emission_sample_weights(examples, state, class_balance)
        model_weights, means, scales, history = train_logistic_model(
            features,
            labels,
            weights,
            np.ones(len(labels), dtype=bool),
            seed=seed + offset * 97,
            iterations=iterations,
            learning_rate=0.012,
            l2=0.035,
        )
        emissions[state] = {
            "weights": [round(float(value), 10) for value in model_weights],
            "means": [round(float(value), 10) for value in means],
            "scales": [round(float(value), 10) for value in scales],
            "history": history,
        }
    return {
        "type": "one-v-rest-standardized-logistic-hand-emissions-v1",
        "featureNames": list(HAND_OCCUPANCY_FEATURE_NAMES),
        "classBalance": rounded(class_balance),
        "states": emissions,
    }


def transition_probabilities(
    segments: list[list[dict[str, Any]]], smoothing: float = 1.0
) -> tuple[list[float], list[list[float]]]:
    starts = np.full(len(STATE_NAMES), float(smoothing), dtype=float)
    transitions = np.full(
        (len(STATE_NAMES), len(STATE_NAMES)), float(smoothing), dtype=float
    )
    for segment in segments:
        if not segment:
            continue
        starts[STATE_INDEX[segment[0]["targetState"]]] += 1.0
        for left, right in zip(segment, segment[1:]):
            transitions[
                STATE_INDEX[left["targetState"]],
                STATE_INDEX[right["targetState"]],
            ] += 1.0
    starts /= starts.sum()
    transitions /= transitions.sum(axis=1, keepdims=True)
    return starts.tolist(), transitions.tolist()


def fit_model(
    segments: list[list[dict[str, Any]]],
    *,
    class_balance: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    examples = all_examples(segments)
    if len(examples) < 100:
        raise ValueError("Hand-sequence training requires at least 100 examples")
    starts, transitions = transition_probabilities(segments)
    return {
        "emissions": fit_emissions(
            examples,
            class_balance=class_balance,
            iterations=iterations,
            seed=seed,
        ),
        "initialProbabilities": [rounded(value) for value in starts],
        "transitionProbabilities": [
            [rounded(value) for value in row] for row in transitions
        ],
    }


def emission_probabilities(
    segment: list[dict[str, Any]], model: dict[str, Any]
) -> np.ndarray:
    matrix = np.asarray([example["features"] for example in segment], dtype=float)
    columns = []
    for state in STATE_NAMES:
        config = model["emissions"]["states"][state]
        weights = np.asarray(config["weights"], dtype=float)
        means = np.asarray(config["means"], dtype=float)
        scales = np.asarray(config["scales"], dtype=float)
        logits = ((matrix - means) / np.maximum(1e-9, np.abs(scales))) @ weights
        columns.append(1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0))))
    probabilities = np.column_stack(columns)
    probabilities /= np.maximum(1e-12, probabilities.sum(axis=1, keepdims=True))
    return probabilities


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def posterior_probabilities(
    segment: list[dict[str, Any]],
    model: dict[str, Any],
    *,
    transition_strength: float,
) -> np.ndarray:
    emissions = np.maximum(1e-12, emission_probabilities(segment, model))
    log_emissions = np.log(emissions)
    log_initial = np.log(
        np.maximum(1e-12, np.asarray(model["initialProbabilities"], dtype=float))
    )
    log_transitions = np.log(
        np.maximum(
            1e-12, np.asarray(model["transitionProbabilities"], dtype=float)
        )
    ) * max(0.0, float(transition_strength))
    length, state_count = log_emissions.shape
    forward = np.full((length, state_count), -math.inf, dtype=float)
    backward = np.full((length, state_count), -math.inf, dtype=float)
    forward[0] = log_initial * max(0.0, float(transition_strength)) + log_emissions[0]
    for time in range(1, length):
        for state in range(state_count):
            forward[time, state] = log_emissions[time, state] + logsumexp(
                forward[time - 1] + log_transitions[:, state]
            )
    backward[-1] = 0.0
    for time in range(length - 2, -1, -1):
        for state in range(state_count):
            backward[time, state] = logsumexp(
                log_transitions[state]
                + log_emissions[time + 1]
                + backward[time + 1]
            )
    posterior = forward + backward
    posterior -= np.max(posterior, axis=1, keepdims=True)
    posterior = np.exp(posterior)
    posterior /= np.maximum(1e-12, posterior.sum(axis=1, keepdims=True))
    return posterior


def evaluate(
    segments: list[list[dict[str, Any]]],
    model: dict[str, Any] | None = None,
    *,
    transition_strength: float = 0.0,
    threshold: float = 1.1,
    margin: float = 1.1,
    preserve_melody: bool = True,
) -> dict[str, Any]:
    suppression_tp = suppression_fp = suppression_fn = 0
    pitch_tp = pitch_fp = pitch_fn = 0
    exact = occupancy_correct = posterior_correct = examples_count = suppressions = 0
    for segment in segments:
        posteriors = (
            posterior_probabilities(
                segment, model, transition_strength=transition_strength
            )
            if model is not None
            else np.zeros((len(segment), len(STATE_NAMES)), dtype=float)
        )
        for example, posterior in zip(segment, posteriors):
            examples_count += 1
            candidate = example["candidateGroup"]
            right = hand_notes(candidate, "right")
            target_left = example["targetState"] == "left-only"
            left_probability = float(posterior[STATE_INDEX["left-only"]])
            competing = max(
                float(posterior[STATE_INDEX["right-only"]]),
                float(posterior[STATE_INDEX["both"]]),
            )
            has_melody = any(
                note.get("arrangementRole") == "melody" for note in right
            )
            suppress = bool(
                model is not None
                and example["candidateState"] == "both"
                and left_probability >= threshold
                and left_probability - competing >= margin
                and not (preserve_melody and has_melody)
            )
            suppression_tp += int(suppress and target_left)
            suppression_fp += int(suppress and not target_left)
            suppression_fn += int(
                not suppress and target_left and example["candidateState"] == "both"
            )
            suppressions += int(suppress)
            selected = hand_notes(candidate, "left") if suppress else candidate
            selected_pcs = pitch_set(selected, True)
            target_pcs = set(int(value) for value in example["targetPitchClasses"])
            pitch_tp += len(selected_pcs & target_pcs)
            pitch_fp += len(selected_pcs - target_pcs)
            pitch_fn += len(target_pcs - selected_pcs)
            exact += int(selected_pcs == target_pcs)
            predicted_occupancy = "left-only" if suppress else example["candidateState"]
            occupancy_correct += int(predicted_occupancy == example["targetState"])
            if model is not None:
                posterior_correct += int(
                    STATE_NAMES[int(np.argmax(posterior))] == example["targetState"]
                )
    suppression_precision = suppression_tp / max(1, suppression_tp + suppression_fp)
    suppression_recall = suppression_tp / max(1, suppression_tp + suppression_fn)
    pitch_precision = pitch_tp / max(1, pitch_tp + pitch_fp)
    pitch_recall = pitch_tp / max(1, pitch_tp + pitch_fn)
    pitch_f1 = 2.0 * pitch_precision * pitch_recall / max(
        1e-9, pitch_precision + pitch_recall
    )
    return {
        "examples": examples_count,
        "suppressions": suppressions,
        "trueSuppressions": suppression_tp,
        "wrongSuppressions": suppression_fp,
        "missedSuppressions": suppression_fn,
        "suppressionPrecision": rounded(suppression_precision),
        "suppressionRecall": rounded(suppression_recall),
        "pitchClassPrecision": rounded(pitch_precision),
        "pitchClassRecall": rounded(pitch_recall),
        "pitchClassF1": rounded(pitch_f1),
        "exactPitchClassSets": exact,
        "exactPitchClassSetRate": rounded(exact / max(1, examples_count)),
        "occupancyAccuracy": rounded(occupancy_correct / max(1, examples_count)),
        "posteriorStateAccuracy": rounded(posterior_correct / max(1, examples_count))
        if model is not None
        else None,
    }


def search_cross_validation(
    songs: dict[str, list[list[dict[str, Any]]]],
    *,
    class_balance_grid: list[float],
    transition_strength_grid: list[float],
    threshold_grid: list[float],
    margin_grid: list[float],
    preserve_melody_grid: list[bool],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    song_names = sorted(songs)
    baseline = {song: evaluate(songs[song]) for song in song_names}
    trials: list[dict[str, Any]] = []
    for class_balance in class_balance_grid:
        fold_models = {
            held_out: fit_model(
                [
                    segment
                    for song, segments in songs.items()
                    if song != held_out
                    for segment in segments
                ],
                class_balance=class_balance,
                iterations=iterations,
                seed=seed,
            )
            for held_out in song_names
        }
        for transition_strength in transition_strength_grid:
            for threshold in threshold_grid:
                for margin in margin_grid:
                    for preserve_melody in preserve_melody_grid:
                        folds = []
                        weighted_pitch = weighted_occupancy = 0.0
                        total = 0
                        for song in song_names:
                            result = evaluate(
                                songs[song],
                                fold_models[song],
                                transition_strength=transition_strength,
                                threshold=threshold,
                                margin=margin,
                                preserve_melody=preserve_melody,
                            )
                            control = baseline[song]
                            folds.append(
                                {
                                    "heldOutSong": song,
                                    "baseline": control,
                                    "candidate": result,
                                    "pitchClassF1Delta": rounded(
                                        result["pitchClassF1"]
                                        - control["pitchClassF1"]
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
                            weighted_pitch += result["pitchClassF1"] * result["examples"]
                            weighted_occupancy += result["occupancyAccuracy"] * result["examples"]
                            total += result["examples"]
                        suppressions = sum(row["candidate"]["suppressions"] for row in folds)
                        wrong = sum(row["candidate"]["wrongSuppressions"] for row in folds)
                        precision = (suppressions - wrong) / max(1, suppressions)
                        worst_pitch = min(row["pitchClassF1Delta"] for row in folds)
                        worst_exact = min(row["exactSetRateDelta"] for row in folds)
                        worst_occupancy = min(row["occupancyAccuracyDelta"] for row in folds)
                        aggregate_pitch = weighted_pitch / max(1, total)
                        aggregate_occupancy = weighted_occupancy / max(1, total)
                        score = (
                            aggregate_pitch
                            + 0.20 * aggregate_occupancy
                            + min(0.0, worst_pitch) * 8.0
                            + min(0.0, worst_exact) * 3.0
                            + min(0.0, worst_occupancy) * 2.0
                            + min(0.0, precision - 0.75) * 0.7
                        )
                        trials.append(
                            {
                                "classBalance": class_balance,
                                "transitionStrength": transition_strength,
                                "threshold": threshold,
                                "margin": margin,
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
        "songs": song_names,
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
        "--class-balance-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.25,0.5,0.75,1"),
    )
    parser.add_argument(
        "--transition-strength-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.25,0.5,1,1.5,2"),
    )
    parser.add_argument(
        "--threshold-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.55,0.65,0.75,0.82,0.88,0.93"),
    )
    parser.add_argument(
        "--margin-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.1,0.2,0.3"),
    )
    parser.add_argument(
        "--preserve-melody-grid", default="true,false", choices=("true", "false", "true,false")
    )
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0x53455148)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    songs: dict[str, list[list[dict[str, Any]]]] = {}
    dataset = []
    for row in manifest.get("songs") or []:
        segments, description = build_song_sequence(
            row,
            onset_window=max(0.005, float(args.onset_window_seconds)),
            maximum_time_distance=max(
                0.05, float(args.maximum_time_distance_seconds)
            ),
            gap_cost=max(0.05, float(args.gap_cost)),
        )
        songs[description["id"]] = segments
        dataset.append(description)
    preserve_grid = (
        [True, False]
        if args.preserve_melody_grid == "true,false"
        else [args.preserve_melody_grid == "true"]
    )
    cross_validation = search_cross_validation(
        songs,
        class_balance_grid=args.class_balance_grid,
        transition_strength_grid=args.transition_strength_grid,
        threshold_grid=args.threshold_grid,
        margin_grid=args.margin_grid,
        preserve_melody_grid=preserve_grid,
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    best = cross_validation["best"]
    all_segments = [segment for segments in songs.values() for segment in segments]
    model = fit_model(
        all_segments,
        class_balance=float(best["classBalance"]),
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    for state in STATE_NAMES:
        model["emissions"]["states"][state].pop("history", None)
    profile = {
        "schema": "polymath-pianist-hand-sequence-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
        "onsetWindowSeconds": rounded(args.onset_window_seconds),
        "states": list(STATE_NAMES),
        "model": model,
        "policy": {
            "transitionStrength": rounded(float(best["transitionStrength"])),
            "threshold": rounded(float(best["threshold"])),
            "margin": rounded(float(best["margin"])),
            "preserveMelody": bool(best["preserveMelody"]),
            "allowedChange": "both-to-left-only-only",
        },
        "training": {
            "manifest": str(manifest_path),
            "method": "whole-song leave-one-out logistic-emission Markov smoothing",
            "commercialUseAllowed": False,
            "decision": "EXPERIMENTAL_ONLY",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_profile = Path(args.output_profile).resolve()
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    qualifies = bool(
        int(best["suppressions"]) >= 5
        and float(best["suppressionPrecision"]) >= 0.80
        and float(best["worstSongPitchClassF1Delta"]) >= 0
        and float(best["worstSongExactSetRateDelta"]) >= 0
        and float(best["worstSongOccupancyAccuracyDelta"]) >= 0
    )
    report = {
        "schema": "polymath-pianist-hand-sequence-training-v1",
        "evidenceBoundary": (
            "Private research only; complete-song folds; Kiss Me material at/after "
            "02:30 excluded; no onset or note invention."
        ),
        "manifest": str(manifest_path),
        "outputProfile": str(output_profile),
        "dataset": dataset,
        "crossValidation": cross_validation,
        "finalInSample": evaluate(
            all_segments,
            model,
            transition_strength=float(best["transitionStrength"]),
            threshold=float(best["threshold"]),
            margin=float(best["margin"]),
            preserve_melody=bool(best["preserveMelody"]),
        ),
        "transitionProbabilities": {
            state: {
                following: model["transitionProbabilities"][STATE_INDEX[state]][
                    STATE_INDEX[following]
                ]
                for following in STATE_NAMES
            }
            for state in STATE_NAMES
        },
        "decision": "CANDIDATE_FOR_APPLICATION_TEST" if qualifies else "REJECT_OR_RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputProfile": str(output_profile),
                "report": str(report_path),
                "best": best,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
