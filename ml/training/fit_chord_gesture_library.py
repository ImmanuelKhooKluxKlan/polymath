"""Fit a transposition-invariant library of authored piano chord gestures.

Each prototype pairs a local full-mix chroma context with the pitch-class
intervals chosen by an approved pianist.  Retrieval is evaluated by leaving a
whole song out.  This is deliberately separate from onset selection: the
onset model decides *when* to play, while this library proposes *what chord
shape* to play at that moment.
"""

from __future__ import annotations

import argparse
import bisect
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

from piano_arranger_adapter import (  # noqa: E402
    arrangement_role,
    instrument_family,
    normalize_source_notes,
)

try:
    from .fit_onset_gesture_ranker import (
        in_source_windows,
        load_json,
        map_time,
        target_window,
        trusted_windows,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from fit_onset_gesture_ranker import (  # type: ignore
        in_source_windows,
        load_json,
        map_time,
        target_window,
        trusted_windows,
    )


FAMILY_WEIGHTS = {
    "bass": 1.35,
    "piano": 1.18,
    "guitar": 1.0,
    "pad": 0.82,
    "lead": 0.68,
    "other": 0.72,
}


def aligned_target_left_notes(
    target: dict[str, Any],
    report: dict[str, Any],
    reference_hand_split: int,
) -> list[dict[str, Any]]:
    windows = trusted_windows(report)
    anchors = report.get("anchors") or []
    direct_source_timeline = any(
        isinstance(note, dict)
        and note.get("trainingEligible") is not None
        and note.get("originalTime") is not None
        for note in target.get("notes", [])
    )
    target_items = [item for item in target.get("notes", []) if isinstance(item, dict)]
    has_explicit_hands = any(
        str(item.get("hand") or "").strip().lower() in {"left", "right"}
        for item in target_items
    )
    mapped: list[dict[str, Any]] = []
    for item in target_items:
        try:
            midi = int(round(float(item["midi"])))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
        except (KeyError, TypeError, ValueError):
            continue
        if has_explicit_hands:
            if str(item.get("hand") or "").strip().lower() != "left":
                continue
        elif midi >= reference_hand_split:
            continue
        if direct_source_timeline:
            if not bool(item.get("trainingEligible")) or not in_source_windows(time, windows):
                continue
            source_time = time
        else:
            if target_window(time, windows) is None:
                continue
            source_time = map_time(time, anchors)
            if not in_source_windows(source_time, windows):
                continue
        mapped.append({**item, "midi": midi, "time": source_time})
    return sorted(mapped, key=lambda note: (float(note["time"]), int(note["midi"])))


def group_nearby_notes(
    notes: list[dict[str, Any]], tolerance: float = 0.035
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in notes:
        if not groups or float(note["time"]) - float(groups[-1][0]["time"]) > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def source_onset_times(notes: list[dict[str, Any]], windows: list[dict[str, Any]]) -> list[float]:
    return sorted(
        {
            round(float(note["time"]), 3)
            for note in notes
            if instrument_family(str(note.get("instrument") or "")) != "voice"
            and in_source_windows(float(note["time"]), windows)
        }
    )


def nearest_sequence_index(sequence_times: list[float], onset: float) -> int:
    """Return the nearest selected gesture index without leaking target notes.

    The sequence itself is known at inference time because the onset decoder has
    already decided *when* the accompaniment will play.  Only pitch content is
    predicted here.
    """

    if not sequence_times:
        return 0
    position = bisect.bisect_left(sequence_times, onset)
    choices = [
        index
        for index in (position - 1, position)
        if 0 <= index < len(sequence_times)
    ]
    return min(choices, key=lambda index: abs(sequence_times[index] - onset))


def match_target_groups_to_source(
    groups: list[list[dict[str, Any]]],
    source_times: list[float],
    tolerance: float,
) -> list[tuple[list[dict[str, Any]], float]]:
    possibilities: list[tuple[float, int, int]] = []
    for group_index, group in enumerate(groups):
        target_time = float(group[0]["time"])
        position = bisect.bisect_left(source_times, target_time)
        for source_index in range(max(0, position - 4), min(len(source_times), position + 5)):
            distance = abs(source_times[source_index] - target_time)
            if distance <= tolerance:
                possibilities.append((distance, group_index, source_index))
    used_groups: set[int] = set()
    used_sources: set[int] = set()
    matched: list[tuple[list[dict[str, Any]], float]] = []
    for _distance, group_index, source_index in sorted(possibilities):
        if group_index in used_groups or source_index in used_sources:
            continue
        used_groups.add(group_index)
        used_sources.add(source_index)
        matched.append((groups[group_index], source_times[source_index]))
    return sorted(matched, key=lambda item: item[1])


def harmonic_context(
    notes: list[dict[str, Any]], times: list[float], onset: float, radius: float
) -> tuple[list[float], int, list[float]]:
    left = bisect.bisect_left(times, onset - radius)
    right = bisect.bisect_right(times, onset + radius)
    chroma = [0.0] * 12
    bass_chroma = [0.0] * 12
    immediate_pc_families: dict[int, set[str]] = defaultdict(set)
    for note in notes[left:right]:
        family = instrument_family(str(note.get("instrument") or ""))
        if family == "voice":
            continue
        distance = abs(float(note["time"]) - onset)
        salience = (
            math.exp(-distance / 0.18)
            * FAMILY_WEIGHTS.get(family, 0.72)
            * (0.30 + min(1.0, float(note.get("velocity", 0.7))))
            * (0.30 + min(1.4, float(note.get("duration", 0.2))))
        )
        pitch_class = int(note["midi"]) % 12
        chroma[pitch_class] += salience
        if family == "bass" or arrangement_role(note) == "bass":
            bass_chroma[pitch_class] += salience
        if distance <= 0.10:
            immediate_pc_families[pitch_class].add(family)

    total = max(1e-9, sum(chroma))
    bass_total = sum(bass_chroma)
    if bass_total > 1e-9:
        anchor = max(range(12), key=lambda pc: bass_chroma[pc])
    else:
        templates: list[tuple[float, int]] = []
        for root in range(12):
            templates.append(
                (
                    chroma[root]
                    + 0.88 * chroma[(root + 4) % 12]
                    + 0.84 * chroma[(root + 7) % 12],
                    root,
                )
            )
            templates.append(
                (
                    chroma[root]
                    + 0.88 * chroma[(root + 3) % 12]
                    + 0.84 * chroma[(root + 7) % 12],
                    root,
                )
            )
        anchor = max(templates, key=lambda item: (item[0], -item[1]))[1]

    rotated_chroma = [chroma[(anchor + interval) % 12] / total for interval in range(12)]
    rotated_bass = [
        bass_chroma[(anchor + interval) % 12] / max(1e-9, bass_total)
        for interval in range(12)
    ]
    immediate_support = [
        min(1.0, len(immediate_pc_families.get((anchor + interval) % 12, set())) / 3.0)
        for interval in range(12)
    ]
    context = rotated_chroma + rotated_bass + immediate_support
    return context, anchor, rotated_chroma


def harmonic_sequence_context(
    notes: list[dict[str, Any]],
    times: list[float],
    onset: float,
    radius: float,
    sequence_times: list[float],
) -> tuple[list[float], int, list[float]]:
    """Add neighbouring selected-gesture evidence to the local chord context.

    Pianist reductions frequently alternate a bass strike, an inner tone and an
    upper dyad while the underlying audio still contains the same sustained
    chord.  A local chroma snapshot cannot distinguish those positions.  This
    representation retains the v1 local context, then adds source-only harmonic
    shapes and root motion at the previous/next selected onsets plus bounded IOI
    features.  The weights keep local evidence dominant.
    """

    current, anchor, support = harmonic_context(notes, times, onset, radius)
    if not sequence_times:
        return current + [0.0] * 54, anchor, support

    index = nearest_sequence_index(sequence_times, onset)
    previous_time = sequence_times[index - 1] if index > 0 else None
    next_time = sequence_times[index + 1] if index + 1 < len(sequence_times) else None

    def neighbour_features(value: float | None) -> tuple[list[float], list[float]]:
        if value is None:
            return [0.0] * 12, [0.0] * 12
        neighbour, neighbour_anchor, _ = harmonic_context(notes, times, value, radius)
        chroma = [0.45 * component for component in neighbour[:12]]
        root_motion = [0.0] * 12
        root_motion[(neighbour_anchor - anchor) % 12] = 0.35
        return chroma, root_motion

    previous_chroma, previous_motion = neighbour_features(previous_time)
    next_chroma, next_motion = neighbour_features(next_time)

    previous_gap = onset - previous_time if previous_time is not None else 0.0
    next_gap = next_time - onset if next_time is not None else 0.0
    gap_total = max(1e-6, previous_gap + next_gap)
    rhythm = [
        0.25 * min(1.0, max(0.0, previous_gap) / 1.5),
        0.25 * min(1.0, max(0.0, next_gap) / 1.5),
        0.25 * (previous_gap / gap_total if previous_time is not None else 0.0),
        0.25 * (next_gap / gap_total if next_time is not None else 0.0),
        0.25 if previous_time is None else 0.0,
        0.25 if next_time is None else 0.0,
    ]
    return (
        current
        + previous_chroma
        + next_chroma
        + previous_motion
        + next_motion
        + rhythm,
        anchor,
        support,
    )


def gesture_context(
    notes: list[dict[str, Any]],
    times: list[float],
    onset: float,
    radius: float,
    sequence_times: list[float],
    mode: str,
) -> tuple[list[float], int, list[float]]:
    if mode == "selected-sequence-v2":
        return harmonic_sequence_context(
            notes, times, onset, radius, sequence_times
        )
    if mode != "local-v1":
        raise ValueError(f"Unsupported chord gesture context mode: {mode}")
    return harmonic_context(notes, times, onset, radius)


def squared_distance(first: list[float], second: list[float]) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second))


def predict_intervals(
    context: list[float],
    prototypes: list[dict[str, Any]],
    size: int,
    neighbors: int,
    temperature: float,
) -> set[int]:
    nearest = sorted(
        prototypes,
        key=lambda prototype: squared_distance(context, prototype["context"]),
    )[:neighbors]
    votes = [0.0] * 12
    for prototype in nearest:
        distance = squared_distance(context, prototype["context"])
        weight = float(prototype.get("weight", 1.0)) * math.exp(
            -distance / max(1e-6, temperature)
        )
        for interval in prototype["intervals"]:
            votes[int(interval) % 12] += weight
    return set(
        sorted(range(12), key=lambda interval: (votes[interval], -interval), reverse=True)[
            : max(1, min(3, size))
        ]
    )


def predict_from_ranked_neighbors(
    ranked: list[tuple[float, dict[str, Any]]],
    size: int,
    neighbors: int,
    temperature: float,
) -> set[int]:
    votes = [0.0] * 12
    for distance, prototype in ranked[:neighbors]:
        weight = float(prototype.get("weight", 1.0)) * math.exp(
            -distance / max(1e-6, temperature)
        )
        for interval in prototype["intervals"]:
            votes[int(interval) % 12] += weight
    return set(
        sorted(range(12), key=lambda interval: (votes[interval], -interval), reverse=True)[
            : max(1, min(3, size))
        ]
    )


def predict_size_from_ranked_neighbors(
    ranked: list[tuple[float, dict[str, Any]]],
    neighbors: int,
    temperature: float,
) -> tuple[int, float, float]:
    """Infer pianist chord size instead of borrowing the target's size.

    The v1 evaluator passed ``prototype['size']`` into the interval decoder.
    That is useful as an oracle diagnostic but is unavailable at inference.
    Weighted neighbour voting makes the validation contract honest and gives
    the offline application stage a bounded one/two/three-note prediction.
    """

    votes = {1: 0.0, 2: 0.0, 3: 0.0}
    for distance, prototype in ranked[:neighbors]:
        weight = float(prototype.get("weight", 1.0)) * math.exp(
            -distance / max(1e-6, temperature)
        )
        size = max(1, min(3, int(prototype.get("size", 1))))
        votes[size] += weight
    ordered = sorted(votes, key=lambda size: (votes[size], -abs(size - 2), -size), reverse=True)
    total = max(1e-9, sum(votes.values()))
    winner = ordered[0]
    confidence = votes[winner] / total
    margin = (votes[winner] - votes[ordered[1]]) / total
    return winner, confidence, margin


def chord_size_metrics(predicted: list[int], targets: list[int]) -> dict[str, float]:
    if not targets:
        return {"chordSizeAccuracy": 0.0, "chordSizeMae": 0.0}
    return {
        "chordSizeAccuracy": round(
            sum(left == right for left, right in zip(predicted, targets))
            / len(targets),
            6,
        ),
        "chordSizeMae": round(
            sum(abs(left - right) for left, right in zip(predicted, targets))
            / len(targets),
            6,
        ),
    }


def membership_metrics(predictions: list[set[int]], targets: list[set[int]]) -> dict[str, float]:
    true_positive = sum(len(predicted & target) for predicted, target in zip(predictions, targets))
    predicted_total = sum(len(predicted) for predicted in predictions)
    target_total = sum(len(target) for target in targets)
    exact = sum(predicted == target for predicted, target in zip(predictions, targets))
    precision = true_positive / max(1, predicted_total)
    recall = true_positive / max(1, target_total)
    return {
        "membershipPrecision": round(precision, 6),
        "membershipRecall": round(recall, 6),
        "membershipF1": round(2 * precision * recall / max(1e-9, precision + recall), 6),
        "exactSetShare": round(exact / max(1, len(targets)), 6),
    }


def build_song_prototypes(
    pair: dict[str, Any],
    reference_hand_split: int,
    tolerance: float,
    radius: float,
    context_mode: str = "local-v1",
) -> list[dict[str, Any]]:
    source = load_json(Path(pair["source"]).resolve())
    target = load_json(Path(pair["target"]).resolve())
    report = load_json(Path(pair["alignmentReport"]).resolve())
    windows = trusted_windows(report)
    notes = normalize_source_notes(source.get("notes", []))
    note_times = [float(note["time"]) for note in notes]
    target_groups = group_nearby_notes(
        aligned_target_left_notes(target, report, reference_hand_split)
    )
    matched = match_target_groups_to_source(
        target_groups, source_onset_times(notes, windows), tolerance
    )
    selected_times = [float(source_time) for _group, source_time in matched]
    prototypes: list[dict[str, Any]] = []
    for group, source_time in matched:
        context, anchor, support = gesture_context(
            notes,
            note_times,
            source_time,
            radius,
            selected_times,
            context_mode,
        )
        intervals = sorted({(int(note["midi"]) % 12 - anchor) % 12 for note in group})
        if not intervals:
            continue
        prototypes.append(
            {
                "songId": str(pair["id"]),
                "time": round(source_time, 4),
                "context": [round(value, 7) for value in context],
                "intervals": intervals,
                "size": min(3, len(intervals)),
                "support": [round(value, 7) for value in support],
                "weight": max(0.01, float(pair.get("weight", 1.0))),
            }
        )
    return prototypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--match-tolerance", type=float, default=0.12)
    parser.add_argument("--context-radius", type=float, default=0.35)
    parser.add_argument(
        "--context-mode",
        choices=("local-v1", "selected-sequence-v2"),
        default="local-v1",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    by_song = {
        str(pair["id"]): build_song_prototypes(
            pair,
            args.reference_hand_split,
            args.match_tolerance,
            args.context_radius,
            args.context_mode,
        )
        for pair in manifest.get("pairs", [])
    }
    if len(by_song) < 3:
        raise ValueError("At least three songs are required for whole-song validation.")

    # Context distances do not depend on k or temperature.  Calculate each
    # song fold once with vectorized NumPy, then reuse the ordered neighbours
    # for the complete hyperparameter grid.
    fold_rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]] = {}
    for song_id, held_out in by_song.items():
        training = [
            prototype
            for other_id, prototypes in by_song.items()
            if other_id != song_id
            for prototype in prototypes
        ]
        training_contexts = np.asarray(
            [prototype["context"] for prototype in training], dtype=np.float32
        )
        held_contexts = np.asarray(
            [prototype["context"] for prototype in held_out], dtype=np.float32
        )
        distances = np.sum(
            (held_contexts[:, None, :] - training_contexts[None, :, :]) ** 2,
            axis=2,
        )
        orders = np.argsort(distances, axis=1)
        fold_rankings[song_id] = [
            [(float(distances[row, index]), training[int(index)]) for index in order[:40]]
            for row, order in enumerate(orders)
        ]

    grid: list[dict[str, Any]] = []
    for neighbors in (3, 5, 9, 15, 25, 40):
        for temperature in (0.015, 0.03, 0.06, 0.12, 0.24):
            fold_metrics: dict[str, Any] = {}
            all_predictions: list[set[int]] = []
            all_targets: list[set[int]] = []
            all_size_predictions: list[int] = []
            all_target_sizes: list[int] = []
            for song_id, held_out in by_song.items():
                size_predictions = [
                    predict_size_from_ranked_neighbors(
                        ranked,
                        neighbors,
                        temperature,
                    )[0]
                    for ranked in fold_rankings[song_id]
                ]
                predictions = [
                    predict_from_ranked_neighbors(
                        ranked,
                        predicted_size,
                        neighbors,
                        temperature,
                    )
                    for predicted_size, ranked in zip(
                        size_predictions, fold_rankings[song_id]
                    )
                ]
                oracle_size_predictions = [
                    predict_from_ranked_neighbors(
                        ranked,
                        prototype["size"],
                        neighbors,
                        temperature,
                    )
                    for prototype, ranked in zip(
                        held_out, fold_rankings[song_id]
                    )
                ]
                targets = [set(prototype["intervals"]) for prototype in held_out]
                target_sizes = [int(prototype["size"]) for prototype in held_out]
                fold_metrics[song_id] = {
                    **membership_metrics(predictions, targets),
                    **chord_size_metrics(size_predictions, target_sizes),
                    "oracleSizeMembershipF1": membership_metrics(
                        oracle_size_predictions, targets
                    )["membershipF1"],
                }
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_size_predictions.extend(size_predictions)
                all_target_sizes.extend(target_sizes)
            aggregate = {
                **membership_metrics(all_predictions, all_targets),
                **chord_size_metrics(all_size_predictions, all_target_sizes),
            }
            grid.append(
                {
                    "neighbors": neighbors,
                    "temperature": temperature,
                    "aggregate": aggregate,
                    "folds": fold_metrics,
                }
            )
    winner = max(
        grid,
        key=lambda row: (
            row["aggregate"]["membershipF1"],
            row["aggregate"]["exactSetShare"],
            -row["aggregate"]["chordSizeMae"],
            -row["neighbors"],
        ),
    )
    prototypes = [prototype for values in by_song.values() for prototype in values]
    profile = {
        "id": args.profile_id,
        "type": "transposition-invariant-nearest-chord-gesture-v1",
        "contextMode": args.context_mode,
        "contextRadiusSeconds": args.context_radius,
        "neighbors": winner["neighbors"],
        "temperature": winner["temperature"],
        "sizePrediction": {
            "enabled": True,
            "neighbors": winner["neighbors"],
            "temperature": winner["temperature"],
            "minimumSize": 1,
            "maximumSize": 3,
        },
        "prototypes": prototypes,
        "training": {
            "schema": "polymath-chord-gesture-library-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "policy": (
                "whole-song leave-one-out; trusted windows; key-relative chroma; "
                f"context={args.context_mode}"
            ),
            "commercialUseAllowed": False,
            "warning": "Research-only calibration library; do not deploy without rights clearance and sealed evaluation.",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report = {
        "schema": "polymath-chord-gesture-library-report-v1",
        "profile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "prototypeCounts": {song: len(values) for song, values in by_song.items()},
        "winner": winner,
        "leaderboard": sorted(
            grid,
            key=lambda row: (row["aggregate"]["membershipF1"], row["aggregate"]["exactSetShare"]),
            reverse=True,
        )[:10],
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
