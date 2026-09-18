"""Forensic comparison of an arranged performance and an authored pianist.

The ordinary arranger metrics answer whether notes match.  This report answers
*how* they differ at the level a listener hears: one physical hammer gesture.
It keeps velocity, duration, register, chord membership, source provenance and
selection decisions together instead of averaging them into one F1 score.

The reference is mapped through a frozen alignment and only pre-approved time
ranges are admitted.  A hard reference/source cutoff can be supplied for a
partially trusted performance (for example, Kiss Me before 02:30).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

try:  # Support module and direct-script execution.
    from .analyze_full_mix_arranger import greedy_match_indices
    from .evaluate_piano_arranger import (
        map_time,
        monotonic_anchors,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover - CLI compatibility path.
    from analyze_full_mix_arranger import greedy_match_indices
    from evaluate_piano_arranger import map_time, monotonic_anchors, trusted_source_ranges


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PERCUSSION = {"drums", "percussion", "timpani"}


def finite(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if math.isfinite(numeric) else fallback


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def quantile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    if not kept:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {
        "count": len(kept),
        "mean": round(sum(kept) / len(kept), 6),
        "p10": round(float(quantile(kept, 0.10)), 6),
        "median": round(float(quantile(kept, 0.50)), 6),
        "p90": round(float(quantile(kept, 0.90)), 6),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_energy = sum((a - left_mean) ** 2 for a in left)
    right_energy = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_energy * right_energy)
    return round(numerator / denominator, 6) if denominator > 1e-12 else None


def normalize_notes(
    payload: dict[str, Any],
    *,
    transpose: int = 0,
    hard_end: float | None = None,
    source_indices: bool = False,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for payload_index, item in enumerate(payload.get("notes") or []):
        if not isinstance(item, dict):
            continue
        midi = int(round(finite(item.get("midi", item.get("pitch")), -999))) + transpose
        time = finite(item.get("time", item.get("startTime", item.get("start"))), -1.0)
        duration = finite(item.get("duration"), 0.2)
        if not (0 <= midi <= 127 and time >= 0 and duration > 0):
            continue
        if hard_end is not None and time >= hard_end:
            continue
        note = {
            **item,
            "midi": midi,
            "time": time,
            "duration": duration,
            "scoreDuration": max(
                0.01, finite(item.get("scoreDuration"), duration)
            ),
            "audioDuration": max(
                0.01, finite(item.get("audioDuration"), duration)
            ),
            "velocity": clamp(finite(item.get("velocity"), 0.72), 0.01, 1.0),
            "_payloadIndex": payload_index,
        }
        if source_indices:
            note["sourceIndex"] = payload_index
        notes.append(note)
    return sorted(notes, key=lambda item: (float(item["time"]), int(item["midi"]), int(item["_payloadIndex"])))


def map_reference(
    notes: list[dict[str, Any]], anchors: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for note in notes:
        start = map_time(float(note["time"]), anchors)
        end = map_time(float(note["time"]) + float(note["duration"]), anchors)
        mapped.append(
            {
                **note,
                "referenceTime": float(note["time"]),
                "referenceDuration": float(note["duration"]),
                "time": max(0.0, start),
                "duration": max(0.01, end - start),
            }
        )
    return sorted(mapped, key=lambda item: (float(item["time"]), int(item["midi"])))


def inside_ranges(time: float, ranges: list[tuple[float, float]] | None) -> bool:
    return ranges is None or any(start <= time < end for start, end in ranges)


def group_onsets(notes: list[dict[str, Any]], window: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in sorted(notes, key=lambda item: (float(item["time"]), int(item["midi"]))):
        if not groups or float(note["time"]) - float(groups[-1][0]["time"]) > window:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def pitch_set(group: list[dict[str, Any]], octave_equivalent: bool = False) -> set[int]:
    return {
        int(note["midi"]) % 12 if octave_equivalent else int(note["midi"])
        for note in group
    }


def set_f1(left: set[int], right: set[int]) -> float:
    overlap = len(left & right)
    if not left and not right:
        return 1.0
    return 2.0 * overlap / max(1, len(left) + len(right))


def match_gestures(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    tolerance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily pair nearby onsets, prioritising pitch-class agreement.

    A very close onset may still be paired when every key differs so the report
    can distinguish a wrong chord from a missing beat.  A distant, unrelated
    onset is left unmatched instead of manufacturing a comparison.
    """

    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for reference_index, target in enumerate(reference):
        target_time = float(target[0]["time"])
        target_pc = pitch_set(target, True)
        target_exact = pitch_set(target, False)
        options: list[tuple[tuple[float, ...], int]] = []
        for candidate_index, observed in enumerate(candidate):
            if candidate_index in used:
                continue
            distance = abs(float(observed[0]["time"]) - target_time)
            if distance > tolerance:
                continue
            pc_f1 = set_f1(target_pc, pitch_set(observed, True))
            exact_f1 = set_f1(target_exact, pitch_set(observed, False))
            if pc_f1 <= 0 and distance > min(0.055, tolerance):
                continue
            options.append(
                (
                    (
                        -pc_f1,
                        -exact_f1,
                        distance,
                        abs(len(target_exact) - len(pitch_set(observed, False))),
                        candidate_index,
                    ),
                    candidate_index,
                )
            )
        if not options:
            continue
        candidate_index = min(options)[1]
        used.add(candidate_index)
        matches.append((reference_index, candidate_index))
    matched_reference = {left for left, _right in matches}
    return (
        matches,
        [index for index in range(len(reference)) if index not in matched_reference],
        [index for index in range(len(candidate)) if index not in used],
    )


def velocity_band(value: float) -> str:
    if value < 0.55:
        return "quiet-<0.55"
    if value < 0.65:
        return "soft-0.55-0.64"
    if value < 0.75:
        return "medium-0.65-0.74"
    if value < 0.85:
        return "strong-0.75-0.84"
    return "accent->=0.85"


def duration_band(value: float) -> str:
    if value < 0.18:
        return "short-<0.18s"
    if value < 0.55:
        return "medium-0.18-0.54s"
    return "long->=0.55s"


def pitch_band(midi: int) -> str:
    if midi < 48:
        return "bass-<C3"
    if midi < 60:
        return "lower-C3-B3"
    if midi < 72:
        return "middle-C4-B4"
    if midi < 84:
        return "upper-C5-B5"
    return "high->=C6"


def gesture_velocity(group: list[dict[str, Any]]) -> float:
    return float(median(float(note.get("velocity", 0.72)) for note in group))


def gesture_duration(group: list[dict[str, Any]]) -> float:
    return float(median(float(note.get("duration", 0.2)) for note in group))


def gesture_score_duration(group: list[dict[str, Any]]) -> float:
    return float(
        median(
            float(note.get("scoreDuration", note.get("duration", 0.2)))
            for note in group
        )
    )


def gesture_audio_duration(group: list[dict[str, Any]]) -> float:
    return float(
        median(
            float(note.get("audioDuration", note.get("duration", 0.2)))
            for note in group
        )
    )


def gesture_category(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> str:
    exact_left = pitch_set(reference)
    exact_right = pitch_set(candidate)
    pc_left = pitch_set(reference, True)
    pc_right = pitch_set(candidate, True)
    if exact_left == exact_right:
        return "exact-key-set"
    if pc_left == pc_right:
        return "same-pitch-classes-different-octave"
    if pc_right and pc_right < pc_left:
        return "candidate-subset"
    if pc_left and pc_left < pc_right:
        return "candidate-superset"
    if pc_left & pc_right:
        return "partial-overlap"
    return "wrong-chord"


def grouped_error_rows(
    records: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    rows: list[dict[str, Any]] = []
    for name, values in groups.items():
        errors = [float(item["velocityError"]) for item in values]
        duration_errors = [float(item["durationError"]) for item in values]
        audio_duration_errors = [
            float(item["audioDurationError"]) for item in values
        ]
        rows.append(
            {
                "name": name,
                "gestures": len(values),
                "referenceMeanVelocity": round(
                    sum(float(item["referenceVelocity"]) for item in values)
                    / len(values),
                    6,
                ),
                "candidateMeanVelocity": round(
                    sum(float(item["candidateVelocity"]) for item in values)
                    / len(values),
                    6,
                ),
                "velocityBias": round(sum(errors) / len(errors), 6),
                "velocityMae": round(sum(abs(value) for value in errors) / len(errors), 6),
                "referenceMeanDuration": round(
                    sum(float(item["referenceDuration"]) for item in values)
                    / len(values),
                    6,
                ),
                "candidateMeanDuration": round(
                    sum(float(item["candidateDuration"]) for item in values)
                    / len(values),
                    6,
                ),
                "durationBias": round(
                    sum(duration_errors) / len(duration_errors), 6
                ),
                "durationMae": round(
                    sum(abs(value) for value in duration_errors)
                    / len(duration_errors),
                    6,
                ),
                "referenceMeanAudioDuration": round(
                    sum(float(item["referenceAudioDuration"]) for item in values)
                    / len(values),
                    6,
                ),
                "candidateMeanAudioDuration": round(
                    sum(float(item["candidateAudioDuration"]) for item in values)
                    / len(values),
                    6,
                ),
                "audioDurationBias": round(
                    sum(audio_duration_errors) / len(audio_duration_errors), 6
                ),
                "audioDurationMae": round(
                    sum(abs(value) for value in audio_duration_errors)
                    / len(audio_duration_errors),
                    6,
                ),
                "meanChordSizeChange": round(
                    sum(int(item["candidateChordSize"]) - int(item["referenceChordSize"]) for item in values)
                    / len(values),
                    6,
                ),
                "meanPitchClassF1": round(
                    sum(float(item["pitchClassF1"]) for item in values) / len(values), 6
                ),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["gestures"]), str(row["name"])))


def strongest_alignment_matches(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    best: dict[int, tuple[tuple[int, float], dict[str, Any]]] = {}
    for item in report.get("matches") or []:
        if not isinstance(item, dict):
            continue
        observed = item.get("observed") or {}
        try:
            source_index = int(observed["sourceIndex"])
            residual = abs(finite(item.get("coarseResidual", item.get("timingResidualSeconds")), 9e9))
        except (KeyError, TypeError, ValueError):
            continue
        rank = (int(bool(item.get("exactPitch"))), -residual)
        if source_index not in best or rank > best[source_index][0]:
            best[source_index] = (rank, item)
    return {index: value[1] for index, value in best.items()}


def source_cluster_sizes(notes: list[dict[str, Any]], window: float) -> dict[int, int]:
    output: dict[int, int] = {}
    for group in group_onsets(notes, window):
        size = len(group)
        for note in group:
            output[int(note["sourceIndex"])] = size
    return output


def decision_rows(
    notes: list[dict[str, Any]],
    desired: set[int],
    selected: set[int],
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for note in notes:
        index = int(note["sourceIndex"])
        row = groups[key(note)]
        row["source"] += 1
        row["desired"] += int(index in desired)
        row["selected"] += int(index in selected)
        row["desiredSelected"] += int(index in desired and index in selected)
        row["unsupportedSelected"] += int(index not in desired and index in selected)
    output: list[dict[str, Any]] = []
    for name, row in groups.items():
        output.append(
            {
                "name": name,
                **dict(row),
                "desiredRetention": round(row["desiredSelected"] / max(1, row["desired"]), 6),
                "selectedPrecisionProxy": round(row["desiredSelected"] / max(1, row["selected"]), 6),
            }
        )
    return sorted(output, key=lambda row: (-int(row.get("unsupportedSelected", 0)), -int(row["source"]), str(row["name"])))


def candidate_source_indices(candidate: list[dict[str, Any]]) -> set[int]:
    output: set[int] = set()
    for note in candidate:
        try:
            output.add(int(note["sourceIndex"]))
        except (KeyError, TypeError, ValueError):
            continue
    return output


def analyze(
    reference_payload: dict[str, Any],
    source_payload: dict[str, Any],
    alignment: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    reference_transpose: int = 0,
    reference_end: float | None = None,
    candidate_end: float | None = None,
    onset_window: float = 0.035,
    gesture_match_window: float = 0.25,
) -> dict[str, Any]:
    anchors = monotonic_anchors(alignment)
    ranges = trusted_source_ranges(alignment)
    reference_original = normalize_notes(
        reference_payload,
        transpose=reference_transpose,
        hard_end=reference_end,
    )
    reference = [
        note
        for note in map_reference(reference_original, anchors)
        if inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    candidate = [
        note
        for note in normalize_notes(candidate_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    source = [
        note
        for note in normalize_notes(source_payload, hard_end=candidate_end, source_indices=True)
        if inside_ranges(float(note["time"]), ranges)
        and str(note.get("instrument") or "").lower() not in PERCUSSION
    ]

    reference_groups = group_onsets(reference, onset_window)
    candidate_groups = group_onsets(candidate, onset_window)
    matches, missing_gestures, extra_gestures = match_gestures(
        reference_groups, candidate_groups, gesture_match_window
    )

    records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    for reference_index, candidate_index in matches:
        target = reference_groups[reference_index]
        observed = candidate_groups[candidate_index]
        target_velocity = gesture_velocity(target)
        observed_velocity = gesture_velocity(observed)
        category = gesture_category(target, observed)
        category_counts[category] += 1
        transition_counts[f"{len(pitch_set(target))}->{len(pitch_set(observed))}"] += 1
        records.append(
            {
                "referenceGesture": reference_index,
                "candidateGesture": candidate_index,
                "sourceTime": round(float(target[0]["time"]), 6),
                "referenceTime": round(float(target[0].get("referenceTime", 0.0)), 6),
                "referenceWindow5s": (
                    f"{int(float(target[0].get('referenceTime', 0.0)) // 5) * 5:03d}-"
                    f"{int(float(target[0].get('referenceTime', 0.0)) // 5) * 5 + 5:03d}s"
                ),
                "onsetErrorSeconds": round(float(observed[0]["time"]) - float(target[0]["time"]), 6),
                "category": category,
                "referenceVelocityBand": velocity_band(target_velocity),
                "referenceDurationBand": duration_band(gesture_duration(target)),
                "referenceChordSize": len(pitch_set(target)),
                "candidateChordSize": len(pitch_set(observed)),
                "referenceVelocity": round(target_velocity, 6),
                "candidateVelocity": round(observed_velocity, 6),
                "velocityError": round(observed_velocity - target_velocity, 6),
                "referenceDuration": round(gesture_duration(target), 6),
                "candidateDuration": round(gesture_duration(observed), 6),
                "durationError": round(gesture_duration(observed) - gesture_duration(target), 6),
                "referenceAudioDuration": round(gesture_audio_duration(target), 6),
                "candidateAudioDuration": round(gesture_audio_duration(observed), 6),
                "audioDurationError": round(
                    gesture_audio_duration(observed)
                    - gesture_audio_duration(target),
                    6,
                ),
                "exactKeyF1": round(set_f1(pitch_set(target), pitch_set(observed)), 6),
                "pitchClassF1": round(set_f1(pitch_set(target, True), pitch_set(observed, True)), 6),
                "referenceKeys": [note_name(value) for value in sorted(pitch_set(target))],
                "candidateKeys": [note_name(value) for value in sorted(pitch_set(observed))],
                "candidateRoles": sorted({str(note.get("arrangementRole") or "unknown") for note in observed}),
                "candidateSourceInstruments": sorted({str(note.get("sourceInstrument") or note.get("instrument") or "unknown") for note in observed}),
            }
        )
        records[-1]["candidateRoleSignature"] = "+".join(records[-1]["candidateRoles"])
        records[-1]["candidateSourceSignature"] = "+".join(
            records[-1]["candidateSourceInstruments"]
        )

    ref_velocities = [gesture_velocity(group) for group in reference_groups]
    candidate_velocities = [gesture_velocity(group) for group in candidate_groups]
    ref_durations = [gesture_duration(group) for group in reference_groups]
    candidate_durations = [gesture_duration(group) for group in candidate_groups]
    ref_audio_durations = [
        gesture_audio_duration(group) for group in reference_groups
    ]
    candidate_audio_durations = [
        gesture_audio_duration(group) for group in candidate_groups
    ]
    velocity_errors = [float(item["velocityError"]) for item in records]
    duration_errors = [float(item["durationError"]) for item in records]
    audio_duration_errors = [
        float(item["audioDurationError"]) for item in records
    ]
    onset_errors = [float(item["onsetErrorSeconds"]) for item in records]
    reference_onset_gaps = [
        float(following[0]["time"]) - float(previous[0]["time"])
        for previous, following in zip(reference_groups, reference_groups[1:])
    ]
    candidate_onset_gaps = [
        float(following[0]["time"]) - float(previous[0]["time"])
        for previous, following in zip(candidate_groups, candidate_groups[1:])
    ]
    reference_chord_velocities: dict[int, list[float]] = defaultdict(list)
    candidate_chord_velocities: dict[int, list[float]] = defaultdict(list)
    for group in reference_groups:
        reference_chord_velocities[min(8, len(pitch_set(group)))].append(gesture_velocity(group))
    for group in candidate_groups:
        candidate_chord_velocities[min(8, len(pitch_set(group)))].append(gesture_velocity(group))

    exact_pairs = greedy_match_indices(reference, candidate, 0.1)
    pitch_class_pairs = greedy_match_indices(reference, candidate, 0.25, octave_equivalent=True)
    exact_reference = {left for left, _right in exact_pairs}
    pitch_class_reference = {left for left, _right in pitch_class_pairs}
    missing_reference_notes = [
        note for index, note in enumerate(reference) if index not in pitch_class_reference
    ]
    exact_offset_counts: Counter[str] = Counter()
    for left, right in pitch_class_pairs:
        exact_offset_counts[str(int(candidate[right]["midi"]) - int(reference[left]["midi"]))] += 1

    source_lookup = {int(note["sourceIndex"]): note for note in source}
    aligned = {
        index: item
        for index, item in strongest_alignment_matches(alignment).items()
        if index in source_lookup
    }
    desired = set(aligned)
    selected = candidate_source_indices(candidate) & set(source_lookup)
    source_sizes = source_cluster_sizes(source, onset_window)
    decorated = [
        {**note, "sourceClusterSize": source_sizes[int(note["sourceIndex"])]}
        for note in source
    ]

    source_velocity_values: list[float] = []
    target_velocity_values: list[float] = []
    for source_index, match in aligned.items():
        reference_note = match.get("reference") or {}
        source_velocity_values.append(float(source_lookup[source_index]["velocity"]))
        target_velocity_values.append(clamp(finite(reference_note.get("velocity"), 0.72), 0.01, 1.0))

    selected_probability_supported: list[float] = []
    selected_probability_unsupported: list[float] = []
    for note in candidate:
        try:
            source_index = int(note["sourceIndex"])
            probability = float(note["selectionProbability"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_index in desired:
            selected_probability_supported.append(probability)
        else:
            selected_probability_unsupported.append(probability)

    def chord_rows(values: dict[int, list[float]]) -> list[dict[str, Any]]:
        return [
            {"chordSize": size, **distribution(group_values)}
            for size, group_values in sorted(values.items())
        ]

    reference_spreads = [
        max(float(note["velocity"]) for note in group) - min(float(note["velocity"]) for note in group)
        for group in reference_groups
    ]
    candidate_spreads = [
        max(float(note["velocity"]) for note in group) - min(float(note["velocity"]) for note in group)
        for group in candidate_groups
    ]

    return {
        "schema": "polymath-pianist-gesture-forensic-v1",
        "evidencePolicy": {
            "referenceEndSecondsExclusive": reference_end,
            "candidateEndSecondsExclusive": candidate_end,
            "trustedSourceRanges": ranges,
            "warning": "Alignment support is a strong proxy, not proof that every authored or detected note has a one-to-one cause.",
        },
        "notes": {
            "reference": len(reference),
            "candidate": len(candidate),
            "exactMatches100ms": len(exact_reference),
            "pitchClassMatches250ms": len(pitch_class_reference),
            "pitchClassOffsetSemitones": dict(sorted(exact_offset_counts.items(), key=lambda item: int(item[0]))),
            "missingReferenceByVelocity": dict(Counter(velocity_band(float(note["velocity"])) for note in missing_reference_notes)),
            "missingReferenceKeys": [
                {"key": key, "notes": count}
                for key, count in Counter(note_name(int(note["midi"])) for note in missing_reference_notes).most_common(20)
            ],
        },
        "gestures": {
            "reference": len(reference_groups),
            "candidate": len(candidate_groups),
            "paired": len(matches),
            "missingReferenceGestures": len(missing_gestures),
            "extraCandidateGestures": len(extra_gestures),
            "categories": dict(category_counts.most_common()),
            "chordSizeTransitions": [
                {"transition": name, "gestures": count}
                for name, count in transition_counts.most_common()
            ],
            "referenceVelocity": distribution(ref_velocities),
            "candidateVelocity": distribution(candidate_velocities),
            "pairedVelocityError": {
                **distribution(velocity_errors),
                "mae": round(sum(abs(value) for value in velocity_errors) / max(1, len(velocity_errors)), 6),
                "correlation": pearson(
                    [float(item["referenceVelocity"]) for item in records],
                    [float(item["candidateVelocity"]) for item in records],
                ),
            },
            "referenceDuration": distribution(ref_durations),
            "candidateDuration": distribution(candidate_durations),
            "pairedDurationError": {
                **distribution(duration_errors),
                "mae": round(
                    sum(abs(value) for value in duration_errors)
                    / max(1, len(duration_errors)),
                    6,
                ),
                "correlation": pearson(
                    [float(item["referenceDuration"]) for item in records],
                    [float(item["candidateDuration"]) for item in records],
                ),
            },
            "referenceAudioDuration": distribution(ref_audio_durations),
            "candidateAudioDuration": distribution(candidate_audio_durations),
            "pairedAudioDurationError": {
                **distribution(audio_duration_errors),
                "mae": round(
                    sum(abs(value) for value in audio_duration_errors)
                    / max(1, len(audio_duration_errors)),
                    6,
                ),
                "correlation": pearson(
                    [float(item["referenceAudioDuration"]) for item in records],
                    [float(item["candidateAudioDuration"]) for item in records],
                ),
            },
            "pairedOnsetError": {
                **distribution(onset_errors),
                "mae": round(
                    sum(abs(value) for value in onset_errors)
                    / max(1, len(onset_errors)),
                    6,
                ),
            },
            "referenceInterOnsetInterval": distribution(reference_onset_gaps),
            "candidateInterOnsetInterval": distribution(candidate_onset_gaps),
            "referenceVelocityByChordSize": chord_rows(reference_chord_velocities),
            "candidateVelocityByChordSize": chord_rows(candidate_chord_velocities),
            "referenceWithinGestureVelocitySpread": distribution(reference_spreads),
            "candidateWithinGestureVelocitySpread": distribution(candidate_spreads),
            "byReferenceVelocityBand": grouped_error_rows(records, "referenceVelocityBand"),
            "byReferenceDurationBand": grouped_error_rows(records, "referenceDurationBand"),
            "byReferenceChordSize": grouped_error_rows(records, "referenceChordSize"),
            "byMatchCategory": grouped_error_rows(records, "category"),
            "byReferenceWindow5s": grouped_error_rows(records, "referenceWindow5s"),
            "byCandidateRoleSignature": grouped_error_rows(records, "candidateRoleSignature"),
            "byCandidateSourceSignature": grouped_error_rows(records, "candidateSourceSignature"),
        },
        "sourceDecisions": {
            "eligiblePitchedSourceNotes": len(source),
            "alignmentSupportedSourceNotes": len(desired),
            "selectedSourceNotes": len(selected),
            "supportedAndSelected": len(desired & selected),
            "supportedButIgnored": len(desired - selected),
            "selectedWithoutAlignmentSupport": len(selected - desired),
            "sourceToTargetVelocityCorrelation": pearson(source_velocity_values, target_velocity_values),
            "selectedProbabilitySupported": distribution(selected_probability_supported),
            "selectedProbabilityUnsupported": distribution(selected_probability_unsupported),
            "byInstrument": decision_rows(decorated, desired, selected, lambda note: str(note.get("instrument") or "unknown")),
            "byVelocity": decision_rows(decorated, desired, selected, lambda note: velocity_band(float(note["velocity"]))),
            "byDuration": decision_rows(decorated, desired, selected, lambda note: duration_band(float(note["duration"]))),
            "byPitchBand": decision_rows(decorated, desired, selected, lambda note: pitch_band(int(note["midi"]))),
            "byClusterSize": decision_rows(
                decorated,
                desired,
                selected,
                lambda note: str(min(8, int(note["sourceClusterSize"]))) + ("+" if int(note["sourceClusterSize"]) >= 8 else ""),
            ),
            "jointRiskPatterns": decision_rows(
                decorated,
                desired,
                selected,
                lambda note: "|".join(
                    (
                        str(note.get("instrument") or "unknown"),
                        duration_band(float(note["duration"])),
                        "cluster-" + str(min(8, int(note["sourceClusterSize"]))) + ("+" if int(note["sourceClusterSize"]) >= 8 else ""),
                        pitch_band(int(note["midi"])),
                    )
                ),
            )[:40],
        },
        "worstPairedGestures": sorted(
            records,
            key=lambda item: (
                float(item["pitchClassF1"]),
                -abs(float(item["velocityError"])),
                -abs(float(item["durationError"])),
            ),
        )[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--reference-end-seconds", type=float)
    parser.add_argument("--candidate-end-seconds", type=float)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--gesture-match-window-seconds", type=float, default=0.25)
    args = parser.parse_args()

    inputs = {
        "reference": Path(args.reference).resolve(),
        "source": Path(args.source).resolve(),
        "alignment": Path(args.alignment).resolve(),
        "candidate": Path(args.candidate).resolve(),
    }
    report = {
        "inputs": {name: str(path) for name, path in inputs.items()},
        **analyze(
            json.loads(inputs["reference"].read_text(encoding="utf-8-sig")),
            json.loads(inputs["source"].read_text(encoding="utf-8-sig")),
            json.loads(inputs["alignment"].read_text(encoding="utf-8-sig")),
            json.loads(inputs["candidate"].read_text(encoding="utf-8-sig")),
            reference_transpose=args.reference_transpose_semitones,
            reference_end=args.reference_end_seconds,
            candidate_end=args.candidate_end_seconds,
            onset_window=args.onset_window_seconds,
            gesture_match_window=args.gesture_match_window_seconds,
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "notes": report["notes"],
                "gestureCounts": {
                    key: report["gestures"][key]
                    for key in (
                        "reference",
                        "candidate",
                        "paired",
                        "missingReferenceGestures",
                        "extraCandidateGestures",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
