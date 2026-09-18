"""Measure when an arranged chord should become a pianist gesture sequence.

The ordinary note matcher treats a vertical chord and a short broken chord as
roughly the same local pitch material.  A listener does not: one sounds like a
block and the other sounds like a pianist.  This audit assigns mapped reference
gestures to cells centred on the current arranger's onsets, then measures when
the authored performance keeps, drops, or unfolds each cell.

This is an analysis tool.  It never changes a candidate and never stores song
identity or absolute timestamps as inference features.
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
    from .analyze_pianist_gesture_patterns import (
        PERCUSSION,
        distribution,
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
except ImportError:  # pragma: no cover - CLI compatibility path.
    from analyze_pianist_gesture_patterns import (
        PERCUSSION,
        distribution,
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


def rounded(value: float) -> float:
    return round(float(value), 6)


def family(instrument: Any) -> str:
    text = str(instrument or "unknown").lower()
    if "voice" in text or "vocal" in text or "choir" in text:
        return "voice"
    if "guitar" in text:
        return "guitar"
    if "bass" in text or "contrabass" in text:
        return "bass"
    if "piano" in text or "keyboard" in text:
        return "piano"
    if "string" in text or "violin" in text or "cello" in text:
        return "strings"
    return "other"


def gap_band(value: float) -> str:
    if value < 0.18:
        return "fast-<0.18s"
    if value < 0.27:
        return "short-0.18-0.26s"
    if value < 0.37:
        return "pulse-0.27-0.36s"
    if value < 0.56:
        return "slow-0.37-0.55s"
    return "held->=0.56s"


def duration_band(value: float) -> str:
    if value < 0.18:
        return "short-<0.18s"
    if value < 0.55:
        return "medium-0.18-0.54s"
    return "long->=0.55s"


def size_band(value: int) -> str:
    return str(value) if value < 5 else "5+"


def target_count_band(value: int) -> str:
    return str(value) if value < 3 else "3+"


def gesture_time(group: list[dict[str, Any]]) -> float:
    return float(group[0]["time"])


def gesture_duration(group: list[dict[str, Any]]) -> float:
    return float(
        median(
            finite(note.get("scoreDuration", note.get("duration")), 0.2)
            for note in group
        )
    )


def gesture_role_signature(group: list[dict[str, Any]]) -> str:
    roles = sorted({str(note.get("arrangementRole") or "harmony") for note in group})
    return "+".join(roles)


def source_signature(notes: list[dict[str, Any]]) -> str:
    counts = Counter(family(note.get("instrument")) for note in notes)
    return "+".join(sorted(name for name, count in counts.items() if count)) or "unknown"


def nearest_source_notes(
    source: list[dict[str, Any]],
    time: float,
    *,
    onset_radius: float = 0.14,
    release_slack: float = 0.08,
) -> list[dict[str, Any]]:
    """Return local attacks plus notes physically sustaining at ``time``."""

    output: list[dict[str, Any]] = []
    for note in source:
        onset = float(note["time"])
        duration = max(0.03, float(note.get("duration", 0.2)))
        if abs(onset - time) <= onset_radius or (
            onset <= time <= onset + duration + release_slack
        ):
            output.append(note)
    return output


def source_pitch_class_evidence(
    local_source: list[dict[str, Any]],
    source_attacks: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    time: float,
) -> list[dict[str, Any]]:
    """Describe each local pitch class without consulting the reference.

    A binary ``sourcePitchClasses`` flag cannot distinguish a fresh piano
    attack from a distant-register guitar tone that merely sustains through a
    bass gesture.  Those cases require very different arrangement decisions.
    This fixed twelve-row representation preserves attack, family, register,
    duration, and velocity evidence for inference-safe downstream models.
    """

    candidate_midis = [int(note["midi"]) for note in candidate]
    candidate_centre = float(median(candidate_midis))
    output: list[dict[str, Any]] = []
    for pitch_class in range(12):
        notes = [
            note for note in local_source if int(note["midi"]) % 12 == pitch_class
        ]
        attacks = [
            note for note in source_attacks if int(note["midi"]) % 12 == pitch_class
        ]
        counts = Counter(family(note.get("instrument")) for note in notes)
        count = len(notes)
        if notes:
            nearest = min(
                notes,
                key=lambda note: (
                    abs(float(note["time"]) - time),
                    abs(int(note["midi"]) - candidate_centre),
                ),
            )
            closest_register = min(
                notes,
                key=lambda note: abs(int(note["midi"]) - candidate_centre),
            )
            nearest_offset = float(nearest["time"]) - time
            register_offset = int(closest_register["midi"]) - candidate_centre
            durations = [max(0.03, finite(note.get("duration"), 0.2)) for note in notes]
            velocities = [finite(note.get("velocity"), 0.72) for note in notes]
        else:
            nearest_offset = 0.0
            register_offset = 0.0
            durations = []
            velocities = []
        output.append(
            {
                "pitchClass": pitch_class,
                "noteCount": count,
                "attackCount": len(attacks),
                "sustainOnlyCount": max(0, count - len(attacks)),
                "nearestOnsetOffsetSeconds": rounded(nearest_offset),
                "medianDurationSeconds": (
                    rounded(median(durations)) if durations else 0.0
                ),
                "maximumVelocity": rounded(max(velocities)) if velocities else 0.0,
                "minimumRegisterDistanceSemitones": rounded(abs(register_offset)),
                "closestRegisterOffsetSemitones": rounded(register_offset),
                "exactCandidateMidiMatch": any(
                    int(note["midi"]) in candidate_midis for note in notes
                ),
                "familyShares": {
                    name: rounded(counts[name] / max(1, count))
                    for name in ("voice", "guitar", "bass", "piano", "strings", "other")
                },
            }
        )
    return output


def is_pitch_class_decoder_note(note: dict[str, Any]) -> bool:
    generated = str(note.get("generatedBy") or "").lower()
    source = str(note.get("source") or "").lower()
    return "pianist-pitch-class-decoder" in generated or (
        "pianist-pitch-class-decoder" in source
    )


def candidate_pitch_class_evidence(
    candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose frozen decoder provenance without reference-derived labels."""

    output: list[dict[str, Any]] = []
    for pitch_class in range(12):
        notes = [note for note in candidate if int(note["midi"]) % 12 == pitch_class]
        generated = [note for note in notes if is_pitch_class_decoder_note(note)]
        probabilities = [
            finite(note.get("pitchClassProbability"), 0.0) for note in generated
        ]
        role_counts = Counter(
            str(note.get("arrangementRole") or "harmony") for note in notes
        )
        family_counts = Counter(
            family(note.get("sourceInstrument") or note.get("instrument"))
            for note in notes
        )
        count = len(notes)
        output.append(
            {
                "pitchClass": pitch_class,
                "noteCount": count,
                "decoderGeneratedCount": len(generated),
                "decoderGeneratedShare": rounded(len(generated) / max(1, count)),
                "minimumDecoderProbability": (
                    rounded(min(probabilities)) if probabilities else 0.0
                ),
                "medianDecoderProbability": (
                    rounded(median(probabilities)) if probabilities else 0.0
                ),
                "melodyShare": rounded(role_counts["melody"] / max(1, count)),
                "bassShare": rounded(role_counts["bass"] / max(1, count)),
                "harmonyShare": rounded(role_counts["harmony"] / max(1, count)),
                "leftHandShare": rounded(
                    sum(int(note["midi"]) < 60 for note in notes) / max(1, count)
                ),
                "meanMidiCentered": (
                    rounded((sum(int(note["midi"]) for note in notes) / count - 66.0) / 24.0)
                    if notes
                    else 0.0
                ),
                "sourceFamilyShares": {
                    name: rounded(family_counts[name] / max(1, count))
                    for name in ("voice", "guitar", "bass", "piano", "strings", "other")
                },
            }
        )
    return output


def candidate_cells(
    groups: list[list[dict[str, Any]]],
    *,
    edge_seconds: float = 0.35,
) -> list[tuple[float, float]]:
    """Create non-overlapping Voronoi cells around candidate onsets."""

    if not groups:
        return []
    times = [gesture_time(group) for group in groups]
    cells: list[tuple[float, float]] = []
    for index, time in enumerate(times):
        left = (
            (times[index - 1] + time) / 2.0
            if index
            else time - min(edge_seconds, max(0.08, (times[1] - time) / 2.0))
            if len(times) > 1
            else time - edge_seconds
        )
        right = (
            (time + times[index + 1]) / 2.0
            if index + 1 < len(times)
            else time + min(edge_seconds, max(0.08, (time - times[index - 1]) / 2.0))
            if index
            else time + edge_seconds
        )
        cells.append((left, right))
    return cells


def assign_groups_to_cells(
    groups: list[list[dict[str, Any]]],
    cells: list[tuple[float, float]],
) -> list[list[list[dict[str, Any]]]]:
    assigned: list[list[list[dict[str, Any]]]] = [[] for _cell in cells]
    if not cells:
        return assigned
    cell_index = 0
    for group in groups:
        time = gesture_time(group)
        while cell_index + 1 < len(cells) and time >= cells[cell_index][1]:
            cell_index += 1
        left, right = cells[cell_index]
        if left <= time < right or (
            cell_index == len(cells) - 1 and math.isclose(time, right)
        ):
            assigned[cell_index].append(group)
    return assigned


def support_share(
    groups: list[list[dict[str, Any]]], source: list[dict[str, Any]]
) -> float:
    desired = 0
    supported = 0
    for group in groups:
        evidence = nearest_source_notes(source, gesture_time(group))
        source_pcs = pitch_set(evidence, True)
        target_pcs = pitch_set(group, True)
        desired += len(target_pcs)
        supported += len(target_pcs & source_pcs)
    return supported / max(1, desired)


def relative_pitch_direction(groups: list[list[dict[str, Any]]]) -> str:
    if len(groups) < 2:
        return "single"
    centres = [median(int(note["midi"]) for note in group) for group in groups]
    movements = [right - left for left, right in zip(centres, centres[1:])]
    if all(abs(value) <= 1 for value in movements):
        return "level"
    if all(value >= -1 for value in movements) and any(value > 1 for value in movements):
        return "ascending"
    if all(value <= 1 for value in movements) and any(value < -1 for value in movements):
        return "descending"
    return "alternating"


def summarize_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["textureLabel"]) for row in rows)
    target_counts = Counter(target_count_band(int(row["targetGestureCount"])) for row in rows)
    return {
        "cells": len(rows),
        "labels": dict(sorted(labels.items())),
        "targetGestureCounts": dict(sorted(target_counts.items())),
        "keepRate": rounded(sum(row["textureLabel"] == "keep-vertical" for row in rows) / max(1, len(rows))),
        "expansionRate": rounded(sum(str(row["textureLabel"]).startswith("unfold") for row in rows) / max(1, len(rows))),
        "dropRate": rounded(sum(row["textureLabel"] == "drop" for row in rows) / max(1, len(rows))),
        "meanTargetGesturesPerCell": rounded(sum(int(row["targetGestureCount"]) for row in rows) / max(1, len(rows))),
        "sourcePitchClassSupport": distribution(float(row["sourcePitchClassSupport"]) for row in rows),
    }


def grouped_summaries(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    output = [{"name": name, **summarize_labels(values)} for name, values in buckets.items()]
    return sorted(output, key=lambda item: (-int(item["cells"]), str(item["name"])))


def cell_example(
    song_id: str,
    index: int,
    candidate: list[dict[str, Any]],
    references: list[list[dict[str, Any]]],
    source: list[dict[str, Any]],
    next_gap: float,
    local_candidate_density: float,
    *,
    previous_gap: float | None = None,
    previous_pitch_class_similarity: float = 0.0,
    next_pitch_class_similarity: float = 0.0,
    interior_source: list[dict[str, Any]] | None = None,
    previous_candidate_chord_size: int | None = None,
    next_candidate_chord_size: int | None = None,
) -> dict[str, Any]:
    time = gesture_time(candidate)
    local_source = nearest_source_notes(source, time)
    target_count = len(references)
    if target_count == 0:
        label = "drop"
    elif target_count == 1:
        label = "keep-vertical"
    elif target_count == 2:
        label = "unfold-2"
    else:
        label = "unfold-3+"
    candidate_pcs = pitch_set(candidate, True)
    reference_pc_union = set().union(*(pitch_set(group, True) for group in references)) if references else set()
    offsets = [gesture_time(group) - time for group in references]
    within_ioi = [
        gesture_time(right) - gesture_time(left)
        for left, right in zip(references, references[1:])
    ]
    source_counts = Counter(family(note.get("instrument")) for note in local_source)
    count = max(1, sum(source_counts.values()))
    source_pcs = pitch_set(local_source, True)
    candidate_midis = sorted({int(note["midi"]) for note in candidate})
    candidate_velocity = median(finite(note.get("velocity"), 0.72) for note in candidate)
    candidate_source_velocity = median(
        finite(note.get("sourceVelocityBeforeArrangement"), candidate_velocity)
        for note in candidate
    )
    source_attacks = [
        note for note in source if abs(float(note["time"]) - time) <= 0.035
    ]
    per_pitch_class_evidence = source_pitch_class_evidence(
        local_source,
        source_attacks,
        candidate,
        time=time,
    )
    per_candidate_pitch_class_evidence = candidate_pitch_class_evidence(candidate)
    interior_source = list(interior_source or [])
    interior_groups = group_onsets(interior_source, 0.035)
    interior_pcs = pitch_set(interior_source, True)
    interior_counts = Counter(family(note.get("instrument")) for note in interior_source)
    interior_count = max(1, sum(interior_counts.values()))
    interior_times = [gesture_time(group) for group in interior_groups]
    off_centre_reference = [
        group for group in references if abs(gesture_time(group) - time) > 0.055
    ]
    raw_aligned_reference = 0
    for group in off_centre_reference:
        target_time = gesture_time(group)
        target_pcs = pitch_set(group, True)
        if any(
            abs(gesture_time(raw_group) - target_time) <= 0.09
            and bool(target_pcs & pitch_set(raw_group, True))
            for raw_group in interior_groups
        ):
            raw_aligned_reference += 1
    previous_gap = next_gap if previous_gap is None else max(0.03, previous_gap)
    return {
        "songId": song_id,
        "cellIndex": index,
        "sourceTime": rounded(time),
        "textureLabel": label,
        "targetGestureCount": target_count,
        "candidateChordSize": len(candidate_midis),
        "candidatePitchClassCount": len(candidate_pcs),
        "candidateRoleSignature": gesture_role_signature(candidate),
        "candidateDuration": rounded(gesture_duration(candidate)),
        "candidateDurationBand": duration_band(gesture_duration(candidate)),
        "candidateVelocity": rounded(candidate_velocity),
        "candidateSourceVelocity": rounded(candidate_source_velocity),
        "candidateMeanMidiCentered": rounded((sum(candidate_midis) / len(candidate_midis) - 66.0) / 24.0),
        "candidatePitchSpan": max(candidate_midis) - min(candidate_midis),
        "candidateIntervalSignature": [midi - candidate_midis[0] for midi in candidate_midis],
        "candidateLeftShare": rounded(sum(int(note["midi"]) < 60 for note in candidate) / len(candidate)),
        "candidateMelodyShare": rounded(sum(str(note.get("arrangementRole")) == "melody" for note in candidate) / len(candidate)),
        "candidateBassShare": rounded(sum(str(note.get("arrangementRole")) == "bass" for note in candidate) / len(candidate)),
        "previousGapSeconds": rounded(previous_gap),
        "nextGapSeconds": rounded(next_gap),
        "nextGapBand": gap_band(next_gap),
        "previousPitchClassSimilarity": rounded(previous_pitch_class_similarity),
        "nextPitchClassSimilarity": rounded(next_pitch_class_similarity),
        "previousCandidateChordSize": previous_candidate_chord_size or len(candidate_midis),
        "nextCandidateChordSize": next_candidate_chord_size or len(candidate_midis),
        "localCandidateOnsetsPerSecond": rounded(local_candidate_density),
        "sourceSignature": source_signature(local_source),
        "sourceNoteCount": len(local_source),
        "sourceAttackNoteCount": len(source_attacks),
        "sourceAttackPitchClassCount": len(pitch_set(source_attacks, True)),
        "sourcePitchClassCount": len(source_pcs),
        "sourceMedianDuration": rounded(median(float(note.get("duration", 0.2)) for note in local_source)) if local_source else 0.0,
        "sourceAttackMedianDuration": rounded(median(float(note.get("duration", 0.2)) for note in source_attacks)) if source_attacks else 0.0,
        "interiorSourceOnsetCount": len(interior_groups),
        "interiorSourceNoteCount": len(interior_source),
        "interiorSourcePitchClassCount": len(interior_pcs),
        "interiorSourceFirstOffsetSeconds": rounded(interior_times[0] - time) if interior_times else 0.0,
        "interiorSourceLastOffsetSeconds": rounded(interior_times[-1] - time) if interior_times else 0.0,
        "interiorSourceVelocity": rounded(median(finite(note.get("velocity"), 0.72) for note in interior_source)) if interior_source else 0.0,
        "interiorSourceDuration": rounded(median(finite(note.get("duration"), 0.2) for note in interior_source)) if interior_source else 0.0,
        "interiorSourceVoiceShare": rounded(interior_counts["voice"] / interior_count),
        "interiorSourceGuitarShare": rounded(interior_counts["guitar"] / interior_count),
        "interiorSourceBassShare": rounded(interior_counts["bass"] / interior_count),
        "interiorSourcePitchClassSimilarity": rounded(set_f1(candidate_pcs, interior_pcs)),
        "sourceVoiceShare": rounded(source_counts["voice"] / count),
        "sourceGuitarShare": rounded(source_counts["guitar"] / count),
        "sourceBassShare": rounded(source_counts["bass"] / count),
        "sourcePitchClassSupport": rounded(support_share(references, source)),
        "candidateToReferencePitchClassF1": rounded(set_f1(candidate_pcs, reference_pc_union)),
        "referenceGestureSizes": [len(pitch_set(group)) for group in references],
        "referenceSizeSignature": "-".join(str(len(pitch_set(group))) for group in references) or "none",
        "referenceRelativeOffsetsSeconds": [rounded(value) for value in offsets],
        "referenceRelativeOffsetsByNextGap": [rounded(value / max(0.03, next_gap)) for value in offsets],
        "referenceWithinCellIoiSeconds": [rounded(value) for value in within_ioi],
        "referencePitchDirection": relative_pitch_direction(references),
        "offCentreReferenceGestures": len(off_centre_reference),
        "offCentreReferenceRawAttackMatches": raw_aligned_reference,
        "offCentreReferenceRawAttackMatchShare": rounded(
            raw_aligned_reference / max(1, len(off_centre_reference))
        ),
        "referencePitchClasses": [sorted(pitch_set(group, True)) for group in references],
        "candidatePitchClasses": sorted(candidate_pcs),
        "sourcePitchClasses": sorted(source_pcs),
        "sourcePitchClassEvidence": per_pitch_class_evidence,
        "candidatePitchClassEvidence": per_candidate_pitch_class_evidence,
    }


def local_density(times: list[float], index: int, radius: float = 2.0) -> float:
    centre = times[index]
    count = sum(abs(value - centre) <= radius for value in times)
    return count / max(0.1, radius * 2.0)


def analyze_song(row: dict[str, Any], onset_window: float) -> dict[str, Any]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every manifest song requires an id")
    reference_path = Path(str(row["reference"])).resolve()
    source_path = Path(str(row["source"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = Path(str(row.get("textureCandidate") or row.get("baseline") or row["candidate"])).resolve()
    reference_end = finite(row.get("referenceEndSeconds"), math.nan)
    reference_end = reference_end if math.isfinite(reference_end) else None
    candidate_end = finite(row.get("candidateEndSeconds"), math.nan)
    candidate_end = candidate_end if math.isfinite(candidate_end) else None
    transpose = int(row.get("referenceTransposeSemitones") or 0)

    reference_payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    source_payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    ranges = trusted_source_ranges(alignment)
    anchors = monotonic_anchors(alignment)
    normalized_reference = normalize_notes(
        reference_payload, transpose=transpose, hard_end=reference_end
    )
    mapped_reference = (
        normalized_reference
        if bool(row.get("referenceAlreadyAligned"))
        else map_reference(normalized_reference, anchors)
    )
    reference = [
        note
        for note in mapped_reference
        if inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    source = [
        note
        for note in normalize_notes(source_payload, hard_end=candidate_end, source_indices=True)
        if inside_ranges(float(note["time"]), ranges)
        and str(note.get("instrument") or "").lower() not in PERCUSSION
    ]
    candidate = [
        note
        for note in normalize_notes(candidate_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    reference_groups = group_onsets(reference, onset_window)
    candidate_groups = group_onsets(candidate, onset_window)
    cells = candidate_cells(candidate_groups)
    assigned = assign_groups_to_cells(reference_groups, cells)
    times = [gesture_time(group) for group in candidate_groups]
    rows: list[dict[str, Any]] = []
    for index, (group, references) in enumerate(zip(candidate_groups, assigned)):
        previous_gap = (
            times[index] - times[index - 1]
            if index
            else times[index + 1] - times[index]
            if index + 1 < len(times)
            else 0.3
        )
        next_gap = (
            times[index + 1] - times[index]
            if index + 1 < len(times)
            else times[index] - times[index - 1]
            if index
            else 0.3
        )
        next_time = times[index] + max(0.03, next_gap)
        interior_source = [
            note
            for note in source
            if times[index] + onset_window < float(note["time"])
            < next_time - onset_window
        ]
        rows.append(
            cell_example(
                song_id,
                index,
                group,
                references,
                source,
                max(0.03, next_gap),
                local_density(times, index),
                previous_gap=max(0.03, previous_gap),
                previous_pitch_class_similarity=(
                    set_f1(pitch_set(group, True), pitch_set(candidate_groups[index - 1], True))
                    if index
                    else 0.0
                ),
                next_pitch_class_similarity=(
                    set_f1(pitch_set(group, True), pitch_set(candidate_groups[index + 1], True))
                    if index + 1 < len(candidate_groups)
                    else 0.0
                ),
                interior_source=interior_source,
                previous_candidate_chord_size=(
                    len(pitch_set(candidate_groups[index - 1])) if index else len(pitch_set(group))
                ),
                next_candidate_chord_size=(
                    len(pitch_set(candidate_groups[index + 1]))
                    if index + 1 < len(candidate_groups)
                    else len(pitch_set(group))
                ),
            )
        )
    anchored = [
        item
        for item in rows
        if item["targetGestureCount"] > 0
        and (
            item["candidateToReferencePitchClassF1"] > 0
            or item["sourcePitchClassSupport"] >= 0.5
        )
    ]
    unfolded = [item for item in anchored if str(item["textureLabel"]).startswith("unfold")]
    return {
        "id": song_id,
        "inputs": {
            "reference": str(reference_path),
            "source": str(source_path),
            "alignment": str(alignment_path),
            "candidate": str(candidate_path),
        },
        "boundary": {
            "referenceAlreadyAligned": bool(row.get("referenceAlreadyAligned")),
            "referenceTransposeSemitones": transpose,
            "referenceEndSeconds": reference_end,
            "candidateEndSeconds": candidate_end,
            "trustedSourceRanges": ranges,
        },
        "counts": {
            "referenceGestures": len(reference_groups),
            "candidateGestures": len(candidate_groups),
            "cells": len(rows),
            "anchoredCells": len(anchored),
            "unfoldedAnchoredCells": len(unfolded),
        },
        "allCells": summarize_labels(rows),
        "anchoredCells": summarize_labels(anchored),
        "byCandidateChordSize": grouped_summaries(anchored, lambda item: size_band(int(item["candidateChordSize"]))),
        "byNextGap": grouped_summaries(anchored, lambda item: str(item["nextGapBand"])),
        "byCandidateDuration": grouped_summaries(anchored, lambda item: str(item["candidateDurationBand"])),
        "byRoleSignature": grouped_summaries(anchored, lambda item: str(item["candidateRoleSignature"])),
        "bySourceSignature": grouped_summaries(anchored, lambda item: str(item["sourceSignature"])),
        "byVoicePresence": grouped_summaries(anchored, lambda item: "voice-present" if float(item["sourceVoiceShare"]) > 0 else "no-voice"),
        "unfoldedPatterns": [
            {"name": name, "cells": count}
            for name, count in Counter(
                f"{item['referenceSizeSignature']}|{item['referencePitchDirection']}"
                for item in unfolded
            ).most_common(30)
        ],
        "unfoldedOffsetSeconds": distribution(
            offset
            for item in unfolded
            for offset in item["referenceRelativeOffsetsSeconds"][1:]
        ),
        "unfoldedWithinCellIoiSeconds": distribution(
            value
            for item in unfolded
            for value in item["referenceWithinCellIoiSeconds"]
        ),
        "highConfidenceUnfoldedExamples": sorted(
            unfolded,
            key=lambda item: (
                -float(item["sourcePitchClassSupport"]),
                -float(item["candidateToReferencePitchClassF1"]),
                float(item["sourceTime"]),
            ),
        )[:120],
        "cells": rows,
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item for report in reports for item in report["cells"]]
    anchored = [
        item
        for item in rows
        if item["targetGestureCount"] > 0
        and (
            item["candidateToReferencePitchClassF1"] > 0
            or item["sourcePitchClassSupport"] >= 0.5
        )
    ]
    return {
        "allCells": summarize_labels(rows),
        "anchoredCells": summarize_labels(anchored),
        "bySong": [
            {"name": report["id"], **report["anchoredCells"]} for report in reports
        ],
        "byCandidateChordSize": grouped_summaries(anchored, lambda item: size_band(int(item["candidateChordSize"]))),
        "byNextGap": grouped_summaries(anchored, lambda item: str(item["nextGapBand"])),
        "byRoleSignature": grouped_summaries(anchored, lambda item: str(item["candidateRoleSignature"])),
        "bySourceSignature": grouped_summaries(anchored, lambda item: str(item["sourceSignature"])),
        "byVoicePresence": grouped_summaries(anchored, lambda item: "voice-present" if float(item["sourceVoiceShare"]) > 0 else "no-voice"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = manifest.get("songs") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("The manifest must contain at least one row in songs")
    reports = [
        analyze_song(row, max(0.005, float(args.onset_window_seconds)))
        for row in rows
    ]
    payload = {
        "schema": "polymath-pianist-texture-audit-v1",
        "evidenceBoundary": (
            "Private development references only. Kiss Me material at/after 02:30 is excluded. "
            "Absolute song time and song identity are audit fields, never inference features."
        ),
        "manifest": str(manifest_path),
        "aggregate": aggregate_reports(reports),
        "songs": reports,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "songs": [
                    {
                        "id": report["id"],
                        "candidateGestures": report["counts"]["candidateGestures"],
                        "referenceGestures": report["counts"]["referenceGestures"],
                        "anchored": report["anchoredCells"],
                    }
                    for report in reports
                ],
                "aggregate": payload["aggregate"]["anchoredCells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
