"""Monotonically align arranger gestures with an authored piano performance.

Nearest-onset matching can misread one shifted rhythmic stream as alternating
"missing" and "extra" notes.  This audit uses global sequence order, pitch
classes, exact register and onset distance together.  It is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support module and direct-script execution.
    from .analyze_pianist_gesture_patterns import (
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
except ImportError:  # pragma: no cover
    from analyze_pianist_gesture_patterns import (
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


def gesture_time(group: list[dict[str, Any]]) -> float:
    return float(group[0]["time"])


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


def match_cost(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    maximum_time_distance: float,
) -> float:
    distance = abs(gesture_time(reference) - gesture_time(candidate))
    if distance > maximum_time_distance:
        return math.inf
    pc_f1 = set_f1(pitch_set(reference, True), pitch_set(candidate, True))
    exact_f1 = set_f1(pitch_set(reference), pitch_set(candidate))
    size_delta = abs(len(pitch_set(reference)) - len(pitch_set(candidate)))
    # Two gaps cost 1.70.  A close wrong chord may be retained as a timing
    # correspondence, but a distant wrong chord is cheaper to leave unmatched.
    value = (
        0.42 * distance / max(0.03, maximum_time_distance)
        + 0.82 * (1.0 - pc_f1)
        + 0.18 * (1.0 - exact_f1)
        + 0.045 * min(5, size_delta)
    )
    if pc_f1 <= 0.0:
        value += 0.38
    return value


def align_sequences(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    *,
    maximum_time_distance: float = 0.55,
    gap_cost: float = 0.85,
) -> tuple[list[tuple[int, int]], list[int], list[int], float]:
    """Needleman-Wunsch alignment with a music-aware substitution cost."""

    rows = len(reference) + 1
    columns = len(candidate) + 1
    scores = np.full((rows, columns), np.inf, dtype=np.float64)
    back = np.zeros((rows, columns), dtype=np.int8)
    scores[0, :] = np.arange(columns) * gap_cost
    scores[:, 0] = np.arange(rows) * gap_cost
    back[0, 1:] = 2  # candidate insertion
    back[1:, 0] = 1  # reference deletion
    for i in range(1, rows):
        reference_time = gesture_time(reference[i - 1])
        for j in range(1, columns):
            candidate_time = gesture_time(candidate[j - 1])
            substitution = math.inf
            if abs(reference_time - candidate_time) <= maximum_time_distance:
                substitution = scores[i - 1, j - 1] + match_cost(
                    reference[i - 1],
                    candidate[j - 1],
                    maximum_time_distance=maximum_time_distance,
                )
            deletion = scores[i - 1, j] + gap_cost
            insertion = scores[i, j - 1] + gap_cost
            best = min((substitution, 0), (deletion, 1), (insertion, 2))
            scores[i, j], back[i, j] = best
    i, j = len(reference), len(candidate)
    matches: list[tuple[int, int]] = []
    missing: list[int] = []
    extra: list[int] = []
    while i or j:
        action = int(back[i, j])
        if i and j and action == 0:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (not j or action == 1):
            missing.append(i - 1)
            i -= 1
        else:
            extra.append(j - 1)
            j -= 1
    return (
        list(reversed(matches)),
        list(reversed(missing)),
        list(reversed(extra)),
        float(scores[-1, -1]),
    )


def grouped_transition_rows(
    transitions: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in transitions:
        groups[str(item[key])].append(item)
    output: list[dict[str, Any]] = []
    for name, values in groups.items():
        output.append(
            {
                "name": name,
                "transitions": len(values),
                "candidateIoi": distribution(item["candidateIoi"] for item in values),
                "referenceIoi": distribution(item["referenceIoi"] for item in values),
                "ioiError": distribution(item["ioiError"] for item in values),
                "ioiRatio": distribution(item["ioiRatio"] for item in values),
            }
        )
    return sorted(output, key=lambda item: (-int(item["transitions"]), str(item["name"])))


def analyze_song(
    row: dict[str, Any],
    *,
    onset_window: float,
    maximum_time_distance: float,
    gap_cost: float,
) -> dict[str, Any]:
    song_id = str(row.get("id") or "").strip()
    reference_path = Path(str(row["reference"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = Path(
        str(row.get("textureCandidate") or row.get("baseline") or row["candidate"])
    ).resolve()
    reference_end_value = finite(row.get("referenceEndSeconds"), math.nan)
    reference_end = reference_end_value if math.isfinite(reference_end_value) else None
    candidate_end_value = finite(row.get("candidateEndSeconds"), math.nan)
    candidate_end = candidate_end_value if math.isfinite(candidate_end_value) else None
    transpose = int(row.get("referenceTransposeSemitones") or 0)
    alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    anchors = monotonic_anchors(alignment)
    ranges = trusted_source_ranges(alignment)
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    normalized_reference = normalize_notes(
        reference_payload, transpose=transpose, hard_end=reference_end
    )
    mapped_reference = (
        normalized_reference
        if bool(row.get("referenceAlreadyAligned"))
        else map_reference(normalized_reference, anchors)
    )
    reference_notes = [
        note
        for note in mapped_reference
        if inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    candidate_notes = [
        note
        for note in normalize_notes(candidate_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    reference = group_onsets(reference_notes, onset_window)
    candidate = group_onsets(candidate_notes, onset_window)
    matches, missing, extra, score = align_sequences(
        reference,
        candidate,
        maximum_time_distance=maximum_time_distance,
        gap_cost=gap_cost,
    )
    pairs: list[dict[str, Any]] = []
    exact = 0
    pc_values: list[float] = []
    for reference_index, candidate_index in matches:
        target = reference[reference_index]
        observed = candidate[candidate_index]
        reference_midis = pitch_set(target)
        candidate_midis = pitch_set(observed)
        reference_pitch_classes = pitch_set(target, True)
        candidate_pitch_classes = pitch_set(observed, True)
        pc_f1 = set_f1(reference_pitch_classes, candidate_pitch_classes)
        exact_f1 = set_f1(reference_midis, candidate_midis)
        exact += int(exact_f1 == 1.0)
        pc_values.append(pc_f1)
        pairs.append(
            {
                "referenceIndex": reference_index,
                "candidateIndex": candidate_index,
                "referenceTime": round(gesture_time(target), 6),
                "candidateTime": round(gesture_time(observed), 6),
                "onsetErrorSeconds": round(gesture_time(observed) - gesture_time(target), 6),
                "pitchClassF1": round(pc_f1, 6),
                "exactF1": round(exact_f1, 6),
                "referenceChordSize": len(pitch_set(target)),
                "candidateChordSize": len(pitch_set(observed)),
                "referenceMidis": sorted(reference_midis),
                "candidateMidis": sorted(candidate_midis),
                "referencePitchClasses": sorted(reference_pitch_classes),
                "candidatePitchClasses": sorted(candidate_pitch_classes),
                "missingPitchClasses": sorted(
                    reference_pitch_classes - candidate_pitch_classes
                ),
                "extraPitchClasses": sorted(
                    candidate_pitch_classes - reference_pitch_classes
                ),
            }
        )
    transitions: list[dict[str, Any]] = []
    for left, right in zip(pairs, pairs[1:]):
        if (
            int(right["referenceIndex"]) != int(left["referenceIndex"]) + 1
            or int(right["candidateIndex"]) != int(left["candidateIndex"]) + 1
        ):
            continue
        reference_ioi = float(right["referenceTime"]) - float(left["referenceTime"])
        candidate_ioi = float(right["candidateTime"]) - float(left["candidateTime"])
        if reference_ioi <= 0 or candidate_ioi <= 0:
            continue
        transitions.append(
            {
                "referenceIndex": int(left["referenceIndex"]),
                "candidateIndex": int(left["candidateIndex"]),
                "referenceIoi": round(reference_ioi, 6),
                "candidateIoi": round(candidate_ioi, 6),
                "ioiError": round(candidate_ioi - reference_ioi, 6),
                "ioiRatio": round(reference_ioi / candidate_ioi, 6),
                "candidateGapBand": gap_band(candidate_ioi),
                "candidateChordSize": str(min(5, int(left["candidateChordSize"])))
                + ("+" if int(left["candidateChordSize"]) >= 5 else ""),
            }
        )
    residuals = [float(item["onsetErrorSeconds"]) for item in pairs]
    return {
        "id": song_id,
        "inputs": {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "alignment": str(alignment_path),
        },
        "boundary": {
            "referenceAlreadyAligned": bool(row.get("referenceAlreadyAligned")),
            "referenceTransposeSemitones": transpose,
            "referenceEndSeconds": reference_end,
            "candidateEndSeconds": candidate_end,
        },
        "alignment": {
            "score": round(score, 6),
            "referenceGestures": len(reference),
            "candidateGestures": len(candidate),
            "matchedGestures": len(matches),
            "missingReferenceGestures": len(missing),
            "extraCandidateGestures": len(extra),
            "matchRateAgainstReference": round(len(matches) / max(1, len(reference)), 6),
            "meanPitchClassF1": round(sum(pc_values) / max(1, len(pc_values)), 6),
            "exactGestureSets": exact,
            "onsetError": distribution(residuals),
            "onsetAbsoluteError": distribution(abs(value) for value in residuals),
        },
        "consecutiveTransitions": {
            "count": len(transitions),
            "candidateIoi": distribution(item["candidateIoi"] for item in transitions),
            "referenceIoi": distribution(item["referenceIoi"] for item in transitions),
            "ioiError": distribution(item["ioiError"] for item in transitions),
            "ioiRatio": distribution(item["ioiRatio"] for item in transitions),
            "byCandidateGap": grouped_transition_rows(transitions, "candidateGapBand"),
            "byCandidateChordSize": grouped_transition_rows(transitions, "candidateChordSize"),
        },
        "pairs": pairs,
        "missingReferenceIndices": missing,
        "extraCandidateIndices": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--maximum-time-distance-seconds", type=float, default=0.55)
    parser.add_argument("--gap-cost", type=float, default=0.85)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = manifest.get("songs") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("The manifest must contain at least one row in songs")
    reports = [
        analyze_song(
            row,
            onset_window=max(0.005, float(args.onset_window_seconds)),
            maximum_time_distance=max(0.05, float(args.maximum_time_distance_seconds)),
            gap_cost=max(0.05, float(args.gap_cost)),
        )
        for row in rows
    ]
    payload = {
        "schema": "polymath-pianist-sequence-alignment-v1",
        "evidenceBoundary": (
            "Evaluation only; complete-song monotonic alignment. Kiss Me material at/after 02:30 is excluded."
        ),
        "manifest": str(manifest_path),
        "configuration": {
            "onsetWindowSeconds": args.onset_window_seconds,
            "maximumTimeDistanceSeconds": args.maximum_time_distance_seconds,
            "gapCost": args.gap_cost,
        },
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
                    {"id": report["id"], **report["alignment"]}
                    for report in reports
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
