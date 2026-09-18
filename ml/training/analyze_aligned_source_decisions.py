"""Explain which aligned source events an arranger keeps or ignores.

This diagnostic joins three frozen artifacts by ``sourceIndex``:

* the index-stable transcription used by the arranger;
* an approved source-to-reference alignment;
* the arranged candidate.

It separates upstream misses from decoder decisions and reports whether source
velocity, instrument family, chord size, or pitch position predicts an error.
Alignment matches are evidence, not perfect ground truth: an authored pianist
may invent a note or choose a different member of an equivalent chord.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "median": round(median(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def trusted_source_ranges(report: dict[str, Any]) -> list[tuple[float, float]] | None:
    windows = report.get("qualityWindows")
    if not isinstance(windows, list):
        return None
    ranges: list[tuple[float, float]] = []
    for window in windows:
        if not isinstance(window, dict) or window.get("status") not in {
            "trusted",
            "accepted-manually",
        }:
            continue
        start = window.get("sourceStartSeconds", window.get("sourceStart"))
        end = window.get("sourceEndSeconds", window.get("sourceEnd"))
        start_value = finite(start, -1.0)
        end_value = finite(end, -1.0)
        if end_value > start_value >= 0:
            ranges.append((start_value, end_value))
    return ranges


def inside_ranges(time: float, ranges: list[tuple[float, float]] | None) -> bool:
    return ranges is None or any(start <= time < end for start, end in ranges)


def source_notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("notes") or []):
        if not isinstance(item, dict):
            continue
        midi = int(round(finite(item.get("midi", item.get("pitch")), -1)))
        time = finite(item.get("time", item.get("startTime")), -1.0)
        duration = finite(item.get("duration"), 0.2)
        if 21 <= midi <= 108 and time >= 0 and duration > 0:
            notes.append(
                {
                    **item,
                    "sourceIndex": index,
                    "midi": midi,
                    "time": time,
                    "duration": duration,
                    "velocity": max(0.01, min(1.0, finite(item.get("velocity"), 0.72))),
                    "instrument": str(item.get("instrument") or "unknown").lower(),
                }
            )
    return notes


def candidate_by_source_index(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payload.get("notes") or []:
        if not isinstance(item, dict):
            continue
        try:
            source_index = int(item["sourceIndex"])
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime")))
        except (KeyError, TypeError, ValueError):
            continue
        result[source_index].append({**item, "midi": midi, "time": time})
    return dict(result)


def strongest_alignment_matches(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    strongest: dict[int, tuple[tuple[int, float], dict[str, Any]]] = {}
    for item in report.get("matches") or []:
        if not isinstance(item, dict):
            continue
        observed = item.get("observed") or {}
        reference = item.get("reference") or {}
        try:
            source_index = int(observed["sourceIndex"])
            residual = abs(float(item.get("coarseResidual", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        quality = (int(bool(item.get("exactPitch"))), -residual)
        if source_index not in strongest or quality > strongest[source_index][0]:
            strongest[source_index] = (quality, item)
    return {index: value[1] for index, value in strongest.items()}


def velocity_band(note: dict[str, Any]) -> str:
    value = float(note["velocity"])
    if value < 0.55:
        return "quiet-<0.55"
    if value < 0.80:
        return "medium-0.55-0.79"
    return "strong->=0.80"


def duration_band(note: dict[str, Any]) -> str:
    value = float(note["duration"])
    if value < 0.18:
        return "short-<0.18s"
    if value < 0.55:
        return "medium-0.18-0.54s"
    return "long->=0.55s"


def pitch_band(note: dict[str, Any]) -> str:
    midi = int(note["midi"])
    if midi < 48:
        return "bass-<C3"
    if midi < 60:
        return "lower-C3-B3"
    if midi < 72:
        return "middle-C4-B4"
    if midi < 84:
        return "upper-C5-B5"
    return "high->=C6"


def group_decisions(
    notes: list[dict[str, Any]],
    desired_indices: set[int],
    selected_indices: set[int],
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"source": 0, "desired": 0, "selected": 0, "desiredSelected": 0}
    )
    for note in notes:
        name = key(note)
        source_index = int(note["sourceIndex"])
        row = groups[name]
        row["source"] += 1
        if source_index in desired_indices:
            row["desired"] += 1
        if source_index in selected_indices:
            row["selected"] += 1
        if source_index in desired_indices and source_index in selected_indices:
            row["desiredSelected"] += 1
    output = []
    for name, row in groups.items():
        output.append(
            {
                "name": name,
                **row,
                "desiredRetention": round(row["desiredSelected"] / max(1, row["desired"]), 6),
                "selectedPrecisionProxy": round(
                    row["desiredSelected"] / max(1, row["selected"]), 6
                ),
            }
        )
    return sorted(output, key=lambda row: (-row["desired"], row["name"]))


def onset_cluster_sizes(notes: list[dict[str, Any]]) -> dict[int, int]:
    groups: Counter[int] = Counter(int(round(float(note["time"]) * 1000)) for note in notes)
    return {
        int(note["sourceIndex"]): groups[int(round(float(note["time"]) * 1000))]
        for note in notes
    }


def analyze(
    source: dict[str, Any],
    alignment: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_transpose: int = 0,
) -> dict[str, Any]:
    ranges = trusted_source_ranges(alignment)
    notes = [
        note
        for note in source_notes(source)
        if inside_ranges(float(note["time"]), ranges)
        and str(note["instrument"]) not in {"drums", "percussion"}
    ]
    note_lookup = {int(note["sourceIndex"]): note for note in notes}
    matches = strongest_alignment_matches(alignment)
    matches = {
        index: item
        for index, item in matches.items()
        if index in note_lookup
    }
    candidates = candidate_by_source_index(candidate)
    selected_indices = set(candidates) & set(note_lookup)
    desired_indices = set(matches)
    retained = desired_indices & selected_indices
    ignored = desired_indices - selected_indices
    unsupported_selected = selected_indices - desired_indices

    exact_selected = 0
    pitch_class_selected = 0
    target_velocity_errors: list[float] = []
    raw_to_candidate_velocity: list[float] = []
    ignored_keys: Counter[str] = Counter()
    kept_keys: Counter[str] = Counter()
    ignored_target_velocities: list[float] = []
    kept_target_velocities: list[float] = []
    for source_index, match in matches.items():
        reference = match.get("reference") or {}
        reference_midi = int(round(finite(reference.get("midi"), -99))) + reference_transpose
        reference_velocity = max(0.01, min(1.0, finite(reference.get("velocity"), 0.72)))
        key_name = note_name(reference_midi)
        outputs = candidates.get(source_index, [])
        if not outputs:
            ignored_keys[key_name] += 1
            ignored_target_velocities.append(reference_velocity)
            continue
        kept_keys[key_name] += 1
        kept_target_velocities.append(reference_velocity)
        best = min(
            outputs,
            key=lambda note: (
                int(int(note["midi"]) % 12 != reference_midi % 12),
                abs(int(note["midi"]) - reference_midi),
            ),
        )
        exact_selected += int(int(best["midi"]) == reference_midi)
        pitch_class_selected += int(int(best["midi"]) % 12 == reference_midi % 12)
        candidate_velocity = max(0.01, min(1.0, finite(best.get("velocity"), 0.72)))
        target_velocity_errors.append(candidate_velocity - reference_velocity)
        raw_to_candidate_velocity.append(candidate_velocity - note_lookup[source_index]["velocity"])

    cluster_sizes = onset_cluster_sizes(notes)
    notes_with_cluster = [
        {**note, "onsetClusterSize": cluster_sizes[int(note["sourceIndex"])]}
        for note in notes
    ]

    def top(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
        return [{"name": name, "notes": count} for name, count in counter.most_common(limit)]

    velocity_changes = [
        finite(note.get("velocity"), 0.72)
        - finite(
            note.get("sourceVelocityBeforeArrangement"),
            note.get("velocityBeforeGestureCoherence", note.get("velocity")),
        )
        for rows in candidates.values()
        for note in rows
    ]
    return {
        "schema": "polymath-aligned-source-decision-audit-v1",
        "evidenceBoundary": (
            "Alignment-supported source events are a precision/retention proxy, not "
            "complete ground truth for an authored piano arrangement."
        ),
        "counts": {
            "eligiblePitchedSourceNotes": len(notes),
            "alignmentSupportedSourceNotes": len(desired_indices),
            "selectedSourceNotes": len(selected_indices),
            "supportedAndSelected": len(retained),
            "supportedButIgnored": len(ignored),
            "selectedWithoutAlignmentSupport": len(unsupported_selected),
            "supportedRetention": round(len(retained) / max(1, len(desired_indices)), 6),
            "selectedPrecisionProxy": round(len(retained) / max(1, len(selected_indices)), 6),
            "selectedWithExactTargetPitch": exact_selected,
            "selectedWithTargetPitchClass": pitch_class_selected,
        },
        "ignoredReferenceKeys": top(ignored_keys),
        "retainedReferenceKeys": top(kept_keys),
        "breakdown": {
            "sourceInstrument": group_decisions(
                notes_with_cluster, desired_indices, selected_indices, lambda note: str(note["instrument"])
            ),
            "sourceVelocity": group_decisions(
                notes_with_cluster, desired_indices, selected_indices, velocity_band
            ),
            "sourceDuration": group_decisions(
                notes_with_cluster, desired_indices, selected_indices, duration_band
            ),
            "sourcePitchBand": group_decisions(
                notes_with_cluster, desired_indices, selected_indices, pitch_band
            ),
            "sourceOnsetClusterSize": group_decisions(
                notes_with_cluster,
                desired_indices,
                selected_indices,
                lambda note: str(min(8, int(note["onsetClusterSize"])))
                + ("+" if int(note["onsetClusterSize"]) >= 8 else ""),
            ),
        },
        "velocity": {
            "retainedTarget": distribution(kept_target_velocities),
            "ignoredTarget": distribution(ignored_target_velocities),
            "selectedTargetError": distribution(target_velocity_errors),
            "selectedRawToCandidateChange": distribution(raw_to_candidate_velocity),
            "allCandidateChangeFromSource": distribution(velocity_changes),
            "candidateRaisedAtLeast010": sum(value >= 0.10 for value in velocity_changes),
            "candidateLoweredAtLeast010": sum(value <= -0.10 for value in velocity_changes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    alignment_path = Path(args.alignment).resolve()
    candidate_path = Path(args.candidate).resolve()
    report = {
        "inputs": {
            "source": str(source_path),
            "alignment": str(alignment_path),
            "candidate": str(candidate_path),
            "referenceTransposeSemitones": args.reference_transpose_semitones,
        },
        **analyze(
            json.loads(source_path.read_text(encoding="utf-8-sig")),
            json.loads(alignment_path.read_text(encoding="utf-8-sig")),
            json.loads(candidate_path.read_text(encoding="utf-8-sig")),
            reference_transpose=args.reference_transpose_semitones,
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
