"""Audit pianist gesture dynamics and note-selection errors.

This report deliberately separates three questions that a single accuracy
number hides:

* Which reference keys were omitted, and which candidate keys were extra?
* Did matching notes become louder or softer than the approved performance?
* Do simultaneous notes behave like one pianist gesture or conflicting stems?

The utility is evaluation-only. It never modifies a model or candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, fraction))
    left = int(math.floor(position))
    right = min(len(ordered) - 1, left + 1)
    ratio = position - left
    return ordered[left] * (1.0 - ratio) + ordered[right] * ratio


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def load_notes(
    path: Path,
    transpose: int = 0,
    training_only: bool = False,
    hand_split_midi: int = 60,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    include_ranges: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    output: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("notes", [])):
        if not isinstance(item, dict):
            continue
        if training_only and item.get("trainingEligible") is False:
            continue
        midi = int(round(finite(item.get("midi", item.get("pitch")), -1))) + transpose
        onset = finite(item.get("time", item.get("startTime")), -1)
        duration = finite(item.get("duration"), 0.2)
        velocity = finite(item.get("velocity"), 0.72)
        if not 21 <= midi <= 108 or onset < 0 or duration <= 0:
            continue
        if start_seconds is not None and onset < start_seconds:
            continue
        if end_seconds is not None and onset >= end_seconds:
            continue
        if include_ranges is not None and not any(
            start <= onset < end for start, end in include_ranges
        ):
            continue
        output.append(
            {
                **item,
                "_index": index,
                "midi": midi,
                "note": note_name(midi),
                "time": onset,
                "duration": duration,
                "velocity": max(0.01, min(1.0, velocity)),
                # Recompute the hand after transposition.  Keeping a stale
                # source-side ``hand`` label makes octave-normalized reference
                # notes appear to belong to the wrong hand in the audit.
                "hand": "left" if midi < hand_split_midi else "right",
            }
        )
    return sorted(output, key=lambda note: (note["time"], note["midi"], note["_index"]))


def trusted_source_ranges(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    ranges: list[tuple[float, float]] = []
    for window in payload.get("qualityWindows") or []:
        if not isinstance(window, dict) or window.get("trainingEligible") is False:
            continue
        if window.get("status") not in {"trusted", "accepted-manually"}:
            continue
        start = finite(
            window.get("sourceStart", window.get("sourceStartSeconds")), math.nan
        )
        end = finite(
            window.get("sourceEnd", window.get("sourceEndSeconds")), math.nan
        )
        if math.isfinite(start) and math.isfinite(end) and end > start:
            ranges.append((start, end))
    if not ranges:
        raise ValueError(f"No trusted source ranges found in {path}")
    return ranges


def onset_groups(notes: list[dict[str, Any]], window: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in notes:
        if not groups or note["time"] - groups[-1][0]["time"] > window:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def greedy_match(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
    pitch_class: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], set[int], set[int]]:
    available = set(range(len(candidate)))
    matched_reference: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference_index, expected in enumerate(reference):
        options = [
            index
            for index in available
            if abs(candidate[index]["time"] - expected["time"]) <= tolerance
            and (
                candidate[index]["midi"] % 12 == expected["midi"] % 12
                if pitch_class
                else candidate[index]["midi"] == expected["midi"]
            )
        ]
        if not options:
            continue
        selected = min(
            options,
            key=lambda index: (
                abs(candidate[index]["time"] - expected["time"]),
                abs(candidate[index]["midi"] - expected["midi"]),
                index,
            ),
        )
        available.remove(selected)
        matched_reference.add(reference_index)
        pairs.append((expected, candidate[selected]))
    return pairs, matched_reference, set(range(len(candidate))) - available


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6) if values else 0.0,
        "p10": round(quantile(values, 0.10), 6),
        "median": round(quantile(values, 0.50), 6),
        "p90": round(quantile(values, 0.90), 6),
        "minimum": round(min(values), 6) if values else 0.0,
        "maximum": round(max(values), 6) if values else 0.0,
    }


def group_summary(notes: list[dict[str, Any]], window: float) -> dict[str, Any]:
    groups = onset_groups(notes, window)
    multi = [group for group in groups if len(group) > 1]
    spreads = [
        max(note["velocity"] for note in group) - min(note["velocity"] for note in group)
        for group in multi
    ]
    by_size: dict[int, list[float]] = defaultdict(list)
    for group in groups:
        size = min(8, len({int(note["midi"]) for note in group}))
        by_size[size].append(median(note["velocity"] for note in group))
    return {
        "groups": len(groups),
        "multiNoteGroups": len(multi),
        "exactSharedVelocityGroups": sum(spread <= 1e-9 for spread in spreads),
        "sharedVelocityGroupShare": round(
            sum(spread <= 1e-9 for spread in spreads) / max(1, len(spreads)), 6
        ),
        "withinGestureSpread": distribution(spreads),
        "gestureVelocities": distribution(
            [median(note["velocity"] for note in group) for group in groups]
        ),
        "velocityByChordSize": [
            {"size": size, **distribution(values)}
            for size, values in sorted(by_size.items())
        ],
    }


def candidate_change_breakdown(
    notes: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for note in notes:
        if "velocityBeforeGestureCoherence" not in note:
            continue
        before = finite(note.get("velocityBeforeGestureCoherence"), note["velocity"])
        change = note["velocity"] - before
        groups[str(note.get(field) or "unknown")].append(change)
    return [
        {
            "name": name,
            "raised": sum(change > 0.005 for change in values),
            "lowered": sum(change < -0.005 for change in values),
            "unchanged": sum(abs(change) <= 0.005 for change in values),
            "change": distribution(values),
        }
        for name, values in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def grouped_match_recall(
    reference: list[dict[str, Any]], matched_indices: set[int], field: str
) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for index, note in enumerate(reference):
        if field == "velocityBand":
            velocity = note["velocity"]
            value = "quiet-<0.55" if velocity < 0.55 else "medium-0.55-0.79" if velocity < 0.80 else "strong->=0.80"
        elif field == "durationBand":
            duration = note["duration"]
            value = "short-<0.18s" if duration < 0.18 else "medium-0.18-0.54s" if duration < 0.55 else "long->=0.55s"
        else:
            value = str(note.get(field) or "unknown")
        totals[value] += 1
        if index in matched_indices:
            matched[value] += 1
    return [
        {
            "name": name,
            "referenceNotes": total,
            "matchedNotes": matched[name],
            "recall": round(matched[name] / total, 6),
        }
        for name, total in sorted(totals.items())
    ]


def analyze(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
    onset_window: float,
) -> dict[str, Any]:
    exact_pairs, exact_reference, exact_candidate = greedy_match(
        reference, candidate, tolerance, False
    )
    pc_pairs, pc_reference, pc_candidate = greedy_match(
        reference, candidate, tolerance, True
    )
    exact_errors = [
        observed["velocity"] - expected["velocity"]
        for expected, observed in exact_pairs
    ]
    del exact_reference, exact_candidate
    errors = [observed["velocity"] - expected["velocity"] for expected, observed in pc_pairs]
    changed = [
        finite(note.get("velocity"), 0.72)
        - finite(note.get("velocityBeforeGestureCoherence"), finite(note.get("velocity"), 0.72))
        for note in candidate
        if "velocityBeforeGestureCoherence" in note
    ]

    missing_by_key: Counter[str] = Counter()
    for index, note in enumerate(reference):
        if index not in pc_reference:
            missing_by_key[note["note"]] += 1
    extra_by_key: Counter[str] = Counter()
    extra_by_source: Counter[str] = Counter()
    extra_by_role: Counter[str] = Counter()
    for index, note in enumerate(candidate):
        if index in pc_candidate:
            continue
        extra_by_key[note["note"]] += 1
        extra_by_source[str(note.get("sourceInstrument") or note.get("instrument") or "unknown")] += 1
        extra_by_role[str(note.get("arrangementRole") or "unknown")] += 1

    def top(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
        return [{"name": name, "notes": count} for name, count in counter.most_common(limit)]

    return {
        "matching": {
            "toleranceSeconds": tolerance,
            "exactPitchMatches": len(exact_pairs),
            "pitchClassMatches": len(pc_pairs),
            "referenceNotes": len(reference),
            "candidateNotes": len(candidate),
            "pitchClassPrecision": round(len(pc_pairs) / max(1, len(candidate)), 6),
            "pitchClassRecall": round(len(pc_pairs) / max(1, len(reference)), 6),
        },
        "velocity": {
            "reference": distribution([note["velocity"] for note in reference]),
            "candidate": distribution([note["velocity"] for note in candidate]),
            "matchedExactPitch": {
                "notes": len(exact_errors),
                "meanSignedError": (
                    round(sum(exact_errors) / len(exact_errors), 6)
                    if exact_errors
                    else 0.0
                ),
                "meanAbsoluteError": (
                    round(sum(abs(error) for error in exact_errors) / len(exact_errors), 6)
                    if exact_errors
                    else 0.0
                ),
                "raisedByAtLeast010": sum(error >= 0.10 for error in exact_errors),
                "loweredByAtLeast010": sum(error <= -0.10 for error in exact_errors),
                "within010": sum(abs(error) <= 0.10 for error in exact_errors),
            },
            "matchedPitchClass": {
                "meanSignedError": round(sum(errors) / len(errors), 6) if errors else 0.0,
                "meanAbsoluteError": round(sum(abs(error) for error in errors) / len(errors), 6) if errors else 0.0,
                "raisedByAtLeast010": sum(error >= 0.10 for error in errors),
                "loweredByAtLeast010": sum(error <= -0.10 for error in errors),
                "within010": sum(abs(error) <= 0.10 for error in errors),
            },
            "candidateChangeFromPreGesture": {
                "changedNotes": len(changed),
                "raised": sum(change > 0.005 for change in changed),
                "lowered": sum(change < -0.005 for change in changed),
                "unchanged": sum(abs(change) <= 0.005 for change in changed),
                "change": distribution(changed),
                "byHand": candidate_change_breakdown(candidate, "hand"),
                "byRole": candidate_change_breakdown(candidate, "arrangementRole"),
                "bySourceInstrument": candidate_change_breakdown(
                    candidate, "sourceInstrument"
                ),
            },
        },
        "gestureCoherence": {
            "reference": group_summary(reference, onset_window),
            "candidate": group_summary(candidate, onset_window),
        },
        "omissions": {
            "notes": len(reference) - len(pc_pairs),
            "topKeys": top(missing_by_key),
            "recallByHand": grouped_match_recall(reference, pc_reference, "hand"),
            "recallByVelocity": grouped_match_recall(reference, pc_reference, "velocityBand"),
            "recallByDuration": grouped_match_recall(reference, pc_reference, "durationBand"),
        },
        "extras": {
            "notes": len(candidate) - len(pc_pairs),
            "topKeys": top(extra_by_key),
            "bySourceInstrument": top(extra_by_source),
            "byArrangementRole": top(extra_by_role),
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    match = report["matching"]
    velocity = report["velocity"]
    reference_groups = report["gestureCoherence"]["reference"]
    candidate_groups = report["gestureCoherence"]["candidate"]
    lines = [
        "# Piano velocity and note-selection audit",
        "",
        "## Headline",
        "",
        f"- Pitch-class/onset matches: {match['pitchClassMatches']} of {match['referenceNotes']} reference notes.",
        f"- Exact-key velocity MAE: {velocity['matchedExactPitch']['meanAbsoluteError']:.4f}.",
        f"- Octave-agnostic velocity MAE: {velocity['matchedPitchClass']['meanAbsoluteError']:.4f}.",
        f"- Reference shared-velocity chord groups: {reference_groups['sharedVelocityGroupShare']:.1%}.",
        f"- Candidate shared-velocity chord groups: {candidate_groups['sharedVelocityGroupShare']:.1%}.",
        f"- Candidate notes raised by the gesture pass: {velocity['candidateChangeFromPreGesture']['raised']}.",
        f"- Candidate notes lowered by the gesture pass: {velocity['candidateChangeFromPreGesture']['lowered']}.",
        "",
        "## Most frequently missed reference keys",
        "",
    ]
    lines.extend(
        f"- {item['name']}: {item['notes']}" for item in report["omissions"]["topKeys"]
    )
    lines.extend(["", "## Most frequent extra candidate sources", ""])
    lines.extend(
        f"- {item['name']}: {item['notes']}" for item in report["extras"]["bySourceInstrument"]
    )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This is a fixed-song diagnostic. It measures the supplied reference and cannot prove unseen-song generalization or commercial rights.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--hand-split-midi", type=int, default=60)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument(
        "--candidate-alignment",
        help="Restrict candidate notes to trusted source windows from this alignment report.",
    )
    parser.add_argument("--candidate-start-seconds", type=float)
    parser.add_argument("--candidate-end-seconds", type=float)
    args = parser.parse_args()
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    candidate_alignment_path = (
        Path(args.candidate_alignment).resolve() if args.candidate_alignment else None
    )
    candidate_ranges = (
        trusted_source_ranges(candidate_alignment_path)
        if candidate_alignment_path is not None
        else None
    )
    report = {
        "schema": "polymath-piano-velocity-selection-audit-v1",
        "inputs": {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "referenceTransposeSemitones": args.reference_transpose_semitones,
            "handSplitMidi": args.hand_split_midi,
            "candidateAlignment": (
                str(candidate_alignment_path) if candidate_alignment_path else None
            ),
            "candidateStartSeconds": args.candidate_start_seconds,
            "candidateEndSeconds": args.candidate_end_seconds,
            "candidateTrustedSourceRanges": candidate_ranges,
        },
        **analyze(
            load_notes(
                reference_path,
                args.reference_transpose_semitones,
                args.training_only,
                args.hand_split_midi,
            ),
            load_notes(
                candidate_path,
                hand_split_midi=args.hand_split_midi,
                start_seconds=args.candidate_start_seconds,
                end_seconds=args.candidate_end_seconds,
                include_ranges=candidate_ranges,
            ),
            max(0.001, args.tolerance_seconds),
            max(0.001, args.onset_window_seconds),
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        markdown_path = Path(args.markdown).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **report["matching"], **report["velocity"]["matchedPitchClass"]}, indent=2))


if __name__ == "__main__":
    main()
