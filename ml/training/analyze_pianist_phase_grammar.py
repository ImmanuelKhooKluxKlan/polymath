"""Audit beat-phase, hand and voicing grammar in approved piano targets.

This script is analysis-only.  It maps each approved reference onto the source
clock, estimates its shortest recurring rhythmic pulse, and expresses every
gesture with transposition-invariant properties.  It never copies target notes
into an inference payload and never modifies model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from .analyze_pianist_gesture_patterns import (
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )
    from .analyze_pianist_reduction_grammar import explicit_hand, family
except ImportError:  # pragma: no cover - direct CLI execution
    from analyze_pianist_gesture_patterns import (
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )
    from analyze_pianist_reduction_grammar import explicit_hand, family


def rounded(value: float) -> float:
    return round(float(value), 6)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(ordered),
        "minimum": rounded(ordered[0]),
        "median": rounded(median(ordered)),
        "mean": rounded(sum(ordered) / len(ordered)),
        "maximum": rounded(ordered[-1]),
    }


def gesture_time(group: list[dict[str, Any]]) -> float:
    return float(group[0]["time"])


def gesture_occupancy(group: list[dict[str, Any]]) -> str:
    hands = {explicit_hand(note) for note in group}
    left = "left" in hands
    right = "right" in hands
    if left and right:
        return "both"
    if left:
        return "left-only"
    return "right-only"


def estimate_pulse(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Fit the small pulse whose integer multiples best explain target IOIs."""

    gaps = [
        gesture_time(right) - gesture_time(left)
        for left, right in zip(groups, groups[1:])
        if 0.055 <= gesture_time(right) - gesture_time(left) <= 0.90
    ]
    trials: list[tuple[float, float, float]] = []
    for step in range(80, 321):
        pulse = step / 1000.0
        residuals = []
        exactish = 0
        for gap in gaps:
            multiple = max(1, min(8, round(gap / pulse)))
            residual = abs(gap - multiple * pulse) / pulse
            residuals.append(min(1.0, residual))
            exactish += residual <= 0.12
        mean_residual = sum(residuals) / max(1, len(residuals))
        # Prefer a genuinely short subdivision, but only after fit and coverage.
        score = mean_residual + 0.18 * (1.0 - exactish / max(1, len(gaps))) + 0.025 * pulse
        trials.append((score, pulse, exactish / max(1, len(gaps))))
    score, pulse, coverage = min(trials)
    multiples = Counter(
        str(max(1, min(8, round(gap / pulse))))
        for gap in gaps
    )
    return {
        "seconds": rounded(pulse),
        "fitScore": rounded(score),
        "withinTwelvePercentShare": rounded(coverage),
        "gapMultiples": dict(sorted(multiples.items(), key=lambda item: int(item[0]))),
    }


def load_song(row: dict[str, Any], onset_window: float) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    alignment = json.loads(Path(row["alignment"]).read_text(encoding="utf-8-sig"))
    ranges = trusted_source_ranges(alignment)
    reference_end_value = finite(row.get("referenceEndSeconds"), math.nan)
    reference_end = reference_end_value if math.isfinite(reference_end_value) else None
    candidate_end_value = finite(row.get("candidateEndSeconds"), math.nan)
    candidate_end = candidate_end_value if math.isfinite(candidate_end_value) else None
    reference_payload = json.loads(Path(row["reference"]).read_text(encoding="utf-8-sig"))
    reference = normalize_notes(
        reference_payload,
        transpose=int(row.get("referenceTransposeSemitones") or 0),
        hard_end=reference_end,
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
    source_payload = json.loads(Path(row["source"]).read_text(encoding="utf-8-sig"))
    source = [
        note
        for note in normalize_notes(source_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
        and family(note.get("instrument")) not in {"voice", "percussion"}
    ]
    return group_onsets(reference, onset_window), group_onsets(source, onset_window)


def source_pool(
    groups: list[list[dict[str, Any]]], time: float, radius: float = 0.25
) -> list[dict[str, Any]]:
    return [
        note
        for group in groups
        if abs(gesture_time(group) - time) <= radius
        for note in group
    ]


def phase_rows(
    reference: list[list[dict[str, Any]]],
    source: list[list[dict[str, Any]]],
    pulse: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_index = 0
    for index, group in enumerate(reference):
        time = gesture_time(group)
        previous_gap = time - gesture_time(reference[index - 1]) if index else 9.0
        if index == 0 or previous_gap >= max(0.75, 4.5 * pulse):
            run_index = 0
        else:
            run_index += max(1, round(previous_gap / pulse))
        local = source_pool(source, time)
        source_midis = sorted({int(note["midi"]) for note in local})
        source_pcs = {midi % 12 for midi in source_midis}
        target_midis = sorted({int(note["midi"]) for note in group})
        target_pcs = {midi % 12 for midi in target_midis}
        exact_supported = sum(midi in source_midis for midi in target_midis)
        pc_supported = sum(midi % 12 in source_pcs for midi in target_midis)
        source_shifts = []
        for midi in target_midis:
            equivalent = [source_midi for source_midi in source_midis if source_midi % 12 == midi % 12]
            if equivalent:
                anchor = min(equivalent, key=lambda source_midi: (abs(source_midi - midi), source_midi))
                source_shifts.append(midi - anchor)
        left = [midi for midi in target_midis if midi < 60]
        right = [midi for midi in target_midis if midi >= 60]
        octave_pairs = sum(
            1
            for position, midi in enumerate(target_midis)
            if any(other % 12 == midi % 12 for other in target_midis[position + 1 :])
        )
        rows.append(
            {
                "time": rounded(time),
                "phase2": run_index % 2,
                "phase4": run_index % 4,
                "phase8": run_index % 8,
                "occupancy": gesture_occupancy(group),
                "chordSize": len(target_midis),
                "leftSize": len(left),
                "rightSize": len(right),
                "octavePairs": octave_pairs,
                "exactSupported": exact_supported,
                "pitchClassSupported": pc_supported,
                "targetNotes": len(target_midis),
                "sourceShifts": source_shifts,
                "velocity": rounded(median(finite(note.get("velocity"), 0.7) for note in group)),
                "duration": rounded(median(finite(note.get("duration"), 0.2) for note in group)),
            }
        )
    return rows


def summarize_phase(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[key])].append(row)
    output = []
    for phase in sorted(grouped):
        values = grouped[phase]
        occupancy = Counter(str(row["occupancy"]) for row in values)
        output.append(
            {
                "phase": phase,
                "gestures": len(values),
                "occupancy": dict(occupancy),
                "occupancyShare": {
                    name: rounded(count / len(values)) for name, count in occupancy.items()
                },
                "meanChordSize": rounded(sum(int(row["chordSize"]) for row in values) / len(values)),
                "meanLeftSize": rounded(sum(int(row["leftSize"]) for row in values) / len(values)),
                "meanRightSize": rounded(sum(int(row["rightSize"]) for row in values) / len(values)),
                "octavePairGestureShare": rounded(sum(int(row["octavePairs"]) > 0 for row in values) / len(values)),
                "velocity": distribution(float(row["velocity"]) for row in values),
            }
        )
    return output


def analyze_song(row: dict[str, Any], onset_window: float) -> dict[str, Any]:
    reference, source = load_song(row, onset_window)
    pulse = estimate_pulse(reference)
    rows = phase_rows(reference, source, float(pulse["seconds"]))
    occupancies = Counter(str(row["occupancy"]) for row in rows)
    source_shifts = Counter(
        str(shift)
        for row in rows
        for shift in row["sourceShifts"]
    )
    octave_shifted = sum(
        count for shift, count in source_shifts.items() if int(shift) != 0
    )
    supported = sum(source_shifts.values())
    return {
        "id": row.get("id"),
        "referenceGestures": len(reference),
        "sourceAttackGroups": len(source),
        "pulse": pulse,
        "occupancy": dict(occupancies),
        "chordSize": dict(Counter(str(row["chordSize"]) for row in rows)),
        "meanChordSize": rounded(sum(int(row["chordSize"]) for row in rows) / max(1, len(rows))),
        "exactSourceSupportShare": rounded(sum(int(row["exactSupported"]) for row in rows) / max(1, sum(int(row["targetNotes"]) for row in rows))),
        "pitchClassSourceSupportShare": rounded(sum(int(row["pitchClassSupported"]) for row in rows) / max(1, sum(int(row["targetNotes"]) for row in rows))),
        "nearestSourceEquivalentShiftSemitones": dict(
            sorted(source_shifts.items(), key=lambda item: int(item[0]))
        ),
        "octaveRevoicingShareAmongPitchClassSupportedNotes": rounded(
            octave_shifted / max(1, supported)
        ),
        "phase2": summarize_phase(rows, "phase2"),
        "phase4": summarize_phase(rows, "phase4"),
        "phase8": summarize_phase(rows, "phase8"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    songs = [
        analyze_song(row, max(0.005, float(args.onset_window_seconds)))
        for row in manifest.get("songs") or []
    ]
    payload = {
        "schema": "polymath-pianist-phase-grammar-audit-v1",
        "evidenceBoundary": "Analysis only; references are used as labels, never as inference inputs. Kiss Me at/after 02:30 is excluded.",
        "manifest": str(manifest_path),
        "songs": songs,
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
                        "id": song["id"],
                        "pulse": song["pulse"],
                        "meanChordSize": song["meanChordSize"],
                        "exactSourceSupportShare": song["exactSourceSupportShare"],
                        "pitchClassSourceSupportShare": song["pitchClassSourceSupportShare"],
                    }
                    for song in songs
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
