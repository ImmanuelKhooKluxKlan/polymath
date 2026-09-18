"""Compare two arranger candidates against identical trusted pianist labels.

The report is designed for iterative listening work. It keeps exact-key and
octave-agnostic evidence separate, compares velocity only on the same matched
reference notes, and exposes where a challenger gains or loses notes by hand,
key, velocity, duration, chord size, and five-second section.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

try:
    from ml.training.analyze_piano_velocity_style import (
        load_notes,
        note_name,
        onset_groups,
        trusted_source_ranges,
    )
except ModuleNotFoundError:  # Allow direct `python ml/training/...py` use.
    from analyze_piano_velocity_style import (
        load_notes,
        note_name,
        onset_groups,
        trusted_source_ranges,
    )


def match_indices(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
    *,
    pitch_class: bool,
) -> dict[int, int]:
    available = set(range(len(candidate)))
    result: dict[int, int] = {}
    for reference_index, expected in enumerate(reference):
        options = [
            index
            for index in available
            if abs(float(candidate[index]["time"]) - float(expected["time"])) <= tolerance
            and (
                int(candidate[index]["midi"]) % 12 == int(expected["midi"]) % 12
                if pitch_class
                else int(candidate[index]["midi"]) == int(expected["midi"])
            )
        ]
        if not options:
            continue
        selected = min(
            options,
            key=lambda index: (
                abs(float(candidate[index]["time"]) - float(expected["time"])),
                abs(int(candidate[index]["midi"]) - int(expected["midi"])),
                index,
            ),
        )
        available.remove(selected)
        result[reference_index] = selected
    return result


def velocity_band(value: float) -> str:
    if value < 0.45:
        return "00-quiet-<0.45"
    if value < 0.55:
        return "01-quiet-0.45-0.54"
    if value < 0.65:
        return "02-soft-0.55-0.64"
    if value < 0.75:
        return "03-medium-0.65-0.74"
    if value < 0.85:
        return "04-strong-0.75-0.84"
    return "05-strong->=0.85"


def duration_band(value: float) -> str:
    if value < 0.18:
        return "short-<0.18s"
    if value < 0.55:
        return "medium-0.18-0.54s"
    return "long->=0.55s"


def pitch_band(midi: int) -> str:
    if midi < 48:
        return "A1-B2"
    if midi < 60:
        return "C3-B3"
    if midi < 72:
        return "C4-B4"
    if midi < 84:
        return "C5-B5"
    return "C6-C7"


def reference_chord_sizes(
    reference: list[dict[str, Any]], onset_window: float
) -> dict[int, int]:
    result: dict[int, int] = {}
    by_original_index = {int(note["_index"]): index for index, note in enumerate(reference)}
    for group in onset_groups(reference, onset_window):
        size = min(6, len({int(note["midi"]) for note in group}))
        for note in group:
            result[by_original_index[int(note["_index"])]] = size
    return result


def error_summary(values: list[float]) -> dict[str, Any]:
    return {
        "notes": len(values),
        "meanSignedError": round(sum(values) / len(values), 6) if values else None,
        "meanAbsoluteError": (
            round(sum(abs(value) for value in values) / len(values), 6)
            if values
            else None
        ),
        "within005": sum(abs(value) <= 0.05 for value in values),
        "within010": sum(abs(value) <= 0.10 for value in values),
    }


def group_comparison(
    reference: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    baseline_matches: dict[int, int],
    candidate_matches: dict[int, int],
    labels: list[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)

    output: list[dict[str, Any]] = []
    for label, indices in groups.items():
        baseline_indices = [index for index in indices if index in baseline_matches]
        candidate_indices = [index for index in indices if index in candidate_matches]
        common = [
            index
            for index in indices
            if index in baseline_matches and index in candidate_matches
        ]
        baseline_errors = [
            float(baseline[baseline_matches[index]]["velocity"])
            - float(reference[index]["velocity"])
            for index in common
        ]
        candidate_errors = [
            float(candidate[candidate_matches[index]]["velocity"])
            - float(reference[index]["velocity"])
            for index in common
        ]
        baseline_closer = 0
        candidate_closer = 0
        tied = 0
        for baseline_error, candidate_error in zip(baseline_errors, candidate_errors):
            difference = abs(candidate_error) - abs(baseline_error)
            if difference < -0.005:
                candidate_closer += 1
            elif difference > 0.005:
                baseline_closer += 1
            else:
                tied += 1
        baseline_summary = error_summary(baseline_errors)
        candidate_summary = error_summary(candidate_errors)
        output.append(
            {
                "name": label,
                "referenceNotes": len(indices),
                "baselineMatches": len(baseline_indices),
                "candidateMatches": len(candidate_indices),
                "candidateMinusBaselineMatches": len(candidate_indices)
                - len(baseline_indices),
                "commonMatches": len(common),
                "baselineVelocity": baseline_summary,
                "candidateVelocity": candidate_summary,
                "candidateMinusBaselineVelocityMae": (
                    round(
                        float(candidate_summary["meanAbsoluteError"])
                        - float(baseline_summary["meanAbsoluteError"]),
                        6,
                    )
                    if common
                    else None
                ),
                "candidateCloserVelocity": candidate_closer,
                "baselineCloserVelocity": baseline_closer,
                "velocityTies": tied,
            }
        )
    return sorted(output, key=lambda row: (-row["referenceNotes"], row["name"]))


def top_unmatched(
    notes: list[dict[str, Any]], matches: dict[int, int]
) -> dict[str, Any]:
    matched = set(matches.values())
    unmatched = [note for index, note in enumerate(notes) if index not in matched]

    def top(key: Callable[[dict[str, Any]], str], limit: int = 20) -> list[dict[str, Any]]:
        counts = Counter(key(note) for note in unmatched)
        return [{"name": name, "notes": count} for name, count in counts.most_common(limit)]

    return {
        "notes": len(unmatched),
        "keys": top(lambda note: note_name(int(note["midi"]))),
        "sourceInstruments": top(
            lambda note: str(note.get("sourceInstrument") or note.get("instrument") or "unknown")
        ),
        "roles": top(lambda note: str(note.get("arrangementRole") or "unknown")),
        "velocityBands": top(lambda note: velocity_band(float(note["velocity"]))),
    }


def compare(
    reference: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
    onset_window: float,
) -> dict[str, Any]:
    baseline_exact = match_indices(reference, baseline, tolerance, pitch_class=False)
    candidate_exact = match_indices(reference, candidate, tolerance, pitch_class=False)
    baseline_pc = match_indices(reference, baseline, tolerance, pitch_class=True)
    candidate_pc = match_indices(reference, candidate, tolerance, pitch_class=True)
    chord_sizes = reference_chord_sizes(reference, onset_window)

    categories = {
        "hand": [str(note["hand"]) for note in reference],
        "referenceVelocityBand": [
            velocity_band(float(note["velocity"])) for note in reference
        ],
        "referenceDurationBand": [
            duration_band(float(note["duration"])) for note in reference
        ],
        "referencePitchBand": [pitch_band(int(note["midi"])) for note in reference],
        "referenceKey": [note_name(int(note["midi"])) for note in reference],
        "referenceChordSize": [str(chord_sizes.get(index, 1)) for index in range(len(reference))],
        "sourceFiveSecondWindow": [
            f"{int(float(note['time']) // 5) * 5:03d}-{int(float(note['time']) // 5) * 5 + 5:03d}s"
            for note in reference
        ],
    }
    common = sorted(set(baseline_exact) & set(candidate_exact))
    baseline_common_errors = [
        float(baseline[baseline_exact[index]]["velocity"])
        - float(reference[index]["velocity"])
        for index in common
    ]
    candidate_common_errors = [
        float(candidate[candidate_exact[index]]["velocity"])
        - float(reference[index]["velocity"])
        for index in common
    ]
    candidate_closer = sum(
        abs(candidate_error) < abs(baseline_error) - 0.005
        for baseline_error, candidate_error in zip(
            baseline_common_errors, candidate_common_errors
        )
    )
    baseline_closer = sum(
        abs(baseline_error) < abs(candidate_error) - 0.005
        for baseline_error, candidate_error in zip(
            baseline_common_errors, candidate_common_errors
        )
    )
    return {
        "schema": "polymath-pianist-candidate-comparison-v1",
        "evidenceBoundary": "Fixed-song trusted-window diagnostic; not unseen-song proof.",
        "headline": {
            "referenceNotes": len(reference),
            "baselineNotes": len(baseline),
            "candidateNotes": len(candidate),
            "baselineExactMatches": len(baseline_exact),
            "candidateExactMatches": len(candidate_exact),
            "candidateMinusBaselineExactMatches": len(candidate_exact) - len(baseline_exact),
            "baselinePitchClassMatches": len(baseline_pc),
            "candidatePitchClassMatches": len(candidate_pc),
            "candidateMinusBaselinePitchClassMatches": len(candidate_pc) - len(baseline_pc),
            "commonExactMatches": len(common),
            "baselineCommonVelocity": error_summary(baseline_common_errors),
            "candidateCommonVelocity": error_summary(candidate_common_errors),
            "candidateCloserVelocity": candidate_closer,
            "baselineCloserVelocity": baseline_closer,
            "velocityTies": len(common) - candidate_closer - baseline_closer,
        },
        "byReference": {
            name: group_comparison(
                reference,
                baseline,
                candidate,
                baseline_exact,
                candidate_exact,
                labels,
            )
            for name, labels in categories.items()
        },
        "unmatchedCandidate": {
            "baseline": top_unmatched(baseline, baseline_exact),
            "candidate": top_unmatched(candidate, candidate_exact),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--candidate-end-seconds", type=float)
    args = parser.parse_args()

    reference_path = Path(args.reference).resolve()
    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    alignment_path = Path(args.alignment).resolve()
    ranges = trusted_source_ranges(alignment_path)
    reference = load_notes(
        reference_path,
        transpose=args.reference_transpose_semitones,
        training_only=True,
    )
    common_load = {
        "include_ranges": ranges,
        "end_seconds": args.candidate_end_seconds,
    }
    baseline = load_notes(baseline_path, **common_load)
    candidate = load_notes(candidate_path, **common_load)
    report = {
        "inputs": {
            "reference": str(reference_path),
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "alignment": str(alignment_path),
            "referenceTransposeSemitones": args.reference_transpose_semitones,
            "candidateEndSeconds": args.candidate_end_seconds,
            "trustedSourceRanges": ranges,
        },
        **compare(
            reference,
            baseline,
            candidate,
            max(0.001, args.tolerance_seconds),
            max(0.001, args.onset_window_seconds),
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["headline"]}, indent=2))


if __name__ == "__main__":
    main()
