"""Explain where a full-mix piano reduction succeeds and fails.

This is a diagnostic companion to ``evaluate_piano_arranger.py``.  It uses the
same frozen alignment, trusted time ranges, and greedy matching rules, but
keeps the arranger metadata attached to every note.  The report answers three
different questions that a single F1 score cannot:

* Did the upstream transcription contain the desired pitch class at all?
* If it did, did the piano arranger retain a useful note near that time?
* Which source instruments, arrangement roles, pitch bands, and time windows
  account for the remaining false positives?

The tool never trains a profile and never changes a production checkpoint.
Development songs can therefore be inspected repeatedly without pretending
that they remain untouched holdouts.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:  # Support both ``python -m ml.training...`` and direct script execution.
    from .evaluate_piano_arranger import (
        evaluate,
        load_json,
        map_reference_notes,
        monotonic_anchors,
        normalize_notes,
        notes_inside_ranges,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover - exercised by the CLI integration path.
    from evaluate_piano_arranger import (
        evaluate,
        load_json,
        map_reference_notes,
        monotonic_anchors,
        normalize_notes,
        notes_inside_ranges,
        trusted_source_ranges,
    )


PITCH_BANDS = (
    (0, 47, "bass-A0-B2"),
    (48, 59, "lower-C3-B3"),
    (60, 71, "middle-C4-B4"),
    (72, 83, "upper-C5-B5"),
    (84, 127, "high-C6-G9"),
)

PERCUSSION_INSTRUMENTS = {"drums", "timpani", "percussion"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pitch_band(midi: int) -> str:
    for minimum, maximum, name in PITCH_BANDS:
        if minimum <= midi <= maximum:
            return name
    return "invalid"


def normalize_notes_with_metadata(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize notes while retaining provenance used by the arranger."""

    normalized: list[dict[str, Any]] = []
    for payload_index, item in enumerate(payload.get("notes", [])):
        if not isinstance(item, dict):
            continue
        try:
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        if not (0 <= midi <= 127 and time >= 0):
            continue
        if not (math.isfinite(time) and math.isfinite(duration)):
            continue
        note = dict(item)
        note.update(
            {
                "midi": midi,
                "time": time,
                "duration": duration,
                "visualDuration": _optional_duration(item, "visualDuration", duration),
                "audioDuration": _optional_duration(item, "audioDuration", duration),
                "_analysisPayloadIndex": payload_index,
            }
        )
        normalized.append(note)
    normalized.sort(key=lambda note: (float(note["time"]), int(note["midi"]), int(note["_analysisPayloadIndex"])))
    for analysis_index, note in enumerate(normalized):
        note["_analysisIndex"] = analysis_index
    return normalized


def _optional_duration(item: dict[str, Any], field: str, fallback: float) -> float:
    try:
        value = float(item.get(field, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback
    return max(0.01, value) if math.isfinite(value) else fallback


def filter_metadata_notes(
    notes: list[dict[str, Any]],
    ranges: list[tuple[float, float]] | None,
) -> list[dict[str, Any]]:
    if ranges is None:
        return notes
    return [
        note
        for note in notes
        if any(start <= float(note["time"]) < end for start, end in ranges)
    ]


def greedy_match_indices(
    reference: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    tolerance: float,
    *,
    octave_equivalent: bool = False,
) -> list[tuple[int, int]]:
    """Return stable reference/observed indices using evaluator semantics."""

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, note in enumerate(observed):
        key = int(note["midi"]) % 12 if octave_equivalent else int(note["midi"])
        grouped[key].append((index, note))
    times = {
        key: [float(note["time"]) for _index, note in values]
        for key, values in grouped.items()
    }
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for reference_index, target in enumerate(reference):
        key = int(target["midi"]) % 12 if octave_equivalent else int(target["midi"])
        pool = grouped.get(key, [])
        if not pool:
            continue
        target_time = float(target["time"])
        position = bisect.bisect_left(times[key], target_time)
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        left = position - 1
        while left >= 0 and target_time - times[key][left] <= tolerance:
            observed_index, candidate = pool[left]
            if observed_index not in used:
                candidates.append((abs(float(candidate["time"]) - target_time), observed_index, candidate))
            left -= 1
        right = position
        while right < len(pool) and times[key][right] - target_time <= tolerance:
            observed_index, candidate = pool[right]
            if observed_index not in used:
                candidates.append((abs(float(candidate["time"]) - target_time), observed_index, candidate))
            right += 1
        if not candidates:
            continue
        _distance, observed_index, _best = min(
            candidates,
            key=lambda item: (
                item[0],
                abs(int(item[2]["midi"]) - int(target["midi"])),
                item[1],
            ),
        )
        used.add(observed_index)
        matches.append((reference_index, observed_index))
    return matches


def group_precision_rows(
    notes: list[dict[str, Any]],
    exact_observed: set[int],
    pitch_class_observed: set[int],
    field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        value = pitch_band(int(note["midi"])) if field == "pitchBand" else str(note.get(field) or "unknown")
        groups[value].append(index)
    rows = []
    for name, indices in groups.items():
        count = len(indices)
        exact = sum(index in exact_observed for index in indices)
        pitch_class = sum(index in pitch_class_observed for index in indices)
        matched_probabilities = [
            float(notes[index]["selectionProbability"])
            for index in indices
            if index in pitch_class_observed
            and isinstance(notes[index].get("selectionProbability"), (int, float))
        ]
        unmatched_probabilities = [
            float(notes[index]["selectionProbability"])
            for index in indices
            if index not in pitch_class_observed
            and isinstance(notes[index].get("selectionProbability"), (int, float))
        ]
        rows.append(
            {
                "name": name,
                "notes": count,
                "share": round(count / max(1, len(notes)), 6),
                "exact100Matches": exact,
                "exact100Precision": round(exact / max(1, count), 6),
                "pitchClass250Matches": pitch_class,
                "pitchClass250Precision": round(pitch_class / max(1, count), 6),
                "unmatchedAtPitchClass250": count - pitch_class,
                "matchedMedianSelectionProbability": (
                    round(median(matched_probabilities), 6)
                    if matched_probabilities
                    else None
                ),
                "unmatchedMedianSelectionProbability": (
                    round(median(unmatched_probabilities), 6)
                    if unmatched_probabilities
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["unmatchedAtPitchClass250"]), str(row["name"])))


def diagnostic_for_output(
    reference: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_matches = greedy_match_indices(reference, observed, 0.1)
    pitch_class_matches = greedy_match_indices(reference, observed, 0.25, octave_equivalent=True)
    exact_observed = {observed_index for _reference_index, observed_index in exact_matches}
    pitch_class_observed = {observed_index for _reference_index, observed_index in pitch_class_matches}
    return {
        "metrics": evaluate(reference, observed),
        "bySourceInstrument": group_precision_rows(
            observed, exact_observed, pitch_class_observed, "sourceInstrument"
        ),
        "byArrangementRole": group_precision_rows(
            observed, exact_observed, pitch_class_observed, "arrangementRole"
        ),
        "byPitchBand": group_precision_rows(
            observed, exact_observed, pitch_class_observed, "pitchBand"
        ),
    }


def upstream_recall_ceiling(
    reference: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    # Percussion events can carry arbitrary MIDI numbers in a transcription.
    # Counting a snare labelled MIDI 38 as support for a sung D creates a false
    # melody ceiling and falsely blames the arranger when it correctly removes
    # drums.  The product contract is pitched piano reduction, so only pitched
    # upstream events are admissible evidence here.
    pitched_raw = [
        note
        for note in raw
        if str(note.get("instrument") or "").strip().lower()
        not in PERCUSSION_INSTRUMENTS
    ]
    raw_matches = greedy_match_indices(
        reference, pitched_raw, 0.25, octave_equivalent=True
    )
    candidate_matches = greedy_match_indices(reference, candidate, 0.25, octave_equivalent=True)
    raw_reference = {reference_index for reference_index, _observed_index in raw_matches}
    candidate_reference = {reference_index for reference_index, _observed_index in candidate_matches}
    both = raw_reference & candidate_reference
    arranger_missed = raw_reference - candidate_reference
    upstream_missing = set(range(len(reference))) - raw_reference
    return {
        "referenceNotes": len(reference),
        "rawNotes": len(raw),
        "rawPitchedNotes": len(pitched_raw),
        "excludedPercussionNotes": len(raw) - len(pitched_raw),
        "rawPitchClassMatches250ms": len(raw_reference),
        "rawPitchClassRecallCeiling250ms": round(len(raw_reference) / max(1, len(reference)), 6),
        "candidatePitchClassMatches250ms": len(candidate_reference),
        "candidatePitchClassRecall250ms": round(len(candidate_reference) / max(1, len(reference)), 6),
        "retainedFromRawSupport": len(both),
        "arrangerMissedDespiteRawSupport": len(arranger_missed),
        "upstreamMissingAt250ms": len(upstream_missing),
        "diagnosticCaveat": (
            "Pitch-class support excludes percussion but remains an optimistic "
            "ceiling; dense pitched stems can contain coincidental same-class notes."
        ),
    }


def window_diagnostics(
    reference: list[dict[str, Any]],
    baseline: list[dict[str, Any]] | None,
    candidate: list[dict[str, Any]],
    ranges: list[tuple[float, float]] | None,
    window_seconds: float,
) -> list[dict[str, Any]]:
    if ranges is None:
        maximum = max(
            [float(note["time"]) for note in reference + candidate] or [0.0]
        )
        ranges = [(0.0, maximum + window_seconds)]
    rows: list[dict[str, Any]] = []
    for range_start, range_end in ranges:
        start = range_start
        while start < range_end:
            end = min(range_end, start + window_seconds)
            ref_notes = [note for note in reference if start <= float(note["time"]) < end]
            candidate_notes = [note for note in candidate if start <= float(note["time"]) < end]
            if ref_notes or candidate_notes:
                candidate_metrics = evaluate(ref_notes, candidate_notes)
                baseline_metrics = None
                if baseline is not None:
                    baseline_notes = [note for note in baseline if start <= float(note["time"]) < end]
                    baseline_metrics = evaluate(ref_notes, baseline_notes)
                row = {
                    "startSeconds": round(start, 4),
                    "endSeconds": round(end, 4),
                    "referenceNotes": len(ref_notes),
                    "candidateNotes": len(candidate_notes),
                    "candidateExactF1_100ms": candidate_metrics["exactPitchOnset100ms"]["f1"],
                    "candidatePitchClassF1_250ms": candidate_metrics["pitchClassOnset250ms"]["f1"],
                }
                if baseline_metrics is not None:
                    row["baselineExactF1_100ms"] = baseline_metrics["exactPitchOnset100ms"]["f1"]
                    row["baselinePitchClassF1_250ms"] = baseline_metrics["pitchClassOnset250ms"]["f1"]
                    row["exactF1Delta"] = round(
                        float(row["candidateExactF1_100ms"])
                        - float(row["baselineExactF1_100ms"]),
                        6,
                    )
                    row["pitchClassF1Delta"] = round(
                        float(row["candidatePitchClassF1_250ms"])
                        - float(row["baselinePitchClassF1_250ms"]),
                        6,
                    )
                rows.append(row)
            start = end
    return sorted(
        rows,
        key=lambda row: (
            float(row["candidatePitchClassF1_250ms"]),
            float(row["candidateExactF1_100ms"]),
            float(row["startSeconds"]),
        ),
    )


def compact_note(note: dict[str, Any]) -> dict[str, Any]:
    return {
        key: note.get(key)
        for key in (
            "midi",
            "time",
            "duration",
            "audioDuration",
            "sourceInstrument",
            "arrangementRole",
            "selectionProbability",
            "sourceIndex",
        )
        if key in note
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--worst-windows", type=int, default=12)
    args = parser.parse_args()

    paths = {
        "reference": Path(args.reference).resolve(),
        "alignment": Path(args.alignment).resolve(),
        "candidate": Path(args.candidate).resolve(),
        "raw": Path(args.raw).resolve(),
    }
    if args.baseline:
        paths["baseline"] = Path(args.baseline).resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")
    if args.reference_transpose_semitones % 12 != 0:
        raise ValueError("reference transpose must be an octave multiple")
    if args.window_seconds <= 0:
        raise ValueError("window seconds must be positive")

    alignment = load_json(paths["alignment"])
    anchors = monotonic_anchors(alignment)
    ranges = trusted_source_ranges(alignment)
    reference = normalize_notes(
        load_json(paths["reference"]),
        transpose_semitones=args.reference_transpose_semitones,
    )
    reference = notes_inside_ranges(map_reference_notes(reference, anchors), ranges)
    candidate = filter_metadata_notes(
        normalize_notes_with_metadata(load_json(paths["candidate"])), ranges
    )
    raw = filter_metadata_notes(normalize_notes_with_metadata(load_json(paths["raw"])), ranges)
    baseline = None
    if "baseline" in paths:
        baseline = filter_metadata_notes(
            normalize_notes_with_metadata(load_json(paths["baseline"])), ranges
        )

    candidate_analysis = diagnostic_for_output(reference, candidate)
    baseline_analysis = diagnostic_for_output(reference, baseline) if baseline is not None else None
    windows = window_diagnostics(
        reference, baseline, candidate, ranges, args.window_seconds
    )
    exact_matches = greedy_match_indices(reference, candidate, 0.1)
    pitch_class_matches = greedy_match_indices(reference, candidate, 0.25, octave_equivalent=True)
    exact_observed = {observed_index for _reference_index, observed_index in exact_matches}
    pitch_class_observed = {observed_index for _reference_index, observed_index in pitch_class_matches}
    worst_false_positives = [
        compact_note(note)
        for index, note in enumerate(candidate)
        if index not in pitch_class_observed
    ][:100]

    report: dict[str, Any] = {
        "schema": "polymath-full-mix-arranger-diagnostic-v1",
        "fixedEvidencePolicy": (
            "Uses the supplied frozen alignment and only its pre-approved trusted ranges."
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "referenceTransposeSemitones": args.reference_transpose_semitones,
        "approvedSourceRanges": ranges,
        "referenceNotes": len(reference),
        "rawNotes": len(raw),
        "candidate": candidate_analysis,
        "upstreamVsReduction": upstream_recall_ceiling(reference, raw, candidate),
        "worstWindows": windows[: max(0, args.worst_windows)],
        "candidateUnmatchedSample": worst_false_positives,
        "candidateMatchCounts": {
            "exact100": len(exact_observed),
            "pitchClass250": len(pitch_class_observed),
        },
    }
    if baseline_analysis is not None:
        report["baseline"] = baseline_analysis
        report["candidateMinusBaseline"] = {
            "exactF1_100ms": round(
                float(candidate_analysis["metrics"]["exactPitchOnset100ms"]["f1"])
                - float(baseline_analysis["metrics"]["exactPitchOnset100ms"]["f1"]),
                6,
            ),
            "pitchClassF1_250ms": round(
                float(candidate_analysis["metrics"]["pitchClassOnset250ms"]["f1"])
                - float(baseline_analysis["metrics"]["pitchClassOnset250ms"]["f1"]),
                6,
            ),
        }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "referenceNotes": len(reference),
                "rawNotes": len(raw),
                "candidateNotes": len(candidate),
                "upstreamVsReduction": report["upstreamVsReduction"],
                "candidateMinusBaseline": report.get("candidateMinusBaseline"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
