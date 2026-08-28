"""Diagnose individual-instrument transcription errors against reviewed labels.

The report deliberately keeps raw model errors visible.  It separates missed
notes, false notes, wrong instruments/octaves, rapid retriggers, chopped or
overlong sustains, chord completeness, pitch ranges, and time hotspots.  This
is an evaluation tool, not playback cleanup.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from ml.training.muscriptor_tokens import (
    TokenEncodingError,
    canonical_instrument_name,
)


def _instrument(value: Any, default: str = "acoustic_piano") -> str:
    try:
        return canonical_instrument_name(value or default)
    except TokenEncodingError:
        return str(value or default).strip().lower().replace(" ", "_")


def _payload_notes(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    candidates = [payload.get("notes")]
    for parent in ("result", "output", "raw"):
        value = payload.get(parent)
        if isinstance(value, dict):
            candidates.append(value.get("notes"))
    return next((item for item in candidates if isinstance(item, list)), [])


def normalize_notes(
    source: Iterable[Any],
    default_instrument: str = "acoustic_piano",
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for source_index, note in enumerate(source):
        try:
            midi = int(note.get("midi", note.get("pitch")))
            start = float(note.get("time", note.get("start", note.get("startTime"))))
            duration = float(note.get("duration", 0.1))
            velocity = float(note.get("velocity", 0.75))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= midi <= 127 and start >= 0 and duration > 0:
            notes.append({
                "index": len(notes),
                "sourceIndex": source_index,
                "midi": midi,
                "time": start,
                "duration": duration,
                "velocity": max(0.0, min(1.0, velocity)),
                "instrument": _instrument(note.get("instrument"), default_instrument),
                "continuedFromPreviousClip": bool(note.get("continuedFromPreviousClip")),
                "continuesIntoNextClip": bool(note.get("continuesIntoNextClip")),
            })
    notes.sort(key=lambda item: (item["time"], item["instrument"], item["midi"]))
    for index, note in enumerate(notes):
        note["index"] = index
    return notes


def load_notes(path: Path, default_instrument: str = "acoustic_piano") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return normalize_notes(_payload_notes(payload), default_instrument)


def _better(candidate: tuple[int, float], incumbent: tuple[int, float]) -> bool:
    return candidate[0] > incumbent[0] or (
        candidate[0] == incumbent[0] and candidate[1] < incumbent[1] - 1e-12
    )


def _match_ordered_group(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    onset_tolerance: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Maximum-cardinality, minimum-error monotonic matching for one key."""

    rows, columns = len(reference), len(predicted)
    scores = [[(0, 0.0) for _ in range(columns + 1)] for _ in range(rows + 1)]
    choices = [["" for _ in range(columns + 1)] for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        choices[row][0] = "reference"
    for column in range(1, columns + 1):
        choices[0][column] = "predicted"
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best = scores[row - 1][column]
            choice = "reference"
            left = scores[row][column - 1]
            if _better(left, best):
                best, choice = left, "predicted"
            target = reference[row - 1]
            candidate = predicted[column - 1]
            onset_error = abs(candidate["time"] - target["time"])
            if onset_error <= onset_tolerance:
                previous = scores[row - 1][column - 1]
                duration_error = abs(candidate["duration"] - target["duration"])
                diagonal = (previous[0] + 1, previous[1] + onset_error + 0.02 * duration_error)
                if _better(diagonal, best) or diagonal == best:
                    best, choice = diagonal, "match"
            scores[row][column] = best
            choices[row][column] = choice
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    row, column = rows, columns
    while row or column:
        choice = choices[row][column]
        if choice == "match":
            matches.append((reference[row - 1], predicted[column - 1]))
            row -= 1
            column -= 1
        elif choice == "reference":
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return matches


def match_notes(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    onset_tolerance: float,
    instrument_aware: bool = True,
) -> list[dict[str, Any]]:
    reference_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    prediction_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for note in reference:
        key = (note["instrument"], note["midi"]) if instrument_aware else (note["midi"],)
        reference_groups[key].append(note)
    for note in predicted:
        key = (note["instrument"], note["midi"]) if instrument_aware else (note["midi"],)
        prediction_groups[key].append(note)
    matches: list[dict[str, Any]] = []
    for key, targets in reference_groups.items():
        candidates = prediction_groups.get(key, [])
        for target, candidate in _match_ordered_group(targets, candidates, onset_tolerance):
            matches.append({
                "reference": target,
                "predicted": candidate,
                "onsetErrorSeconds": candidate["time"] - target["time"],
                "offsetErrorSeconds": (
                    candidate["time"] + candidate["duration"]
                    - target["time"] - target["duration"]
                ),
            })
    return sorted(matches, key=lambda item: item["reference"]["time"])


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    mix = position - low
    return ordered[low] * (1 - mix) + ordered[high] * mix


def _prf(matched: int, reference_count: int, predicted_count: int) -> dict[str, float]:
    precision = matched / max(1, predicted_count)
    recall = matched / max(1, reference_count)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _frame_score(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    frame_seconds: float = 0.02,
) -> dict[str, Any]:
    def occupied(notes: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
        frames: set[tuple[str, int, int]] = set()
        for note in notes:
            first = max(0, math.floor(note["time"] / frame_seconds))
            last = max(first, math.ceil((note["time"] + note["duration"]) / frame_seconds))
            for frame in range(first, last):
                frames.add((note["instrument"], note["midi"], frame))
        return frames

    target_frames = occupied(reference)
    candidate_frames = occupied(predicted)
    matched = len(target_frames & candidate_frames)
    return {
        "frameMilliseconds": round(frame_seconds * 1000),
        "referenceFrames": len(target_frames),
        "predictedFrames": len(candidate_frames),
        "matchedFrames": matched,
        **_prf(matched, len(target_frames), len(candidate_frames)),
    }


def _is_cut_off(match: dict[str, Any]) -> bool:
    target = match["reference"]
    candidate = match["predicted"]
    early_by = -match["offsetErrorSeconds"]
    return (
        target["duration"] >= 0.15
        and candidate["duration"] < target["duration"] * 0.60
        and early_by > max(0.10, target["duration"] * 0.25)
    )


def _is_overlong(match: dict[str, Any]) -> bool:
    target = match["reference"]
    candidate = match["predicted"]
    return (
        target["duration"] >= 0.10
        and candidate["duration"] > target["duration"] * 1.75
        and match["offsetErrorSeconds"] > 0.15
    )


def _pitch_band(midi: int) -> str:
    if midi < 48:
        return "low_A0_to_B2"
    if midi < 72:
        return "middle_C3_to_B4"
    return "high_C5_to_G9"


def _rapid_retrigger_indices(notes: list[dict[str, Any]], seconds: float = 0.075) -> set[int]:
    result: set[int] = set()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        groups[(note["instrument"], note["midi"])].append(note)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item["time"])
        for previous, current in zip(ordered, ordered[1:]):
            if current["time"] - previous["time"] <= seconds:
                result.add(current["index"])
    return result


def _same_key_gap_profile(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def count(notes: list[dict[str, Any]], threshold: float) -> int:
        groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for note in notes:
            groups[(note["instrument"], note["midi"])].append(note)
        total = 0
        for group in groups.values():
            ordered = sorted(group, key=lambda item: item["time"])
            total += sum(
                current["time"] - previous["time"] <= threshold
                for previous, current in zip(ordered, ordered[1:])
            )
        return total

    result = []
    for milliseconds in (75, 100, 125, 150, 200, 250):
        threshold = milliseconds / 1000
        target = count(reference, threshold)
        candidate = count(predicted, threshold)
        result.append({
            "maximumGapMs": milliseconds,
            "referenceRepeats": target,
            "predictedRepeats": candidate,
            "excessPredictedRepeats": candidate - target,
        })
    return result


def _diagnose_substitutions(
    false_negatives: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    onset_tolerance: float,
) -> tuple[dict[int, str], dict[int, str]]:
    fn_causes: dict[int, str] = {}
    fp_causes: dict[int, str] = {}
    priorities = (
        ("wrongInstrument", lambda target, candidate: target["midi"] == candidate["midi"] and target["instrument"] != candidate["instrument"] and abs(target["time"] - candidate["time"]) <= onset_tolerance),
        ("octaveSubstitution", lambda target, candidate: target["instrument"] == candidate["instrument"] and target["midi"] != candidate["midi"] and abs(target["midi"] - candidate["midi"]) in {12, 24} and abs(target["time"] - candidate["time"]) <= onset_tolerance),
        ("nearPitchSubstitution", lambda target, candidate: target["instrument"] == candidate["instrument"] and 0 < abs(target["midi"] - candidate["midi"]) <= 2 and abs(target["time"] - candidate["time"]) <= onset_tolerance),
        ("timingNearMiss", lambda target, candidate: target["instrument"] == candidate["instrument"] and target["midi"] == candidate["midi"] and onset_tolerance < abs(target["time"] - candidate["time"]) <= 0.25),
    )
    for cause, predicate in priorities:
        candidates: list[tuple[float, int, int]] = []
        for target in false_negatives:
            if target["index"] in fn_causes:
                continue
            for candidate in false_positives:
                if candidate["index"] in fp_causes or not predicate(target, candidate):
                    continue
                cost = abs(target["time"] - candidate["time"]) + 0.01 * abs(target["midi"] - candidate["midi"])
                candidates.append((cost, target["index"], candidate["index"]))
        for _, target_index, candidate_index in sorted(candidates):
            if target_index not in fn_causes and candidate_index not in fp_causes:
                fn_causes[target_index] = cause
                fp_causes[candidate_index] = cause
    return fn_causes, fp_causes


def _chord_analysis(
    reference: list[dict[str, Any]],
    matched_reference_indices: set[int],
    grouping_seconds: float = 0.04,
) -> dict[str, int]:
    chords: list[list[dict[str, Any]]] = []
    for instrument in sorted({note["instrument"] for note in reference}):
        notes = sorted((note for note in reference if note["instrument"] == instrument), key=lambda item: item["time"])
        current: list[dict[str, Any]] = []
        anchor = -1.0
        for note in notes:
            if not current or note["time"] - anchor <= grouping_seconds:
                if not current:
                    anchor = note["time"]
                current.append(note)
            else:
                if len(current) >= 2:
                    chords.append(current)
                current = [note]
                anchor = note["time"]
        if len(current) >= 2:
            chords.append(current)
    complete = partial = missed = 0
    for chord in chords:
        found = sum(note["index"] in matched_reference_indices for note in chord)
        if found == len(chord):
            complete += 1
        elif found:
            partial += 1
        else:
            missed += 1
    return {"referenceChords": len(chords), "complete": complete, "partial": partial, "missed": missed}


def _instrument_breakdown(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    cut_off_indices: set[int],
    overlong_indices: set[int],
    rapid_indices: set[int],
) -> list[dict[str, Any]]:
    result = []
    instruments = sorted({note["instrument"] for note in [*reference, *predicted]})
    for instrument in instruments:
        targets = [note for note in reference if note["instrument"] == instrument]
        candidates = [note for note in predicted if note["instrument"] == instrument]
        instrument_matches = [
            match for match in matches if match["reference"]["instrument"] == instrument
        ]
        result.append({
            "instrument": instrument,
            "referenceNotes": len(targets),
            "predictedNotes": len(candidates),
            "matchedNotes": len(instrument_matches),
            "ignoredNotes": len(targets) - len(instrument_matches),
            "extraNotes": len(candidates) - len(instrument_matches),
            "cutOffNotes": sum(match["reference"]["index"] in cut_off_indices for match in instrument_matches),
            "overlongNotes": sum(match["reference"]["index"] in overlong_indices for match in instrument_matches),
            "rapidRetriggers": sum(note["index"] in rapid_indices for note in candidates),
            **_prf(len(instrument_matches), len(targets), len(candidates)),
        })
    return result


def analyze_errors(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    onset_tolerance: float = 0.05,
    window_seconds: float = 5.0,
    example_limit: int = 24,
) -> dict[str, Any]:
    reference = normalize_notes(reference)
    predicted = normalize_notes(predicted)
    matches = match_notes(reference, predicted, onset_tolerance, instrument_aware=True)
    matched_reference = {match["reference"]["index"] for match in matches}
    matched_predicted = {match["predicted"]["index"] for match in matches}
    false_negatives = [note for note in reference if note["index"] not in matched_reference]
    false_positives = [note for note in predicted if note["index"] not in matched_predicted]
    fn_causes, fp_causes = _diagnose_substitutions(false_negatives, false_positives, onset_tolerance)
    rapid_indices = _rapid_retrigger_indices(predicted)
    for note in false_positives:
        if note["index"] not in fp_causes:
            fp_causes[note["index"]] = "rapidRetrigger" if note["index"] in rapid_indices else "spuriousExtra"
    for note in false_negatives:
        fn_causes.setdefault(note["index"], "ignored")

    cut_off = [match for match in matches if _is_cut_off(match)]
    overlong = [match for match in matches if _is_overlong(match)]
    cut_off_indices = {match["reference"]["index"] for match in cut_off}
    overlong_indices = {match["reference"]["index"] for match in overlong}
    onset_errors = [abs(match["onsetErrorSeconds"]) for match in matches]
    offset_errors = [abs(match["offsetErrorSeconds"]) for match in matches]
    offset_matches = sum(
        abs(match["offsetErrorSeconds"]) <= max(0.05, 0.20 * match["reference"]["duration"])
        for match in matches
    )

    windows: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "ignored": 0,
        "extra": 0,
        "cutOff": 0,
        "rapidRetriggers": 0,
    })
    for note in false_negatives:
        windows[int(note["time"] // window_seconds)]["ignored"] += 1
    for note in false_positives:
        bucket = windows[int(note["time"] // window_seconds)]
        bucket["extra"] += 1
        bucket["rapidRetriggers"] += note["index"] in rapid_indices
    for match in cut_off:
        windows[int(match["reference"]["time"] // window_seconds)]["cutOff"] += 1
    hotspots = []
    for index, counts in windows.items():
        severity = counts["ignored"] + counts["extra"] + 2 * counts["cutOff"] + counts["rapidRetriggers"]
        hotspots.append({
            "startSeconds": round(index * window_seconds, 3),
            "endSeconds": round((index + 1) * window_seconds, 3),
            "severity": severity,
            **counts,
        })
    hotspots.sort(key=lambda item: (-item["severity"], item["startSeconds"]))

    error_causes: dict[str, int] = defaultdict(int)
    for cause in fp_causes.values():
        error_causes[cause] += 1
    pitch_bands = {}
    for band in ("low_A0_to_B2", "middle_C3_to_B4", "high_C5_to_G9"):
        targets = [note for note in reference if _pitch_band(note["midi"]) == band]
        candidates = [note for note in predicted if _pitch_band(note["midi"]) == band]
        band_matches = [match for match in matches if _pitch_band(match["reference"]["midi"]) == band]
        pitch_bands[band] = {
            "referenceNotes": len(targets),
            "predictedNotes": len(candidates),
            "matchedNotes": len(band_matches),
            **_prf(len(band_matches), len(targets), len(candidates)),
        }

    core = _prf(len(matches), len(reference), len(predicted))
    return {
        "schema": "polymath-instrument-error-analysis-v2",
        "onsetToleranceSeconds": onset_tolerance,
        "referenceNotes": len(reference),
        "predictedNotes": len(predicted),
        "matchedNotes": len(matches),
        "falsePositiveNotes": len(false_positives),
        "falseNegativeNotes": len(false_negatives),
        "ignoredNotes": len(false_negatives),
        "cutOffNotes": len(cut_off),
        "overlongNotes": len(overlong),
        "rapidRetriggers": len(rapid_indices),
        **core,
        "onsetOnly": core,
        "onsetAndOffset": {
            "matchedNotes": offset_matches,
            **_prf(offset_matches, len(reference), len(predicted)),
        },
        "frame": _frame_score(reference, predicted),
        "timing": {
            "medianOnsetErrorMs": round((median(onset_errors) if onset_errors else 0) * 1000, 3),
            "p95OnsetErrorMs": round((percentile(onset_errors, 0.95) or 0) * 1000, 3),
            "medianOffsetErrorMs": round((median(offset_errors) if offset_errors else 0) * 1000, 3),
            "p95OffsetErrorMs": round((percentile(offset_errors, 0.95) or 0) * 1000, 3),
        },
        "errorCauses": dict(sorted(error_causes.items())),
        "instruments": _instrument_breakdown(
            reference, predicted, matches, cut_off_indices, overlong_indices, rapid_indices,
        ),
        "patternRecognition": {
            "chords": _chord_analysis(reference, matched_reference),
            "sameKeyRepeatProfile": _same_key_gap_profile(reference, predicted),
            "pitchBands": pitch_bands,
            "worstFiveSecondWindows": hotspots[:10],
        },
        "examples": {
            "ignored": [
                {**note, "cause": fn_causes[note["index"]]}
                for note in false_negatives[:example_limit]
            ],
            "extra": [
                {**note, "cause": fp_causes[note["index"]]}
                for note in false_positives[:example_limit]
            ],
            "cutOff": [
                {
                    "reference": match["reference"],
                    "predicted": match["predicted"],
                    "earlyByMs": round(-match["offsetErrorSeconds"] * 1000, 3),
                }
                for match in cut_off[:example_limit]
            ],
        },
    }


def evaluate(
    reference: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    onset_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Backward-compatible compact metrics backed by the robust matcher."""

    report = analyze_errors(reference, predicted, onset_tolerance)
    return {
        "schema": "polymath-note-evaluation-v2",
        "onsetToleranceSeconds": onset_tolerance,
        "referenceNotes": report["referenceNotes"],
        "predictedNotes": report["predictedNotes"],
        "matchedNotes": report["matchedNotes"],
        "falsePositiveNotes": report["falsePositiveNotes"],
        "falseNegativeNotes": report["falseNegativeNotes"],
        "precision": report["precision"],
        "recall": report["recall"],
        "f1": report["f1"],
        **report["timing"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--instrument", default="acoustic_piano")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reference = load_notes(args.reference.resolve(), args.instrument)
    predicted = load_notes(args.predicted.resolve(), args.instrument)
    result = (
        evaluate(reference, predicted, args.onset_tolerance)
        if args.compact
        else analyze_errors(reference, predicted, args.onset_tolerance)
    )
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.resolve().write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
