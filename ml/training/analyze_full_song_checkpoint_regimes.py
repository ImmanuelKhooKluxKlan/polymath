"""Explain how full-song checkpoint outputs differ from a frozen baseline.

The checkpoint gate answers whether a candidate is safe.  This diagnostic
answers *why* it passed or failed by breaking exact-pitch onset behavior down
by held-out window and register, then accounting for event additions,
removals, recovered reference notes, and lost reference notes.

It is analysis-only.  It does not train, mutate, or promote a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from ml.training.evaluate_piano_arranger import (
    evaluate,
    greedy_matches,
    load_json,
    normalize_notes,
    prepare_reference_notes,
)


DEFAULT_PITCH_BANDS: tuple[tuple[str, int, int], ...] = (
    ("A0-B1", 21, 35),
    ("C2-B2", 36, 47),
    ("C3-B3", 48, 59),
    ("C4-B4", 60, 71),
    ("C5-B5", 72, 83),
    ("C6-C8", 84, 108),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_window(
    notes: Iterable[dict[str, float | int]],
    start: float,
    end: float,
) -> list[dict[str, float | int]]:
    return [
        note
        for note in notes
        if start <= float(note["time"]) < end
    ]


def notes_inside_windows(
    notes: Iterable[dict[str, float | int]],
    windows: list[dict[str, float]],
) -> list[dict[str, float | int]]:
    return [
        note
        for note in notes
        if any(
            window["start"] <= float(note["time"]) < window["end"]
            for window in windows
        )
    ]


def _inside_pitch_band(
    notes: Iterable[dict[str, float | int]],
    minimum: int,
    maximum: int,
) -> list[dict[str, float | int]]:
    return [
        note
        for note in notes
        if minimum <= int(note["midi"]) <= maximum
    ]


def _exact_summary(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
) -> dict[str, Any]:
    metrics = evaluate(reference, observed)
    return {
        "referenceNotes": len(reference),
        "predictedNotes": len(observed),
        "rapidRetriggersUnder100ms": int(metrics["rapidRetriggersUnder100ms"]),
        **{
            f"{milliseconds}ms": metrics[f"exactPitchOnset{milliseconds}ms"]
            for milliseconds in (50, 100, 250)
        },
    }


def _rounded_delta(candidate: float | int, baseline: float | int) -> float:
    return round(float(candidate) - float(baseline), 6)


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "predictedNotes": int(candidate["predictedNotes"]) - int(baseline["predictedNotes"]),
        "rapidRetriggersUnder100ms": (
            int(candidate["rapidRetriggersUnder100ms"])
            - int(baseline["rapidRetriggersUnder100ms"])
        ),
        **{
            f"{milliseconds}ms": {
                name: _rounded_delta(
                    candidate[f"{milliseconds}ms"][name],
                    baseline[f"{milliseconds}ms"][name],
                )
                for name in ("matches", "precision", "recall", "f1")
            }
            for milliseconds in (50, 100, 250)
        },
    }


def _identity_set(
    notes: Iterable[dict[str, float | int]],
) -> set[int]:
    return {id(note) for note in notes}


def _matched_note_ids(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    tolerance: float,
) -> tuple[set[int], set[int]]:
    matches = greedy_matches(reference, observed, tolerance)
    return (
        {id(target) for target, _candidate in matches},
        {id(candidate) for _target, candidate in matches},
    )


def _counter_rows(counter: Counter[int], limit: int = 24) -> list[dict[str, int]]:
    return [
        {"midi": midi, "count": count}
        for midi, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _event_change_summary(
    reference: list[dict[str, float | int]],
    baseline: list[dict[str, float | int]],
    candidate: list[dict[str, float | int]],
    *,
    equivalence_tolerance: float = 0.02,
) -> dict[str, Any]:
    baseline_to_candidate = greedy_matches(
        baseline,
        candidate,
        equivalence_tolerance,
    )
    retained_baseline_ids = {id(note) for note, _candidate in baseline_to_candidate}
    retained_candidate_ids = {id(note) for _baseline, note in baseline_to_candidate}
    removed = [note for note in baseline if id(note) not in retained_baseline_ids]
    added = [note for note in candidate if id(note) not in retained_candidate_ids]

    result: dict[str, Any] = {
        "equivalenceToleranceMs": int(round(equivalence_tolerance * 1000)),
        "retainedEvents": len(baseline_to_candidate),
        "addedEvents": len(added),
        "removedEvents": len(removed),
        "addedByMidi": _counter_rows(Counter(int(note["midi"]) for note in added)),
        "removedByMidi": _counter_rows(Counter(int(note["midi"]) for note in removed)),
    }
    for milliseconds in (100, 250):
        tolerance = milliseconds / 1000.0
        baseline_reference_ids, baseline_match_ids = _matched_note_ids(
            reference,
            baseline,
            tolerance,
        )
        candidate_reference_ids, candidate_match_ids = _matched_note_ids(
            reference,
            candidate,
            tolerance,
        )
        recovered_reference_ids = candidate_reference_ids - baseline_reference_ids
        lost_reference_ids = baseline_reference_ids - candidate_reference_ids
        added_ids = _identity_set(added)
        removed_ids = _identity_set(removed)
        result[f"{milliseconds}ms"] = {
            "recoveredReferenceNotes": len(recovered_reference_ids),
            "lostReferenceNotes": len(lost_reference_ids),
            "netReferenceMatches": len(candidate_reference_ids) - len(baseline_reference_ids),
            "addedEventsUsedAsMatches": len(candidate_match_ids & added_ids),
            "removedEventsPreviouslyMatched": len(baseline_match_ids & removed_ids),
        }
    return result


def _timing_shift_summary(
    baseline: list[dict[str, float | int]],
    candidate: list[dict[str, float | int]],
    tolerance: float = 0.25,
) -> dict[str, Any]:
    matches = greedy_matches(baseline, candidate, tolerance)
    absolute_shifts = [
        abs(float(candidate_note["time"]) - float(baseline_note["time"]))
        for baseline_note, candidate_note in matches
    ]
    return {
        "matchedSamePitchEvents": len(matches),
        "toleranceMs": int(round(tolerance * 1000)),
        "medianAbsoluteOnsetShiftMs": (
            round(median(absolute_shifts) * 1000.0, 3)
            if absolute_shifts
            else None
        ),
        "maximumAbsoluteOnsetShiftMs": (
            round(max(absolute_shifts) * 1000.0, 3)
            if absolute_shifts
            else None
        ),
    }


def analyze_predictions(
    reference: list[dict[str, float | int]],
    predictions: dict[str, list[dict[str, float | int]]],
    windows: list[dict[str, float]],
    *,
    baseline_label: str,
) -> dict[str, Any]:
    if baseline_label not in predictions:
        raise ValueError(f"Unknown baseline label: {baseline_label}")
    if not windows:
        raise ValueError("At least one analysis window is required")

    held_reference = notes_inside_windows(reference, windows)
    held_predictions = {
        label: notes_inside_windows(notes, windows)
        for label, notes in predictions.items()
    }
    baseline_notes = held_predictions[baseline_label]
    baseline_aggregate = _exact_summary(held_reference, baseline_notes)

    analyses: dict[str, Any] = {}
    for label, observed in held_predictions.items():
        aggregate = _exact_summary(held_reference, observed)
        per_window: list[dict[str, Any]] = []
        for index, window in enumerate(windows):
            window_reference = _inside_window(
                held_reference,
                window["start"],
                window["end"],
            )
            window_observed = _inside_window(
                observed,
                window["start"],
                window["end"],
            )
            window_baseline = _inside_window(
                baseline_notes,
                window["start"],
                window["end"],
            )
            metrics = _exact_summary(window_reference, window_observed)
            baseline_metrics = _exact_summary(window_reference, window_baseline)
            per_window.append(
                {
                    "index": index,
                    "start": window["start"],
                    "end": window["end"],
                    "metrics": metrics,
                    "deltaAgainstBaseline": _metric_delta(metrics, baseline_metrics),
                }
            )

        per_pitch_band: dict[str, Any] = {}
        for band, minimum, maximum in DEFAULT_PITCH_BANDS:
            band_reference = _inside_pitch_band(held_reference, minimum, maximum)
            band_observed = _inside_pitch_band(observed, minimum, maximum)
            band_baseline = _inside_pitch_band(baseline_notes, minimum, maximum)
            metrics = _exact_summary(band_reference, band_observed)
            baseline_metrics = _exact_summary(band_reference, band_baseline)
            per_pitch_band[band] = {
                "midiMinimum": minimum,
                "midiMaximum": maximum,
                "metrics": metrics,
                "deltaAgainstBaseline": _metric_delta(metrics, baseline_metrics),
            }

        analyses[label] = {
            "aggregate": aggregate,
            "deltaAgainstBaseline": _metric_delta(aggregate, baseline_aggregate),
            "perWindow": per_window,
            "perPitchBand": per_pitch_band,
            "eventChangesAgainstBaseline": _event_change_summary(
                held_reference,
                baseline_notes,
                observed,
            ),
            "timingShiftsAgainstBaseline": _timing_shift_summary(
                baseline_notes,
                observed,
            ),
        }
    return {
        "schema": "polymath-full-song-checkpoint-regime-analysis-v1",
        "baselineLabel": baseline_label,
        "windows": windows,
        "heldOutReferenceNotes": len(held_reference),
        "pitchBands": [
            {"name": name, "midiMinimum": minimum, "midiMaximum": maximum}
            for name, minimum, maximum in DEFAULT_PITCH_BANDS
        ],
        "predictions": analyses,
    }


def _parse_window(value: str) -> dict[str, float]:
    try:
        raw_start, raw_end = value.split(":", 1)
        start = float(raw_start)
        end = float(raw_end)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise argparse.ArgumentTypeError("window boundaries must be finite and END > START >= 0")
    return {"start": start, "end": end}


def _parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError("prediction needs a non-empty label and existing file")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--alignment-report", required=True)
    parser.add_argument("--prediction", action="append", required=True, type=_parse_prediction)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--window", action="append", required=True, type=_parse_window)
    parser.add_argument("--reference-already-aligned", action="store_true")
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser().resolve()
    alignment_path = Path(args.alignment_report).expanduser().resolve()
    if not reference_path.is_file() or not alignment_path.is_file():
        raise FileNotFoundError("reference and alignment report must exist")
    windows = sorted(args.window, key=lambda item: item["start"])
    for previous, current in zip(windows, windows[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("analysis windows cannot overlap")

    prediction_paths = dict(args.prediction)
    if len(prediction_paths) != len(args.prediction):
        raise ValueError("prediction labels must be unique")
    reference = prepare_reference_notes(
        {
            "referenceAlreadyAligned": bool(args.reference_already_aligned),
            "referenceTransposeSemitones": int(args.reference_transpose_semitones),
        },
        load_json(reference_path),
        load_json(alignment_path),
    )
    predictions = {
        label: normalize_notes(load_json(path))
        for label, path in prediction_paths.items()
    }
    report = analyze_predictions(
        reference,
        predictions,
        windows,
        baseline_label=args.baseline_label,
    )
    report["inputs"] = {
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "alignmentReport": {"path": str(alignment_path), "sha256": sha256(alignment_path)},
        "predictions": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in prediction_paths.items()
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "sha256": sha256(output_path),
        "baselineLabel": args.baseline_label,
        "predictions": {
            label: analysis["deltaAgainstBaseline"]
            for label, analysis in report["predictions"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
