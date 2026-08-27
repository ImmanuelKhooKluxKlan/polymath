"""Measure piano note accuracy for a candidate model against reviewed labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def load_notes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        payload.get("notes") if isinstance(payload, dict) else None,
        payload.get("result", {}).get("notes") if isinstance(payload, dict) and isinstance(payload.get("result"), dict) else None,
        payload.get("output", {}).get("notes") if isinstance(payload, dict) and isinstance(payload.get("output"), dict) else None,
    ]
    source = next((item for item in candidates if isinstance(item, list)), [])
    notes: list[dict[str, Any]] = []
    for index, note in enumerate(source):
        try:
            midi = int(note.get("midi", note.get("pitch")))
            start = float(note.get("time", note.get("start", note.get("startTime"))))
            duration = float(note.get("duration", 0.1))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= midi <= 127 and start >= 0 and duration > 0:
            notes.append({"index": index, "midi": midi, "time": start, "duration": duration})
    return sorted(notes, key=lambda note: (note["time"], note["midi"]))


def match_notes(reference: list[dict[str, Any]], predicted: list[dict[str, Any]], onset_tolerance: float) -> list[dict[str, Any]]:
    by_pitch: dict[int, list[dict[str, Any]]] = {}
    for note in predicted:
        by_pitch.setdefault(note["midi"], []).append(note)
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    for target in reference:
        candidates = [candidate for candidate in by_pitch.get(target["midi"], []) if (
            candidate["index"] not in used
            and abs(candidate["time"] - target["time"]) <= onset_tolerance
        )]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: (
            abs(candidate["time"] - target["time"]),
            abs(candidate["duration"] - target["duration"]),
        ))
        used.add(best["index"])
        matches.append({
            "reference": target,
            "predicted": best,
            "onsetErrorSeconds": best["time"] - target["time"],
            "offsetErrorSeconds": (best["time"] + best["duration"]) - (target["time"] + target["duration"]),
        })
    return matches


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    mix = position - low
    return ordered[low] * (1 - mix) + ordered[high] * mix


def evaluate(reference: list[dict[str, Any]], predicted: list[dict[str, Any]], onset_tolerance: float = 0.05) -> dict[str, Any]:
    matches = match_notes(reference, predicted, onset_tolerance)
    true_positive = len(matches)
    false_negative = len(reference) - true_positive
    false_positive = len(predicted) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    onset_errors = [abs(match["onsetErrorSeconds"]) for match in matches]
    offset_errors = [abs(match["offsetErrorSeconds"]) for match in matches]
    return {
        "schema": "polymath-note-evaluation-v1",
        "onsetToleranceSeconds": onset_tolerance,
        "referenceNotes": len(reference),
        "predictedNotes": len(predicted),
        "matchedNotes": true_positive,
        "falsePositiveNotes": false_positive,
        "falseNegativeNotes": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "medianOnsetErrorMs": round((median(onset_errors) if onset_errors else 0) * 1000, 3),
        "p95OnsetErrorMs": round((percentile(onset_errors, 0.95) or 0) * 1000, 3),
        "medianOffsetErrorMs": round((median(offset_errors) if offset_errors else 0) * 1000, 3),
        "p95OffsetErrorMs": round((percentile(offset_errors, 0.95) or 0) * 1000, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(
        load_notes(args.reference.resolve()),
        load_notes(args.predicted.resolve()),
        args.onset_tolerance,
    )
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.resolve().write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
