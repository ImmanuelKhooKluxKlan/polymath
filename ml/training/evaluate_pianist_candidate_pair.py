"""Compare two arranged scores against one aligned pianist reference.

This is the small, direct counterpart to the manifest/grid search utilities.
It reports the same strict note, gesture, structure, velocity, and quality
metrics for an incumbent and challenger without changing either file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_piano_arranger import load_json, prepare_reference_notes
from .search_default_piano_pipeline import clip_candidate, compact_metrics, measure


def evaluate_pair(
    reference_payload: dict[str, Any],
    alignment: dict[str, Any],
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    reference_already_aligned: bool = False,
    reference_transpose_semitones: int = 0,
    reference_end_seconds: float | None = None,
    candidate_end_seconds: float | None = None,
) -> dict[str, Any]:
    row = {
        "referenceAlreadyAligned": reference_already_aligned,
        "referenceTransposeSemitones": reference_transpose_semitones,
        "referenceEndSeconds": reference_end_seconds,
        "candidateEndSeconds": candidate_end_seconds,
    }
    reference = prepare_reference_notes(row, reference_payload, alignment)
    if candidate_end_seconds is not None:
        reference = [
            note for note in reference if float(note["time"]) < candidate_end_seconds
        ]
    baseline = compact_metrics(measure(reference, clip_candidate(baseline_payload, row, alignment)))
    candidate = compact_metrics(measure(reference, clip_candidate(candidate_payload, row, alignment)))
    return {
        "schema": "polymath-pianist-candidate-pair-v1",
        "referenceNotes": len(reference),
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            key: round(float(candidate[key]) - float(baseline[key]), 6)
            for key in (
                "quality",
                "exactF1_100ms",
                "exactF1_250ms",
                "pitchClassF1_250ms",
                "pitchClassRecall_250ms",
                "velocityMeanAbsoluteError",
                "chordSizeDistance",
                "handOccupancyDistance",
            )
        },
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-already-aligned", action="store_true")
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    parser.add_argument("--reference-end-seconds", type=float)
    parser.add_argument("--candidate-end-seconds", type=float)
    args = parser.parse_args()
    result = evaluate_pair(
        load_json(args.reference.resolve()),
        load_json(args.alignment.resolve()),
        load_json(args.baseline.resolve()),
        load_json(args.candidate.resolve()),
        reference_already_aligned=args.reference_already_aligned,
        reference_transpose_semitones=args.reference_transpose_semitones,
        reference_end_seconds=args.reference_end_seconds,
        candidate_end_seconds=args.candidate_end_seconds,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
