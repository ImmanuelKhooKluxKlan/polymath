"""Audit complete pianist candidates against an immutable baseline manifest.

The report combines note/structure deltas with duration, cutoff, and rapid-
retrigger gates.  It is deliberately read-only: candidate and baseline JSON
files are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_pianist_candidate_pair import evaluate_pair
from .evaluate_piano_arranger import load_json, prepare_reference_notes
from .evaluate_raw_support_recovery_loso import duration_gate, performance_summary
from .search_default_piano_pipeline import clip_candidate, measure


METRIC_KEYS = (
    "quality",
    "exactF1_100ms",
    "exactF1_250ms",
    "pitchClassF1_250ms",
    "pitchClassRecall_250ms",
    "velocityMeanAbsoluteError",
    "chordSizeDistance",
    "handOccupancyDistance",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_gates(deltas: dict[str, float], tolerance: float = 0.0005) -> dict[str, bool]:
    """Reject note-quality regressions while allowing evaluator rounding noise."""

    return {
        "qualityNotWorse": float(deltas["quality"]) >= -tolerance,
        "exactF1At100msNotWorse": float(deltas["exactF1_100ms"]) >= -tolerance,
        "exactF1At250msNotWorse": float(deltas["exactF1_250ms"]) >= -tolerance,
        "pitchClassF1At250msNotWorse": (
            float(deltas["pitchClassF1_250ms"]) >= -tolerance
        ),
        "pitchClassRecallAt250msNotWorse": (
            float(deltas["pitchClassRecall_250ms"]) >= -tolerance
        ),
    }


def weighted_delta(rows: list[dict[str, Any]], key: str) -> float:
    denominator = sum(int(row["evaluation"]["referenceNotes"]) for row in rows)
    numerator = sum(
        int(row["evaluation"]["referenceNotes"])
        * float(row["evaluation"]["deltas"][key])
        for row in rows
    )
    return round(numerator / max(1, denominator), 6)


def error_ceiling(metrics: dict[str, Any]) -> dict[str, float]:
    """Separate register, onset-window, and remaining pitch-class headroom.

    These are diagnostic gaps, not promises that an independent model can
    recover every point.  Pitch-class matching removes octave placement from
    the comparison, while the wider onset window isolates coarse timing.
    """

    notes = metrics["notes"]
    exact_100 = float(notes["exactPitchOnset100ms"]["f1"])
    pitch_class_100 = float(notes["pitchClassOnset100ms"]["f1"])
    pitch_class_250 = float(notes["pitchClassOnset250ms"]["f1"])
    return {
        "exactPitchF1At100ms": round(exact_100, 6),
        "pitchClassF1At100ms": round(pitch_class_100, 6),
        "pitchClassF1At250ms": round(pitch_class_250, 6),
        "registerPlacementGapAt100ms": round(
            max(0.0, pitch_class_100 - exact_100), 6
        ),
        "coarseTimingGap100To250ms": round(
            max(0.0, pitch_class_250 - pitch_class_100), 6
        ),
        "remainingPitchClassOrEventGapAt250ms": round(
            max(0.0, 1.0 - pitch_class_250), 6
        ),
    }


def evaluate_manifest(
    manifest: dict[str, Any], selected_song_ids: set[str] | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row.get("id") or "")
        if not song_id:
            raise ValueError("Every song requires a non-empty id")
        if selected_song_ids and song_id not in selected_song_ids:
            continue
        if not row.get("baseline"):
            continue

        reference_payload = load_json(Path(str(row["reference"])).resolve())
        alignment_payload = load_json(Path(str(row["alignment"])).resolve())
        baseline_payload = load_json(Path(str(row["baseline"])).resolve())
        candidate_payload = load_json(Path(str(row["candidate"])).resolve())

        evaluation = evaluate_pair(
            reference_payload,
            alignment_payload,
            baseline_payload,
            candidate_payload,
            reference_already_aligned=bool(row.get("referenceAlreadyAligned")),
            reference_transpose_semitones=int(
                row.get("referenceTransposeSemitones") or 0
            ),
            reference_end_seconds=(
                float(row["referenceEndSeconds"])
                if row.get("referenceEndSeconds") is not None
                else None
            ),
            candidate_end_seconds=(
                float(row["candidateEndSeconds"])
                if row.get("candidateEndSeconds") is not None
                else None
            ),
        )

        reference = prepare_reference_notes(row, reference_payload, alignment_payload)
        if row.get("candidateEndSeconds") is not None:
            end = float(row["candidateEndSeconds"])
            reference = [note for note in reference if float(note["time"]) < end]
        baseline_metrics = measure(
            reference, clip_candidate(baseline_payload, row, alignment_payload)
        )
        candidate_metrics = measure(
            reference, clip_candidate(candidate_payload, row, alignment_payload)
        )
        baseline_performance = performance_summary(baseline_metrics)
        candidate_performance = performance_summary(candidate_metrics)
        performance_gates = duration_gate(
            baseline_performance, candidate_performance
        )
        note_gates = metric_gates(evaluation["deltas"])
        rows.append(
            {
                "id": song_id,
                "evaluation": evaluation,
                "baselinePerformance": baseline_performance,
                "candidatePerformance": candidate_performance,
                "candidateErrorCeiling": error_ceiling(candidate_metrics),
                "metricGates": note_gates,
                "performanceGates": performance_gates,
                "passes": all(note_gates.values())
                and all(performance_gates.values()),
            }
        )

    if not rows:
        raise ValueError("No manifest songs with baseline and candidate files were selected")

    weighted = {key: weighted_delta(rows, key) for key in METRIC_KEYS}
    return {
        "schema": "polymath-pianist-candidate-manifest-audit-v1",
        "purpose": manifest.get("purpose"),
        "commercialUseAllowed": bool(manifest.get("commercialUseAllowed", False)),
        "songs": rows,
        "summary": {
            "songs": len(rows),
            "referenceNotes": sum(
                int(row["evaluation"]["referenceNotes"]) for row in rows
            ),
            "weightedDeltas": weighted,
            "allSongsPass": all(bool(row["passes"]) for row in rows),
            "materialQualityImprovement": weighted["quality"] >= 0.001,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--song", action="append", default=[])
    args = parser.parse_args()
    selected = {str(value) for value in args.song if str(value)} or None
    result = evaluate_manifest(load_json(args.manifest.resolve()), selected)
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
