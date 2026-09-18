"""Audit promotion invariants for a pitch-only swap-benefit candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .apply_raw_support_recovery import load_json
from .evaluate_pianist_candidate_pair import evaluate_pair
from .evaluate_piano_arranger import evaluate, prepare_reference_notes
from .search_default_piano_pipeline import clip_candidate


FROZEN_NOTE_FIELDS = (
    "time",
    "startTime",
    "start",
    "duration",
    "scoreDuration",
    "visualDuration",
    "audioDuration",
    "releaseSeconds",
    "velocity",
    "hand",
    "arrangementRole",
    "articulation",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def numeric_delta(candidate: Any, baseline: Any) -> float | None:
    try:
        left = float(candidate)
        right = float(baseline)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return round(left - right, 6)


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

    row = {
        "referenceAlreadyAligned": bool(args.reference_already_aligned),
        "referenceTransposeSemitones": int(args.reference_transpose_semitones),
        "referenceEndSeconds": args.reference_end_seconds,
        "candidateEndSeconds": args.candidate_end_seconds,
    }
    reference_payload = load_json(args.reference.resolve())
    alignment = load_json(args.alignment.resolve())
    baseline_payload = load_json(args.baseline.resolve())
    candidate_payload = load_json(args.candidate.resolve())
    baseline_notes = baseline_payload.get("notes") or []
    candidate_notes = candidate_payload.get("notes") or []
    note_count_same = len(baseline_notes) == len(candidate_notes)
    field_regressions: list[dict[str, Any]] = []
    changed_pitches: list[dict[str, Any]] = []
    if note_count_same:
        for index, (baseline_note, candidate_note) in enumerate(zip(baseline_notes, candidate_notes)):
            old_midi = int(round(float(baseline_note.get("midi", baseline_note.get("pitch")))))
            new_midi = int(round(float(candidate_note.get("midi", candidate_note.get("pitch")))))
            if old_midi != new_midi:
                changed_pitches.append(
                    {
                        "index": index,
                        "time": baseline_note.get("time", baseline_note.get("startTime", baseline_note.get("start"))),
                        "oldMidi": old_midi,
                        "newMidi": new_midi,
                        "hand": candidate_note.get("hand"),
                        "role": candidate_note.get("arrangementRole"),
                        "benefitProbability": candidate_note.get("rawSupportBenefitProbability"),
                    }
                )
            for field in FROZEN_NOTE_FIELDS:
                if baseline_note.get(field) != candidate_note.get(field):
                    field_regressions.append(
                        {
                            "index": index,
                            "field": field,
                            "baseline": baseline_note.get(field),
                            "candidate": candidate_note.get(field),
                        }
                    )

    pair = evaluate_pair(
        reference_payload,
        alignment,
        baseline_payload,
        candidate_payload,
        reference_already_aligned=bool(args.reference_already_aligned),
        reference_transpose_semitones=int(args.reference_transpose_semitones),
        reference_end_seconds=args.reference_end_seconds,
        candidate_end_seconds=args.candidate_end_seconds,
    )
    reference = prepare_reference_notes(row, reference_payload, alignment)
    baseline_detailed = evaluate(reference, clip_candidate(baseline_payload, row, alignment))
    candidate_detailed = evaluate(reference, clip_candidate(candidate_payload, row, alignment))
    duration_deltas: dict[str, Any] = {}
    for name in ("duration", "visualDuration", "physicalDuration"):
        baseline_values = baseline_detailed[name]
        candidate_values = candidate_detailed[name]
        duration_deltas[name] = {
            "matchedNotes": int(candidate_values["matchedNotes"]) - int(baseline_values["matchedNotes"]),
            "medianAbsoluteErrorSeconds": numeric_delta(
                candidate_values["medianAbsoluteErrorSeconds"], baseline_values["medianAbsoluteErrorSeconds"]
            ),
            "medianRelativeError": numeric_delta(
                candidate_values["medianRelativeError"], baseline_values["medianRelativeError"]
            ),
            "severeCutoffs": int(candidate_values["severeCutoffs"]) - int(baseline_values["severeCutoffs"]),
            "severeCutoffRate": numeric_delta(
                candidate_values["severeCutoffRate"], baseline_values["severeCutoffRate"]
            ),
            "onsetAndOffset250msF1": numeric_delta(
                candidate_values["onsetAndOffset250ms"]["f1"],
                baseline_values["onsetAndOffset250ms"]["f1"],
            ),
        }

    gates = {
        "noteCountFrozen": note_count_same,
        "timingDurationVelocityAndRolesFrozen": not field_regressions,
        "atLeastOnePitchChanged": bool(changed_pitches),
        "qualityImproved": float(pair["deltas"]["quality"]) > 0,
        "exact100Improved": float(pair["deltas"]["exactF1_100ms"]) > 0,
        "exact250Improved": float(pair["deltas"]["exactF1_250ms"]) > 0,
        "pitchClassF1NotRegressed": float(pair["deltas"]["pitchClassF1_250ms"]) >= 0,
        "pitchClassRecallNotRegressed": float(pair["deltas"]["pitchClassRecall_250ms"]) >= 0,
        "rapidRetriggersNotRegressed": int(candidate_detailed["rapidRetriggersUnder100ms"])
        <= int(baseline_detailed["rapidRetriggersUnder100ms"]),
        "physicalDurationMaeNotRegressed": float(candidate_detailed["physicalDuration"]["medianAbsoluteErrorSeconds"])
        <= float(baseline_detailed["physicalDuration"]["medianAbsoluteErrorSeconds"]),
        "physicalCutoffRateNotRegressed": float(candidate_detailed["physicalDuration"]["severeCutoffRate"])
        <= float(baseline_detailed["physicalDuration"]["severeCutoffRate"]),
    }
    report = {
        "schema": "polymath-raw-support-swap-benefit-promotion-audit-v1",
        "evidenceBoundary": "The policy and both stacked models were frozen before this transfer score was opened.",
        "pair": pair,
        "changedPitchCount": len(changed_pitches),
        "changedPitches": changed_pitches,
        "frozenFieldRegressions": field_regressions,
        "detailed": {
            "baseline": baseline_detailed,
            "candidate": candidate_detailed,
            "durationDeltas": duration_deltas,
        },
        "gates": gates,
        "allAutomatedResearchGatesPassed": all(gates.values()),
        "productionDecision": "KEEP_PHASE45",
        "productionDecisionReason": (
            "This is a research-layer transfer, not a pristine project-wide holdout; "
            "commercial licensing and blind listening approval also remain unresolved."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "changedPitchCount": len(changed_pitches),
                "deltas": pair["deltas"],
                "durationDeltas": duration_deltas,
                "gates": gates,
                "allAutomatedResearchGatesPassed": report["allAutomatedResearchGatesPassed"],
                "productionDecision": report["productionDecision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
