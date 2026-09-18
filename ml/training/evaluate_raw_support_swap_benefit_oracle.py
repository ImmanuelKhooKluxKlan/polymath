"""Measure the label-defined ceiling of the post-arranger swap layer.

This is an offline diagnostic, never an inference path: it uses reference-only
labels to choose at most one safe exact-recovery swap per gesture, then runs the
ordinary song-level evaluator.  The result answers whether better ranking can
materially improve this architecture or whether a larger arranger change is
needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .apply_raw_support_recovery import load_json
from .evaluate_pianist_candidate_pair import evaluate_pair
from .fit_raw_support_swap_benefit_ranker import build_song_examples
from .raw_support_swap_benefit import apply_scored_proposals


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def weighted_delta(rows: list[dict[str, Any]], key: str) -> float:
    denominator = sum(int(row["evaluation"]["referenceNotes"]) for row in rows)
    return round(
        sum(
            float(row["evaluation"]["deltas"][key])
            * int(row["evaluation"]["referenceNotes"])
            for row in rows
        )
        / max(1, denominator),
        6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--minimum-source-midi", type=int, default=0)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    parser.add_argument("--register-shifts", default="0,12")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    selector_root = args.selector_profiles_root.resolve()
    output_root = args.output_root.resolve()
    manifest = load_json(manifest_path)
    register_shifts = tuple(
        dict.fromkeys(int(value.strip()) for value in args.register_shifts.split(",") if value.strip())
    )
    rows: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row["id"])
        selector_path = selector_root / song_id / "profile.json"
        examples = build_song_examples(
            row,
            load_json(selector_path),
            source_radius=max(0.01, float(args.source_radius_seconds)),
            minimum_source_midi=int(args.minimum_source_midi),
            maximum_source_midi=int(args.maximum_source_midi),
            minimum_output_midi=int(args.minimum_output_midi),
            maximum_output_midi=int(args.maximum_output_midi),
            register_shifts=register_shifts,
        )
        baseline = load_json(Path(str(row["candidate"])).resolve())
        scores = [
            0.90 + 0.01 * min(8.0, max(0.0, float(utility))) if label >= 0.5 else 0.0
            for label, utility in zip(examples["labels"], examples["utilities"])
        ]
        oracle, diagnostics = apply_scored_proposals(
            baseline,
            examples["proposals"],
            scores,
            {"id": "reference-label-oracle-not-for-inference"},
            examples["diagnostics"],
            threshold=0.5,
            minimum_selector_probability=0.0,
            minimum_probability_gain=-1.0,
            maximum_replacements_per_gesture=1,
        )
        reference = load_json(Path(str(row["reference"])).resolve())
        alignment = load_json(Path(str(row["alignment"])).resolve())
        evaluation = evaluate_pair(
            reference,
            alignment,
            baseline,
            oracle,
            reference_already_aligned=bool(row.get("referenceAlreadyAligned")),
            reference_transpose_semitones=int(row.get("referenceTransposeSemitones") or 0),
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
        atomic_json(output_root / "oracle-candidates" / f"{song_id}.json", oracle)
        rows.append(
            {
                "id": song_id,
                "positiveProposals": int(examples["positives"]),
                "oracleReplacements": int(diagnostics["replacements"]),
                "evaluation": evaluation,
            }
        )

    metric_keys = (
        "quality",
        "exactF1_100ms",
        "exactF1_250ms",
        "pitchClassF1_250ms",
        "pitchClassRecall_250ms",
        "velocityMeanAbsoluteError",
        "chordSizeDistance",
        "handOccupancyDistance",
    )
    report = {
        "schema": "polymath-raw-support-swap-benefit-oracle-v1",
        "warning": "Reference-derived diagnostic only; impossible to use at inference.",
        "manifest": str(manifest_path),
        "weightedDeltas": {key: weighted_delta(rows, key) for key in metric_keys},
        "songs": [
            {
                "id": row["id"],
                "positiveProposals": row["positiveProposals"],
                "oracleReplacements": row["oracleReplacements"],
                "deltas": row["evaluation"]["deltas"],
            }
            for row in rows
        ],
    }
    atomic_json(output_root / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
