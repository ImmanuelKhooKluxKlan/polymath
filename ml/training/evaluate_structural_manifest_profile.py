"""Replay one frozen arranger profile against the structural audit corpus."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from ml.training.evaluate_piano_arranger import load_json, prepare_reference_notes
from ml.training.evaluate_raw_support_recovery_loso import (
    duration_gate,
    performance_summary,
)
from ml.training.search_default_piano_pipeline import (
    atomic_json,
    clip_candidate,
    compact_metrics,
    load_rows,
    measure,
    scalar_quality,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.manifest.resolve())
    profile = load_json(args.profile.resolve())
    repo_root = Path(__file__).resolve().parents[2]
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

    output_dir = args.output_dir.resolve()
    results: dict[str, Any] = {}
    deltas: list[float] = []
    performance_gates_by_song: list[dict[str, bool]] = []
    for row in rows:
        song_id = str(row["id"])
        alignment = load_json(Path(str(row["alignment"])).resolve())
        reference = prepare_reference_notes(
            row,
            load_json(Path(str(row["reference"])).resolve()),
            alignment,
        )
        baseline_notes = clip_candidate(
            load_json(Path(str(row["baseline"])).resolve()), row, alignment
        )
        baseline = measure(reference, baseline_notes)
        candidate_payload = arrange_payload(
            copy.deepcopy(load_json(Path(str(row["source"])).resolve())),
            "full",
            style_profile=copy.deepcopy(profile),
        )
        candidate = measure(
            reference,
            clip_candidate(candidate_payload, row, alignment),
        )
        baseline_performance = performance_summary(baseline)
        candidate_performance = performance_summary(candidate)
        performance_gates = duration_gate(
            baseline_performance, candidate_performance
        )
        performance_gates_by_song.append(performance_gates)
        delta = scalar_quality(candidate) - scalar_quality(baseline)
        deltas.append(delta)
        results[song_id] = {
            "baseline": compact_metrics(baseline),
            "candidate": compact_metrics(candidate),
            "qualityDelta": round(delta, 6),
            "performance": {
                "baseline": baseline_performance,
                "candidate": candidate_performance,
                "gates": performance_gates,
                "passes": all(performance_gates.values()),
            },
            "arrangerDiagnostics": candidate_payload.get("pianoArrangement"),
        }
        atomic_json(output_dir / "candidates" / f"{song_id}.json", candidate_payload)

    duration_and_cutoff_gates = {
        key: all(gates[key] for gates in performance_gates_by_song)
        for key in performance_gates_by_song[0]
    }
    mean_delta = sum(deltas) / max(1, len(deltas))
    worst_delta = min(deltas)
    passes_structural_gate = mean_delta > 0 and worst_delta >= -0.0005
    passes_duration_and_cutoff_gate = all(duration_and_cutoff_gates.values())
    report = {
        "schema": "polymath-structural-manifest-profile-evaluation-v1",
        "evidenceBoundary": (
            "This utility only replays a frozen profile. Whether results are holdout "
            "evidence depends on whether each reference participated in profile fitting."
        ),
        "manifest": str(args.manifest.resolve()),
        "profile": str(args.profile.resolve()),
        "meanQualityDelta": round(mean_delta, 6),
        "worstQualityDelta": round(worst_delta, 6),
        "improvedSongs": sum(delta > 0 for delta in deltas),
        "structuralGate": {
            "meanQualityImproves": mean_delta > 0,
            "noSongRegressesOverTolerance": worst_delta >= -0.0005,
            "passes": passes_structural_gate,
        },
        "durationAndCutoffGates": duration_and_cutoff_gates,
        "passesDurationAndCutoffGate": passes_duration_and_cutoff_gate,
        "decision": (
            "CANDIDATE_FOR_BLIND_LISTENING"
            if passes_structural_gate and passes_duration_and_cutoff_gate
            else "REJECT"
        ),
        "songs": results,
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "meanQualityDelta": report["meanQualityDelta"],
                "worstQualityDelta": report["worstQualityDelta"],
                "improvedSongs": report["improvedSongs"],
                "durationAndCutoffGates": duration_and_cutoff_gates,
                "decision": report["decision"],
                "qualityDeltas": {
                    song_id: result["qualityDelta"]
                    for song_id, result in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
