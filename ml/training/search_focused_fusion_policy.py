"""Search a guarded full-mix + focused-melody fusion policy across songs.

This is development tooling. It never edits a checkpoint or production
profile. Each trial fuses two immutable transcriptions of the same real-song
audio, runs the frozen Piano arranger, and scores the result against fixed
target-to-source time warps. Only a compact leaderboard and the winning
artifacts are retained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ml.training.evaluate_piano_arranger import (
    at_least,
    at_most,
    evaluate,
    load_json,
    map_reference_notes,
    monotonic_anchors,
    normalize_notes,
    notes_inside_ranges,
    reference_transpose_semitones,
    trusted_source_ranges,
    weighted_average,
)
from ml.training.merge_focused_transcription import fuse_focused_transcription


def parse_grid(value: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("grid requires finite numeric values")
    return values


def parse_choices(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    if not values:
        raise argparse.ArgumentTypeError("choice list cannot be empty")
    return values


def parse_booleans(value: str) -> tuple[bool, ...]:
    choices: list[bool] = []
    for item in parse_choices(value):
        if item not in {"true", "false"}:
            raise argparse.ArgumentTypeError("boolean grid accepts true,false")
        choice = item == "true"
        if choice not in choices:
            choices.append(choice)
    return tuple(choices)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arranger(repo_root: Path):
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

    return arrange_payload


def flattened_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for tolerance in (50, 100, 250):
        for prefix, field in (
            ("exact", "exactPitchOnset"),
            ("pitchClass", "pitchClassOnset"),
        ):
            row = metrics[f"{field}{tolerance}ms"]
            for statistic in ("precision", "recall", "f1"):
                output[f"{prefix}{statistic.title()}_{tolerance}ms"] = float(
                    row[statistic]
                )
    output.update(
        {
            "durationMedianAbsoluteErrorSeconds": float(
                metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0
            ),
            "visualSevereCutoffRate": float(
                metrics["visualDuration"]["severeCutoffRate"]
            ),
            "physicalDurationMedianAbsoluteErrorSeconds": float(
                metrics["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0
            ),
            "physicalSevereCutoffRate": float(
                metrics["physicalDuration"]["severeCutoffRate"]
            ),
            "rapidRetriggersUnder100ms": float(
                metrics["rapidRetriggersUnder100ms"]
            ),
        }
    )
    return output


def aggregate_song_metrics(
    rows: Iterable[tuple[float, dict[str, Any]]]
) -> dict[str, float]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for weight, metrics in rows:
        for name, value in flattened_metrics(metrics).items():
            values[name].append((value, weight))
    return {
        name: round(weighted_average(weighted_values), 6)
        for name, weighted_values in values.items()
    }


def promotion_gates(
    baseline: dict[str, float],
    candidate: dict[str, float],
    trusted_regressions: list[str],
) -> dict[str, bool]:
    return {
        "exactF1_100ms_improves": at_least(
            candidate["exactF1_100ms"], baseline["exactF1_100ms"] + 0.005
        ),
        "exactRecall_100ms_improves": at_least(
            candidate["exactRecall_100ms"], baseline["exactRecall_100ms"] + 0.003
        ),
        "exactF1_250ms_does_not_regress": at_least(
            candidate["exactF1_250ms"], baseline["exactF1_250ms"] - 0.002
        ),
        "pitchClassF1_250ms_does_not_regress": at_least(
            candidate["pitchClassF1_250ms"],
            baseline["pitchClassF1_250ms"] - 0.002,
        ),
        "pitchClassRecall_250ms_does_not_regress": at_least(
            candidate["pitchClassRecall_250ms"],
            baseline["pitchClassRecall_250ms"] - 0.002,
        ),
        "no_trusted_song_regresses_over_1_point": not trusted_regressions,
        "duration_error_not_over_10_percent_worse": at_most(
            candidate["durationMedianAbsoluteErrorSeconds"],
            baseline["durationMedianAbsoluteErrorSeconds"] * 1.10,
        ),
        "visual_cutoff_rate_not_worse": at_most(
            candidate["visualSevereCutoffRate"],
            baseline["visualSevereCutoffRate"] * 1.05 + 0.005,
        ),
        "physical_duration_error_not_over_10_percent_worse": at_most(
            candidate["physicalDurationMedianAbsoluteErrorSeconds"],
            baseline["physicalDurationMedianAbsoluteErrorSeconds"] * 1.10,
        ),
        "physical_cutoff_rate_not_worse": at_most(
            candidate["physicalSevereCutoffRate"],
            baseline["physicalSevereCutoffRate"] * 1.05 + 0.005,
        ),
        "rapid_retriggers_not_over_5_percent_worse": at_most(
            candidate["rapidRetriggersUnder100ms"],
            baseline["rapidRetriggersUnder100ms"] * 1.05 + 1.0,
        ),
    }


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    deltas = row["deltas"]
    candidate = row["candidate"]
    return (
        float(row["passedAllGates"]),
        float(row["passedGateCount"]),
        deltas["exactF1_100ms"],
        deltas["exactRecall_100ms"],
        deltas["pitchClassF1_250ms"],
        deltas["pitchClassRecall_250ms"],
        -max(0.0, deltas["physicalSevereCutoffRate"]),
        -candidate["physicalDurationMedianAbsoluteErrorSeconds"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arranger-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--support-scopes", type=parse_choices, default=("primary-pitched",)
    )
    parser.add_argument("--pitch-modes", type=parse_choices, default=("exact",))
    parser.add_argument(
        "--support-tolerances", type=parse_grid, default=(0.05, 0.075, 0.10, 0.15)
    )
    parser.add_argument(
        "--minimum-support-ratios", type=parse_grid, default=(0.50, 0.60, 0.65, 0.70)
    )
    parser.add_argument(
        "--retain-unmatched-primary", type=parse_booleans, default=(False,)
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest.resolve()
    profile_path = args.arranger_profile.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_json(manifest_path)
    style_profile = load_json(profile_path)
    arrange_payload = load_arranger(repo_root)
    prepared: list[dict[str, Any]] = []
    for pair in manifest.get("pairs") or []:
        pair_id = str(pair["id"])
        paths = {
            name: Path(pair[field]).resolve()
            for name, field in (
                ("primary", "primary"),
                ("focused", "focused"),
                ("target", "target"),
                ("alignment", "alignmentReport"),
                ("baseline", "baseline"),
            )
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{pair_id} {name} does not exist: {path}")
        alignment = load_json(paths["alignment"])
        ranges = trusted_source_ranges(alignment)
        if ranges == []:
            raise ValueError(f"{pair_id}: alignment has zero trusted ranges")
        target = normalize_notes(
            load_json(paths["target"]),
            transpose_semitones=reference_transpose_semitones(pair),
        )
        reference = notes_inside_ranges(
            map_reference_notes(target, monotonic_anchors(alignment)), ranges
        )
        baseline_notes = notes_inside_ranges(
            normalize_notes(load_json(paths["baseline"])), ranges
        )
        if len(reference) < 100:
            raise ValueError(f"{pair_id}: fewer than 100 fixed reference notes")
        prepared.append(
            {
                "id": pair_id,
                "weight": float(pair.get("weight", 1.0)),
                "primary": load_json(paths["primary"]),
                "focused": load_json(paths["focused"]),
                "reference": reference,
                "ranges": ranges,
                "baselineMetrics": evaluate(reference, baseline_notes),
                "paths": {name: str(path) for name, path in paths.items()},
            }
        )

    baseline_aggregate = aggregate_song_metrics(
        (item["weight"], item["baselineMetrics"]) for item in prepared
    )
    combinations = list(
        itertools.product(
            args.support_scopes,
            args.pitch_modes,
            args.support_tolerances,
            args.minimum_support_ratios,
            args.retain_unmatched_primary,
        )
    )
    rows: list[dict[str, Any]] = []
    best_artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for scope, pitch_mode, tolerance, minimum_ratio, retain_primary in combinations:
        song_rows: dict[str, Any] = {}
        candidate_metrics: list[tuple[float, dict[str, Any]]] = []
        artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        trusted_regressions: list[str] = []
        for item in prepared:
            fused = fuse_focused_transcription(
                item["primary"],
                item["focused"],
                ("voice",),
                known_shared_audio=True,
                strategy="corroborated-union",
                support_scope=scope,
                support_pitch_mode=pitch_mode,
                support_tolerance_seconds=tolerance,
                minimum_pass_support_ratio=minimum_ratio,
                retain_unmatched_primary=retain_primary,
            )
            arranged = arrange_payload(
                fused, "full", style_profile=copy.deepcopy(style_profile)
            )
            metrics = evaluate(
                item["reference"],
                notes_inside_ranges(normalize_notes(arranged), item["ranges"]),
            )
            weight = item["weight"]
            candidate_metrics.append((weight, metrics))
            exact_delta = (
                float(metrics["exactPitchOnset100ms"]["f1"])
                - float(item["baselineMetrics"]["exactPitchOnset100ms"]["f1"])
            )
            if weight >= 0.75 and exact_delta < -0.01:
                trusted_regressions.append(item["id"])
            fusion = fused["focusedTranscriptionFusion"]
            song_rows[item["id"]] = {
                "weight": weight,
                "focusedPassApplied": fusion["applied"],
                "focusedSupportRatio": fusion["focusedSupportRatio"],
                "focusedNotesInserted": fusion["focusedNotesInserted"],
                "outputNotes": metrics["observedNotes"],
                "exactF1_100ms": metrics["exactPitchOnset100ms"]["f1"],
                "pitchClassF1_250ms": metrics["pitchClassOnset250ms"]["f1"],
                "pitchClassRecall_250ms": metrics["pitchClassOnset250ms"]["recall"],
                "physicalSevereCutoffRate": metrics["physicalDuration"]["severeCutoffRate"],
            }
            artifacts[item["id"]] = (fused, arranged)
        candidate_aggregate = aggregate_song_metrics(candidate_metrics)
        deltas = {
            name: round(candidate_aggregate[name] - value, 6)
            for name, value in baseline_aggregate.items()
        }
        gates = promotion_gates(
            baseline_aggregate, candidate_aggregate, trusted_regressions
        )
        row = {
            "policy": {
                "strategy": "corroborated-union",
                "supportScope": scope,
                "supportPitchMode": pitch_mode,
                "supportToleranceSeconds": tolerance,
                "minimumPassSupportRatio": minimum_ratio,
                "retainUnmatchedPrimary": retain_primary,
            },
            "passedAllGates": all(gates.values()),
            "passedGateCount": sum(gates.values()),
            "failedGates": [name for name, passed in gates.items() if not passed],
            "candidate": candidate_aggregate,
            "deltas": deltas,
            "songs": song_rows,
        }
        rows.append(row)
        if len(rows) == 1 or ranking_key(row) > ranking_key(max(rows[:-1], key=ranking_key)):
            best_artifacts = artifacts

    rows.sort(key=ranking_key, reverse=True)
    best = rows[0]
    # Recompute the winning artifacts after sorting. This avoids depending on
    # incidental grid traversal order and makes the saved policy reproducible.
    winner = best["policy"]
    best_artifacts = {}
    for item in prepared:
        fused = fuse_focused_transcription(
            item["primary"],
            item["focused"],
            ("voice",),
            known_shared_audio=True,
            strategy=winner["strategy"],
            support_scope=winner["supportScope"],
            support_pitch_mode=winner["supportPitchMode"],
            support_tolerance_seconds=winner["supportToleranceSeconds"],
            minimum_pass_support_ratio=winner["minimumPassSupportRatio"],
            retain_unmatched_primary=winner["retainUnmatchedPrimary"],
        )
        best_artifacts[item["id"]] = (
            fused,
            arrange_payload(
                fused, "full", style_profile=copy.deepcopy(style_profile)
            ),
        )
    for pair_id, (fused, arranged) in best_artifacts.items():
        atomic_json(output_dir / "best-fused" / f"{pair_id}.json", fused)
        atomic_json(output_dir / "best-candidate" / f"{pair_id}.json", arranged)
    report = {
        "schema": "polymath-focused-fusion-policy-search-v1",
        "developmentOnly": True,
        "manifest": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "arrangerProfile": str(profile_path),
        "arrangerProfileSha256": sha256_file(profile_path),
        "trialCount": len(rows),
        "baseline": baseline_aggregate,
        "best": best,
        "leaderboard": rows[: max(1, min(args.top_k, len(rows)))],
    }
    atomic_json(output_dir / "best-policy.json", winner)
    atomic_json(output_dir / "search-report.json", report)
    print(
        json.dumps(
            {
                "trials": len(rows),
                "passed": sum(1 for row in rows if row["passedAllGates"]),
                "bestPolicy": winner,
                "bestDeltas": best["deltas"],
                "failedGates": best["failedGates"],
                "output": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
