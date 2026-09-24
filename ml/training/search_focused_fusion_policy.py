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


def load_production_pipeline(repo_root: Path):
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger_pipeline import run_pipeline

    return run_pipeline


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


def subset_trial(row: dict[str, Any], song_ids: set[str]) -> dict[str, Any]:
    selected = [
        song for song_id, song in row["songs"].items() if song_id in song_ids
    ]
    baseline = {
        name: round(
            weighted_average(
                [
                    (float(song["baselineFlattened"][name]), float(song["weight"]))
                    for song in selected
                ]
            ),
            6,
        )
        for name in selected[0]["baselineFlattened"]
    }
    candidate = {
        name: round(
            weighted_average(
                [
                    (float(song["candidateFlattened"][name]), float(song["weight"]))
                    for song in selected
                ]
            ),
            6,
        )
        for name in selected[0]["candidateFlattened"]
    }
    deltas = {
        name: round(candidate[name] - value, 6) for name, value in baseline.items()
    }
    trusted_regressions = [
        song_id
        for song_id, song in row["songs"].items()
        if song_id in song_ids
        and float(song["weight"]) >= 0.75
        and (
            float(song["candidateFlattened"]["exactF1_100ms"])
            - float(song["baselineFlattened"]["exactF1_100ms"])
        )
        < -0.01
    ]
    gates = promotion_gates(baseline, candidate, trusted_regressions)
    return {
        "policy": row["policy"],
        "passedAllGates": all(gates.values()),
        "passedGateCount": sum(gates.values()),
        "failedGates": [name for name, passed in gates.items() if not passed],
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "songs": row["songs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arranger-profile", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        choices=("arranger", "production-v003"),
        default="arranger",
        help="Render candidates through the legacy arranger or the complete current v003 pipeline.",
    )
    parser.add_argument("--recovery-profile", type=Path)
    parser.add_argument("--register-profile", type=Path)
    parser.add_argument(
        "--melody-decoder-config",
        type=Path,
        help="Optional focused-vocal melody decoder profile applied after a pass is accepted.",
    )
    parser.add_argument("--decoder-maximum-unanchored-polyphony", type=parse_grid)
    parser.add_argument("--decoder-upper-candidate-preferences", type=parse_grid)
    parser.add_argument("--decoder-jump-penalties", type=parse_grid)
    parser.add_argument("--decoder-decode-all-candidates", type=parse_booleans)
    parser.add_argument("--decoder-retain-unmatched-primary", type=parse_booleans)
    parser.add_argument("--decoder-minimum-input-notes-per-second", type=parse_grid)
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
    run_pipeline = None
    recovery_profile_path = None
    register_profile_path = None
    recovery_profile = None
    register_profile = None
    melody_decoder_path = (
        args.melody_decoder_config.resolve() if args.melody_decoder_config else None
    )
    melody_decoder = None
    if melody_decoder_path:
        melody_decoder = load_json(melody_decoder_path)
        melody_decoder["enabled"] = True
    decoder_variants: list[dict[str, Any] | None] = [None]
    if melody_decoder is not None:
        polyphony_grid = args.decoder_maximum_unanchored_polyphony or (
            float(melody_decoder.get("maximum_unanchored_polyphony", 3)),
        )
        upper_grid = args.decoder_upper_candidate_preferences or (
            float(melody_decoder.get("upper_candidate_preference", 0.0)),
        )
        jump_grid = args.decoder_jump_penalties or (
            float(melody_decoder.get("jump_penalty_per_semitone", 0.08)),
        )
        decode_all_grid = args.decoder_decode_all_candidates or (
            bool(melody_decoder.get("decode_all_candidates_after_acceptance", False)),
        )
        recover_primary_grid = args.decoder_retain_unmatched_primary or (
            bool(melody_decoder.get("retain_unmatched_primary_when_applied", True)),
        )
        minimum_density_grid = args.decoder_minimum_input_notes_per_second or (
            float(melody_decoder.get("minimum_input_notes_per_second", 0.0)),
        )
        decoder_variants = []
        for (
            maximum_polyphony,
            upper_preference,
            jump_penalty,
            decode_all,
            recover_primary,
            minimum_density,
        ) in itertools.product(
            polyphony_grid,
            upper_grid,
            jump_grid,
            decode_all_grid,
            recover_primary_grid,
            minimum_density_grid,
        ):
            variant = copy.deepcopy(melody_decoder)
            variant["maximum_unanchored_polyphony"] = int(round(maximum_polyphony))
            variant["upper_candidate_preference"] = float(upper_preference)
            variant["jump_penalty_per_semitone"] = float(jump_penalty)
            variant["decode_all_candidates_after_acceptance"] = bool(decode_all)
            variant["retain_unmatched_primary_when_applied"] = bool(recover_primary)
            variant["minimum_input_notes_per_second"] = float(minimum_density)
            decoder_variants.append(variant)
    if args.pipeline == "production-v003":
        recovery_profile_path = (
            args.recovery_profile
            or repo_root / "server" / "models" / "piano-arranger" / "raw-support-selector-v001.json"
        ).resolve()
        register_profile_path = (
            args.register_profile
            or repo_root / "server" / "models" / "piano-arranger" / "adaptive-register-v003.json"
        ).resolve()
        recovery_profile = load_json(recovery_profile_path)
        register_profile = load_json(register_profile_path)
        run_pipeline = load_production_pipeline(repo_root)

    def render(source: dict[str, Any]) -> dict[str, Any]:
        if run_pipeline is None:
            return arrange_payload(
                source, "full", style_profile=copy.deepcopy(style_profile)
            )
        output, _diagnostics = run_pipeline(
            source,
            "full",
            copy.deepcopy(style_profile),
            copy.deepcopy(recovery_profile),
            copy.deepcopy(register_profile),
        )
        return output
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
            decoder_variants,
        )
    )
    rows: list[dict[str, Any]] = []
    best_artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    render_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for (
        scope,
        pitch_mode,
        tolerance,
        minimum_ratio,
        retain_primary,
        decoder_variant,
    ) in combinations:
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
                focused_melody_decoder=copy.deepcopy(decoder_variant),
            )
            note_fingerprint = hashlib.sha256(
                json.dumps(
                    fused.get("notes") or [],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            cache_key = (item["id"], note_fingerprint)
            cached = render_cache.get(cache_key)
            if cached is None:
                arranged = render(copy.deepcopy(fused))
                metrics = evaluate(
                    item["reference"],
                    notes_inside_ranges(normalize_notes(arranged), item["ranges"]),
                )
                render_cache[cache_key] = (arranged, metrics)
            else:
                arranged, metrics = cached
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
                "baselineFlattened": flattened_metrics(item["baselineMetrics"]),
                "candidateFlattened": flattened_metrics(metrics),
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
                "focusedMelodyDecoder": copy.deepcopy(decoder_variant),
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
    song_ids = [item["id"] for item in prepared]
    folds: list[dict[str, Any]] = []
    for holdout_id in song_ids:
        training_ids = set(song_ids) - {holdout_id}
        fold_trials = [subset_trial(row, training_ids) for row in rows]
        fold_trials.sort(key=ranking_key, reverse=True)
        winner = fold_trials[0]
        heldout = winner["songs"][holdout_id]
        heldout_baseline = heldout["baselineFlattened"]
        heldout_candidate = heldout["candidateFlattened"]
        heldout_deltas = {
            name: round(
                float(heldout_candidate[name]) - float(heldout_baseline[name]), 6
            )
            for name in heldout_baseline
        }
        folds.append(
            {
                "holdoutSong": holdout_id,
                "trainingSongs": sorted(training_ids),
                "selectedPolicy": winner["policy"],
                "trainingPassedAllGates": winner["passedAllGates"],
                "trainingFailedGates": winner["failedGates"],
                "trainingDeltas": winner["deltas"],
                "heldout": {
                    "focusedPassApplied": heldout["focusedPassApplied"],
                    "focusedSupportRatio": heldout["focusedSupportRatio"],
                    "focusedNotesInserted": heldout["focusedNotesInserted"],
                    "baseline": heldout_baseline,
                    "candidate": heldout_candidate,
                    "deltas": heldout_deltas,
                },
            }
        )

    loso_baseline = {
        name: round(
            weighted_average(
                [
                    (
                        float(fold["heldout"]["baseline"][name]),
                        float(next(item["weight"] for item in prepared if item["id"] == fold["holdoutSong"])),
                    )
                    for fold in folds
                ]
            ),
            6,
        )
        for name in folds[0]["heldout"]["baseline"]
    }
    loso_candidate = {
        name: round(
            weighted_average(
                [
                    (
                        float(fold["heldout"]["candidate"][name]),
                        float(next(item["weight"] for item in prepared if item["id"] == fold["holdoutSong"])),
                    )
                    for fold in folds
                ]
            ),
            6,
        )
        for name in folds[0]["heldout"]["candidate"]
    }
    loso_deltas = {
        name: round(loso_candidate[name] - value, 6)
        for name, value in loso_baseline.items()
    }
    loso_nonregressing = all(
        float(fold["heldout"]["deltas"]["exactF1_100ms"]) >= -0.005
        for fold in folds
    )
    loso_gates = promotion_gates(
        loso_baseline,
        loso_candidate,
        [
            fold["holdoutSong"]
            for fold in folds
            if float(fold["heldout"]["deltas"]["exactF1_100ms"]) < -0.01
        ],
    )
    loso_decision = (
        "CANDIDATE_FOR_NEW_SEALED_HOLDOUT"
        if all(loso_gates.values()) and loso_nonregressing
        else "REJECT_NO_WHOLE_SONG_TRANSFER"
    )
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
            focused_melody_decoder=copy.deepcopy(
                winner.get("focusedMelodyDecoder")
            ),
        )
        best_artifacts[item["id"]] = (
            fused,
            render(fused),
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
        "pipeline": args.pipeline,
        "recoveryProfile": str(recovery_profile_path) if recovery_profile_path else None,
        "recoveryProfileSha256": (
            sha256_file(recovery_profile_path) if recovery_profile_path else None
        ),
        "registerProfile": str(register_profile_path) if register_profile_path else None,
        "registerProfileSha256": (
            sha256_file(register_profile_path) if register_profile_path else None
        ),
        "melodyDecoderProfile": str(melody_decoder_path) if melody_decoder_path else None,
        "melodyDecoderProfileSha256": (
            sha256_file(melody_decoder_path) if melody_decoder_path else None
        ),
        "trialCount": len(rows),
        "uniqueRenderedSongCandidates": len(render_cache),
        "baseline": baseline_aggregate,
        "best": best,
        "wholeSongLeaveOneOut": {
            "decision": loso_decision,
            "allHeldoutExact100WithinHalfPoint": loso_nonregressing,
            "gates": loso_gates,
            "baseline": loso_baseline,
            "candidate": loso_candidate,
            "deltas": loso_deltas,
            "folds": folds,
        },
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
                "leaveOneSongOutDecision": loso_decision,
                "leaveOneSongOutDeltas": loso_deltas,
                "output": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
