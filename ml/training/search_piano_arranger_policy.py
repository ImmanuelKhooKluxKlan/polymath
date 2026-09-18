"""Search a small, auditable Piano-route policy on one development song.

This is deliberately *not* a checkpoint trainer and it never edits the
production profile.  It reuses one frozen upstream transcription, applies a
bounded grid of arranger policies, and scores every result against the same
pre-frozen alignment and baseline.  Only the winning profile/output and a
compact leaderboard are retained.

The song supplied here becomes development evidence after the first score.  A
different, sealed song is still required for any promotion claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ml.training.blend_piano_arranger_profiles import blend_selection_models
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
)


def parse_grid(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        number = float(raw)
        if not math.isfinite(number):
            raise argparse.ArgumentTypeError("grid values must be finite")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return tuple(dict.fromkeys(values))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_profile_hash(profile: dict[str, Any]) -> str:
    canonical = copy.deepcopy(profile)
    canonical.pop("profileSha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_trial_profile(
    template: dict[str, Any],
    *,
    base_selection_model: dict[str, Any],
    other_selection_model: dict[str, Any],
    high_base_share: float,
    high_density: float,
    threshold_offset: float,
    quota_backfill: float,
    short_source_duration_weight: float,
    long_source_duration_weight: float,
    profile_id: str,
    created_at: str,
    bass_legato_bridge_seconds: float | None = None,
    harmony_legato_bridge_seconds: float | None = None,
    bass_physical_hold_seconds: float | None = None,
    harmony_physical_hold_seconds: float | None = None,
    physical_adaptation_low_nps: float | None = None,
    physical_adaptation_high_nps: float | None = None,
    physical_adaptation_minimum_voice_ratio: float = 0.05,
    physical_adaptation_maximum_piano_ratio: float = 0.05,
) -> dict[str, Any]:
    """Return one bounded research profile without mutating its template."""

    profile = copy.deepcopy(template)
    profile.pop("profileSha256", None)
    profile["id"] = profile_id
    profile["createdAt"] = created_at
    decoder = profile.setdefault("decoder", {})
    density_policy = decoder.get("adaptiveSourceDensity")
    if not isinstance(density_policy, dict) or not density_policy.get("enabled"):
        raise ValueError("template requires an enabled adaptiveSourceDensity policy")
    selection_policy = decoder.get("adaptiveSelectionBlend")
    if not isinstance(selection_policy, dict) or not selection_policy.get("enabled"):
        raise ValueError("template requires an enabled adaptiveSelectionBlend policy")

    bounded_share = clamp(high_base_share, 0.0, 1.0)
    bounded_density = clamp(high_density, 0.5, 3.0)
    bounded_backfill = clamp(quota_backfill, 0.0, 1.0)
    bounded_short_source_duration = clamp(short_source_duration_weight, 0.0, 1.0)
    bounded_long_source_duration = clamp(long_source_duration_weight, 0.0, 1.0)

    high_model = blend_selection_models(
        base_selection_model,
        other_selection_model,
        bounded_share,
    )
    high_model["threshold"] = round(
        clamp(float(high_model["threshold"]) + threshold_offset, 0.01, 0.99),
        6,
    )
    selection_policy["highSourceBaseShare"] = round(bounded_share, 6)
    selection_policy["highSourceSelectionModel"] = high_model
    density_policy["highDensityMultiplier"] = round(bounded_density, 4)
    density_policy["shortSourceDurationWeight"] = round(
        bounded_short_source_duration, 4
    )
    density_policy["longSourceDurationWeight"] = round(
        bounded_long_source_duration, 4
    )
    # Confidence backfill becomes stricter only for a dense, vocal,
    # piano-light mix. Sparse, instrumental, and piano-rich sources retain the
    # template's conservative fill so this fix cannot silently delete notes.
    base_backfill = clamp(float(decoder.get("quotaBackfillRatio", 1.0)), 0.0, 1.0)
    density_policy["lowQuotaBackfillRatio"] = round(base_backfill, 4)
    density_policy["highQuotaBackfillRatio"] = round(bounded_backfill, 4)
    density_policy["nonVocalQuotaBackfillRatio"] = round(base_backfill, 4)

    physical_parameters = (
        bass_legato_bridge_seconds,
        harmony_legato_bridge_seconds,
        bass_physical_hold_seconds,
        harmony_physical_hold_seconds,
    )
    if any(value is not None for value in physical_parameters):
        physical = decoder.setdefault("physicalPerformance", {})
        bridges = physical.setdefault("maximumLegatoBridgeSeconds", {})
        holds = physical.setdefault("maximumPhysicalHoldSeconds", {})
        requested_bridges = {
            "melody": float(bridges.get("melody", 0.9)),
            "bass": float(
                bass_legato_bridge_seconds
                if bass_legato_bridge_seconds is not None
                else bridges.get("bass", 1.8)
            ),
            "harmony": float(
                harmony_legato_bridge_seconds
                if harmony_legato_bridge_seconds is not None
                else bridges.get("harmony", 2.4)
            ),
        }
        requested_holds = {
            "melody": float(holds.get("melody", 1.35)),
            "bass": float(
                bass_physical_hold_seconds
                if bass_physical_hold_seconds is not None
                else holds.get("bass", 2.2)
            ),
            "harmony": float(
                harmony_physical_hold_seconds
                if harmony_physical_hold_seconds is not None
                else holds.get("harmony", 2.6)
            ),
        }
        adaptation_requested = (
            physical_adaptation_low_nps is not None
            or physical_adaptation_high_nps is not None
        )
        if adaptation_requested:
            if (
                physical_adaptation_low_nps is None
                or physical_adaptation_high_nps is None
                or physical_adaptation_high_nps <= physical_adaptation_low_nps
            ):
                raise ValueError(
                    "physical adaptation requires low/high NPS with high greater than low"
                )
            high_bridges = {
                "melody": float(bridges.get("melody", 0.9)),
                "bass": float(bridges.get("bass", 1.8)),
                "harmony": float(bridges.get("harmony", 2.4)),
            }
            high_holds = {
                "melody": float(holds.get("melody", 1.35)),
                "bass": float(holds.get("bass", 2.2)),
                "harmony": float(holds.get("harmony", 2.6)),
            }
            bridges.update({role: round(clamp(value, 0.05, 4.0), 4) for role, value in high_bridges.items()})
            holds.update({role: round(clamp(value, 0.055, 4.0), 4) for role, value in high_holds.items()})
            physical["adaptiveSourceDensity"] = {
                "enabled": True,
                "lowSourceNotesPerSecond": round(float(physical_adaptation_low_nps), 4),
                "highSourceNotesPerSecond": round(float(physical_adaptation_high_nps), 4),
                "minimumVoiceRatio": round(clamp(physical_adaptation_minimum_voice_ratio, 0.0, 1.0), 4),
                "maximumPianoRatio": round(clamp(physical_adaptation_maximum_piano_ratio, 0.0, 1.0), 4),
                "lowMaximumLegatoBridgeSeconds": {
                    role: round(clamp(value, 0.05, 4.0), 4)
                    for role, value in requested_bridges.items()
                },
                "highMaximumLegatoBridgeSeconds": {
                    role: round(clamp(value, 0.05, 4.0), 4)
                    for role, value in high_bridges.items()
                },
                "lowMaximumPhysicalHoldSeconds": {
                    role: round(clamp(value, 0.055, 4.0), 4)
                    for role, value in requested_holds.items()
                },
                "highMaximumPhysicalHoldSeconds": {
                    role: round(clamp(value, 0.055, 4.0), 4)
                    for role, value in high_holds.items()
                },
            }
        else:
            bridges.update({role: round(clamp(value, 0.05, 4.0), 4) for role, value in requested_bridges.items()})
            holds.update({role: round(clamp(value, 0.055, 4.0), 4) for role, value in requested_holds.items()})

    profile["searchTrial"] = {
        "schema": "polymath-full-mix-policy-trial-v1",
        "developmentOnly": True,
        "highSourceBaseShare": round(bounded_share, 6),
        "highDensityMultiplier": round(bounded_density, 4),
        "highSelectionThresholdOffset": round(float(threshold_offset), 6),
        "quotaBackfillRatio": round(bounded_backfill, 4),
        "shortSourceDurationWeight": round(bounded_short_source_duration, 4),
        "longSourceDurationWeight": round(bounded_long_source_duration, 4),
        "bassLegatoBridgeSeconds": bass_legato_bridge_seconds,
        "harmonyLegatoBridgeSeconds": harmony_legato_bridge_seconds,
        "bassPhysicalHoldSeconds": bass_physical_hold_seconds,
        "harmonyPhysicalHoldSeconds": harmony_physical_hold_seconds,
        "physicalAdaptationLowSourceNotesPerSecond": physical_adaptation_low_nps,
        "physicalAdaptationHighSourceNotesPerSecond": physical_adaptation_high_nps,
        "physicalAdaptationMinimumVoiceRatio": physical_adaptation_minimum_voice_ratio,
        "physicalAdaptationMaximumPianoRatio": physical_adaptation_maximum_piano_ratio,
    }
    profile["profileSha256"] = canonical_profile_hash(profile)
    return profile


def gate_result(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, float]]:
    deltas = {
        "exactF1_50ms": round(
            float(candidate["exactPitchOnset50ms"]["f1"])
            - float(baseline["exactPitchOnset50ms"]["f1"]),
            6,
        ),
        "pitchClassF1_50ms": round(
            float(candidate["pitchClassOnset50ms"]["f1"])
            - float(baseline["pitchClassOnset50ms"]["f1"]),
            6,
        ),
        "exactF1_100ms": round(
            float(candidate["exactPitchOnset100ms"]["f1"])
            - float(baseline["exactPitchOnset100ms"]["f1"]),
            6,
        ),
        "exactRecall_100ms": round(
            float(candidate["exactPitchOnset100ms"]["recall"])
            - float(baseline["exactPitchOnset100ms"]["recall"]),
            6,
        ),
        "pitchClassF1_100ms": round(
            float(candidate["pitchClassOnset100ms"]["f1"])
            - float(baseline["pitchClassOnset100ms"]["f1"]),
            6,
        ),
        "exactF1_250ms": round(
            float(candidate["exactPitchOnset250ms"]["f1"])
            - float(baseline["exactPitchOnset250ms"]["f1"]),
            6,
        ),
        "pitchClassF1_250ms": round(
            float(candidate["pitchClassOnset250ms"]["f1"])
            - float(baseline["pitchClassOnset250ms"]["f1"]),
            6,
        ),
        "pitchClassRecall_250ms": round(
            float(candidate["pitchClassOnset250ms"]["recall"])
            - float(baseline["pitchClassOnset250ms"]["recall"]),
            6,
        ),
        "durationMedianAbsoluteErrorSeconds": round(
            float(candidate["duration"]["medianAbsoluteErrorSeconds"] or 0.0)
            - float(baseline["duration"]["medianAbsoluteErrorSeconds"] or 0.0),
            6,
        ),
        "visualSevereCutoffRate": round(
            float(candidate["visualDuration"]["severeCutoffRate"])
            - float(baseline["visualDuration"]["severeCutoffRate"]),
            6,
        ),
        "physicalDurationMedianAbsoluteErrorSeconds": round(
            float(candidate["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0)
            - float(
                baseline["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0
            ),
            6,
        ),
        "physicalSevereCutoffRate": round(
            float(candidate["physicalDuration"]["severeCutoffRate"])
            - float(baseline["physicalDuration"]["severeCutoffRate"]),
            6,
        ),
        "rapidRetriggersUnder100ms": round(
            float(candidate["rapidRetriggersUnder100ms"])
            - float(baseline["rapidRetriggersUnder100ms"]),
            6,
        ),
    }
    gates = {
        "exactF1_100ms_improves": at_least(
            float(candidate["exactPitchOnset100ms"]["f1"]),
            float(baseline["exactPitchOnset100ms"]["f1"]) + 0.005,
        ),
        "exactRecall_100ms_improves": at_least(
            float(candidate["exactPitchOnset100ms"]["recall"]),
            float(baseline["exactPitchOnset100ms"]["recall"]) + 0.003,
        ),
        "exactF1_250ms_does_not_regress": at_least(
            float(candidate["exactPitchOnset250ms"]["f1"]),
            float(baseline["exactPitchOnset250ms"]["f1"]) - 0.002,
        ),
        "pitchClassF1_250ms_does_not_regress": at_least(
            float(candidate["pitchClassOnset250ms"]["f1"]),
            float(baseline["pitchClassOnset250ms"]["f1"]) - 0.002,
        ),
        "pitchClassRecall_250ms_does_not_regress": at_least(
            float(candidate["pitchClassOnset250ms"]["recall"]),
            float(baseline["pitchClassOnset250ms"]["recall"]) - 0.002,
        ),
        "duration_error_not_over_10_percent_worse": at_most(
            float(candidate["duration"]["medianAbsoluteErrorSeconds"] or 0.0),
            float(baseline["duration"]["medianAbsoluteErrorSeconds"] or 0.0)
            * 1.10,
        ),
        "visual_cutoff_rate_not_worse": at_most(
            float(candidate["visualDuration"]["severeCutoffRate"]),
            float(baseline["visualDuration"]["severeCutoffRate"]) * 1.05
            + 0.005,
        ),
        "physical_duration_error_not_over_10_percent_worse": at_most(
            float(
                candidate["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0
            ),
            float(
                baseline["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0
            )
            * 1.10,
        ),
        "physical_cutoff_rate_not_worse": at_most(
            float(candidate["physicalDuration"]["severeCutoffRate"]),
            float(baseline["physicalDuration"]["severeCutoffRate"]) * 1.05
            + 0.005,
        ),
        "rapid_retriggers_not_over_5_percent_worse": at_most(
            float(candidate["rapidRetriggersUnder100ms"]),
            float(baseline["rapidRetriggersUnder100ms"]) * 1.05 + 1.0,
        ),
    }
    return gates, deltas


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    deltas = row["deltas"]
    # Passing all frozen gates dominates a larger but unsafe F1 gain.  Within
    # that group, strict timing and then pitch-class coverage lead; density is
    # only a final tie-breaker rather than a target in itself.
    return (
        float(row["passedAllGates"]),
        float(row["passedGateCount"]),
        float(deltas["exactF1_100ms"]),
        float(deltas["exactRecall_100ms"]),
        float(deltas["pitchClassF1_100ms"]),
        float(deltas["exactF1_250ms"]),
        float(deltas["pitchClassF1_250ms"]),
        float(deltas["pitchClassRecall_250ms"]),
        -float(metrics["physicalDurationMedianAbsoluteErrorSeconds"] or 0.0),
        -max(0.0, float(deltas["physicalSevereCutoffRate"])),
        -abs(float(metrics["noteCountRatio"]) - 1.0),
    )


def load_arranger(repo_root: Path) -> Callable[..., dict[str, Any]]:
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

    return arrange_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--template-profile", type=Path, required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--other-profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--high-base-shares", type=parse_grid, required=True)
    parser.add_argument("--high-densities", type=parse_grid, required=True)
    parser.add_argument("--threshold-offsets", type=parse_grid, default=(0.0,))
    parser.add_argument("--quota-backfills", type=parse_grid, default=(1.0,))
    parser.add_argument(
        "--source-duration-weights", type=parse_grid, default=(1.0,)
    )
    parser.add_argument(
        "--long-source-duration-weights",
        type=parse_grid,
        help=(
            "Optional separate long-source hold grid. When omitted, each short "
            "weight is also used for the long-source endpoint."
        ),
    )
    parser.add_argument(
        "--bass-legato-bridges", type=parse_grid, default=(1.8,)
    )
    parser.add_argument(
        "--harmony-legato-bridges", type=parse_grid, default=(2.4,)
    )
    parser.add_argument(
        "--bass-physical-holds", type=parse_grid, default=(2.2,)
    )
    parser.add_argument(
        "--harmony-physical-holds", type=parse_grid, default=(2.6,)
    )
    parser.add_argument("--physical-adaptation-low-nps", type=float)
    parser.add_argument("--physical-adaptation-high-nps", type=float)
    parser.add_argument(
        "--physical-adaptation-minimum-voice-ratio", type=float, default=0.05
    )
    parser.add_argument(
        "--physical-adaptation-maximum-piano-ratio", type=float, default=0.05
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--profile-id",
        help="Optional stable id; valid only when the grid contains one trial.",
    )
    parser.add_argument(
        "--best-profile-output",
        type=Path,
        help="Optional second path for the winning profile (for candidate freezing).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    paths = {
        "input": args.input.resolve(),
        "templateProfile": args.template_profile.resolve(),
        "baseProfile": args.base_profile.resolve(),
        "otherProfile": args.other_profile.resolve(),
        "manifest": args.manifest.resolve(),
        "baselineDirectory": args.baseline_dir.resolve(),
    }
    manifest = load_json(paths["manifest"])
    pairs = manifest.get("pairs") or []
    if len(pairs) != 1:
        raise ValueError("policy search requires a manifest with exactly one pair")
    pair = pairs[0]
    pair_id = str(pair["id"])
    target_path = Path(pair["target"]).resolve()
    alignment_path = Path(pair["alignmentReport"]).resolve()
    baseline_path = paths["baselineDirectory"] / f"{pair_id}-arranged.json"
    for name, path in {
        **paths,
        "target": target_path,
        "alignmentReport": alignment_path,
        "baseline": baseline_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    source_payload = load_json(paths["input"])
    template = load_json(paths["templateProfile"])
    base = load_json(paths["baseProfile"])
    other = load_json(paths["otherProfile"])
    alignment = load_json(alignment_path)
    ranges = trusted_source_ranges(alignment)
    if ranges == []:
        raise ValueError("alignment has zero trusted source ranges")
    target = normalize_notes(
        load_json(target_path),
        transpose_semitones=reference_transpose_semitones(pair),
    )
    mapped_target = notes_inside_ranges(
        map_reference_notes(target, monotonic_anchors(alignment)), ranges
    )
    baseline_notes = notes_inside_ranges(
        normalize_notes(load_json(baseline_path)), ranges
    )
    if len(mapped_target) < 100:
        raise ValueError("fewer than 100 trusted reference notes are available")
    baseline_metrics = evaluate(mapped_target, baseline_notes)
    arrange_payload = load_arranger(repo_root)

    duration_pairs = (
        tuple(itertools.product(args.source_duration_weights, args.long_source_duration_weights))
        if args.long_source_duration_weights is not None
        else tuple((weight, weight) for weight in args.source_duration_weights)
    )
    combinations = list(
        itertools.product(
            args.high_base_shares,
            args.high_densities,
            args.threshold_offsets,
            args.quota_backfills,
            duration_pairs,
            args.bass_legato_bridges,
            args.harmony_legato_bridges,
            args.bass_physical_holds,
            args.harmony_physical_holds,
        )
    )
    if args.profile_id and len(combinations) != 1:
        raise ValueError("--profile-id requires a one-trial grid")
    created_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for index, (
        share,
        density,
        threshold_offset,
        backfill,
        duration_pair,
        bass_bridge,
        harmony_bridge,
        bass_hold,
        harmony_hold,
    ) in enumerate(combinations, start=1):
        short_duration_weight, long_duration_weight = duration_pair
        label = (
            f"s{share:.4f}-d{density:.4f}-t{threshold_offset:+.4f}"
            f"-q{backfill:.4f}-us{short_duration_weight:.4f}"
            f"-ul{long_duration_weight:.4f}-bb{bass_bridge:.4f}"
            f"-hb{harmony_bridge:.4f}-bc{bass_hold:.4f}"
            f"-hc{harmony_hold:.4f}"
        ).replace(".", "p").replace("+", "plus").replace("-", "minus")
        profile = make_trial_profile(
            template,
            base_selection_model=base.get("selectionModel") or {},
            other_selection_model=other.get("selectionModel") or {},
            high_base_share=share,
            high_density=density,
            threshold_offset=threshold_offset,
            quota_backfill=backfill,
            short_source_duration_weight=short_duration_weight,
            long_source_duration_weight=long_duration_weight,
            profile_id=(
                args.profile_id
                if args.profile_id
                else f"pianella-full-mix-search-{label}"
            ),
            created_at=created_at,
            bass_legato_bridge_seconds=bass_bridge,
            harmony_legato_bridge_seconds=harmony_bridge,
            bass_physical_hold_seconds=bass_hold,
            harmony_physical_hold_seconds=harmony_hold,
            physical_adaptation_low_nps=args.physical_adaptation_low_nps,
            physical_adaptation_high_nps=args.physical_adaptation_high_nps,
            physical_adaptation_minimum_voice_ratio=(
                args.physical_adaptation_minimum_voice_ratio
            ),
            physical_adaptation_maximum_piano_ratio=(
                args.physical_adaptation_maximum_piano_ratio
            ),
        )
        arranged = arrange_payload(source_payload, "full", style_profile=profile)
        candidate_notes = notes_inside_ranges(normalize_notes(arranged), ranges)
        metrics = evaluate(mapped_target, candidate_notes)
        gates, deltas = gate_result(baseline_metrics, metrics)
        row = {
            "trial": label,
            "parameters": profile["searchTrial"],
            "profileSha256": profile["profileSha256"],
            "passedAllGates": all(gates.values()),
            "passedGateCount": sum(1 for passed in gates.values() if passed),
            "failedGates": [name for name, passed in gates.items() if not passed],
            "deltas": deltas,
            "metrics": {
                "observedNotes": metrics["observedNotes"],
                "noteCountRatio": metrics["noteCountRatio"],
                "exactF1_50ms": metrics["exactPitchOnset50ms"]["f1"],
                "exactF1_100ms": metrics["exactPitchOnset100ms"]["f1"],
                "pitchClassF1_100ms": metrics["pitchClassOnset100ms"]["f1"],
                "exactF1_250ms": metrics["exactPitchOnset250ms"]["f1"],
                "pitchClassF1_250ms": metrics["pitchClassOnset250ms"]["f1"],
                "durationMedianAbsoluteErrorSeconds": metrics["duration"][
                    "medianAbsoluteErrorSeconds"
                ],
                "visualSevereCutoffRate": metrics["visualDuration"][
                    "severeCutoffRate"
                ],
                "physicalDurationMedianAbsoluteErrorSeconds": metrics[
                    "physicalDuration"
                ]["medianAbsoluteErrorSeconds"],
                "physicalSevereCutoffRate": metrics["physicalDuration"][
                    "severeCutoffRate"
                ],
                "rapidRetriggersUnder100ms": metrics[
                    "rapidRetriggersUnder100ms"
                ],
            },
            "gates": gates,
        }
        rows.append(row)
        ranked = ranking_key(row)
        if best is None or ranked > best[0]:
            best = (ranked, row, profile, arranged)
        if args.progress_every > 0 and (
            index % args.progress_every == 0 or index == len(combinations)
        ):
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(combinations),
                        "passing": sum(1 for item in rows if item["passedAllGates"]),
                        "currentBest": best[1]["trial"] if best else None,
                    }
                ),
                flush=True,
            )

    if best is None:
        raise RuntimeError("search produced no candidates")
    rows.sort(key=ranking_key, reverse=True)
    best_row, best_profile, best_output = best[1], best[2], best[3]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_paths = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            **paths,
            "target": target_path,
            "alignmentReport": alignment_path,
            "baseline": baseline_path,
        }.items()
        if isinstance(path, Path) and path.is_file()
    }
    report = {
        "schema": "polymath-full-mix-policy-search-v1",
        "createdAt": created_at,
        "developmentOnly": True,
        "promotionWarning": (
            "This song was scored during search and is no longer an untouched holdout. "
            "A different sealed real-song test is required."
        ),
        "pairId": pair_id,
        "trustedReferenceNotes": len(mapped_target),
        "trialCount": len(rows),
        "passingTrialCount": sum(1 for row in rows if row["passedAllGates"]),
        "baselineMetrics": baseline_metrics,
        "best": best_row,
        "top": rows[: max(1, args.top_k)],
        "inputs": provenance_paths,
    }
    atomic_json(args.output_dir / "search-report.json", report)
    atomic_json(args.output_dir / "best-profile.json", best_profile)
    atomic_json(args.output_dir / f"{pair_id}.json", best_output)
    if args.best_profile_output is not None:
        atomic_json(args.best_profile_output.resolve(), best_profile)
    print(
        json.dumps(
            {
                "decision": "DEVELOPMENT_PASS"
                if best_row["passedAllGates"]
                else "DEVELOPMENT_REJECT",
                "best": best_row,
                "report": str(args.output_dir / "search-report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
