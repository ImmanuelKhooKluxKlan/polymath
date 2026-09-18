"""Search a source-supported missing-pulse supplement for a frozen arranger.

The base arrangement and its existing onsets stay frozen.  Candidate actions
may add a small bass, inner, or upper gesture only where (a) the source exposes
a stable short pulse, (b) no compact harmony onset already occupies the slot,
and (c) no detected vocal-melody onset is nearby.  The destination target is
used to select the development pattern, never while applying that pattern.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from ml.training.evaluate_piano_arranger import load_json, prepare_reference_notes
from ml.training.search_default_piano_pipeline import (
    atomic_json,
    clip_candidate,
    compact_metrics,
    load_rows,
    measure,
    notes_in_time_window,
    scalar_quality,
)


ACTIONS = (
    "rest",
    "inner",
    "upper",
    "root",
    "root+upper",
    "low-root+upper",
    "low-inner+upper",
)


def configured_profile(
    base: dict[str, Any],
    pattern: list[str],
    *,
    quiet_radius: float,
    minimum_onset_distance: float,
    broad_gate: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    result["id"] = "pianella-compact-pulse-supplement-search-v1"
    decoder = result.setdefault("decoder", {})
    decoder["defaultPipeline"] = True
    full_mix = decoder.setdefault("defaultFullMix", {})
    cyclic = full_mix.setdefault("sourceSupportedCyclicHarmony", {})
    cyclic.update(
        {
            "enabled": True,
            "onsetPolicy": "compact",
            "phaseClock": "pulse-grid",
            "supplementalPhasePattern": pattern,
            "supplementalMinimumOnsetDistanceSeconds": minimum_onset_distance,
            "supplementalMelodyQuietRadiusSeconds": quiet_radius,
        }
    )
    if broad_gate:
        cyclic["maximumPulseSeconds"] = 0.23
        cyclic["minimumGuitarShare"] = 0.50
    result["training"] = {
        "status": "RESEARCH_ONLY",
        "method": "same-song coordinate search of source-supported missing pulses",
        "commercialUseAllowed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--development-song", default="kiss-me")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-count", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--melody-quiet-radius", type=float, default=0.45)
    parser.add_argument("--minimum-onset-distance", type=float, default=0.08)
    parser.add_argument("--training-start-seconds", type=float, default=0.0)
    parser.add_argument("--training-end-seconds", type=float)
    parser.add_argument("--validation-start-seconds", type=float)
    parser.add_argument("--validation-end-seconds", type=float)
    parser.add_argument(
        "--fixed-phase-pattern",
        help="Comma-separated source-supported actions to evaluate unchanged.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the fixed pattern without consulting reference labels to change it.",
    )
    args = parser.parse_args()
    if not 1 <= args.phase_count <= 32:
        raise ValueError("phase-count must be between 1 and 32")
    if not 0.10 <= args.melody_quiet_radius <= 2.0:
        raise ValueError("melody-quiet-radius must be between 0.10 and 2.0")
    if not 0.04 <= args.minimum_onset_distance <= 0.20:
        raise ValueError("minimum-onset-distance must be between 0.04 and 0.20")

    validation_start = args.validation_start_seconds
    if validation_start is None and args.training_end_seconds is not None:
        validation_start = args.training_end_seconds
    if validation_start is not None:
        training_end = (
            float("inf")
            if args.training_end_seconds is None
            else args.training_end_seconds
        )
        validation_end = (
            float("inf")
            if args.validation_end_seconds is None
            else args.validation_end_seconds
        )
        if (
            args.training_start_seconds < validation_end
            and validation_start < training_end
        ):
            raise ValueError("training and validation windows must not overlap")

    rows = load_rows(args.manifest.resolve())
    selected = [row for row in rows if str(row["id"]) == args.development_song]
    if len(selected) != 1:
        raise ValueError("development song must occur exactly once")
    base_profile = load_json(args.base_profile.resolve())

    repo_root = Path(__file__).resolve().parents[2]
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        alignment = load_json(Path(str(row["alignment"])).resolve())
        reference = prepare_reference_notes(
            row,
            load_json(Path(str(row["reference"])).resolve()),
            alignment,
        )
        baseline_notes = clip_candidate(
            load_json(Path(str(row["baseline"])).resolve()),
            row,
            alignment,
        )
        cache[str(row["id"])] = {
            "row": row,
            "alignment": alignment,
            "reference": reference,
            "source": load_json(Path(str(row["source"])).resolve()),
            "baselineNotes": baseline_notes,
            "baseline": measure(reference, baseline_notes),
        }

    development = cache[args.development_song]
    training_reference = notes_in_time_window(
        development["reference"],
        start_seconds=args.training_start_seconds,
        end_seconds=args.training_end_seconds,
    )
    training_baseline_notes = notes_in_time_window(
        development["baselineNotes"],
        start_seconds=args.training_start_seconds,
        end_seconds=args.training_end_seconds,
    )
    if not training_reference:
        raise ValueError("training window contains no reference notes")
    training_baseline = measure(training_reference, training_baseline_notes)

    validation_reference: list[dict[str, Any]] = []
    validation_baseline: dict[str, Any] | None = None
    if validation_start is not None:
        validation_reference = notes_in_time_window(
            development["reference"],
            start_seconds=validation_start,
            end_seconds=args.validation_end_seconds,
        )
        if not validation_reference:
            raise ValueError("validation window contains no reference notes")
        validation_baseline = measure(
            validation_reference,
            notes_in_time_window(
                development["baselineNotes"],
                start_seconds=validation_start,
                end_seconds=args.validation_end_seconds,
            ),
        )
    trials: list[dict[str, Any]] = []
    output_cache: dict[tuple[str, ...], tuple[dict[str, Any], dict[str, Any]]] = {}

    def evaluate_pattern(pattern: list[str]) -> tuple[float, dict[str, Any], dict[str, Any]]:
        key = tuple(pattern)
        cached = output_cache.get(key)
        if cached is None:
            arranged = arrange_payload(
                copy.deepcopy(development["source"]),
                "full",
                style_profile=configured_profile(
                    base_profile,
                    pattern,
                    quiet_radius=args.melody_quiet_radius,
                    minimum_onset_distance=args.minimum_onset_distance,
                ),
            )
            candidate_notes = clip_candidate(
                arranged,
                development["row"],
                development["alignment"],
            )
            metrics = measure(
                training_reference,
                notes_in_time_window(
                    candidate_notes,
                    start_seconds=args.training_start_seconds,
                    end_seconds=args.training_end_seconds,
                ),
            )
            output_cache[key] = (arranged, metrics)
        arranged, metrics = output_cache[key]
        return scalar_quality(metrics), arranged, metrics

    rest_pattern = ["rest"] * args.phase_count
    control_quality, _control_output, control_metrics = evaluate_pattern(rest_pattern)
    if args.fixed_phase_pattern:
        pattern = [
            value.strip()
            for value in args.fixed_phase_pattern.split(",")
            if value.strip()
        ]
        if not pattern or any(action not in ACTIONS for action in pattern):
            raise ValueError("fixed-phase-pattern contains an unsupported action")
    else:
        pattern = list(rest_pattern)
    if args.evaluate_only and not args.fixed_phase_pattern:
        raise ValueError("evaluate-only requires fixed-phase-pattern")
    for cycle in range(0 if args.evaluate_only else max(1, args.cycles)):
        changed = False
        for phase in range(len(pattern)):
            phase_trials = []
            for action in ACTIONS:
                candidate_pattern = list(pattern)
                candidate_pattern[phase] = action
                quality, _output, metrics = evaluate_pattern(candidate_pattern)
                phase_trials.append(
                    {
                        "cycle": cycle + 1,
                        "phase": phase,
                        "action": action,
                        "pattern": candidate_pattern,
                        "quality": round(quality, 6),
                        "metrics": compact_metrics(metrics),
                    }
                )
            winner = max(phase_trials, key=lambda row: float(row["quality"]))
            trials.extend(phase_trials)
            changed = changed or winner["action"] != pattern[phase]
            pattern = list(winner["pattern"])
        if not changed:
            break

    best_quality, best_output, best_metrics = evaluate_pattern(pattern)
    best_candidate_notes = clip_candidate(
        best_output, development["row"], development["alignment"]
    )
    full_metrics = measure(development["reference"], best_candidate_notes)
    validation_metrics = (
        measure(
            validation_reference,
            notes_in_time_window(
                best_candidate_notes,
                start_seconds=float(validation_start),
                end_seconds=args.validation_end_seconds,
            ),
        )
        if validation_start is not None
        else None
    )
    frozen_profile = configured_profile(
        base_profile,
        pattern,
        quiet_radius=args.melody_quiet_radius,
        minimum_onset_distance=args.minimum_onset_distance,
    )
    transfer: dict[str, Any] = {}
    for song_id, item in cache.items():
        if song_id == args.development_song:
            continue
        arranged = arrange_payload(
            copy.deepcopy(item["source"]),
            "full",
            style_profile=configured_profile(
                base_profile,
                pattern,
                quiet_radius=args.melody_quiet_radius,
                minimum_onset_distance=args.minimum_onset_distance,
                broad_gate=True,
            ),
        )
        metrics = measure(
            item["reference"],
            clip_candidate(arranged, item["row"], item["alignment"]),
        )
        transfer[song_id] = {
            "baseline": compact_metrics(item["baseline"]),
            "candidate": compact_metrics(metrics),
            "qualityDelta": round(
                scalar_quality(metrics) - scalar_quality(item["baseline"]), 6
            ),
            "cyclicDiagnostics": arranged.get("pianoArrangement", {}).get(
                "sourceSupportedCyclicHarmony"
            ),
        }

    output_dir = args.output_dir.resolve()
    atomic_json(output_dir / "frozen-research-profile.json", frozen_profile)
    atomic_json(output_dir / f"{args.development_song}-candidate.json", best_output)
    report = {
        "schema": "polymath-pulse-grid-supplement-search-v1",
        "evidenceBoundary": (
            "Only the declared training window selected this supplement. The "
            "chronological validation window and transfer songs were evaluated "
            "only after freezing."
            if validation_start is not None
            else "The development target selected the supplement. Its score is not "
            "test accuracy. Transfer songs are evaluated only after freezing."
        ),
        "developmentSong": args.development_song,
        "selectionWindow": {
            "startSeconds": args.training_start_seconds,
            "endSeconds": args.training_end_seconds,
        },
        "chronologicalValidationWindow": (
            {
                "startSeconds": validation_start,
                "endSeconds": args.validation_end_seconds,
            }
            if validation_start is not None
            else None
        ),
        "melodyQuietRadiusSeconds": args.melody_quiet_radius,
        "minimumOnsetDistanceSeconds": args.minimum_onset_distance,
        "selectionMode": "fixed-evaluation" if args.evaluate_only else "coordinate-search",
        "control": {
            "selection": compact_metrics(control_metrics),
            "fullBaseline": compact_metrics(development["baseline"]),
            "chronologicalValidationBaseline": (
                compact_metrics(validation_baseline)
                if validation_baseline is not None
                else None
            ),
        },
        "winner": {
            "supplementalPhasePattern": pattern,
            "selectionMetrics": compact_metrics(best_metrics),
            "selectionQualityDeltaVsNoSupplement": round(
                best_quality - control_quality, 6
            ),
            "fullMetrics": compact_metrics(full_metrics),
            "chronologicalValidationMetrics": (
                compact_metrics(validation_metrics)
                if validation_metrics is not None
                else None
            ),
            "chronologicalValidationQualityDeltaVsBaseline": (
                round(
                    scalar_quality(validation_metrics)
                    - scalar_quality(validation_baseline),
                    6,
                )
                if validation_metrics is not None
                and validation_baseline is not None
                else None
            ),
        },
        "crossSongTransfer": transfer,
        "trials": sorted(
            trials, key=lambda row: float(row["quality"]), reverse=True
        ),
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "pattern": pattern,
                "developmentQualityDeltaVsNoSupplement": round(
                    best_quality - control_quality, 6
                ),
                "selection": compact_metrics(best_metrics),
                "chronologicalValidation": (
                    compact_metrics(validation_metrics)
                    if validation_metrics is not None
                    else None
                ),
                "chronologicalValidationQualityDeltaVsBaseline": (
                    round(
                        scalar_quality(validation_metrics)
                        - scalar_quality(validation_baseline),
                        6,
                    )
                    if validation_metrics is not None
                    and validation_baseline is not None
                    else None
                ),
                "transferQualityDeltas": {
                    song_id: row["qualityDelta"]
                    for song_id, row in transfer.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
