"""Fit an interpretable cyclic harmony pattern on one approved development song.

This is intentionally a same-song supervised experiment, not a holdout score.
It searches only symbolic actions (preserve, octave-swap, or hand thinning),
keeps the legacy onset grid, and then reports the frozen pattern on every other
available song.  The output must remain research-only until unseen songs pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from ml.training.evaluate_piano_arranger import (
    load_json,
    prepare_reference_notes,
)
from ml.training.search_default_piano_pipeline import (
    atomic_json,
    clip_candidate,
    compact_metrics,
    load_rows,
    measure,
    notes_in_time_window,
    scalar_quality,
)


COMPACT_ACTIONS = (
    "compact-preserve",
    "compact-octave-root",
    "compact-upper",
    "compact-inner",
    "compact-root+upper",
    "low-root+upper",
    "low-inner+upper",
    "root+upper",
    "upper",
    "inner",
)

PULSE_GRID_ACTIONS = (
    "rest",
    "low-root+upper",
    "low-inner+upper",
    "root+upper",
    "upper",
    "inner",
    "root",
)

DEFAULT_PULSE_PATTERN = (
    "low-root+upper",
    "inner",
    "low-inner+upper",
    "root+upper",
    "inner",
    "upper",
    "upper",
    "inner",
)


def profile(
    pattern: list[str],
    *,
    broad_gate: bool = False,
    onset_policy: str = "compact",
    phase_clock: str = "event-index",
    context_action_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "polymath-piano-arranger-profile-v1",
        "id": f"pianella-source-supported-{onset_policy}-phase-search-v1",
        "decoder": {
            "defaultPipeline": True,
            "defaultFullMix": {
                "harmonyWindowSeconds": 0.30,
                "maximumHarmonyPitchClasses": 4,
                "sourceSupportedCyclicHarmony": {
                    "enabled": True,
                    "onsetPolicy": onset_policy,
                    "phaseClock": phase_clock,
                    "onsetWindowSeconds": 0.035,
                    "minimumPulseSeconds": 0.125,
                    "maximumPulseSeconds": 0.23 if broad_gate else 0.17,
                    "minimumGuitarShare": 0.50 if broad_gate else 0.70,
                    "minimumGridCoverage": 0.78,
                    "minimumStableTransitionShare": 0.45,
                    "minimumRunGroups": 8,
                    "contextRadiusSeconds": 0.26,
                    "phasePattern": pattern,
                    "contextActionMap": dict(context_action_map or {}),
                    "contextMelodyRadiusSeconds": 0.18,
                    "contextHarmonyChangeThreshold": 0.52,
                    **(
                        {
                            "useBassContextForRoot": True,
                            "bassContextRadiusSeconds": 1.20,
                        }
                        if phase_clock == "harmonic-root-event-index"
                        else {}
                    ),
                    **(
                        {"snapToRoleOnsetsSeconds": 0.08}
                        if onset_policy == "pulse-grid"
                        else {}
                    ),
                },
            },
            "gestureDynamics": {
                "enabled": True,
                "onsetWindowSeconds": 0.035,
                "minimumVelocity": 0.38,
                "maximumVelocity": 0.94,
                "sourceBlend": 0.82,
                "melodyPerformanceGain": 1.12,
                "accompanimentGainDuringMelody": 0.88,
                "neutralPerformanceGain": 1.0,
            },
        },
        "training": {
            "status": "RESEARCH_ONLY",
            "method": "same-song coordinate search of transposition-invariant phase actions",
            "commercialUseAllowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-song", default="kiss-me")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--onset-policy",
        choices=("compact", "pulse-grid"),
        default="compact",
    )
    parser.add_argument("--phase-count", type=int, default=8)
    parser.add_argument(
        "--fixed-pattern",
        default="",
        help=(
            "Comma-separated frozen actions. When supplied, no target-driven "
            "coordinate search is performed; the pattern is evaluated only."
        ),
    )
    parser.add_argument(
        "--context-action",
        action="append",
        default=[],
        help="Frozen source-context override in KEY=ACTION form; may be repeated.",
    )
    parser.add_argument(
        "--phase-clock",
        choices=("event-index", "pulse-grid", "harmonic-root-event-index"),
        default="event-index",
    )
    parser.add_argument("--training-start-seconds", type=float, default=0.0)
    parser.add_argument("--training-end-seconds", type=float)
    parser.add_argument("--validation-start-seconds", type=float)
    parser.add_argument("--validation-end-seconds", type=float)
    args = parser.parse_args()
    fixed_pattern = [
        value.strip() for value in args.fixed_pattern.split(",") if value.strip()
    ]
    if fixed_pattern:
        args.phase_count = len(fixed_pattern)
    if not 1 <= args.phase_count <= 32:
        raise ValueError("phase-count must be between 1 and 32")
    actions = (
        PULSE_GRID_ACTIONS
        if args.onset_policy == "pulse-grid"
        else COMPACT_ACTIONS
    )
    context_action_map: dict[str, str] = {}
    for value in args.context_action:
        if "=" not in value:
            raise ValueError("context-action must use KEY=ACTION form")
        key, action = (part.strip() for part in value.split("=", 1))
        if not key or action not in COMPACT_ACTIONS:
            raise ValueError(f"Unsupported context action: {value}")
        context_action_map[key] = action

    rows = load_rows(args.manifest.resolve())
    selected = [row for row in rows if str(row["id"]) == args.development_song]
    if len(selected) != 1:
        raise ValueError("development song must occur exactly once")

    repo_root = Path(__file__).resolve().parents[2]
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

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
                style_profile=profile(
                    pattern,
                    onset_policy=args.onset_policy,
                    phase_clock=args.phase_clock,
                    context_action_map=context_action_map,
                ),
            )
            candidate_notes = clip_candidate(
                arranged, development["row"], development["alignment"]
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

    if fixed_pattern:
        unsupported = sorted(set(fixed_pattern) - set(actions))
        if unsupported:
            raise ValueError(
                "Fixed pattern has unsupported actions: " + ", ".join(unsupported)
            )
        pattern = list(fixed_pattern)
    elif args.onset_policy == "pulse-grid":
        pattern = [
            DEFAULT_PULSE_PATTERN[index % len(DEFAULT_PULSE_PATTERN)]
            for index in range(args.phase_count)
        ]
    else:
        pattern = ["compact-preserve"] * args.phase_count
    baseline_quality = scalar_quality(training_baseline)
    best_quality, _best_output, _best_metrics = evaluate_pattern(pattern)
    for cycle in range(0 if fixed_pattern else max(1, args.cycles)):
        changed = False
        for phase in range(len(pattern)):
            phase_trials = []
            for action in actions:
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
            phase_trials.sort(key=lambda row: float(row["quality"]), reverse=True)
            winner = phase_trials[0]
            trials.extend(phase_trials)
            if winner["action"] != pattern[phase]:
                changed = True
            pattern = list(winner["pattern"])
            best_quality = float(winner["quality"])
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
    frozen_profile = profile(
        pattern,
        onset_policy=args.onset_policy,
        phase_clock=args.phase_clock,
        context_action_map=context_action_map,
    )
    transfer: dict[str, Any] = {}
    for song_id, item in cache.items():
        if song_id == args.development_song:
            continue
        arranged = arrange_payload(
            copy.deepcopy(item["source"]),
            "full",
            style_profile=profile(
                pattern,
                broad_gate=True,
                onset_policy=args.onset_policy,
                phase_clock=args.phase_clock,
                context_action_map=context_action_map,
            ),
        )
        metrics = measure(
            item["reference"],
            clip_candidate(arranged, item["row"], item["alignment"]),
        )
        transfer[song_id] = {
            "baseline": compact_metrics(item["baseline"]),
            "candidate": compact_metrics(metrics),
            "qualityDelta": round(scalar_quality(metrics) - scalar_quality(item["baseline"]), 6),
            "cyclicDiagnostics": arranged.get("pianoArrangement", {}).get(
                "sourceSupportedCyclicHarmony"
            ),
        }

    output_dir = args.output_dir.resolve()
    atomic_json(output_dir / "frozen-research-profile.json", frozen_profile)
    atomic_json(output_dir / f"{args.development_song}-candidate.json", best_output)
    report = {
        "schema": "polymath-cyclic-harmony-phase-search-v1",
        "evidenceBoundary": (
            "Only the declared training window selected this pattern. The chronological "
            "validation window and other songs were scored only after it was frozen."
            if validation_start is not None
            else "The development target selected this pattern. Its score is not test "
            "accuracy. Other songs are reported only after the pattern is frozen."
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
        "onsetPolicy": args.onset_policy,
        "phaseCount": args.phase_count,
        "fixedPatternEvaluation": bool(fixed_pattern),
        "contextActionMap": context_action_map,
        "phaseClock": args.phase_clock,
        "baseline": {
            "selection": compact_metrics(training_baseline),
            "full": compact_metrics(development["baseline"]),
            "chronologicalValidation": (
                compact_metrics(validation_baseline)
                if validation_baseline is not None
                else None
            ),
        },
        "winner": {
            "phasePattern": pattern,
            "selectionMetrics": compact_metrics(best_metrics),
            "selectionQualityDelta": round(best_quality - baseline_quality, 6),
            "fullMetrics": compact_metrics(full_metrics),
            "chronologicalValidationMetrics": (
                compact_metrics(validation_metrics)
                if validation_metrics is not None
                else None
            ),
            "chronologicalValidationQualityDelta": (
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
        "trials": sorted(trials, key=lambda row: float(row["quality"]), reverse=True),
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "pattern": pattern,
                "developmentQualityDelta": round(best_quality - baseline_quality, 6),
                "selection": compact_metrics(best_metrics),
                "chronologicalValidation": (
                    compact_metrics(validation_metrics)
                    if validation_metrics is not None
                    else None
                ),
                "chronologicalValidationQualityDelta": (
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
                    song_id: row["qualityDelta"] for song_id, row in transfer.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
