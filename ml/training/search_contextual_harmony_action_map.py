"""Search a tiny source-context harmony policy with an unseen time split.

Unlike a cyclic phase lookup, this experiment never keys a decision by song
timestamp or a fixed phase number.  It chooses among source-supported voicings
using only three observable states at each compact harmony onset:

* whether detected melody is nearby;
* whether the compact gesture occupies the left, right, or both hands;
* whether the local harmony is stable or has just changed.

The declared training window selects the twelve-cell action map.  A disjoint
chronological window and every other song are evaluated only after the map is
frozen.  Outputs remain research-only until independent songs pass.
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

CONTEXT_KEYS = tuple(
    f"{melody}:{occupancy}:{harmony}"
    for melody in ("melody", "quiet")
    for occupancy in ("left", "right", "both")
    for harmony in ("change", "stable")
)


def make_profile(
    action_map: dict[str, str],
    *,
    melody_radius: float,
    harmony_change_threshold: float,
    broad_gate: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "polymath-piano-arranger-profile-v1",
        "id": "pianella-source-context-harmony-search-v1",
        "decoder": {
            "defaultPipeline": True,
            "defaultFullMix": {
                "harmonyWindowSeconds": 0.30,
                "maximumHarmonyPitchClasses": 4,
                "sourceSupportedCyclicHarmony": {
                    "enabled": True,
                    "onsetPolicy": "compact",
                    "phaseClock": "pulse-grid",
                    "phasePattern": ["compact-preserve"],
                    "onsetWindowSeconds": 0.035,
                    "minimumPulseSeconds": 0.125,
                    "maximumPulseSeconds": 0.23 if broad_gate else 0.17,
                    "minimumGuitarShare": 0.50 if broad_gate else 0.70,
                    "minimumGridCoverage": 0.78,
                    "minimumStableTransitionShare": 0.45,
                    "minimumRunGroups": 8,
                    "contextRadiusSeconds": 0.26,
                    "contextMelodyRadiusSeconds": melody_radius,
                    "contextHarmonyChangeThreshold": harmony_change_threshold,
                    "contextActionMap": dict(action_map),
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
            "method": "chronological source-context action-map coordinate search",
            "commercialUseAllowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-song", default="kiss-me")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--training-start-seconds", type=float, default=0.0)
    parser.add_argument("--training-end-seconds", type=float, required=True)
    parser.add_argument("--validation-start-seconds", type=float, required=True)
    parser.add_argument("--validation-end-seconds", type=float)
    parser.add_argument("--melody-radius", type=float, default=0.18)
    parser.add_argument("--harmony-change-threshold", type=float, default=0.52)
    parser.add_argument(
        "--fixed-context-action",
        action="append",
        default=[],
        metavar="CONTEXT=ACTION",
        help="Seed or freeze a context action; repeat for multiple contexts.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the fixed map without using any reference to change it.",
    )
    args = parser.parse_args()

    if not 0.04 <= args.melody_radius <= 0.60:
        raise ValueError("melody-radius must be between 0.04 and 0.60")
    if not 0.05 <= args.harmony_change_threshold <= 0.95:
        raise ValueError("harmony-change-threshold must be between 0.05 and 0.95")
    training_end = args.training_end_seconds
    validation_end = (
        float("inf")
        if args.validation_end_seconds is None
        else args.validation_end_seconds
    )
    if (
        args.training_start_seconds < validation_end
        and args.validation_start_seconds < training_end
    ):
        raise ValueError("training and validation windows must not overlap")

    rows = load_rows(args.manifest.resolve())
    selected = [row for row in rows if str(row["id"]) == args.development_song]
    if len(selected) != 1:
        raise ValueError("development song must occur exactly once")

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
            load_json(Path(str(row["baseline"])).resolve()), row, alignment
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
    validation_reference = notes_in_time_window(
        development["reference"],
        start_seconds=args.validation_start_seconds,
        end_seconds=args.validation_end_seconds,
    )
    if not training_reference or not validation_reference:
        raise ValueError("training and validation windows must both contain reference notes")
    training_baseline = measure(
        training_reference,
        notes_in_time_window(
            development["baselineNotes"],
            start_seconds=args.training_start_seconds,
            end_seconds=args.training_end_seconds,
        ),
    )
    validation_baseline = measure(
        validation_reference,
        notes_in_time_window(
            development["baselineNotes"],
            start_seconds=args.validation_start_seconds,
            end_seconds=args.validation_end_seconds,
        ),
    )

    output_cache: dict[tuple[str, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
    trials: list[dict[str, Any]] = []

    def evaluate_map(
        action_map: dict[str, str],
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
        key = tuple(action_map[name] for name in CONTEXT_KEYS)
        cached = output_cache.get(key)
        if cached is None:
            arranged = arrange_payload(
                copy.deepcopy(development["source"]),
                "full",
                style_profile=make_profile(
                    action_map,
                    melody_radius=args.melody_radius,
                    harmony_change_threshold=args.harmony_change_threshold,
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
            output_cache[key] = arranged, metrics
        arranged, metrics = output_cache[key]
        return scalar_quality(metrics), arranged, metrics

    action_map = {key: "compact-preserve" for key in CONTEXT_KEYS}
    for assignment in args.fixed_context_action:
        context_key, separator, action = str(assignment).partition("=")
        if not separator or context_key not in CONTEXT_KEYS or action not in ACTIONS:
            raise ValueError(
                "fixed-context-action must be a known CONTEXT=ACTION assignment"
            )
        action_map[context_key] = action
    control_quality, _control_output, control_metrics = evaluate_map(action_map)
    for cycle in range(0 if args.evaluate_only else max(1, args.cycles)):
        changed = False
        for context_key in CONTEXT_KEYS:
            rows_for_context = []
            incumbent_action = action_map[context_key]
            for action in ACTIONS:
                candidate_map = dict(action_map)
                candidate_map[context_key] = action
                quality, _output, metrics = evaluate_map(candidate_map)
                rows_for_context.append(
                    {
                        "cycle": cycle + 1,
                        "context": context_key,
                        "action": action,
                        "quality": round(quality, 6),
                        "metrics": compact_metrics(metrics),
                    }
                )
            winner = max(
                rows_for_context,
                key=lambda row: (
                    float(row["quality"]),
                    int(row["action"] == incumbent_action),
                    -ACTIONS.index(str(row["action"])),
                ),
            )
            trials.extend(rows_for_context)
            changed = changed or winner["action"] != incumbent_action
            action_map[context_key] = str(winner["action"])
        if not changed:
            break

    best_quality, best_output, best_training_metrics = evaluate_map(action_map)
    best_candidate_notes = clip_candidate(
        best_output, development["row"], development["alignment"]
    )
    validation_metrics = measure(
        validation_reference,
        notes_in_time_window(
            best_candidate_notes,
            start_seconds=args.validation_start_seconds,
            end_seconds=args.validation_end_seconds,
        ),
    )
    full_metrics = measure(development["reference"], best_candidate_notes)
    frozen_profile = make_profile(
        action_map,
        melody_radius=args.melody_radius,
        harmony_change_threshold=args.harmony_change_threshold,
    )

    transfer: dict[str, Any] = {}
    for song_id, item in cache.items():
        if song_id == args.development_song:
            continue
        arranged = arrange_payload(
            copy.deepcopy(item["source"]),
            "full",
            style_profile=make_profile(
                action_map,
                melody_radius=args.melody_radius,
                harmony_change_threshold=args.harmony_change_threshold,
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
        "schema": "polymath-contextual-harmony-action-search-v1",
        "evidenceBoundary": (
            "Only the declared training window selected source-context actions. "
            "The disjoint validation window and other songs were scored after freezing."
        ),
        "developmentSong": args.development_song,
        "selectionWindow": {
            "startSeconds": args.training_start_seconds,
            "endSeconds": args.training_end_seconds,
        },
        "chronologicalValidationWindow": {
            "startSeconds": args.validation_start_seconds,
            "endSeconds": args.validation_end_seconds,
        },
        "contextDefinition": {
            "melodyRadiusSeconds": args.melody_radius,
            "harmonyChangeThreshold": args.harmony_change_threshold,
            "keys": list(CONTEXT_KEYS),
        },
        "selectionMode": "fixed-evaluation" if args.evaluate_only else "coordinate-search",
        "control": {
            "selection": compact_metrics(control_metrics),
            "validationBaseline": compact_metrics(validation_baseline),
            "fullBaseline": compact_metrics(development["baseline"]),
        },
        "winner": {
            "contextActionMap": action_map,
            "selectionMetrics": compact_metrics(best_training_metrics),
            "selectionQualityDeltaVsControl": round(
                best_quality - control_quality, 6
            ),
            "chronologicalValidationMetrics": compact_metrics(validation_metrics),
            "chronologicalValidationQualityDeltaVsBaseline": round(
                scalar_quality(validation_metrics) - scalar_quality(validation_baseline),
                6,
            ),
            "fullMetrics": compact_metrics(full_metrics),
            "cyclicDiagnostics": best_output.get("pianoArrangement", {}).get(
                "sourceSupportedCyclicHarmony"
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
                "contextActionMap": action_map,
                "selectionQualityDeltaVsControl": round(
                    best_quality - control_quality, 6
                ),
                "chronologicalValidationQualityDeltaVsBaseline": round(
                    scalar_quality(validation_metrics)
                    - scalar_quality(validation_baseline),
                    6,
                ),
                "selection": compact_metrics(best_training_metrics),
                "chronologicalValidation": compact_metrics(validation_metrics),
                "transferQualityDeltas": {
                    song_id: row["qualityDelta"] for song_id, row in transfer.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
