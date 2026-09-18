"""Test an onset ranker trained without the development song.

The frozen model ranks source attacks in four-second windows.  Voice and bass
events are preserved; only harmony-source attacks outside the selected quota
are removed before the ordinary piano reducer runs.  A first-half grid chooses
the keep ratio and compact window, while the second half remains unseen until
the policy is frozen.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from collections import defaultdict
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
    parse_float_grid,
    parse_int_grid,
    scalar_quality,
)


def selected_onset_keys(
    payload: dict[str, Any], model: dict[str, Any], keep_ratio: float
) -> tuple[set[int], dict[str, Any]]:
    from piano_arranger_adapter import normalize_source_notes, selection_scores

    notes = normalize_source_notes(payload.get("notes", []))
    scores = selection_scores(notes, {"selectionModel": model})
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        grouped[int(round(float(note["time"]) * 1000.0))].append(index)
    onset_scores = {
        key: max(float(scores[index]) for index in indices)
        for key, indices in grouped.items()
    }
    windows: dict[int, list[int]] = defaultdict(list)
    for key in sorted(onset_scores):
        windows[int((key / 1000.0) // 4.0)].append(key)
    selected: set[int] = set()
    for keys in windows.values():
        quota = min(len(keys), max(1, int(round(len(keys) * keep_ratio))))
        selected.update(
            sorted(keys, key=lambda key: (onset_scores[key], -key), reverse=True)[
                :quota
            ]
        )
    return selected, {
        "rawOnsets": len(grouped),
        "selectedOnsets": len(selected),
        "selectedOnsetRatio": round(len(selected) / max(1, len(grouped)), 6),
        "windows": len(windows),
    }


def filtered_harmony_payload(
    payload: dict[str, Any], model: dict[str, Any], keep_ratio: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    from piano_arranger_adapter import instrument_family

    selected, diagnostics = selected_onset_keys(payload, model, keep_ratio)
    output = copy.deepcopy(payload)
    kept: list[dict[str, Any]] = []
    removed = 0
    for note in output.get("notes", []):
        try:
            key = int(round(float(note.get("time", note.get("startTime"))) * 1000.0))
        except (TypeError, ValueError):
            continue
        family = instrument_family(str(note.get("instrument") or ""))
        if family in {"voice", "bass"} or key in selected:
            kept.append(note)
        else:
            removed += 1
    output["notes"] = kept
    diagnostics.update(
        {
            "sourceNotes": len(payload.get("notes", [])),
            "keptNotes": len(kept),
            "removedHarmonyNotes": removed,
        }
    )
    return output, diagnostics


def profile(window: float, maximum_pitch_classes: int) -> dict[str, Any]:
    return {
        "schema": "polymath-piano-arranger-profile-v1",
        "id": "unseen-onset-ranked-default-pipeline-v1",
        "decoder": {
            "defaultPipeline": True,
            "defaultFullMix": {
                "harmonyWindowSeconds": window,
                "maximumHarmonyPitchClasses": maximum_pitch_classes,
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
            "method": "frozen leave-one-song-out onset ranking",
            "commercialUseAllowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--onset-profile", type=Path, required=True)
    parser.add_argument("--development-song", default="kiss-me")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-start-seconds", type=float, default=0.0)
    parser.add_argument("--training-end-seconds", type=float, required=True)
    parser.add_argument("--validation-start-seconds", type=float, required=True)
    parser.add_argument("--validation-end-seconds", type=float)
    parser.add_argument(
        "--keep-ratios",
        type=parse_float_grid,
        default=(0.48, 0.52, 0.56, 0.58, 0.60, 0.64, 0.68),
    )
    parser.add_argument(
        "--harmony-windows",
        type=parse_float_grid,
        default=(0.10, 0.12, 0.15, 0.18, 0.22, 0.30),
    )
    parser.add_argument(
        "--maximum-pitch-classes", type=parse_int_grid, default=(2, 3, 4)
    )
    args = parser.parse_args()

    validation_end = (
        float("inf")
        if args.validation_end_seconds is None
        else args.validation_end_seconds
    )
    if (
        args.training_start_seconds < validation_end
        and args.validation_start_seconds < args.training_end_seconds
    ):
        raise ValueError("training and validation windows must not overlap")
    if any(not 0.10 <= ratio <= 0.95 for ratio in args.keep_ratios):
        raise ValueError("keep ratios must be between 0.10 and 0.95")
    if any(not 0.06 <= window <= 0.40 for window in args.harmony_windows):
        raise ValueError("harmony windows must be between 0.06 and 0.40 seconds")
    if any(not 1 <= value <= 4 for value in args.maximum_pitch_classes):
        raise ValueError("maximum pitch classes must be between 1 and 4")

    rows = load_rows(args.manifest.resolve())
    selected_rows = [row for row in rows if str(row["id"]) == args.development_song]
    if len(selected_rows) != 1:
        raise ValueError("development song must occur exactly once")
    onset_profile = load_json(args.onset_profile.resolve())
    onset_model = onset_profile.get("selectionModel") or {}
    excluded = set(
        str(value)
        for value in (onset_profile.get("training") or {}).get(
            "excludedSongIds", []
        )
    )
    if args.development_song not in excluded:
        raise ValueError(
            "The onset profile must explicitly exclude the development song"
        )

    repo_root = Path(__file__).resolve().parents[2]
    server_path = str(repo_root / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    from piano_arranger import arrange_payload

    row = selected_rows[0]
    alignment = load_json(Path(str(row["alignment"])).resolve())
    reference = prepare_reference_notes(
        row, load_json(Path(str(row["reference"])).resolve()), alignment
    )
    source = load_json(Path(str(row["source"])).resolve())
    baseline_notes = clip_candidate(
        load_json(Path(str(row["baseline"])).resolve()), row, alignment
    )
    training_reference = notes_in_time_window(
        reference,
        start_seconds=args.training_start_seconds,
        end_seconds=args.training_end_seconds,
    )
    validation_reference = notes_in_time_window(
        reference,
        start_seconds=args.validation_start_seconds,
        end_seconds=args.validation_end_seconds,
    )
    training_baseline = measure(
        training_reference,
        notes_in_time_window(
            baseline_notes,
            start_seconds=args.training_start_seconds,
            end_seconds=args.training_end_seconds,
        ),
    )
    validation_baseline = measure(
        validation_reference,
        notes_in_time_window(
            baseline_notes,
            start_seconds=args.validation_start_seconds,
            end_seconds=args.validation_end_seconds,
        ),
    )

    filtered_cache: dict[float, tuple[dict[str, Any], dict[str, Any]]] = {}
    trials: list[dict[str, Any]] = []
    outputs: dict[tuple[float, float, int], dict[str, Any]] = {}
    for keep_ratio, window, pitch_classes in itertools.product(
        args.keep_ratios, args.harmony_windows, args.maximum_pitch_classes
    ):
        if keep_ratio not in filtered_cache:
            filtered_cache[keep_ratio] = filtered_harmony_payload(
                source, onset_model, keep_ratio
            )
        filtered, selection_diagnostics = filtered_cache[keep_ratio]
        candidate_payload = arrange_payload(
            copy.deepcopy(filtered),
            "full",
            style_profile=profile(window, pitch_classes),
        )
        candidate_notes = clip_candidate(candidate_payload, row, alignment)
        metrics = measure(
            training_reference,
            notes_in_time_window(
                candidate_notes,
                start_seconds=args.training_start_seconds,
                end_seconds=args.training_end_seconds,
            ),
        )
        key = (keep_ratio, window, pitch_classes)
        outputs[key] = candidate_payload
        trials.append(
            {
                "keepRatio": keep_ratio,
                "harmonyWindowSeconds": window,
                "maximumHarmonyPitchClasses": pitch_classes,
                "selectionQuality": round(scalar_quality(metrics), 6),
                "selectionQualityDeltaVsBaseline": round(
                    scalar_quality(metrics) - scalar_quality(training_baseline), 6
                ),
                "selectionMetrics": compact_metrics(metrics),
                "onsetSelection": selection_diagnostics,
            }
        )

    trials.sort(
        key=lambda item: (
            float(item["selectionQuality"]),
            -abs(float(item["keepRatio"]) - float(onset_profile.get("recommendedTargetOnsetKeepRatio", 0.58))),
            -float(item["harmonyWindowSeconds"]),
        ),
        reverse=True,
    )
    winner = trials[0]
    winner_key = (
        float(winner["keepRatio"]),
        float(winner["harmonyWindowSeconds"]),
        int(winner["maximumHarmonyPitchClasses"]),
    )
    candidate_payload = outputs[winner_key]
    candidate_notes = clip_candidate(candidate_payload, row, alignment)
    validation_metrics = measure(
        validation_reference,
        notes_in_time_window(
            candidate_notes,
            start_seconds=args.validation_start_seconds,
            end_seconds=args.validation_end_seconds,
        ),
    )
    full_metrics = measure(reference, candidate_notes)

    output_dir = args.output_dir.resolve()
    frozen_policy = {
        "schema": "polymath-onset-ranked-harmony-policy-v1",
        "id": "unseen-kiss-onset-ranked-harmony-v1",
        "onsetProfile": str(args.onset_profile.resolve()),
        "onsetProfileSha256": onset_profile.get("profileSha256"),
        "keepRatio": winner_key[0],
        "harmonyWindowSeconds": winner_key[1],
        "maximumHarmonyPitchClasses": winner_key[2],
        "training": {
            "status": "RESEARCH_ONLY",
            "selectionSong": args.development_song,
            "selectionWindowSeconds": [
                args.training_start_seconds,
                args.training_end_seconds,
            ],
            "commercialUseAllowed": False,
        },
    }
    atomic_json(output_dir / "frozen-research-policy.json", frozen_policy)
    atomic_json(output_dir / f"{args.development_song}-candidate.json", candidate_payload)
    report = {
        "schema": "polymath-unseen-onset-ranked-harmony-search-v1",
        "evidenceBoundary": (
            "The onset model was fitted without the development song. Only the first "
            "time window selected reduction hyperparameters; the second was opened after freezing."
        ),
        "developmentSong": args.development_song,
        "selectionWindow": [args.training_start_seconds, args.training_end_seconds],
        "validationWindow": [args.validation_start_seconds, args.validation_end_seconds],
        "baseline": {
            "selection": compact_metrics(training_baseline),
            "validation": compact_metrics(validation_baseline),
            "full": compact_metrics(measure(reference, baseline_notes)),
        },
        "winner": {
            **winner,
            "validationMetrics": compact_metrics(validation_metrics),
            "validationQualityDeltaVsBaseline": round(
                scalar_quality(validation_metrics) - scalar_quality(validation_baseline), 6
            ),
            "fullMetrics": compact_metrics(full_metrics),
        },
        "leaderboard": trials,
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "winner": frozen_policy,
                "selectionQualityDeltaVsBaseline": winner[
                    "selectionQualityDeltaVsBaseline"
                ],
                "validationQualityDeltaVsBaseline": report["winner"][
                    "validationQualityDeltaVsBaseline"
                ],
                "full": compact_metrics(full_metrics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
