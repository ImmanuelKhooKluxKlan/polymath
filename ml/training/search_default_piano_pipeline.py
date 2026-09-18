"""Search the legacy full-mix piano decoder without leaking a holdout answer.

The raw transcription and alignment are frozen.  Each trial changes only the
time window used to compact harmony and the maximum simultaneous pitch classes.
Training songs select the policy; the named holdout is evaluated only after the
winner has been frozen.  Nothing in this script edits a deployed profile.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ml.training.analyze_pianist_gesture_patterns import group_onsets
from ml.training.evaluate_piano_arranger import (
    evaluate,
    load_json,
    normalize_notes,
    notes_inside_ranges,
    prepare_reference_notes,
    trusted_source_ranges,
)


def parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("grid needs finite comma-separated values")
    return values


def parse_int_grid(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not values:
        raise argparse.ArgumentTypeError("grid needs comma-separated integers")
    return values


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def distribution_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def normalized_counter(values: list[str]) -> dict[str, float]:
    counts = Counter(values)
    total = max(1, sum(counts.values()))
    return {key: count / total for key, count in sorted(counts.items())}


def gesture_structure(notes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_onsets(notes, 0.035)
    sizes: list[str] = []
    hands: list[str] = []
    coherent_velocity = 0
    spreads: list[float] = []
    for group in groups:
        midis = {int(note["midi"]) for note in group}
        size = len(midis)
        sizes.append(str(size) if size < 6 else "6+")
        has_left = any(midi < 60 for midi in midis)
        has_right = any(midi >= 60 for midi in midis)
        hands.append("both" if has_left and has_right else "left-only" if has_left else "right-only")
        velocities = [float(note.get("velocity", 0.72)) for note in group]
        spread = max(velocities) - min(velocities)
        spreads.append(spread)
        coherent_velocity += int(size > 1 and spread <= 0.01)
    multi = sum(int(len({int(note["midi"]) for note in group}) > 1) for group in groups)
    return {
        "gestures": len(groups),
        "chordSizes": normalized_counter(sizes),
        "handOccupancy": normalized_counter(hands),
        "coherentMultiNoteVelocityShare": round(coherent_velocity / max(1, multi), 6),
        "meanWithinGestureVelocitySpread": round(sum(spreads) / max(1, len(spreads)), 6),
    }


def clip_candidate(
    payload: dict[str, Any], row: dict[str, Any], alignment: dict[str, Any]
) -> list[dict[str, Any]]:
    notes = notes_inside_ranges(normalize_notes(payload), trusted_source_ranges(alignment))
    candidate_end = row.get("candidateEndSeconds")
    if candidate_end is not None:
        notes = [note for note in notes if float(note["time"]) < float(candidate_end)]
    return notes


def notes_in_time_window(
    notes: list[dict[str, Any]],
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return notes whose onsets fall inside one fixed source-clock window.

    Model-selection scripts use this after reference alignment and candidate
    clipping.  Keeping the split in the common source clock prevents a target
    timestamp from leaking into a supposedly unseen validation segment.
    """

    start = float(start_seconds)
    end = None if end_seconds is None else float(end_seconds)
    if not math.isfinite(start) or start < 0:
        raise ValueError("start_seconds must be a finite non-negative number")
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("end_seconds must be finite and greater than start_seconds")
    return [
        note
        for note in notes
        if float(note["time"]) >= start
        and (end is None or float(note["time"]) < end)
    ]


def measure(reference: list[dict[str, Any]], observed: list[dict[str, Any]]) -> dict[str, Any]:
    note_metrics = evaluate(reference, observed)
    reference_structure = gesture_structure(reference)
    observed_structure = gesture_structure(observed)
    return {
        "notes": note_metrics,
        "structure": observed_structure,
        "chordSizeDistance": round(
            distribution_distance(reference_structure["chordSizes"], observed_structure["chordSizes"]), 6
        ),
        "handOccupancyDistance": round(
            distribution_distance(reference_structure["handOccupancy"], observed_structure["handOccupancy"]), 6
        ),
        "gestureCountRatio": round(observed_structure["gestures"] / max(1, reference_structure["gestures"]), 6),
    }


def scalar_quality(metrics: dict[str, Any]) -> float:
    notes = metrics["notes"]
    note_ratio_error = abs(math.log(max(1e-6, float(notes["noteCountRatio"]))))
    gesture_ratio_error = abs(math.log(max(1e-6, float(metrics["gestureCountRatio"]))))
    velocity_error = float(notes["velocity"]["meanAbsoluteError"] or 1.0)
    return (
        0.24 * float(notes["pitchClassOnset250ms"]["f1"])
        + 0.18 * float(notes["pitchClassOnset250ms"]["recall"])
        + 0.18 * float(notes["exactPitchOnset250ms"]["f1"])
        + 0.10 * float(notes["exactPitchOnset100ms"]["f1"])
        - 0.10 * float(metrics["chordSizeDistance"])
        - 0.08 * float(metrics["handOccupancyDistance"])
        - 0.055 * note_ratio_error
        - 0.035 * gesture_ratio_error
        - 0.02 * velocity_error
    )


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    notes = metrics["notes"]
    return {
        "quality": round(scalar_quality(metrics), 6),
        "observedNotes": notes["observedNotes"],
        "noteCountRatio": notes["noteCountRatio"],
        "gestureCountRatio": metrics["gestureCountRatio"],
        "exactF1_100ms": notes["exactPitchOnset100ms"]["f1"],
        "exactF1_250ms": notes["exactPitchOnset250ms"]["f1"],
        "pitchClassF1_250ms": notes["pitchClassOnset250ms"]["f1"],
        "pitchClassRecall_250ms": notes["pitchClassOnset250ms"]["recall"],
        "velocityMeanAbsoluteError": notes["velocity"]["meanAbsoluteError"],
        "velocityMeanBias": notes["velocity"]["meanBias"],
        "chordSizeDistance": metrics["chordSizeDistance"],
        "handOccupancyDistance": metrics["handOccupancyDistance"],
        "rapidRetriggersUnder100ms": notes["rapidRetriggersUnder100ms"],
        "structure": metrics["structure"],
    }


def make_profile(window: float, maximum_pitch_classes: int) -> dict[str, Any]:
    return {
        "schema": "polymath-piano-arranger-profile-v1",
        "id": f"default-structure-w{window:.3f}-p{maximum_pitch_classes}",
        "decoder": {
            "defaultPipeline": True,
            "defaultFullMix": {
                "harmonyWindowSeconds": window,
                "maximumHarmonyPitchClasses": maximum_pitch_classes,
            },
        },
    }


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)
    rows = manifest.get("songs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty songs list")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not song_id for song_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("every song id must be non-empty and unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-song", required=True)
    parser.add_argument("--harmony-windows", type=parse_float_grid, default=(0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30))
    parser.add_argument("--maximum-pitch-classes", type=parse_int_grid, default=(2, 3, 4))
    args = parser.parse_args()

    rows = load_rows(args.manifest.resolve())
    holdouts = [row for row in rows if str(row["id"]) == args.holdout_song]
    training_rows = [row for row in rows if str(row["id"]) != args.holdout_song]
    if len(holdouts) != 1 or not training_rows:
        raise ValueError("holdout song must occur exactly once and leave training songs")

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
        source = load_json(Path(str(row["source"])).resolve())
        baseline_payload = load_json(Path(str(row["baseline"])).resolve())
        baseline = measure(reference, clip_candidate(baseline_payload, row, alignment))
        cache[str(row["id"])] = {
            "row": row,
            "alignment": alignment,
            "reference": reference,
            "source": source,
            "baseline": baseline,
        }

    trials: list[dict[str, Any]] = []
    trial_outputs: dict[tuple[float, int], dict[str, dict[str, Any]]] = {}
    for window, pitch_classes in itertools.product(args.harmony_windows, args.maximum_pitch_classes):
        if not 0.06 <= window <= 0.40 or not 1 <= pitch_classes <= 4:
            raise ValueError("grid is outside arranger safety bounds")
        profile = make_profile(window, pitch_classes)
        per_song: dict[str, Any] = {}
        outputs: dict[str, dict[str, Any]] = {}
        deltas: list[float] = []
        for row in training_rows:
            song_id = str(row["id"])
            item = cache[song_id]
            arranged = arrange_payload(copy.deepcopy(item["source"]), "full", style_profile=copy.deepcopy(profile))
            metrics = measure(
                item["reference"],
                clip_candidate(arranged, row, item["alignment"]),
            )
            baseline_quality = scalar_quality(item["baseline"])
            delta = scalar_quality(metrics) - baseline_quality
            deltas.append(delta)
            per_song[song_id] = {
                "baseline": compact_metrics(item["baseline"]),
                "candidate": compact_metrics(metrics),
                "qualityDelta": round(delta, 6),
            }
            outputs[song_id] = arranged
        trials.append(
            {
                "harmonyWindowSeconds": window,
                "maximumHarmonyPitchClasses": pitch_classes,
                "trainingMeanQualityDelta": round(sum(deltas) / len(deltas), 6),
                "trainingWorstSongQualityDelta": round(min(deltas), 6),
                "trainingImprovedSongs": sum(delta > 0 for delta in deltas),
                "songs": per_song,
            }
        )
        trial_outputs[(window, pitch_classes)] = outputs

    trials.sort(
        key=lambda row: (
            int(row["trainingImprovedSongs"]),
            float(row["trainingWorstSongQualityDelta"]),
            float(row["trainingMeanQualityDelta"]),
        ),
        reverse=True,
    )
    winner = trials[0]
    winner_key = (
        float(winner["harmonyWindowSeconds"]),
        int(winner["maximumHarmonyPitchClasses"]),
    )
    frozen_profile = make_profile(*winner_key)

    holdout = holdouts[0]
    holdout_id = str(holdout["id"])
    held = cache[holdout_id]
    holdout_output = arrange_payload(
        copy.deepcopy(held["source"]), "full", style_profile=copy.deepcopy(frozen_profile)
    )
    holdout_metrics = measure(
        held["reference"],
        clip_candidate(holdout_output, holdout, held["alignment"]),
    )
    holdout_delta = scalar_quality(holdout_metrics) - scalar_quality(held["baseline"])

    output_dir = args.output_dir.resolve()
    atomic_json(output_dir / "frozen-profile.json", frozen_profile)
    for song_id, payload in trial_outputs[winner_key].items():
        atomic_json(output_dir / "training-candidates" / f"{song_id}.json", payload)
    atomic_json(output_dir / f"holdout-{holdout_id}.json", holdout_output)
    report = {
        "schema": "polymath-default-piano-pipeline-search-v1",
        "evidenceBoundary": "The holdout target did not participate in policy selection.",
        "manifest": str(args.manifest.resolve()),
        "trainingSongs": [str(row["id"]) for row in training_rows],
        "holdoutSong": holdout_id,
        "winner": winner,
        "heldoutResult": {
            "baseline": compact_metrics(held["baseline"]),
            "candidate": compact_metrics(holdout_metrics),
            "qualityDelta": round(holdout_delta, 6),
        },
        "leaderboard": trials,
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps({"winner": winner_key, "heldoutQualityDelta": round(holdout_delta, 6), "report": str(output_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
