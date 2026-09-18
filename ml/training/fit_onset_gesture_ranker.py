"""Fit a song-independent ranker for pianist-like accompaniment onsets.

The ordinary note selector answers "which detected note should survive?".  A
pianist first answers a different question: "when should the left hand make a
gesture?"  This trainer groups simultaneous source events, learns that timing
decision from trusted source-to-piano alignments, and validates by leaving a
whole song out.  Pitch choice remains the responsibility of the existing
structured chord ranker.

The output is a tiny JSON logistic model used at inference without NumPy.
Training requires NumPy but never changes MuScriptor weights.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    ONSET_GESTURE_SELECTION_FEATURE_NAMES,
    ONSET_GESTURE_SEQUENCE_FEATURE_NAMES,
    instrument_family,
    normalize_source_notes,
    selection_feature_rows,
)

try:
    from .train_piano_arranger_adapter import train_logistic_model
except ImportError:  # pragma: no cover - direct CLI execution
    from train_piano_arranger_adapter import train_logistic_model  # type: ignore


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def map_time(value: float, anchors: list[dict[str, Any]]) -> float:
    """Map target time to source time across either supported report schema."""

    if len(anchors) < 2:
        return value

    def reference(anchor: dict[str, Any]) -> float:
        return float(anchor.get("referenceTime", anchor.get("targetTime", 0.0)))

    def observed(anchor: dict[str, Any]) -> float:
        return float(anchor.get("observedTime", anchor.get("sourceTime", 0.0)))

    ordered = sorted(anchors, key=reference)
    positions = [reference(anchor) for anchor in ordered]
    position = bisect.bisect_right(positions, value)
    if position <= 0:
        left, right = ordered[0], ordered[1]
    elif position >= len(ordered):
        left, right = ordered[-2], ordered[-1]
    else:
        left, right = ordered[position - 1], ordered[position]
    left_reference = reference(left)
    right_reference = reference(right)
    scale = (observed(right) - observed(left)) / max(
        1e-9, right_reference - left_reference
    )
    return observed(left) + (value - left_reference) * scale


def source_window_bounds(window: dict[str, Any]) -> tuple[float, float] | None:
    start = window.get("sourceStartSeconds", window.get("sourceStart"))
    end = window.get("sourceEndSeconds", window.get("sourceEnd"))
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    return (start_value, end_value) if end_value > start_value else None


def target_window_bounds(window: dict[str, Any]) -> tuple[float, float] | None:
    start = window.get("targetStartSeconds", window.get("referenceStart"))
    end = window.get("targetEndSeconds", window.get("referenceEnd"))
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    return (start_value, end_value) if end_value > start_value else None


def trusted_windows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        window
        for window in report.get("qualityWindows") or []
        if isinstance(window, dict)
        and window.get("status") in {"trusted", "accepted-manually"}
        and source_window_bounds(window) is not None
    ]


def in_source_windows(time: float, windows: list[dict[str, Any]]) -> bool:
    return any(
        start <= time < end
        for window in windows
        if (bounds := source_window_bounds(window)) is not None
        for start, end in [bounds]
    )


def target_window(time: float, windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for window in windows:
        bounds = target_window_bounds(window)
        if bounds and bounds[0] <= time < bounds[1]:
            return window
    return None


def aligned_target_left_onsets(
    target: dict[str, Any],
    report: dict[str, Any],
    reference_hand_split: int,
) -> list[float]:
    """Return unique trusted left-hand target onsets on the source timeline."""

    windows = trusted_windows(report)
    anchors = report.get("anchors") or []
    direct_source_timeline = any(
        note.get("trainingEligible") is not None
        and note.get("originalTime") is not None
        for note in target.get("notes", [])
        if isinstance(note, dict)
    )
    mapped: list[float] = []
    for note in target.get("notes", []):
        try:
            midi = int(round(float(note["midi"])))
            time = float(note.get("time", note.get("startTime", note.get("start"))))
        except (KeyError, TypeError, ValueError):
            continue
        if midi >= reference_hand_split:
            continue
        if direct_source_timeline:
            if not bool(note.get("trainingEligible")) or not in_source_windows(time, windows):
                continue
            mapped.append(time)
        else:
            window = target_window(time, windows)
            if not window:
                continue
            mapped_time = map_time(time, anchors)
            if in_source_windows(mapped_time, windows):
                mapped.append(mapped_time)

    # Notes struck within 35 ms form one human gesture.  The first note is the
    # canonical onset, matching the analyser's grouping policy.
    onsets: list[float] = []
    for time in sorted(mapped):
        if not onsets or time - onsets[-1] > 0.035:
            onsets.append(time)
    return onsets


def onset_source_indices(notes: list[dict[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        if instrument_family(str(note.get("instrument") or "")) == "voice":
            continue
        groups[int(round(float(note["time"]) * 1000.0))].append(index)
    return dict(groups)


def one_to_one_positive_keys(
    source_keys: list[int], target_times: list[float], tolerance: float
) -> set[int]:
    """Match each target gesture to at most one detected source onset."""

    source_times = [key / 1000.0 for key in source_keys]
    possibilities: list[tuple[float, int, int]] = []
    for target_index, time in enumerate(target_times):
        position = bisect.bisect_left(source_times, time)
        left = max(0, position - 4)
        right = min(len(source_times), position + 5)
        for source_index in range(left, right):
            distance = abs(source_times[source_index] - time)
            if distance <= tolerance:
                possibilities.append((distance, target_index, source_index))
    used_targets: set[int] = set()
    used_sources: set[int] = set()
    for _distance, target_index, source_index in sorted(possibilities):
        if target_index in used_targets or source_index in used_sources:
            continue
        used_targets.add(target_index)
        used_sources.add(source_index)
    return {source_keys[index] for index in used_sources}


def match_count(
    observed: list[float], reference: list[float], tolerance: float
) -> int:
    possibilities: list[tuple[float, int, int]] = []
    for observed_index, time in enumerate(observed):
        position = bisect.bisect_left(reference, time)
        for reference_index in (position - 1, position):
            if 0 <= reference_index < len(reference):
                distance = abs(time - reference[reference_index])
                if distance <= tolerance:
                    possibilities.append((distance, observed_index, reference_index))
    used_observed: set[int] = set()
    used_reference: set[int] = set()
    for _distance, observed_index, reference_index in sorted(possibilities):
        if observed_index in used_observed or reference_index in used_reference:
            continue
        used_observed.add(observed_index)
        used_reference.add(reference_index)
    return len(used_observed)


def onset_metrics(observed: list[float], reference: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observedOnsets": len(observed),
        "referenceOnsets": len(reference),
    }
    for milliseconds in (50, 100, 250):
        matches = match_count(observed, reference, milliseconds / 1000.0)
        precision = matches / max(1, len(observed))
        recall = matches / max(1, len(reference))
        result[f"onset{milliseconds}ms"] = {
            "matches": matches,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 6),
        }
    return result


def probabilities(
    features: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    logits = ((features - means) / scales) @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def select_by_window(
    keys: list[int], scores: np.ndarray, keep_ratio: float, window_seconds: float = 4.0
) -> list[float]:
    windows: dict[int, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        windows[int((key / 1000.0) // window_seconds)].append(index)
    selected: list[int] = []
    for indices in windows.values():
        quota = min(len(indices), max(1, int(round(len(indices) * keep_ratio))))
        selected.extend(
            sorted(indices, key=lambda index: (float(scores[index]), -keys[index]), reverse=True)[:quota]
        )
    return sorted(keys[index] / 1000.0 for index in selected)


def build_song_examples(
    pair: dict[str, Any],
    reference_hand_split: int,
    tolerance: float,
    feature_names: tuple[str, ...] = ONSET_GESTURE_SELECTION_FEATURE_NAMES,
) -> dict[str, Any]:
    source_path = Path(pair["source"]).resolve()
    target_path = Path(pair["target"]).resolve()
    alignment_path = Path(pair["alignmentReport"]).resolve()
    source = load_json(source_path)
    target = load_json(target_path)
    report = load_json(alignment_path)
    windows = trusted_windows(report)
    if not windows:
        raise ValueError(f"{pair['id']} has no trusted source windows")
    notes = normalize_source_notes(source.get("notes", []))
    all_rows = selection_feature_rows(notes, feature_names)
    groups = onset_source_indices(notes)
    keys = sorted(
        key
        for key in groups
        if in_source_windows(key / 1000.0, windows)
    )
    target_onsets = aligned_target_left_onsets(target, report, reference_hand_split)
    positive_keys = one_to_one_positive_keys(keys, target_onsets, tolerance)
    features = np.asarray([all_rows[groups[key][0]] for key in keys], dtype=np.float64)
    labels = np.asarray([1.0 if key in positive_keys else 0.0 for key in keys])
    song_weight = max(0.01, float(pair.get("weight", 1.0)))
    sample_weights = np.asarray(
        [song_weight * (1.0 if label else 0.45) for label in labels],
        dtype=np.float64,
    )
    return {
        "id": str(pair["id"]),
        "features": features,
        "labels": labels,
        "sampleWeights": sample_weights,
        "keys": keys,
        "targetOnsets": target_onsets,
        "positiveOnsets": int(labels.sum()),
        "rawOnsets": len(keys),
        "learnableTargetShare": round(float(labels.sum()) / max(1, len(target_onsets)), 6),
        "source": str(source_path),
        "target": str(target_path),
        "alignment": str(alignment_path),
        "weight": song_weight,
    }


def train_model(songs: list[dict[str, Any]], seed: int, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = np.vstack([song["features"] for song in songs])
    labels = np.concatenate([song["labels"] for song in songs])
    sample_weights = np.concatenate([song["sampleWeights"] for song in songs])
    training_mask = np.ones(len(labels), dtype=bool)
    weights, means, scales, history = train_logistic_model(
        features,
        labels,
        sample_weights,
        training_mask,
        seed=seed,
        iterations=iterations,
        learning_rate=0.016,
        l2=0.018,
    )
    return weights, means, scales, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--match-tolerance", type=float, default=0.12)
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=0x5049414E)
    parser.add_argument(
        "--feature-contract",
        choices=("v1", "sequence-v2"),
        default="v1",
        help="Use the frozen v1 features or the chord-run/beat-phase v2 contract.",
    )
    parser.add_argument(
        "--exclude-song",
        action="append",
        default=[],
        help="Exclude a complete song from fitting and model selection.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    feature_names = (
        ONSET_GESTURE_SEQUENCE_FEATURE_NAMES
        if args.feature_contract == "sequence-v2"
        else ONSET_GESTURE_SELECTION_FEATURE_NAMES
    )
    all_songs = [
        build_song_examples(
            pair,
            args.reference_hand_split,
            args.match_tolerance,
            feature_names,
        )
        for pair in manifest.get("pairs", [])
    ]
    excluded_songs = {str(value) for value in args.exclude_song if str(value)}
    songs = [song for song in all_songs if song["id"] not in excluded_songs]
    if len(songs) < 3:
        raise ValueError("At least three songs are required for song-level validation.")

    folds: dict[str, Any] = {}
    for held_out in songs:
        training = [song for song in songs if song is not held_out]
        weights, means, scales, _history = train_model(
            training, args.seed, args.iterations
        )
        scores = probabilities(held_out["features"], weights, means, scales)
        numerator = sum(song["positiveOnsets"] * song["weight"] for song in training)
        denominator = sum(song["rawOnsets"] * song["weight"] for song in training)
        keep_ratio = min(0.95, max(0.10, numerator / max(1e-9, denominator)))
        selected = select_by_window(held_out["keys"], scores, keep_ratio)
        folds[held_out["id"]] = {
            "trainingSongs": [song["id"] for song in training],
            "keepRatioLearnedWithoutHeldOutSong": round(keep_ratio, 6),
            "metrics": onset_metrics(selected, held_out["targetOnsets"]),
            "data": {
                "rawOnsets": held_out["rawOnsets"],
                "positiveOnsetsAtTrainingTolerance": held_out["positiveOnsets"],
                "targetOnsets": len(held_out["targetOnsets"]),
                "learnableTargetShare": held_out["learnableTargetShare"],
            },
        }

    weights, means, scales, history = train_model(songs, args.seed, args.iterations)
    numerator = sum(song["positiveOnsets"] * song["weight"] for song in songs)
    denominator = sum(song["rawOnsets"] * song["weight"] for song in songs)
    keep_ratio = min(0.95, max(0.10, numerator / max(1e-9, denominator)))
    model = {
        "type": "standardized-logistic-onset-gesture-ranker-v1",
        "featureNames": list(feature_names),
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "threshold": 0.5,
    }
    profile = {
        "id": args.profile_id,
        "selectionModel": model,
        "recommendedTargetOnsetKeepRatio": round(keep_ratio, 6),
        "training": {
            "schema": "polymath-onset-gesture-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "policy": "whole-song leave-one-out; one-to-one target/source onset labels; trusted windows only",
            "featureContract": args.feature_contract,
            "matchToleranceSeconds": args.match_tolerance,
            "commercialUseAllowed": False,
            "excludedSongIds": sorted(excluded_songs),
            "warning": "Research calibration only. Rights and a sealed independent corpus are required before production promotion.",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    weighted_f1 = {}
    for tolerance in (50, 100, 250):
        values = [
            (folds[song["id"]]["metrics"][f"onset{tolerance}ms"]["f1"], song["weight"])
            for song in songs
        ]
        weighted_f1[f"onset{tolerance}msF1"] = round(
            sum(value * weight for value, weight in values)
            / max(1e-9, sum(weight for _value, weight in values)),
            6,
        )
    report = {
        "schema": "polymath-onset-gesture-training-report-v1",
        "profile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "recommendedTargetOnsetKeepRatio": round(keep_ratio, 6),
        "songs": [
            {
                key: value
                for key, value in song.items()
                if key not in {"features", "labels", "sampleWeights", "keys", "targetOnsets"}
            }
            for song in songs
        ],
        "excludedSongs": sorted(excluded_songs),
        "leaveOneSongOut": folds,
        "weightedLeaveOneSongOut": weighted_f1,
        "optimizationHistory": history,
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
