"""Fit a cross-song nonlinear selector for raw accompaniment support.

The selector answers a narrow question: does a detected low-register event
have a pitch class near a trusted pianist note?  It is used only as evidence
inside conservative gesture swaps/recovery; it does not generate timing or
copy reference coordinates at runtime.

Complete songs are held out for reporting.  Early stopping and threshold
selection use temporal blocks from the remaining training songs, never the
held-out song.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    instrument_family,
    normalize_source_notes,
    selection_feature_rows,
    selection_scores,
)

try:
    from .analyze_pianist_gesture_patterns import (
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )
    from .fit_left_hand_accompaniment import train_mlp_model
    from .train_piano_arranger_adapter import classification_metrics
except ImportError:  # pragma: no cover
    from analyze_pianist_gesture_patterns import (  # type: ignore
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )
    from fit_left_hand_accompaniment import train_mlp_model  # type: ignore
    from train_piano_arranger_adapter import classification_metrics  # type: ignore


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nearest_distance(times: list[float], time: float) -> float:
    position = bisect.bisect_left(times, time)
    candidates = [
        abs(times[index] - time)
        for index in (position - 1, position)
        if 0 <= index < len(times)
    ]
    return min(candidates, default=math.inf)


def aligned_reference_notes(
    row: dict[str, Any], alignment: dict[str, Any]
) -> list[dict[str, Any]]:
    reference_end = row.get("referenceEndSeconds")
    reference = normalize_notes(
        load_json(Path(row["reference"]).resolve()),
        transpose=int(row.get("referenceTransposeSemitones") or 0),
        hard_end=float(reference_end) if reference_end is not None else None,
    )
    if not bool(row.get("referenceAlreadyAligned")):
        reference = map_reference(reference, monotonic_anchors(alignment))
    ranges = trusted_source_ranges(alignment)
    candidate_end = row.get("candidateEndSeconds")
    hard_end = float(candidate_end) if candidate_end is not None else None
    return [
        note
        for note in reference
        if int(note["midi"]) < 60
        and inside_ranges(float(note["time"]), ranges)
        and (hard_end is None or float(note["time"]) < hard_end)
        and (note.get("trainingEligible") is not False)
    ]


def source_family_allowed(
    instrument: str, allowed_source_families: set[str] | None
) -> bool:
    family = instrument_family(str(instrument or ""))
    return family != "voice" and (
        allowed_source_families is None or family in allowed_source_families
    )


def build_song_examples(
    row: dict[str, Any],
    *,
    tolerance: float,
    maximum_source_midi: int,
    source_families: set[str] | None = None,
) -> dict[str, Any]:
    alignment_path = Path(row["alignment"]).resolve()
    source_path = Path(row["source"]).resolve()
    alignment = load_json(alignment_path)
    ranges = trusted_source_ranges(alignment)
    if not ranges:
        raise ValueError(f"{row['id']} has no trusted source ranges")
    source = load_json(source_path)
    notes = normalize_source_notes(source.get("notes") or [])
    feature_rows = selection_feature_rows(
        notes, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
    )
    reference = aligned_reference_notes(row, alignment)
    target_by_pc = {
        pitch_class: sorted(
            float(note["time"])
            for note in reference
            if int(note["midi"]) % 12 == pitch_class
        )
        for pitch_class in range(12)
    }
    eligible_indices: list[int] = []
    features: list[list[float]] = []
    labels: list[float] = []
    temporal_training: list[bool] = []
    example_times: list[float] = []
    for index, (note, features_row) in enumerate(zip(notes, feature_rows)):
        time = float(note["time"])
        if (
            int(note["midi"]) > maximum_source_midi
            or not source_family_allowed(
                str(note.get("instrument") or ""), source_families
            )
            or not inside_ranges(time, ranges)
        ):
            continue
        distance = nearest_distance(target_by_pc[int(note["midi"]) % 12], time)
        eligible_indices.append(index)
        features.append(features_row)
        labels.append(float(distance <= tolerance))
        temporal_training.append(int(time // 20.0) % 5 != 4)
        example_times.append(time)
    if sum(labels) < 10 or len(labels) - sum(labels) < 10:
        raise ValueError(f"{row['id']} lacks both selector classes")
    song_weight = max(0.01, float(row.get("weight", 1.0)))
    positive = max(1, int(sum(labels)))
    negative = max(1, len(labels) - positive)
    sample_weights = [
        song_weight * (0.5 / positive if label else 0.5 / negative) * len(labels)
        for label in labels
    ]
    return {
        "id": str(row["id"]),
        "notes": notes,
        "eligibleIndices": eligible_indices,
        "features": np.asarray(features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.float64),
        "weights": np.asarray(sample_weights, dtype=np.float64),
        "temporalTraining": np.asarray(temporal_training, dtype=bool),
        "times": np.asarray(example_times, dtype=np.float64),
        "source": str(source_path),
        "alignment": str(alignment_path),
        "reference": str(Path(row["reference"]).resolve()),
        "referenceLeftNotes": len(reference),
        "examples": len(labels),
        "positives": positive,
        "positiveShare": round(positive / len(labels), 6),
    }


def combine_songs(songs: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
        np.vstack([song["features"] for song in songs]),
        np.concatenate([song["labels"] for song in songs]),
        np.concatenate([song["weights"] for song in songs]),
        np.concatenate([song["temporalTraining"] for song in songs]),
    )


def best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    trials = [
        classification_metrics(
            labels[mask], probabilities[mask], weights[mask], float(threshold)
        )
        for threshold in np.linspace(0.15, 0.90, 151)
    ]
    return max(trials, key=lambda item: (item["f1"], item["precision"], item["recall"]))


def fit(
    songs: list[dict[str, Any]], *, iterations: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    features, labels, weights, training = combine_songs(songs)
    if training.all() or (~training).all():
        raise ValueError("Temporal training and validation blocks are both required")
    model, probabilities, history = train_mlp_model(
        features,
        labels,
        weights,
        training,
        seed=seed,
        iterations=iterations,
        hidden_size=24,
        learning_rate=0.0035,
        l2=0.0025,
    )
    threshold_metrics = best_threshold(
        labels, probabilities, weights, ~training
    )
    model["threshold"] = threshold_metrics["threshold"]
    return model, {
        "temporalTraining": classification_metrics(
            labels[training], probabilities[training], weights[training], model["threshold"]
        ),
        "temporalValidation": threshold_metrics,
        "history": history,
    }


def evaluate_song(song: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    all_probabilities = selection_scores(song["notes"], {"selectionModel": model})
    probabilities = np.asarray(
        [all_probabilities[index] for index in song["eligibleIndices"]], dtype=float
    )
    labels = song["labels"]
    weights = song["weights"]
    threshold = float(model.get("threshold", 0.5))
    metrics = classification_metrics(
        labels, probabilities, weights, threshold
    )
    metrics["probabilityQuantiles"] = {
        str(fraction): round(float(np.quantile(probabilities, fraction)), 6)
        for fraction in (0.1, 0.5, 0.9)
    }
    return metrics


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument(
        "--fold-profiles-root",
        default="",
        help=(
            "Optional directory for reusable whole-song holdout profiles. "
            "Each <song>/profile.json is trained without that complete song."
        ),
    )
    parser.add_argument("--exclude-song", action="append", default=[])
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument(
        "--source-families",
        default="",
        help=(
            "Optional comma-separated normalized instrument families. Empty "
            "uses every non-voice family."
        ),
    )
    parser.add_argument("--iterations", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=0x53555050)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    excluded = {str(value) for value in args.exclude_song if str(value)}
    source_families = {
        value.strip().lower()
        for value in str(args.source_families).split(",")
        if value.strip()
    } or None
    rows = [
        row
        for row in manifest.get("songs") or []
        if str(row.get("id")) not in excluded
    ]
    if len(rows) < 3:
        raise ValueError("At least three complete songs are required")
    songs = [
        build_song_examples(
            row,
            tolerance=max(0.01, float(args.tolerance_seconds)),
            maximum_source_midi=int(args.maximum_source_midi),
            source_families=source_families,
        )
        for row in rows
    ]
    folds: dict[str, Any] = {}
    fold_profiles_root = (
        Path(args.fold_profiles_root).resolve() if args.fold_profiles_root else None
    )
    for index, held_out in enumerate(songs):
        training = [song for song in songs if song is not held_out]
        model, internal = fit(
            training, iterations=max(200, args.iterations), seed=args.seed + index * 1009
        )
        fold_profile_path = None
        if fold_profiles_root is not None:
            fold_profile = {
                "schema": "polymath-raw-support-selector-profile-v1",
                "id": f"{args.profile_id}-without-{held_out['id']}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "selectionModel": model,
                "training": {
                    "manifest": str(manifest_path),
                    "excludedSongs": sorted(excluded | {held_out["id"]}),
                    "trainingSongs": [song["id"] for song in training],
                    "sourceFamilies": sorted(source_families or []),
                    "method": (
                        "whole-song holdout; temporal-block early stopping inside "
                        "the remaining training songs"
                    ),
                    "commercialUseAllowed": False,
                    "purpose": "private research evaluation",
                },
            }
            fold_profile["profileSha256"] = profile_hash(fold_profile)
            fold_profile_path = fold_profiles_root / held_out["id"] / "profile.json"
            fold_profile_path.parent.mkdir(parents=True, exist_ok=True)
            fold_profile_path.write_text(
                json.dumps(fold_profile, indent=2) + "\n", encoding="utf-8"
            )
        folds[held_out["id"]] = {
            "trainingSongs": [song["id"] for song in training],
            "internalTemporalValidation": internal["temporalValidation"],
            "heldOutSong": evaluate_song(held_out, model),
            "profile": str(fold_profile_path) if fold_profile_path is not None else None,
        }
    final_model, internal = fit(
        songs, iterations=max(200, args.iterations), seed=args.seed
    )
    profile = {
        "schema": "polymath-raw-support-selector-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "selectionModel": final_model,
        "training": {
            "manifest": str(manifest_path),
            "excludedSongs": sorted(excluded),
            "sourceFamilies": sorted(source_families or []),
            "method": "whole-song leave-one-out; temporal-block early stopping inside training folds",
            "commercialUseAllowed": False,
            "purpose": "private research evaluation",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    heldout_f1 = [float(value["heldOutSong"]["f1"]) for value in folds.values()]
    report = {
        "schema": "polymath-raw-support-selector-training-v1",
        "manifest": str(manifest_path),
        "excludedSongs": sorted(excluded),
        "sourceFamilies": sorted(source_families or []),
        "dataset": [
            {
                key: value
                for key, value in song.items()
                if key
                in {
                    "id",
                    "source",
                    "alignment",
                    "reference",
                    "referenceLeftNotes",
                    "examples",
                    "positives",
                    "positiveShare",
                }
            }
            for song in songs
        ],
        "leaveOneSongOut": folds,
        "heldOutSongF1": {
            "mean": round(sum(heldout_f1) / len(heldout_f1), 6),
            "minimum": round(min(heldout_f1), 6),
        },
        "finalInternal": internal,
        "outputProfile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "decision": "CANDIDATE_FOR_END_TO_END_UNSEEN_TEST",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputProfile": str(output_path),
                "report": str(report_path),
                "heldOutSongF1": report["heldOutSongF1"],
                "folds": {
                    song: value["heldOutSong"] for song, value in folds.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
