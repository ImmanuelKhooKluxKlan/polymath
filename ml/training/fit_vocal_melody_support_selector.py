"""Fit a cross-song selector for isolated-vocal melody evidence.

The selector answers a deliberately narrow question: does a note detected in
an isolated vocal stem have the pitch class of a nearby right-hand pianist
note?  It never generates timing and it never sees the eventual test song.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.training.evaluate_piano_arranger import (
    notes_inside_ranges,
    prepare_reference_notes,
    trusted_source_ranges,
)
from ml.training.fit_raw_support_selector import (
    evaluate_song,
    fit,
    load_json,
    nearest_distance,
    profile_hash,
)

import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    instrument_family,
    normalize_source_notes,
    selection_feature_rows,
)


def build_song_examples(
    row: dict[str, Any],
    *,
    tolerance: float,
    minimum_source_midi: int,
    maximum_source_midi: int,
) -> dict[str, Any]:
    alignment_path = Path(str(row["alignment"])).resolve()
    source_path = Path(str(row["vocalSource"])).resolve()
    reference_path = Path(str(row["reference"])).resolve()
    alignment = load_json(alignment_path)
    ranges = trusted_source_ranges(alignment)
    if ranges == []:
        raise ValueError(f"{row['id']} has no trusted source ranges")
    source = load_json(source_path)
    notes = normalize_source_notes(source.get("notes") or [])
    feature_rows = selection_feature_rows(
        notes, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
    )
    reference = prepare_reference_notes(
        row, load_json(reference_path), alignment
    )
    right_reference = [
        note
        for note in reference
        if (
            str(note.get("hand") or "").lower() == "right"
            or (not note.get("hand") and int(note["midi"]) >= 60)
        )
        and note.get("trainingEligible") is not False
    ]
    target_by_pc = {
        pitch_class: sorted(
            float(note["time"])
            for note in right_reference
            if int(note["midi"]) % 12 == pitch_class
        )
        for pitch_class in range(12)
    }
    candidate_end = row.get("candidateEndSeconds")
    eligible_indices: list[int] = []
    features: list[list[float]] = []
    labels: list[float] = []
    temporal_training: list[bool] = []
    example_times: list[float] = []
    for index, (note, feature_row) in enumerate(zip(notes, feature_rows)):
        time = float(note["time"])
        if (
            instrument_family(str(note.get("instrument") or "")) != "voice"
            or int(note["midi"]) < minimum_source_midi
            or int(note["midi"]) > maximum_source_midi
            or not notes_inside_ranges([note], ranges)
            or (
                candidate_end is not None
                and time >= float(candidate_end)
            )
        ):
            continue
        distance = nearest_distance(target_by_pc[int(note["midi"]) % 12], time)
        eligible_indices.append(index)
        features.append(feature_row)
        labels.append(float(distance <= tolerance))
        temporal_training.append(int(time // 20.0) % 5 != 4)
        example_times.append(time)
    if sum(labels) < 10 or len(labels) - sum(labels) < 10:
        raise ValueError(f"{row['id']} lacks both vocal selector classes")
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
        "reference": str(reference_path),
        "referenceRightNotes": len(right_reference),
        "examples": len(labels),
        "positives": positive,
        "positiveShare": round(positive / len(labels), 6),
    }


def write_profile(
    path: Path,
    *,
    profile_id: str,
    model: dict[str, Any],
    manifest_path: Path,
    excluded_songs: set[str],
    training_songs: list[str],
) -> dict[str, Any]:
    profile = {
        "schema": "polymath-vocal-melody-support-selector-profile-v1",
        "id": profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "selectionModel": copy.deepcopy(model),
        "training": {
            "manifest": str(manifest_path),
            "excludedSongs": sorted(excluded_songs),
            "trainingSongs": training_songs,
            "method": (
                "whole-song holdout; temporal-block early stopping inside "
                "the remaining training songs"
            ),
            "commercialUseAllowed": False,
            "purpose": "private research evaluation",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fold-profiles-root", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--exclude-song", action="append", default=[])
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--minimum-source-midi", type=int, default=36)
    parser.add_argument("--maximum-source-midi", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=0x564F4341)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    excluded = {str(value) for value in args.exclude_song if str(value)}
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
            minimum_source_midi=int(args.minimum_source_midi),
            maximum_source_midi=int(args.maximum_source_midi),
        )
        for row in rows
    ]
    folds: dict[str, Any] = {}
    fold_root = Path(args.fold_profiles_root).resolve()
    for index, held_out in enumerate(songs):
        training = [song for song in songs if song is not held_out]
        model, internal = fit(
            training,
            iterations=max(200, args.iterations),
            seed=args.seed + index * 1009,
        )
        fold_path = fold_root / held_out["id"] / "profile.json"
        fold_profile = write_profile(
            fold_path,
            profile_id=f"{args.profile_id}-without-{held_out['id']}",
            model=model,
            manifest_path=manifest_path,
            excluded_songs=excluded | {held_out["id"]},
            training_songs=[song["id"] for song in training],
        )
        folds[held_out["id"]] = {
            "trainingSongs": [song["id"] for song in training],
            "internalTemporalValidation": internal["temporalValidation"],
            "heldOutSong": evaluate_song(held_out, model),
            "profile": str(fold_path),
            "profileSha256": fold_profile["profileSha256"],
        }

    final_model, internal = fit(
        songs, iterations=max(200, args.iterations), seed=args.seed
    )
    output_path = Path(args.output_profile).resolve()
    final_profile = write_profile(
        output_path,
        profile_id=args.profile_id,
        model=final_model,
        manifest_path=manifest_path,
        excluded_songs=excluded,
        training_songs=[song["id"] for song in songs],
    )
    heldout_f1 = [float(value["heldOutSong"]["f1"]) for value in folds.values()]
    report = {
        "schema": "polymath-vocal-melody-support-selector-training-v1",
        "manifest": str(manifest_path),
        "excludedSongs": sorted(excluded),
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
                    "referenceRightNotes",
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
        "profileSha256": final_profile["profileSha256"],
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
                "dataset": report["dataset"],
                "folds": {
                    song: value["heldOutSong"] for song, value in folds.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
