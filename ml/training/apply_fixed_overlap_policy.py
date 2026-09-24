"""Apply one pre-frozen release-preserving overlap policy to decoded clips.

Unlike the policy-search utility, this module never reads reference notes and
never ranks alternatives.  It is the inference-side bridge used after a
radius and onset-pairing tolerance have already been frozen on development
data.  The output is suitable for opened-song listening and later production
integration, but it contains no accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.training.evaluate_checkpoint import stitch_clip_notes
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    predictions_for_records,
    sha256_file,
)
from ml.training.search_overlap_release_preserving_policy import (
    merge_song_predictions_release_preserving,
)
from ml.training.train_muscriptor_piano import read_jsonl


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return cleaned or "song"


def records_by_song(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("songId") or "unknown")].append(record)
    return {
        song_id: sorted(rows, key=lambda row: float(row.get("sourceStart") or 0))
        for song_id, rows in sorted(grouped.items())
    }


def clean_note(note: dict[str, Any]) -> dict[str, Any]:
    return {
        "midi": int(note["midi"]),
        "time": round(float(note["time"]), 6),
        "duration": round(max(0.01, float(note.get("duration") or 0.01)), 6),
        "velocity": round(float(note.get("velocity", 0.75)), 4),
        "instrument": str(note.get("instrument") or "acoustic_piano"),
    }


def checkpoint_identity(evaluation: dict[str, Any]) -> str:
    checkpoint = evaluation.get("checkpoint")
    if isinstance(checkpoint, dict):
        return json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
    return str(checkpoint or "")


def apply_fixed_policy(
    primary_evaluation: dict[str, Any],
    overlap_evaluation: dict[str, Any],
    primary_records: list[dict[str, Any]],
    overlap_records: list[dict[str, Any]],
    *,
    radius_seconds: float,
    onset_match_tolerance_seconds: float,
) -> dict[str, Any]:
    if primary_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Primary input is not a single checkpoint evaluation")
    if overlap_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Overlap input is not a single checkpoint evaluation")
    primary_checkpoint = checkpoint_identity(primary_evaluation)
    overlap_checkpoint = checkpoint_identity(overlap_evaluation)
    if not primary_checkpoint or primary_checkpoint != overlap_checkpoint:
        raise TimelineScoreError("Primary and overlap evaluations must use the same checkpoint")
    primary_predictions = predictions_for_records(primary_evaluation, primary_records)
    overlap_predictions = predictions_for_records(overlap_evaluation, overlap_records)
    primary_songs, primary_boundary_merges = stitch_clip_notes(
        primary_records, primary_predictions, reference=False,
    )
    overlap_songs, overlap_boundary_merges = stitch_clip_notes(
        overlap_records, overlap_predictions, reference=False,
    )
    primary_windows = records_by_song(primary_records)
    overlap_windows = records_by_song(overlap_records)
    song_ids = sorted(set(primary_songs) | set(overlap_songs))
    songs: list[dict[str, Any]] = []
    for song_id in song_ids:
        merged = merge_song_predictions_release_preserving(
            primary_songs.get(song_id, []),
            overlap_songs.get(song_id, []),
            primary_windows.get(song_id, []),
            overlap_windows.get(song_id, []),
            radius_seconds,
            onset_match_tolerance_seconds=onset_match_tolerance_seconds,
        )
        notes = [clean_note(note) for note in merged]
        record_end = max(
            (
                float(record.get("sourceStart") or 0)
                + float(record.get("durationSeconds") or 0)
                for record in primary_windows.get(song_id, [])
            ),
            default=0.0,
        )
        songs.append({
            "songId": song_id,
            "durationSeconds": round(record_end, 6),
            "noteCount": len(notes),
            "notes": notes,
        })
    labels_present = all(
        bool(record.get("labelsPresent", True))
        for record in [*primary_records, *overlap_records]
    )
    return {
        "schema": "polymath-fixed-overlap-inference-v1",
        "checkpoint": primary_evaluation.get("checkpoint"),
        "policy": {
            "schema": "polymath-overlap-release-preserving-v1",
            "radiusSeconds": radius_seconds,
            "onsetMatchToleranceSeconds": onset_match_tolerance_seconds,
            "policyRetunedOnTheseSongs": False,
        },
        "songs": songs,
        "songCount": len(songs),
        "noteCount": sum(song["noteCount"] for song in songs),
        "boundaryAccounting": {
            "primaryPredictionMerges": primary_boundary_merges,
            "overlapPredictionMerges": overlap_boundary_merges,
        },
        "labelsPresent": labels_present,
        "metricsIncluded": False,
        "accuracyEvidenceAllowed": labels_present,
    }


def write_playable_songs(result: dict[str, Any], output_dir: Path) -> list[Path]:
    if output_dir.exists():
        raise TimelineScoreError(f"Refusing to overwrite playable output: {output_dir}")
    output_dir.mkdir(parents=True)
    written: list[Path] = []
    for song in result["songs"]:
        destination = output_dir / f"{safe_filename(song['songId'])}.json"
        payload = {
            "schema": "polymath-ready-to-play-v1",
            "title": song["songId"],
            "instrument": "acoustic_piano",
            "sourceType": "fixed-overlap-inference",
            "checkpoint": result["checkpoint"],
            "inferencePolicy": result["policy"],
            "durationSeconds": song["durationSeconds"],
            "notes": song["notes"],
        }
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--overlap-evaluation", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--overlap-manifest", type=Path, required=True)
    parser.add_argument("--radius-seconds", type=float, required=True)
    parser.add_argument("--onset-match-tolerance-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--playable-output-dir", type=Path)
    args = parser.parse_args()
    paths = {
        "primaryEvaluation": args.primary_evaluation.resolve(),
        "overlapEvaluation": args.overlap_evaluation.resolve(),
        "primaryManifest": args.primary_manifest.resolve(),
        "overlapManifest": args.overlap_manifest.resolve(),
    }
    primary_evaluation = json.loads(paths["primaryEvaluation"].read_text(encoding="utf-8"))
    overlap_evaluation = json.loads(paths["overlapEvaluation"].read_text(encoding="utf-8"))
    primary_records = read_jsonl(paths["primaryManifest"])
    overlap_records = read_jsonl(paths["overlapManifest"])
    result = apply_fixed_policy(
        primary_evaluation,
        overlap_evaluation,
        primary_records,
        overlap_records,
        radius_seconds=args.radius_seconds,
        onset_match_tolerance_seconds=args.onset_match_tolerance_seconds,
    )
    result["sources"] = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
    destination = args.out.resolve()
    if destination.exists():
        raise TimelineScoreError(f"Refusing to overwrite fixed-policy output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    playable = (
        write_playable_songs(result, args.playable_output_dir.resolve())
        if args.playable_output_dir
        else []
    )
    print(json.dumps({
        "output": str(destination),
        "sha256": sha256_file(destination),
        "songs": result["songCount"],
        "notes": result["noteCount"],
        "metricsIncluded": False,
        "playable": [str(path) for path in playable],
    }, indent=2))


if __name__ == "__main__":
    main()
