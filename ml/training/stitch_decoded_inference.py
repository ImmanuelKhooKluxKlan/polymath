"""Stitch one decoded clip evaluation into playable song timelines.

This is the primary-window baseline companion to apply_fixed_overlap_policy.
It exposes decoded notes for listening without calculating or implying
accuracy on unlabeled songs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.training.apply_fixed_overlap_policy import clean_note, write_playable_songs
from ml.training.evaluate_checkpoint import stitch_clip_notes
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    predictions_for_records,
    sha256_file,
)
from ml.training.train_muscriptor_piano import read_jsonl


def stitch_decoded_inference(
    evaluation: dict[str, Any], records: list[dict[str, Any]],
) -> dict[str, Any]:
    if evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Input is not a single checkpoint evaluation")
    if not evaluation.get("checkpoint"):
        raise TimelineScoreError("Evaluation has no checkpoint identity")
    predictions = predictions_for_records(evaluation, records)
    stitched, boundary_merges = stitch_clip_notes(records, predictions, reference=False)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("songId") or "unknown")].append(record)
    songs: list[dict[str, Any]] = []
    for song_id in sorted(stitched):
        rows = grouped[song_id]
        duration = max(
            float(row.get("sourceStart") or 0) + float(row.get("durationSeconds") or 0)
            for row in rows
        )
        notes = [clean_note(note) for note in stitched[song_id]]
        songs.append({
            "songId": song_id,
            "durationSeconds": round(duration, 6),
            "noteCount": len(notes),
            "notes": notes,
        })
    labels_present = all(bool(record.get("labelsPresent", True)) for record in records)
    return {
        "schema": "polymath-stitched-decoded-inference-v1",
        "checkpoint": evaluation.get("checkpoint"),
        "policy": {
            "schema": "polymath-primary-window-stitch-v1",
            "policyRetunedOnTheseSongs": False,
        },
        "songs": songs,
        "songCount": len(songs),
        "noteCount": sum(song["noteCount"] for song in songs),
        "boundaryAccounting": {"primaryPredictionMerges": boundary_merges},
        "labelsPresent": labels_present,
        "metricsIncluded": False,
        "accuracyEvidenceAllowed": labels_present,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--playable-output-dir", type=Path)
    args = parser.parse_args()
    evaluation_path = args.evaluation.resolve()
    manifest_path = args.manifest.resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    records = read_jsonl(manifest_path)
    result = stitch_decoded_inference(evaluation, records)
    result["sources"] = {
        "evaluation": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
    }
    destination = args.out.resolve()
    if destination.exists():
        raise TimelineScoreError(f"Refusing to overwrite stitched output: {destination}")
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
