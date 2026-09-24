"""Score one already-locked fixed-overlap prediction against opened labels.

The prediction is a full-song, reference-free artifact created before sealed
MIDI labels are authorized.  This scorer never changes that prediction.  It
stitches only the reference continuations, computes the standard Phase 94
metrics, and binds the score to both input hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.evaluate_checkpoint import stitch_clip_notes
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    score_song_timelines,
    sha256_file,
)
from ml.training.train_muscriptor_piano import read_jsonl


def score_fixed_overlap(
    prediction: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if prediction.get("schema") != "polymath-fixed-overlap-inference-v1":
        raise TimelineScoreError("Prediction has an unexpected schema")
    if prediction.get("metricsIncluded") is not False:
        raise TimelineScoreError("Prediction was not frozen before scoring")
    songs = prediction.get("songs")
    if not isinstance(songs, list) or not songs:
        raise TimelineScoreError("Prediction contains no songs")
    prediction_by_song: dict[str, list[dict[str, Any]]] = {}
    for song in songs:
        if not isinstance(song, dict):
            raise TimelineScoreError("Prediction contains an invalid song")
        song_id = str(song.get("songId") or "").strip()
        notes = song.get("notes")
        if not song_id or not isinstance(notes, list) or song_id in prediction_by_song:
            raise TimelineScoreError("Prediction has a missing or duplicate song")
        prediction_by_song[song_id] = notes

    if not records:
        raise TimelineScoreError("Reference manifest is empty")
    if any(record.get("split") != "test" for record in records):
        raise TimelineScoreError("Sealed scoring accepts only test records")
    if any(not isinstance(record.get("notes"), list) for record in records):
        raise TimelineScoreError("Reference manifest has missing notes")
    stitched_references, reference_merges = stitch_clip_notes(
        records,
        [list(record["notes"]) for record in records],
        reference=True,
    )
    if set(stitched_references) != set(prediction_by_song):
        raise TimelineScoreError("Prediction and reference contain different songs")

    # score_song_timelines expects clip-local coordinates.  One synthetic clip
    # per song preserves the already-absolute song coordinates without another
    # boundary operation.
    pseudo_records: list[dict[str, Any]] = []
    predictions: list[list[dict[str, Any]]] = []
    for song_id in sorted(stitched_references):
        references = stitched_references[song_id]
        predicted = prediction_by_song[song_id]
        maximum_end = max(
            [float(note["time"]) + float(note.get("duration") or 0) for note in references]
            + [float(note["time"]) + float(note.get("duration") or 0) for note in predicted]
            + [0.01]
        )
        pseudo_records.append({
            "clipId": f"{song_id}-full-song",
            "songId": song_id,
            "sourceStart": 0.0,
            "durationSeconds": maximum_end,
            "notes": references,
        })
        predictions.append(predicted)
    result = score_song_timelines(pseudo_records, predictions)
    result["boundaryAccounting"]["referenceContinuationMerges"] = reference_merges
    result["boundaryAccounting"]["interpretation"] = (
        "Reference continuations were stitched from the opened five-second label "
        "manifest; the pre-label full-song prediction was scored unchanged."
    )
    return result


def score_files(prediction_path: Path, reference_manifest_path: Path) -> dict[str, Any]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8-sig"))
    if not isinstance(prediction, dict):
        raise TimelineScoreError("Prediction file must contain one JSON object")
    records = read_jsonl(reference_manifest_path)
    result = score_fixed_overlap(prediction, records)
    result.update({
        "sourcePrediction": str(prediction_path.resolve()),
        "sourcePredictionSha256": sha256_file(prediction_path),
        "validationManifest": str(reference_manifest_path.resolve()),
        "validationManifestSha256": sha256_file(reference_manifest_path),
        "predictionChangedDuringScoring": False,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prediction_path = args.prediction.resolve()
    reference_path = args.reference_manifest.resolve()
    result = score_files(prediction_path, reference_path)
    destination = args.out.resolve()
    if destination.exists():
        raise TimelineScoreError(f"Refusing to overwrite sealed score: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "songs": result["songs"],
        "microF1_100ms": result["metrics"]["100ms"]["microF1"],
        "lower95_100ms": result["songClusterBootstrap95"]["100ms"]["lower95"],
    }, indent=2))


if __name__ == "__main__":
    main()
