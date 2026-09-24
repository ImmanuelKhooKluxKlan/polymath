"""Rescore decoded five-second clips as continuous song timelines.

The training evaluator deliberately reports clip-local scores, but reviewed
notes that sustain over a five-second boundary appear in two adjacent clip
labels. A real transcription should not re-strike that held note. This audit
merges those artificial boundary continuations before computing the same
exact-pitch onset metrics, while retaining the clip scores for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.training.evaluate_checkpoint import (
    aggregate_clip_scores,
    combined_song_diagnostics,
    song_cluster_bootstrap_f1,
    stitch_clip_notes,
)
from ml.training.train_muscriptor_piano import read_jsonl


class TimelineScoreError(RuntimeError):
    """Raised when an evaluation cannot be safely paired with its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def predictions_for_records(
    evaluation: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    decoded = (evaluation.get("metrics") or {}).get("decodedClips")
    if not isinstance(decoded, list):
        raise TimelineScoreError("Evaluation does not contain decodedClips")
    by_clip: dict[str, dict[str, Any]] = {}
    for item in decoded:
        if not isinstance(item, dict) or not str(item.get("clipId") or ""):
            raise TimelineScoreError("Evaluation contains an invalid decoded clip")
        clip_id = str(item["clipId"])
        if clip_id in by_clip:
            raise TimelineScoreError(f"Evaluation repeats clip {clip_id}")
        by_clip[clip_id] = item

    predictions: list[list[dict[str, Any]]] = []
    for record in records:
        clip_id = str(record.get("clipId") or "")
        item = by_clip.get(clip_id)
        if item is None:
            raise TimelineScoreError(f"Evaluation is missing clip {clip_id}")
        if str(item.get("songId") or "unknown") != str(record.get("songId") or "unknown"):
            raise TimelineScoreError(f"Song mismatch for clip {clip_id}")
        if abs(float(item.get("sourceStart") or 0) - float(record.get("sourceStart") or 0)) > 1e-6:
            raise TimelineScoreError(f"Source-start mismatch for clip {clip_id}")
        notes = item.get("notes")
        if not isinstance(notes, list):
            raise TimelineScoreError(f"Decoded notes are missing for clip {clip_id}")
        predictions.append(notes)
    if len(by_clip) != len(records):
        extras = sorted(set(by_clip) - {str(row.get("clipId") or "") for row in records})
        raise TimelineScoreError(f"Evaluation has clips outside the manifest: {extras[:3]}")
    return predictions


def score_song_timelines(
    records: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise TimelineScoreError("Record and prediction counts differ")
    references = [list(record.get("notes") or []) for record in records]
    stitched_references, reference_merges = stitch_clip_notes(
        records, references, reference=True,
    )
    stitched_predictions, prediction_merges = stitch_clip_notes(
        records, predictions, reference=False,
    )
    song_ids = sorted(set(stitched_references) | set(stitched_predictions))
    reference_songs = [stitched_references.get(song_id, []) for song_id in song_ids]
    prediction_songs = [stitched_predictions.get(song_id, []) for song_id in song_ids]
    per_song = {
        song_id: aggregate_clip_scores([reference], [prediction])
        for song_id, reference, prediction in zip(
            song_ids, reference_songs, prediction_songs, strict=True,
        )
    }
    diagnostics_100ms, per_song_diagnostics_100ms = combined_song_diagnostics(
        stitched_references, stitched_predictions, onset_tolerance=0.10,
    )
    return {
        "schema": "polymath-stitched-song-timeline-score-v1",
        "songs": len(song_ids),
        "songIds": song_ids,
        "metrics": aggregate_clip_scores(reference_songs, prediction_songs),
        "perSong": per_song,
        "songClusterBootstrap95": song_cluster_bootstrap_f1(per_song),
        "diagnostics100ms": diagnostics_100ms,
        "perSongDiagnostics100ms": per_song_diagnostics_100ms,
        "boundaryAccounting": {
            "referenceContinuationMerges": reference_merges,
            "predictedBoundaryMerges": prediction_merges,
            "interpretation": (
                "Reviewed continuation labels at artificial five-second boundaries "
                "are one held note, not a required re-strike."
            ),
        },
    }


def rescore_evaluation(evaluation_path: Path, manifest_path: Path) -> dict[str, Any]:
    return rescore_evaluation_side(evaluation_path, manifest_path)


def evaluation_view(
    payload: dict[str, Any],
    comparison_side: str | None = None,
) -> dict[str, Any]:
    """Return one checkpoint evaluation from a single or paired result."""

    schema = payload.get("schema")
    if schema == "polymath-checkpoint-evaluation-v1":
        if comparison_side is not None:
            raise TimelineScoreError(
                "A comparison side cannot be used with a single-checkpoint evaluation"
            )
        return payload
    if schema != "polymath-checkpoint-comparison-v1":
        raise TimelineScoreError("Unexpected evaluation schema")
    if comparison_side not in {"baseline", "candidate"}:
        raise TimelineScoreError(
            "Paired checkpoint comparisons require --comparison-side baseline or candidate"
        )
    metrics = payload.get(comparison_side)
    if not isinstance(metrics, dict):
        raise TimelineScoreError(f"Comparison does not contain {comparison_side} metrics")
    checkpoint_key = (
        "baseCheckpoint" if comparison_side == "baseline" else "candidateCheckpoint"
    )
    return {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": payload.get(checkpoint_key),
        "validationManifest": payload.get("validationManifest"),
        "clips": payload.get("clips"),
        "instrumentConstraint": payload.get("instrumentConstraint"),
        "metrics": metrics,
    }


def rescore_evaluation_side(
    evaluation_path: Path,
    manifest_path: Path,
    comparison_side: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation = evaluation_view(payload, comparison_side)
    records = read_jsonl(manifest_path)
    predictions = predictions_for_records(evaluation, records)
    result = score_song_timelines(records, predictions)
    source_metrics = evaluation.get("metrics") or {}
    result.update({
        "checkpoint": evaluation.get("checkpoint"),
        "validationManifest": str(manifest_path.resolve()),
        "sourceEvaluation": str(evaluation_path.resolve()),
        "sourceEvaluationSide": comparison_side,
        "sourceEvaluationSha256": sha256_file(evaluation_path),
        "validationManifestSha256": sha256_file(manifest_path),
        "clipLocalMetrics": {
            key: source_metrics.get(key) for key in ("50ms", "100ms", "250ms")
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--comparison-side",
        choices=("baseline", "candidate"),
        help="Select one side when --evaluation is a paired checkpoint comparison.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = rescore_evaluation_side(
        args.evaluation.resolve(),
        args.validation_manifest.resolve(),
        args.comparison_side,
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "songs": result["songs"],
        "microF1": {
            key: result["metrics"][key]["microF1"]
            for key in ("50ms", "100ms", "250ms")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
