"""Evaluate MuScriptor checkpoints against a frozen reviewed clip manifest.

This module keeps decoding evaluation separate from teacher-forced validation
loss.  A lower loss is useful, but only decoded notes reveal whether a candidate
actually improved note precision, recall, and timing.
"""

from __future__ import annotations

import gc
import hashlib
import json
import argparse
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from ml.training.evaluate_predictions import analyze_errors, evaluate, normalize_notes
from ml.training.train_muscriptor_piano import read_jsonl


DEFAULT_TOLERANCES = (0.05, 0.10, 0.25)
PIANO_INSTRUMENTS = ("acoustic_piano",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_notes(events: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert MuScriptor's streamed start/end events into local clip notes."""

    starts: dict[int, Any] = {}
    notes: list[dict[str, Any]] = []
    for event in events:
        if hasattr(event, "start_time") and hasattr(event, "pitch"):
            starts[int(event.index)] = event
        elif hasattr(event, "end_time") and hasattr(event, "start_event"):
            start = event.start_event
            onset = max(0.0, float(start.start_time))
            ending = max(onset + 0.01, float(event.end_time))
            notes.append({
                "midi": int(start.pitch),
                "time": onset,
                "duration": ending - onset,
                "instrument": str(getattr(start, "instrument", "acoustic_piano")),
            })
            starts.pop(int(start.index), None)

    # A start with no end is still a detected onset. The conservative default
    # duration mirrors the production transcription worker.
    for start in starts.values():
        notes.append({
            "midi": int(start.pitch),
            "time": max(0.0, float(start.start_time)),
            "duration": 0.4,
            "instrument": str(getattr(start, "instrument", "acoustic_piano")),
        })
    notes.sort(key=lambda note: (note["time"], note["midi"]))
    return notes


def _indexed(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, note in enumerate(notes):
        result.append({
            "index": index,
            "midi": int(note["midi"]),
            "time": float(note["time"]),
            "duration": max(0.01, float(note.get("duration") or 0.1)),
            "instrument": str(note.get("instrument") or "acoustic_piano"),
            "continuedFromPreviousClip": bool(note.get("continuedFromPreviousClip")),
            "continuesIntoNextClip": bool(note.get("continuesIntoNextClip")),
        })
    return result


def stitch_clip_notes(
    records: list[dict[str, Any]],
    notes_by_clip: list[list[dict[str, Any]]],
    *,
    reference: bool,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Rebuild song timelines and remove artificial five-second boundaries."""

    if len(records) != len(notes_by_clip):
        raise ValueError("Record and decoded clip counts differ")
    songs: dict[str, list[dict[str, Any]]] = {}
    active: dict[tuple[str, str, int], dict[str, Any]] = {}
    boundary_merges = 0
    ordered = sorted(
        zip(records, notes_by_clip, strict=True),
        key=lambda item: (str(item[0].get("songId") or "unknown"), float(item[0].get("sourceStart") or 0)),
    )
    for record, clip_notes in ordered:
        song_id = str(record.get("songId") or "unknown")
        clip_start = float(record.get("sourceStart") or 0)
        instrument_focus = str(record.get("instrumentFocus") or "acoustic_piano")
        destination = songs.setdefault(song_id, [])
        for local in normalize_notes(clip_notes, instrument_focus):
            note = dict(local)
            note["time"] = clip_start + local["time"]
            key = (song_id, note["instrument"], note["midi"])
            previous = active.get(key)
            explicit_continuation = reference and local.get("continuedFromPreviousClip")
            inferred_continuation = (
                not reference
                and local["time"] <= 0.08
                and previous is not None
                and previous["time"] + previous["duration"] >= clip_start - 0.12
            )
            if previous is not None and (explicit_continuation or inferred_continuation):
                ending = note["time"] + note["duration"]
                previous["duration"] = max(previous["duration"], ending - previous["time"])
                previous["continuesIntoNextClip"] = bool(local.get("continuesIntoNextClip"))
                boundary_merges += 1
                if not previous["continuesIntoNextClip"] and reference:
                    active.pop(key, None)
                continue
            destination.append(note)
            if (reference and local.get("continuesIntoNextClip")) or not reference:
                active[key] = note
            elif reference:
                active.pop(key, None)
    for notes in songs.values():
        notes.sort(key=lambda note: (note["time"], note["instrument"], note["midi"]))
    return songs, boundary_merges


def combined_song_diagnostics(
    references: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    onset_tolerance: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_song: dict[str, Any] = {}
    combined_reference: list[dict[str, Any]] = []
    combined_prediction: list[dict[str, Any]] = []
    cursor = 0.0
    for song_id in sorted(set(references) | set(predictions)):
        target = references.get(song_id, [])
        candidate = predictions.get(song_id, [])
        per_song[song_id] = analyze_errors(target, candidate, onset_tolerance)
        duration = max(
            [note["time"] + note["duration"] for note in [*target, *candidate]] or [0.0]
        )
        combined_reference.extend({**note, "time": note["time"] + cursor} for note in target)
        combined_prediction.extend({**note, "time": note["time"] + cursor} for note in candidate)
        cursor += duration + 2.0
    return analyze_errors(combined_reference, combined_prediction, onset_tolerance), per_song


def aggregate_clip_scores(
    references: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCES,
) -> dict[str, Any]:
    """Return micro and macro note scores without matching across clip edges."""

    if len(references) != len(predictions):
        raise ValueError("Reference and prediction clip counts differ")
    result: dict[str, Any] = {}
    for tolerance in tolerances:
        clip_scores = [
            evaluate(_indexed(reference), _indexed(prediction), tolerance)
            for reference, prediction in zip(references, predictions, strict=True)
        ]
        matched = sum(score["matchedNotes"] for score in clip_scores)
        reference_count = sum(score["referenceNotes"] for score in clip_scores)
        predicted_count = sum(score["predictedNotes"] for score in clip_scores)
        precision = matched / max(1, predicted_count)
        recall = matched / max(1, reference_count)
        micro_f1 = 2 * precision * recall / max(1e-12, precision + recall)
        key = f"{round(tolerance * 1000)}ms"
        result[key] = {
            "referenceNotes": reference_count,
            "predictedNotes": predicted_count,
            "matchedNotes": matched,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "microF1": round(micro_f1, 6),
            "macroF1": round(mean(score["f1"] for score in clip_scores), 6),
        }
    return result


def aggregate_song_clip_scores(
    records: list[dict[str, Any]],
    references: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCES,
) -> dict[str, Any]:
    """Calculate the same micro scores separately for every validation song."""

    if not (len(records) == len(references) == len(predictions)):
        raise ValueError("Record, reference, and prediction clip counts differ")
    grouped: dict[str, tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]] = {}
    for record, reference, prediction in zip(records, references, predictions, strict=True):
        song_id = str(record.get("songId") or "unknown")
        targets, candidates = grouped.setdefault(song_id, ([], []))
        targets.append(reference)
        candidates.append(prediction)
    return {
        song_id: aggregate_clip_scores(targets, candidates, tolerances)
        for song_id, (targets, candidates) in sorted(grouped.items())
    }


def song_cluster_bootstrap_f1(
    per_song_scores: dict[str, Any],
    *,
    samples: int = 5000,
    seed: str = "polymath-song-cluster-bootstrap-v1",
) -> dict[str, Any]:
    """Estimate uncertainty by resampling complete songs, never five-second clips."""

    if samples < 100:
        raise ValueError("Song bootstrap requires at least 100 samples")
    song_ids = sorted(per_song_scores)
    if not song_ids:
        return {}

    def f1_for_rows(rows: list[dict[str, Any]]) -> float:
        reference = sum(int(row["referenceNotes"]) for row in rows)
        predicted = sum(int(row["predictedNotes"]) for row in rows)
        matched = sum(int(row["matchedNotes"]) for row in rows)
        precision = matched / max(1, predicted)
        recall = matched / max(1, reference)
        return 2 * precision * recall / max(1e-12, precision + recall)

    result: dict[str, Any] = {}
    for tolerance_key in ("50ms", "100ms", "250ms"):
        rng = random.Random(f"{seed}:{tolerance_key}")
        observed_rows = [per_song_scores[song_id][tolerance_key] for song_id in song_ids]
        draws = []
        for _ in range(samples):
            rows = [
                per_song_scores[song_ids[rng.randrange(len(song_ids))]][tolerance_key]
                for _ in song_ids
            ]
            draws.append(f1_for_rows(rows))
        draws.sort()

        def quantile(fraction: float) -> float:
            position = fraction * (len(draws) - 1)
            lower = int(position)
            upper = min(len(draws) - 1, lower + 1)
            mix = position - lower
            return draws[lower] * (1 - mix) + draws[upper] * mix

        result[tolerance_key] = {
            "pointEstimateMicroF1": round(f1_for_rows(observed_rows), 6),
            "lower95": round(quantile(0.025), 6),
            "median": round(quantile(0.5), 6),
            "upper95": round(quantile(0.975), 6),
            "songs": len(song_ids),
            "bootstrapSamples": samples,
            "resamplingUnit": "complete-song",
        }
    return result


def evaluate_loaded_transcription(
    transcription,
    records: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] | None = PIANO_INSTRUMENTS,
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Decode a frozen panel with an already-loaded checkpoint."""

    import torch

    predictions: list[list[dict[str, Any]]] = []
    transcription._model.eval()
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            predictions.append(decoded_notes(
                transcription.transcribe(
                    str(Path(record["audioClip"])),
                    instruments=list(instruments) if instruments else None,
                ),
            ))
            if progress_callback and (index == 1 or index % 5 == 0 or index == len(records)):
                progress_callback(f"Decoded {index}/{len(records)} validation clips")

    return evaluate_decoded_predictions(
        records,
        predictions,
        include_raw_predictions=include_raw_predictions,
    )


def evaluate_decoded_predictions(
    records: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    *,
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Score already-decoded clips with the exact live-evaluation procedure."""

    if len(records) != len(predictions):
        raise ValueError("Record and decoded clip counts differ")
    references = [list(record["notes"]) for record in records]

    metrics = aggregate_clip_scores(references, predictions)
    metrics["perSongClipScores"] = aggregate_song_clip_scores(
        records, references, predictions,
    )
    metrics["songClusterBootstrap95"] = song_cluster_bootstrap_f1(
        metrics["perSongClipScores"]
    )
    stitched_references, reference_boundary_merges = stitch_clip_notes(
        records, references, reference=True,
    )
    stitched_predictions, prediction_boundary_merges = stitch_clip_notes(
        records, predictions, reference=False,
    )
    diagnostics, per_song = combined_song_diagnostics(
        stitched_references, stitched_predictions,
    )
    metrics["diagnostics50ms"] = diagnostics
    metrics["perSongDiagnostics50ms"] = per_song
    metrics["boundaryAccounting"] = {
        "referenceContinuationMerges": reference_boundary_merges,
        "predictedBoundaryMerges": prediction_boundary_merges,
        "note": "Artificial five-second clip boundaries are merged before duration/pattern analysis.",
    }
    if include_raw_predictions:
        metrics["decodedClips"] = [
            {
                "clipId": str(record.get("clipId") or index),
                "songId": str(record.get("songId") or "unknown"),
                "sourceStart": float(record.get("sourceStart") or 0),
                "notes": notes,
            }
            for index, (record, notes) in enumerate(zip(records, predictions, strict=True))
        ]
    return metrics


def evaluate_checkpoint(
    checkpoint: Path,
    records: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] | None = PIANO_INSTRUMENTS,
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Load one checkpoint, decode every frozen clip, and calculate note scores."""

    import torch
    from muscriptor import TranscriptionModel

    transcription = TranscriptionModel.load_model(checkpoint, device="cuda")
    metrics = evaluate_loaded_transcription(
        transcription,
        records,
        progress_callback,
        instruments,
        include_raw_predictions,
    )
    del transcription
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def compare_checkpoints(
    base: Path,
    candidate: Path,
    validation_manifest: Path,
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] | None = PIANO_INSTRUMENTS,
) -> dict[str, Any]:
    records = read_jsonl(validation_manifest)
    if progress_callback:
        progress_callback("Decoding frozen validation clips with the original checkpoint")
    baseline = evaluate_checkpoint(
        base, records, progress_callback, instruments, include_raw_predictions=True,
    )
    if progress_callback:
        progress_callback("Decoding the same clips with the Phase candidate")
    candidate_metrics = evaluate_checkpoint(
        candidate, records, progress_callback, instruments, include_raw_predictions=True,
    )
    deltas = {
        key: round(candidate_metrics[key]["microF1"] - baseline[key]["microF1"], 6)
        for key in ("50ms", "100ms", "250ms")
    }
    return {
        "schema": "polymath-checkpoint-comparison-v1",
        "baseCheckpoint": str(base),
        "baseCheckpointSha256": sha256_file(base),
        "candidateCheckpoint": str(candidate),
        "candidateCheckpointSha256": sha256_file(candidate),
        "validationManifest": str(validation_manifest),
        "clips": len(records),
        "instrumentConstraint": list(instruments) if instruments else [],
        "baseline": baseline,
        "candidate": candidate_metrics,
        "candidateMinusBaselineMicroF1": deltas,
    }


def evaluate_single_checkpoint(
    checkpoint: Path,
    validation_manifest: Path,
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] | None = PIANO_INSTRUMENTS,
) -> dict[str, Any]:
    """Decode one frozen manifest without wasting a second baseline pass."""

    records = read_jsonl(validation_manifest)
    if progress_callback:
        progress_callback(
            f"Decoding {len(records)} frozen clips with {checkpoint.name}"
        )
    metrics = evaluate_checkpoint(
        checkpoint,
        records,
        progress_callback,
        instruments,
        include_raw_predictions=True,
    )
    return {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256_file(checkpoint),
        "validationManifest": str(validation_manifest),
        "clips": len(records),
        "instrumentConstraint": list(instruments) if instruments else [],
        "metrics": metrics,
    }


def save_comparison(result: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two MuScriptor checkpoints on one frozen manifest."
    )
    parser.add_argument("--base", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate one checkpoint once; mutually exclusive with --base/--candidate.",
    )
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def report(message: str) -> None:
        print(message, flush=True)

    if args.checkpoint:
        if args.base or args.candidate:
            parser.error("--checkpoint cannot be combined with --base or --candidate")
        result = evaluate_single_checkpoint(
            args.checkpoint.resolve(),
            args.validation_manifest.resolve(),
            progress_callback=report,
        )
    else:
        if not args.base or not args.candidate:
            parser.error("comparison mode requires both --base and --candidate")
        result = compare_checkpoints(
            args.base.resolve(),
            args.candidate.resolve(),
            args.validation_manifest.resolve(),
            progress_callback=report,
        )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_comparison(result, destination)
    summary = {
        "output": str(destination),
        "clips": result["clips"],
    }
    if "candidateMinusBaselineMicroF1" in result:
        summary["candidateMinusBaselineMicroF1"] = result[
            "candidateMinusBaselineMicroF1"
        ]
    else:
        summary["microF1"] = {
            key: result["metrics"][key]["microF1"]
            for key in ("50ms", "100ms", "250ms")
        }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
