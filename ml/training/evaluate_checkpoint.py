"""Evaluate MuScriptor checkpoints against a frozen reviewed clip manifest.

This module keeps decoding evaluation separate from teacher-forced validation
loss.  A lower loss is useful, but only decoded notes reveal whether a candidate
actually improved note precision, recall, and timing.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from ml.training.evaluate_predictions import analyze_errors, evaluate, normalize_notes
from ml.training.train_muscriptor_piano import read_jsonl


DEFAULT_TOLERANCES = (0.05, 0.10, 0.25)
PIANO_INSTRUMENTS = ("acoustic_piano",)


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


def evaluate_checkpoint(
    checkpoint: Path,
    records: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] = PIANO_INSTRUMENTS,
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Load one checkpoint, decode every frozen clip, and calculate note scores."""

    import torch
    from muscriptor import TranscriptionModel

    transcription = TranscriptionModel.load_model(checkpoint, device="cuda")
    references: list[list[dict[str, Any]]] = []
    predictions: list[list[dict[str, Any]]] = []
    for index, record in enumerate(records, 1):
        references.append(list(record["notes"]))
        predictions.append(decoded_notes(
            transcription.transcribe(
                str(Path(record["audioClip"])),
                instruments=list(instruments),
            ),
        ))
        if progress_callback and (index == 1 or index % 5 == 0 or index == len(records)):
            progress_callback(f"Decoded {index}/{len(records)} validation clips")

    metrics = aggregate_clip_scores(references, predictions)
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
    del transcription
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def compare_checkpoints(
    base: Path,
    candidate: Path,
    validation_manifest: Path,
    progress_callback: Callable[[str], None] | None = None,
    instruments: tuple[str, ...] = PIANO_INSTRUMENTS,
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
        "validationManifest": str(validation_manifest),
        "clips": len(records),
        "instrumentConstraint": list(instruments),
        "baseline": baseline,
        "candidate": candidate_metrics,
        "candidateMinusBaselineMicroF1": deltas,
    }


def save_comparison(result: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
