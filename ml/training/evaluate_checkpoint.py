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

from ml.training.evaluate_predictions import evaluate
from ml.training.train_muscriptor_piano import read_jsonl


DEFAULT_TOLERANCES = (0.05, 0.10, 0.25)


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
            })
            starts.pop(int(start.index), None)

    # A start with no end is still a detected onset. The conservative default
    # duration mirrors the production transcription worker.
    for start in starts.values():
        notes.append({
            "midi": int(start.pitch),
            "time": max(0.0, float(start.start_time)),
            "duration": 0.4,
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
        })
    return result


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
            transcription.transcribe(str(Path(record["audioClip"])), instruments=None),
        ))
        if progress_callback and (index == 1 or index % 5 == 0 or index == len(records)):
            progress_callback(f"Decoded {index}/{len(records)} validation clips")

    metrics = aggregate_clip_scores(references, predictions)
    del transcription
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def compare_checkpoints(
    base: Path,
    candidate: Path,
    validation_manifest: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    records = read_jsonl(validation_manifest)
    if progress_callback:
        progress_callback("Decoding frozen validation clips with the original checkpoint")
    baseline = evaluate_checkpoint(base, records, progress_callback)
    if progress_callback:
        progress_callback("Decoding the same clips with the Phase candidate")
    candidate_metrics = evaluate_checkpoint(candidate, records, progress_callback)
    deltas = {
        tolerance: round(candidate_metrics[tolerance]["microF1"] - score["microF1"], 6)
        for tolerance, score in baseline.items()
    }
    return {
        "schema": "polymath-checkpoint-comparison-v1",
        "validationManifest": str(validation_manifest),
        "clips": len(records),
        "baseline": baseline,
        "candidate": candidate_metrics,
        "candidateMinusBaselineMicroF1": deltas,
    }


def save_comparison(result: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
