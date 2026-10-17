"""Evaluate held-out time windows while preserving full-song decoder context.

Autoregressive music transcription can behave differently when a five-second
excerpt is decoded alone versus when that same excerpt appears inside a whole
song.  This module transcribes the complete target audio, then scores only
pre-frozen, temporally held-out windows.  It is an opened development gate for
checkpoint selection, never an untouched certification result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from ml.training.evaluate_piano_arranger import (
    evaluate,
    load_json,
    normalize_notes,
    prepare_reference_notes,
)


SCHEMA = "polymath-full-song-target-development-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(config: dict[str, Any], name: str) -> Path:
    item = config.get(name)
    if not isinstance(item, dict):
        raise ValueError(f"Full-song target config is missing {name}")
    path = Path(str(item.get("path") or "")).resolve()
    expected = str(item.get("sha256") or "").lower()
    if not path.is_file() or not expected:
        raise ValueError(f"Full-song target config has an invalid {name}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Full-song target config hash mismatch for {name}: {actual} != {expected}"
        )
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected full-song target schema: {config.get('schema')}")
    windows = config.get("windows")
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("Full-song target config needs at least two windows")
    normalized: list[dict[str, float]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"Window {index} must be an object")
        start = float(window.get("start"))
        end = float(window.get("end"))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Window {index} has invalid boundaries")
        normalized.append({"start": start, "end": end})
    normalized.sort(key=lambda item: item["start"])
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("Full-song target windows cannot overlap")
    config = dict(config)
    constraint = config.get("instrumentConstraint", [])
    if constraint not in ([], ["acoustic_piano"]):
        raise ValueError("Full-song piano development supports only an empty or acoustic-piano output constraint")
    config["instrumentConstraint"] = list(constraint)
    config["windows"] = normalized
    config["resolvedPaths"] = {
        name: str(_verified_path(config, name))
        for name in ("audio", "baselineRaw", "reference", "alignmentReport", "splitReceipt")
    }
    if config.get("decoderMode") == "source-unconditioned-piano-output-v1":
        payload = json.loads(Path(config["resolvedPaths"]["baselineRaw"]).read_text())
        provenance = payload.get("researchProvenance") or {}
        if (
            "sourceConditioning" not in provenance
            or provenance["sourceConditioning"] is not None
            or provenance.get("allowedOutputInstruments") != ["acoustic_piano"]
            or provenance.get("sourceAudioSha256") != config["audio"]["sha256"]
        ):
            raise ValueError("Full-song baseline lacks verified source-unconditioned piano-output provenance")
    return config


def _inside_windows(
    notes: list[dict[str, float | int]],
    windows: list[dict[str, float]],
) -> list[dict[str, float | int]]:
    return [
        note
        for note in notes
        if any(
            window["start"] <= float(note["time"]) < window["end"]
            for window in windows
        )
    ]


def score_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    reference_payload = load_json(Path(config["resolvedPaths"]["reference"]))
    alignment_report = load_json(Path(config["resolvedPaths"]["alignmentReport"]))
    pair = {
        "referenceAlreadyAligned": bool(config.get("referenceAlreadyAligned", True)),
        "referenceTransposeSemitones": int(config.get("referenceTransposeSemitones", 0)),
    }
    reference = _inside_windows(
        prepare_reference_notes(pair, reference_payload, alignment_report),
        config["windows"],
    )
    observed = _inside_windows(normalize_notes(payload), config["windows"])
    if not reference:
        raise ValueError("Full-song target windows contain no approved reference notes")
    scored = evaluate(reference, observed)
    result: dict[str, Any] = {
        "schema": "polymath-full-song-target-development-score-v1",
        "context": "whole-song transcription; held-out windows scored",
        "referenceNotes": len(reference),
        "predictedNotes": len(observed),
        "windows": config["windows"],
    }
    for tolerance in (50, 100, 250):
        exact = scored[f"exactPitchOnset{tolerance}ms"]
        result[f"{tolerance}ms"] = {
            "referenceNotes": len(reference),
            "predictedNotes": len(observed),
            "matchedNotes": int(exact["matches"]),
            "precision": float(exact["precision"]),
            "recall": float(exact["recall"]),
            "microF1": float(exact["f1"]),
        }
    return result


def baseline_score(config: dict[str, Any]) -> dict[str, Any]:
    return score_payload(
        load_json(Path(config["resolvedPaths"]["baselineRaw"])),
        config,
    )


def transcribe_and_score(
    transcription,
    config: dict[str, Any],
    *,
    candidate_id: str,
    optimizer_step: int,
    artifact_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transcribe once and atomically preserve raw plus scored evidence."""

    # Keep dataset/config audits runnable on machines that do not have the
    # heavyweight MuScriptor inference package installed.
    from ml.training.transcribe_batch import transcribe_one

    artifact_directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{optimizer_step:04d}"
    raw_path = artifact_directory / f"{stem}-raw.json"
    score_path = artifact_directory / f"{stem}-score.json"
    if raw_path.exists() or score_path.exists():
        raise FileExistsError(f"Target-development artifact already exists for {stem}")
    payload = transcribe_one(
        transcription,
        Path(config["resolvedPaths"]["audio"]),
        title=f"{candidate_id} target development step {optimizer_step}",
        instruments=config.get("instrumentConstraint") or None,
        model_id=f"{candidate_id}-step-{optimizer_step:04d}",
        provider="Polymath frozen full-song target-development gate",
    )
    score = score_payload(payload, config)
    temporary_raw = raw_path.with_suffix(".json.partial")
    temporary_score = score_path.with_suffix(".json.partial")
    temporary_raw.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_score.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_raw, raw_path)
    os.replace(temporary_score, score_path)
    evidence = {
        "optimizerStep": optimizer_step,
        "raw": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "score": {"path": str(score_path), "sha256": sha256(score_path)},
    }
    return score, evidence
