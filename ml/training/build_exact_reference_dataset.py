"""Build five-second RunPod validation clips from exactly aligned WAV/JSON pairs.

This is for research references whose performance audio and MIDI-derived JSON
share the same clock (for example, MAESTRO).  It must not be used to pretend a
loosely aligned music video is exact ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

from ml.training.dataset_builder import build_dataset
from ml.training.prepare_audio_clips import prepare_manifest, resolve_ffmpeg


SCHEMA = "polymath-exact-reference-dataset-spec-v1"
WINDOW_SECONDS = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                raise ValueError("sample rate is zero")
            return handle.getnframes() / rate
    except (OSError, wave.Error, ValueError) as exc:
        raise ValueError(f"Could not read WAV duration for {path}: {exc}") from exc


def quality_windows(duration: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration - 1e-7:
        end = min(duration, start + WINDOW_SECONDS)
        result.append(
            {
                "id": f"w{index:05d}",
                "sourceStart": round(start, 6),
                "sourceEnd": round(end, 6),
                "referenceStart": round(start, 6),
                "referenceEnd": round(end, 6),
                "status": "trusted",
                "trainingEligible": True,
                "reason": "Audio and performance-MIDI reference share an exact dataset clock.",
            }
        )
        start = end
        index += 1
    return result


def normalize_velocity(value: Any) -> float:
    velocity = float(value if value is not None else 0.75)
    if velocity > 1.0:
        velocity /= 127.0
    return max(0.01, min(1.0, velocity))


def build_exact_supervision_package(
    *,
    song_id: str,
    source_media: Path,
    target_payload: dict[str, Any],
    duration: float,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    target_notes = target_payload.get("notes")
    if not isinstance(target_notes, list) or not target_notes:
        raise ValueError(f"{song_id}: target JSON contains no notes")

    notes: list[dict[str, Any]] = []
    for index, item in enumerate(target_notes):
        if not isinstance(item, dict):
            raise ValueError(f"{song_id}: target note {index} is not an object")
        try:
            midi = int(round(float(item["midi"])))
            start = max(0.0, float(item["time"]))
            note_duration = max(0.01, float(item.get("duration", 0.2)))
            velocity = normalize_velocity(item.get("velocity"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{song_id}: invalid target note {index}: {exc}") from exc
        if not 0 <= midi <= 127 or start >= duration:
            continue
        end = min(duration, start + note_duration)
        window_index = min(int(start // WINDOW_SECONDS), max(0, int((duration - 1e-7) // WINDOW_SECONDS)))
        notes.append(
            {
                "midi": midi,
                "time": round(start, 6),
                "duration": round(max(0.01, end - start), 6),
                "velocity": round(velocity, 4),
                "instrument": "acoustic_piano",
                "qualityStatus": "trusted",
                "qualityWindowId": f"w{window_index:05d}",
                "trainingEligible": True,
            }
        )
    notes.sort(key=lambda item: (item["time"], item["midi"]))
    if not notes:
        raise ValueError(f"{song_id}: no target notes fall inside the WAV duration")

    return {
        "schema": "polymath-supervision-package-v1",
        "songId": song_id,
        "timeline": {
            "sourceDurationSeconds": round(duration, 6),
            "referenceDurationSeconds": round(duration, 6),
            "hardOuterClock": "source-audio",
        },
        "alignment": {
            "method": "exact-performance-midi-clock",
            "qualityWindows": quality_windows(duration),
            "metrics": {"alignmentCoverage": 1.0, "timingGroundTruth": True},
        },
        "provenance": {
            **(provenance or {}),
            "sourceMedia": str(source_media.resolve()),
            "sourceSha256": sha256_file(source_media),
            "labelPolicy": "exact-performance-midi",
        },
        "notes": notes,
    }


def resolve_from(base: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return (path if path.is_absolute() else base / path).resolve()


def build_from_spec(
    spec_path: Path,
    output: Path,
    ffmpeg: Path | None = None,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError(f"spec.schema must be {SCHEMA}")
    dataset_id = str(spec.get("datasetId") or "").strip()
    remote_root = str(spec.get("remoteRoot") or "").strip().rstrip("/")
    songs = spec.get("songs")
    if not dataset_id or not remote_root.startswith("/runpod-volume/training/"):
        raise ValueError("datasetId and a /runpod-volume/training/... remoteRoot are required")
    if not isinstance(songs, list) or not songs:
        raise ValueError("spec.songs must be a non-empty array")

    output.mkdir(parents=True, exist_ok=True)
    packages = output / "supervision"
    manifests = output / "manifests"
    packages.mkdir(exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    base = spec_path.parent
    for item in songs:
        if not isinstance(item, dict):
            raise ValueError("Every song entry must be an object")
        song_id = str(item.get("songId") or "").strip()
        if not song_id or song_id in seen:
            raise ValueError(f"Missing or duplicate songId: {song_id!r}")
        seen.add(song_id)
        split = str(item.get("split") or "validation").strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{song_id}: split must be train, validation, or test")
        source = resolve_from(base, item.get("sourceMedia"))
        target = resolve_from(base, item.get("targetJson"))
        if not source.is_file() or not target.is_file():
            raise ValueError(f"{song_id}: source WAV or target JSON is missing")
        duration = min(wav_duration(source), float(item.get("maximumSeconds") or 10**9))
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
        package = build_exact_supervision_package(
            song_id=song_id,
            source_media=source,
            target_payload=payload,
            duration=duration,
            provenance=item.get("provenance") if isinstance(item.get("provenance"), dict) else None,
        )
        package_path = packages / f"{song_id}.json"
        package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        rights_note = str(item.get("rightsNote") or "").strip()
        if not rights_note:
            raise ValueError(f"{song_id}: rightsNote is required")
        index_rows.append(
            {
                "songId": song_id,
                "sourceMedia": str(source),
                "supervisionPackage": str(package_path),
                "split": split,
                "instrumentFocus": "acoustic_piano",
                "rights": {"allowedForTraining": True, "note": rights_note},
            }
        )

    index_path = output / "index.json"
    index_path.write_text(json.dumps({"songs": index_rows}, indent=2) + "\n", encoding="utf-8")
    dataset_summary = build_dataset(
        index_path,
        manifests,
        clip_seconds=WINDOW_SECONDS,
        hop_seconds=WINDOW_SECONDS,
        minimum_notes=0,
        include_trusted_silence=True,
        maximum_negative_ratio=1.0,
    )

    ffmpeg_path = resolve_ffmpeg(ffmpeg)
    prepared: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        manifest = manifests / f"{split}.jsonl"
        if manifest.is_file() and manifest.stat().st_size:
            prepared[split] = prepare_manifest(
                manifest,
                output,
                ffmpeg_path,
                manifest_audio_root=remote_root,
            )
    summary = {
        "schema": "polymath-exact-reference-dataset-build-v1",
        "datasetId": dataset_id,
        "remoteRoot": remote_root,
        "researchOnly": bool(spec.get("researchOnly", True)),
        "sourceSpec": str(spec_path.resolve()),
        "dataset": dataset_summary,
        "prepared": prepared,
    }
    (output / "build-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    summary = build_from_spec(args.spec.resolve(), args.out.resolve(), args.ffmpeg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
