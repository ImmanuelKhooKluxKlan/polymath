"""Compose prepared manifests while reusing existing RunPod audio objects.

Unlike the staging composer, this utility does not copy WAV files.  Every input
record must already name an absolute ``/runpod-volume/...`` audio path.  It is
intended for cross-validation datasets whose audio has been uploaded once but
whose train/validation manifests change between experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record is not an object")
        records.append(value)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compose(
    train_manifests: Iterable[Path],
    validation_manifests: Iterable[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    inputs = {
        "train": [path.resolve() for path in train_manifests],
        "validation": [path.resolve() for path in validation_manifests],
    }
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen_clips: set[str] = set()
    song_splits: dict[str, str] = {}
    source_rows: list[dict[str, Any]] = []
    for split, manifests in inputs.items():
        if not manifests:
            raise ValueError(f"At least one {split} manifest is required")
        for manifest in manifests:
            if not manifest.is_file():
                raise ValueError(f"Prepared manifest is missing: {manifest}")
            records = read_jsonl(manifest)
            if not records:
                raise ValueError(f"Prepared manifest is empty: {manifest}")
            source_rows.append(
                {
                    "split": split,
                    "manifest": str(manifest),
                    "sha256": sha256(manifest),
                    "records": len(records),
                }
            )
            for record in records:
                clip_id = str(record.get("clipId") or "").strip()
                song_id = str(record.get("songId") or "").strip()
                audio_clip = str(record.get("audioClip") or "").replace("\\", "/")
                if not clip_id or not song_id:
                    raise ValueError(f"{manifest}: clipId and songId are required")
                if clip_id in seen_clips:
                    raise ValueError(f"Duplicate clipId: {clip_id}")
                if not audio_clip.startswith("/runpod-volume/"):
                    raise ValueError(
                        f"{clip_id}: audioClip must be an existing /runpod-volume path"
                    )
                previous_split = song_splits.get(song_id)
                if previous_split and previous_split != split:
                    raise ValueError(
                        f"{song_id}: song leakage between {previous_split} and {split}"
                    )
                seen_clips.add(clip_id)
                song_splits[song_id] = split
                local_audio = str(record.get("localAudioSource") or "").strip()
                inferred_local = manifest.parent / "audio" / split / f"{clip_id}.wav"
                if not local_audio and inferred_local.is_file():
                    local_audio = str(inferred_local.resolve())
                composed = {**record, "split": split, "audioClip": audio_clip}
                if local_audio:
                    composed["localAudioSource"] = local_audio
                output[split].append(composed)
    summary = {
        "schema": "polymath-linked-prepared-dataset-composition-v1",
        "audioPolicy": "Reuse immutable absolute RunPod-volume paths; no WAV files copied.",
        "sources": source_rows,
        "splits": {
            split: {
                "clips": len(records),
                "songs": sorted({str(record["songId"]) for record in records}),
                "audioSeconds": round(
                    sum(float(record.get("durationSeconds") or 0.0) for record in records),
                    6,
                ),
            }
            for split, records in output.items()
        },
    }
    return output, summary


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output, summary = compose(args.train_manifest, args.validation_manifest)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "prepared-train.jsonl", output["train"])
    write_jsonl(destination / "prepared-validation.jsonl", output["validation"])
    (destination / "composition-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
