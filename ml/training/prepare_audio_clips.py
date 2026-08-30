"""Render manifest clips as mono 16 kHz PCM WAV files using FFmpeg."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class ClipPreparationError(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClipPreparationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ClipPreparationError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def resolve_ffmpeg(explicit: Path | None) -> Path:
    if explicit:
        resolved = explicit.resolve()
        if not resolved.is_file():
            raise ClipPreparationError(f"FFmpeg does not exist: {resolved}")
        return resolved
    command = shutil.which("ffmpeg")
    if command:
        return Path(command).resolve()
    raise ClipPreparationError(
        "FFmpeg was not found. Pass --ffmpeg; on this project use the executable "
        "returned by: node -e \"process.stdout.write(require('./server/node_modules/ffmpeg-static'))\""
    )


def render_clip(ffmpeg: Path, record: dict[str, Any], destination: Path) -> None:
    source = Path(str(record.get("sourceMedia") or "")).resolve()
    if not source.is_file():
        raise ClipPreparationError(f"Missing source media for {record.get('clipId')}: {source}")
    start = float(record.get("sourceStart", 0))
    duration = float(record.get("durationSeconds", 0))
    if start < 0 or duration <= 0:
        raise ClipPreparationError(f"Invalid clip time for {record.get('clipId')}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 44:
        temporary.unlink(missing_ok=True)
        raise ClipPreparationError(
            f"FFmpeg failed for {record.get('clipId')}: {completed.stderr.strip()[-1000:]}"
        )
    temporary.replace(destination)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_manifest(
    manifest: Path,
    output_directory: Path,
    ffmpeg: Path,
    overwrite: bool = False,
    manifest_audio_root: str | None = None,
) -> dict[str, Any]:
    records = load_jsonl(manifest)
    split = manifest.stem
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        clip_id = str(record.get("clipId") or f"clip-{index:06d}")
        destination = output_directory / "audio" / split / f"{clip_id}.wav"
        if overwrite or not destination.is_file() or destination.stat().st_size <= 44:
            render_clip(ffmpeg, record, destination)
        manifest_audio_path = (
            str(PurePosixPath(manifest_audio_root) / "audio" / split / destination.name)
            if manifest_audio_root
            else str(destination.resolve())
        )
        prepared.append({**record, "audioClip": manifest_audio_path})
        print(f"[{index}/{len(records)}] {clip_id}")
    prepared_manifest = output_directory / f"prepared-{split}.jsonl"
    write_jsonl(prepared_manifest, prepared)
    return {
        "sourceManifest": str(manifest.resolve()),
        "preparedManifest": str(prepared_manifest.resolve()),
        "clips": len(prepared),
        "sampleRate": 16000,
        "channels": 1,
        "encoding": "PCM signed 16-bit little-endian",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-audio-root",
        help=(
            "Write audioClip paths relative to this runtime root, while still rendering files "
            "under --out. Example: /runpod-volume/training/phase-2-v001"
        ),
    )
    args = parser.parse_args()
    try:
        summary = prepare_manifest(
            args.manifest.resolve(), args.out.resolve(), resolve_ffmpeg(args.ffmpeg),
            args.overwrite, args.manifest_audio_root,
        )
    except ClipPreparationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
