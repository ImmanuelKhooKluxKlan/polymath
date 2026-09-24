"""Prepare deterministic primary and shifted inference clips from one PCM WAV.

This utility is deliberately label-free.  It exists for opened-song transfer
and listening checks, not for validation metrics or model selection.  The two
passes mirror the frozen Phase 94 decoder: primary windows start at 0 seconds
and shifted windows start halfway through the first window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path, PurePosixPath
from typing import Any


class UnlabeledInferenceError(RuntimeError):
    """Raised when a reproducible inference pack cannot be produced."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(duration: float, window_seconds: float, offset_seconds: float) -> list[float]:
    if duration <= 0:
        raise UnlabeledInferenceError("Audio duration must be positive")
    if window_seconds <= 0:
        raise UnlabeledInferenceError("Window size must be positive")
    if not 0 <= offset_seconds < window_seconds:
        raise UnlabeledInferenceError("Offset must be within [0, window size)")
    starts: list[float] = []
    start = offset_seconds
    while start < duration - 1e-9:
        starts.append(round(start, 6))
        start += window_seconds
    return starts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_pass(
    *,
    source: Path,
    raw_frames: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
    frame_count: int,
    song_id: str,
    output_root: Path,
    remote_root: PurePosixPath,
    pass_name: str,
    window_seconds: float,
    offset_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_root = output_root / pass_name / "audio"
    audio_root.mkdir(parents=True, exist_ok=False)
    duration = frame_count / sample_rate
    block_align = channels * sample_width
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for index, start in enumerate(window_starts(duration, window_seconds, offset_seconds)):
        end = min(duration, start + window_seconds)
        start_frame = round(start * sample_rate)
        end_frame = min(frame_count, round(end * sample_rate))
        clip_id = f"{song_id}-{pass_name}-{index:05d}"
        filename = f"{clip_id}.wav"
        destination = audio_root / filename
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            writer.writeframes(raw_frames[start_frame * block_align:end_frame * block_align])
        clip_duration = (end_frame - start_frame) / sample_rate
        records.append({
            "schema": "polymath-unlabeled-inference-clip-v1",
            "clipId": clip_id,
            "songId": song_id,
            "split": "opened-transfer",
            "sourceMedia": str(source.resolve()),
            "sourceStart": round(start_frame / sample_rate, 6),
            "durationSeconds": round(clip_duration, 6),
            "sampleRate": sample_rate,
            "instrumentFocus": "acoustic_piano",
            "notes": [],
            "labelsPresent": False,
            "targetState": "unlabeled-inference-only",
            "audioClip": str(remote_root / pass_name / "audio" / filename),
            "inferencePass": {
                "name": pass_name,
                "offsetSeconds": offset_seconds,
                "windowSeconds": window_seconds,
            },
        })
        files.append({
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "frames": end_frame - start_frame,
        })
    manifest = output_root / pass_name / "manifest.jsonl"
    write_jsonl(manifest, records)
    files.append({
        "path": str(manifest.resolve()),
        "sha256": sha256_file(manifest),
        "bytes": manifest.stat().st_size,
    })
    return records, files


def prepare_unlabeled_inference(
    source: Path,
    output_root: Path,
    remote_root: PurePosixPath,
    *,
    song_id: str,
    window_seconds: float = 5.0,
    overlap_offset_seconds: float = 2.5,
) -> dict[str, Any]:
    source = source.resolve()
    output_root = output_root.resolve()
    if not source.is_file():
        raise UnlabeledInferenceError(f"Source WAV does not exist: {source}")
    if output_root.exists():
        raise UnlabeledInferenceError(f"Refusing to overwrite inference pack: {output_root}")
    if not song_id.strip():
        raise UnlabeledInferenceError("song-id cannot be empty")
    if not 0 < overlap_offset_seconds < window_seconds:
        raise UnlabeledInferenceError("Overlap offset must be inside the window")
    with wave.open(str(source), "rb") as reader:
        if reader.getcomptype() != "NONE":
            raise UnlabeledInferenceError("Source must be an uncompressed PCM WAV")
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        raw_frames = reader.readframes(frame_count)
    if sample_rate != 16000 or channels != 1 or sample_width != 2:
        raise UnlabeledInferenceError(
            "Source must be mono 16 kHz, 16-bit PCM so inference input is immutable"
        )
    output_root.mkdir(parents=True)
    pass_receipts: dict[str, Any] = {}
    for pass_name, offset in (("primary", 0.0), ("overlap", overlap_offset_seconds)):
        records, files = prepare_pass(
            source=source,
            raw_frames=raw_frames,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frame_count=frame_count,
            song_id=song_id,
            output_root=output_root,
            remote_root=remote_root,
            pass_name=pass_name,
            window_seconds=window_seconds,
            offset_seconds=offset,
        )
        pass_receipts[pass_name] = {
            "offsetSeconds": offset,
            "clips": len(records),
            "manifestSha256": files[-1]["sha256"],
            "files": files,
        }
    receipt = {
        "schema": "polymath-unlabeled-inference-pack-v1",
        "purpose": "opened-song transfer and listening only",
        "labelsPresent": False,
        "accuracyEvidenceAllowed": False,
        "songId": song_id,
        "source": str(source),
        "sourceSha256": sha256_file(source),
        "durationSeconds": round(frame_count / sample_rate, 6),
        "sampleRate": sample_rate,
        "windowSeconds": window_seconds,
        "overlapOffsetSeconds": overlap_offset_seconds,
        "remoteRoot": str(remote_root),
        "passes": pass_receipts,
    }
    receipt_path = output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap-offset-seconds", type=float, default=2.5)
    args = parser.parse_args()
    receipt = prepare_unlabeled_inference(
        args.source,
        args.out_root,
        PurePosixPath(args.remote_root),
        song_id=args.song_id,
        window_seconds=args.window_seconds,
        overlap_offset_seconds=args.overlap_offset_seconds,
    )
    print(json.dumps({
        "receipt": str((args.out_root.resolve() / "receipt.json")),
        "sourceSha256": receipt["sourceSha256"],
        "primaryClips": receipt["passes"]["primary"]["clips"],
        "overlapClips": receipt["passes"]["overlap"]["clips"],
        "accuracyEvidenceAllowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
