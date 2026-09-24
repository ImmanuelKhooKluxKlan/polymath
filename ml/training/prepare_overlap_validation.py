"""Build a shifted five-second validation pass for boundary-safe decoding.

The primary Phase94 evaluation uses windows starting at 0, 5, 10, ... seconds.
This development-only manifest starts at a frozen offset (normally 2.5 s), so
events near a primary cut are heard in the middle of another window. It never
opens the sealed split and never changes reference note identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from ml.training.evaluate_checkpoint import stitch_clip_notes
from ml.training.train_muscriptor_piano import read_jsonl


class OverlapManifestError(RuntimeError):
    """Raised when a shifted pass cannot be reproduced safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def offset_starts(duration: float, window_seconds: float, offset_seconds: float) -> list[float]:
    if duration <= 0:
        raise OverlapManifestError("Song duration must be positive")
    if not 0 < window_seconds <= 10:
        raise OverlapManifestError("Window size must be within (0, 10] seconds")
    if not 0 < offset_seconds < window_seconds:
        raise OverlapManifestError("Offset must be inside the window")
    starts: list[float] = []
    start = offset_seconds
    while start < duration - 1e-9:
        starts.append(round(start, 6))
        start += window_seconds
    return starts


def clip_song_notes(
    notes: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for source in notes:
        onset = float(source["time"])
        ending = onset + float(source["duration"])
        if onset >= end or ending <= start:
            continue
        local_start = max(start, onset) - start
        local_end = min(end, ending) - start
        # Keep even a tiny edge fragment. Its adjacent shifted clip carries the
        # continuation, and stitching both pieces restores the true onset. A
        # 10 ms filter here would move valid attacks onto every shifted cut.
        if local_end - local_start <= 0:
            continue
        clipped.append({
            "midi": int(source["midi"]),
            "time": round(local_start, 6),
            "duration": round(local_end - local_start, 6),
            "velocity": round(float(source.get("velocity", 0.75)), 4),
            "instrument": str(source.get("instrument") or "acoustic_piano"),
            "continuedFromPreviousClip": onset < start,
            "continuesIntoNextClip": ending > end,
            "qualityWindowId": f"overlap-{start:.3f}",
        })
    return sorted(clipped, key=lambda note: (note["time"], note["midi"]))


def read_song_audio(records: list[dict[str, Any]]) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    segments = []
    sample_rate: int | None = None
    for record in sorted(records, key=lambda row: float(row.get("sourceStart") or 0)):
        path = Path(str(record.get("localAudioSource") or ""))
        if not path.is_file():
            raise OverlapManifestError(f"Prepared source clip is missing: {path}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = int(rate)
        if int(rate) != sample_rate:
            raise OverlapManifestError("Prepared source clips use different sample rates")
        mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
        expected = round(float(record["durationSeconds"]) * sample_rate)
        if len(mono) < expected:
            shortfall = expected - len(mono)
            if shortfall > round(0.01 * sample_rate):
                raise OverlapManifestError(
                    f"Prepared source clip is shorter than its receipt: {path}"
                )
            # Container/sample rounding can leave the final receipt a fraction
            # of a millisecond longer than its WAV. Zero-pad only that bounded
            # discrepancy; anything above 10 ms still fails closed.
            mono = np.pad(mono, (0, shortfall))
        segments.append(mono[:expected])
    if sample_rate is None or not segments:
        raise OverlapManifestError("Song has no prepared source audio")
    return np.concatenate(segments), sample_rate


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_overlap_manifest(
    manifest_path: Path,
    output_root: Path,
    remote_root: PurePosixPath,
    *,
    offset_seconds: float = 2.5,
    window_seconds: float = 5.0,
) -> dict[str, Any]:
    import soundfile as sf

    records = read_jsonl(manifest_path)
    if any(str(record.get("split") or "") != "validation" for record in records):
        raise OverlapManifestError("Only the already-opened validation split is allowed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("songId") or "unknown")].append(record)
    reference_clips = [list(record.get("notes") or []) for record in records]
    reference_songs, reference_merges = stitch_clip_notes(
        records, reference_clips, reference=True,
    )

    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    song_receipts: list[dict[str, Any]] = []
    for song_id in sorted(grouped):
        song_records = sorted(
            grouped[song_id], key=lambda row: float(row.get("sourceStart") or 0),
        )
        signal, sample_rate = read_song_audio(song_records)
        duration = sum(float(record["durationSeconds"]) for record in song_records)
        expected_samples = round(duration * sample_rate)
        signal = signal[:expected_samples]
        notes = reference_songs.get(song_id, [])
        song_files: list[dict[str, Any]] = []
        for index, start in enumerate(offset_starts(duration, window_seconds, offset_seconds)):
            end = min(duration, start + window_seconds)
            start_sample = round(start * sample_rate)
            end_sample = round(end * sample_rate)
            filename = f"{song_id}-overlap-{index:05d}.wav"
            local_audio = audio_root / filename
            if local_audio.exists():
                raise OverlapManifestError(f"Refusing to overwrite shifted audio: {local_audio}")
            sf.write(local_audio, signal[start_sample:end_sample], sample_rate, subtype="PCM_16")
            clip_notes = clip_song_notes(notes, start, end)
            is_negative = not clip_notes
            output_records.append({
                "schema": "polymath-training-clip-v1",
                "clipId": f"{song_id}-overlap-{index:05d}",
                "songId": song_id,
                "split": "validation",
                "sourceMedia": song_records[0].get("sourceMedia"),
                "sourceStart": start,
                "durationSeconds": round(end - start, 6),
                "sampleRate": sample_rate,
                "instrumentFocus": "acoustic_piano",
                "notes": clip_notes,
                "targetState": "reviewed-silence" if is_negative else "notes",
                "isNegativeExample": is_negative,
                "exampleWeight": 1.0,
                "qualityWindowIds": [f"overlap-{start:.3f}"],
                "audioClip": str(remote_root / "audio" / filename),
                "localAudioSource": str(local_audio.resolve()),
                "overlapPass": {
                    "offsetSeconds": offset_seconds,
                    "windowSeconds": window_seconds,
                    "sourceManifest": str(manifest_path.resolve()),
                },
            })
            song_files.append({
                "path": str(local_audio.resolve()),
                "sha256": sha256_file(local_audio),
                "bytes": local_audio.stat().st_size,
            })
        song_receipts.append({
            "songId": song_id,
            "durationSeconds": round(duration, 6),
            "sampleRate": sample_rate,
            "shiftedClips": len(song_files),
            "files": song_files,
        })

    manifest_out = output_root / "prepared-validation-overlap.jsonl"
    if manifest_out.exists():
        raise OverlapManifestError(f"Refusing to overwrite shifted manifest: {manifest_out}")
    write_jsonl(manifest_out, output_records)
    receipt = {
        "schema": "polymath-overlap-validation-preparation-v1",
        "sourceManifest": str(manifest_path.resolve()),
        "sourceManifestSha256": sha256_file(manifest_path),
        "outputManifest": str(manifest_out.resolve()),
        "outputManifestSha256": sha256_file(manifest_out),
        "offsetSeconds": offset_seconds,
        "windowSeconds": window_seconds,
        "songs": len(grouped),
        "clips": len(output_records),
        "referenceContinuationMerges": reference_merges,
        "sealedTestOpened": False,
        "songReceipts": song_receipts,
    }
    receipt_path = output_root / "preparation-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--offset-seconds", type=float, default=2.5)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    args = parser.parse_args()
    output_root = args.out_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result = prepare_overlap_manifest(
        args.manifest.resolve(),
        output_root,
        PurePosixPath(args.remote_root),
        offset_seconds=args.offset_seconds,
        window_seconds=args.window_seconds,
    )
    print(json.dumps({
        "outputManifest": result["outputManifest"],
        "outputManifestSha256": result["outputManifestSha256"],
        "songs": result["songs"],
        "clips": result["clips"],
    }, indent=2))


if __name__ == "__main__":
    main()
