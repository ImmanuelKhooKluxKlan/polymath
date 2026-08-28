"""Rewrite prepared audio paths for a mounted GPU training volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


class RelocationError(RuntimeError):
    pass


def relocate_manifest(source: Path, destination: Path, audio_root: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    root = PurePosixPath(audio_root)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            clip_id = str(record.get("clipId") or "").strip()
            split = str(record.get("split") or "").strip()
            local_audio = Path(str(record.get("audioClip") or ""))
            if not clip_id or split not in {"train", "validation", "test"}:
                raise RelocationError(f"{source}:{line_number}: invalid clipId or split")
            if not local_audio.is_file() or local_audio.stat().st_size <= 44:
                raise RelocationError(f"{source}:{line_number}: local audio clip is missing")
            records.append({
                **record,
                "audioClip": str(root / "audio" / split / f"{clip_id}.wav"),
                "localAudioSource": str(local_audio.resolve()),
            })
    if not records:
        raise RelocationError(f"Prepared manifest is empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "audioRoot": str(root),
        "clips": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio-root", required=True)
    args = parser.parse_args()
    try:
        result = relocate_manifest(
            args.manifest.resolve(),
            args.out.resolve(),
            args.audio_root,
        )
    except (OSError, json.JSONDecodeError, RelocationError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
