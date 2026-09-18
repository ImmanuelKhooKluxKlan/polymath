"""Create fixed-gain piano/full-mix curriculum WAVs with FFmpeg.

The output is deliberately not loudness-normalized per file.  Each ratio uses
the same linear gain rule so the network cannot learn from mastering changes.
This is a private research data-preparation utility, not an inference step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any


def parse_ratios(value: str) -> list[float]:
    ratios = sorted(
        {float(item.strip()) for item in value.split(",") if item.strip()},
        reverse=True,
    )
    if not ratios or any(not math.isfinite(item) or item < 0 or item > 1 for item in ratios):
        raise argparse.ArgumentTypeError("Ratios must be finite values from 0 through 1")
    return ratios


def ratio_label(ratio: float) -> str:
    return f"p{round(ratio * 100):03d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sampleRate": handle.getframerate(),
            "sampleWidthBytes": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "seconds": round(handle.getnframes() / handle.getframerate(), 6),
        }


def create_mix(
    *,
    ffmpeg: Path,
    piano: Path,
    source: Path,
    output: Path,
    piano_ratio: float,
) -> dict[str, Any]:
    source_ratio = 1.0 - piano_ratio
    filter_graph = (
        f"[0:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"volume={piano_ratio:.8f}[p];"
        f"[1:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"volume={source_ratio:.8f}[s];"
        "[p][s]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95:attack=5:release=50[out]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial" + output.suffix)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(piano),
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(completed.stderr.strip() or f"FFmpeg exited {completed.returncode}")
    temporary.replace(output)
    return {
        "file": str(output),
        "sha256": sha256(output),
        "pianoRatio": piano_ratio,
        "sourceRatio": source_ratio,
        "perFileNormalization": False,
        **inspect_wav(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piano", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("0.75,0.5,0.25"))
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()

    piano = args.piano.resolve()
    source = args.source.resolve()
    ffmpeg = args.ffmpeg.resolve()
    for path, label in ((piano, "piano"), (source, "source"), (ffmpeg, "ffmpeg")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file is missing: {path}")
    rows = [
        create_mix(
            ffmpeg=ffmpeg,
            piano=piano,
            source=source,
            output=args.out.resolve() / f"{args.name}-{ratio_label(ratio)}.wav",
            piano_ratio=ratio,
        )
        for ratio in args.ratios
    ]
    report = {
        "schema": "polymath-piano-curriculum-audio-v1",
        "piano": str(piano),
        "source": str(source),
        "mixes": rows,
        "warning": "Private research augmentation; licensing and held-out validation remain mandatory.",
    }
    report_path = args.out.resolve() / f"{args.name}-curriculum-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
