"""Prepare synchronized source-separation stems for Polymath transcription.

The RunPod transcription worker expects a mono 16 kHz WAV.  Source separators
normally emit stereo 44.1 kHz WAVs.  This helper converts each view with the
same deterministic resampler and refuses inputs whose clocks disagree, so a
vocal or accompaniment pass cannot silently drift away from the full mix.

This script intentionally lives outside the production request path.  It is a
research/data-preparation tool and requires ``soundfile``, ``torch`` and
``julius`` (all installed by Demucs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import julius
import soundfile as sf
import torch


TARGET_SAMPLE_RATE = 16_000
MAX_CLOCK_SKEW_SECONDS = 0.002


def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError(f"Audio view is empty: {path}")
    waveform = torch.from_numpy(samples.T).mean(dim=0, keepdim=True)
    return waveform, int(sample_rate)


def _duration_seconds(waveform: torch.Tensor, sample_rate: int) -> float:
    return float(waveform.shape[-1]) / float(sample_rate)


def prepare_views(
    inputs: list[Path],
    output_dir: Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("At least two synchronized audio views are required.")
    if target_sample_rate <= 0:
        raise ValueError("The target sample rate must be positive.")

    loaded: list[tuple[Path, torch.Tensor, int]] = []
    for path in inputs:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        waveform, sample_rate = _load_mono(resolved)
        loaded.append((resolved, waveform, sample_rate))

    durations = [_duration_seconds(waveform, sample_rate) for _, waveform, sample_rate in loaded]
    clock_skew = max(durations) - min(durations)
    if clock_skew > MAX_CLOCK_SKEW_SECONDS:
        raise ValueError(
            "Audio views do not share one clock: "
            f"duration skew is {clock_skew:.6f}s (limit {MAX_CLOCK_SKEW_SECONDS:.6f}s)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    expected_frames: int | None = None
    for source, waveform, sample_rate in loaded:
        if sample_rate != target_sample_rate:
            waveform = julius.resample_frac(
                waveform,
                sample_rate,
                target_sample_rate,
            )
        waveform = waveform.clamp(-1.0, 1.0)
        if expected_frames is None:
            expected_frames = int(waveform.shape[-1])
        elif abs(int(waveform.shape[-1]) - expected_frames) > 1:
            raise ValueError("Prepared audio views drifted by more than one sample.")

        destination = output_dir / f"{source.stem}-16k-mono.wav"
        sf.write(
            destination,
            waveform.squeeze(0).cpu().numpy(),
            target_sample_rate,
            subtype="PCM_16",
        )
        outputs.append(
            {
                "source": str(source),
                "output": str(destination.resolve()),
                "inputSampleRate": sample_rate,
                "outputSampleRate": target_sample_rate,
                "frames": int(waveform.shape[-1]),
                "durationSeconds": round(
                    _duration_seconds(waveform, target_sample_rate), 6
                ),
            }
        )

    report = {
        "schema": "polymath-synchronized-audio-views-v1",
        "clockSkewSeconds": round(clock_skew, 9),
        "views": outputs,
    }
    report_path = output_dir / "prepared-views.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    args = parser.parse_args()
    report = prepare_views(
        [Path(value) for value in args.input],
        Path(args.output_dir).resolve(),
        target_sample_rate=args.sample_rate,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
