"""Render MIDI-like JSON notes into a deterministic alignment reference WAV.

This is not a production piano synthesizer.  It deliberately uses a neutral,
harmonic tone so audio-chroma alignment can recover the musical clock when an
approved MIDI/JSON target exists but its original reference recording does not.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def finite_number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def normalized_notes(payload: dict[str, Any]) -> list[dict[str, float | int]]:
    notes: list[dict[str, float | int]] = []
    for item in payload.get("notes", []):
        midi = int(round(finite_number(item.get("midi", item.get("pitch")), -1)))
        start = finite_number(item.get("time", item.get("startTime")), -1.0)
        duration = max(0.04, finite_number(item.get("duration"), 0.30))
        velocity = min(1.0, max(0.08, finite_number(item.get("velocity"), 0.72)))
        if 21 <= midi <= 108 and start >= 0:
            notes.append(
                {"midi": midi, "time": start, "duration": min(duration, 8.0), "velocity": velocity}
            )
    return sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))


def render_notes(
    notes: list[dict[str, float | int]],
    *,
    sample_rate: int = 16000,
    release_seconds: float = 0.16,
) -> np.ndarray:
    if not notes:
        raise ValueError("The reference contains no playable piano notes")
    duration = max(float(note["time"]) + float(note["duration"]) for note in notes)
    samples = np.zeros(int(math.ceil((duration + release_seconds + 0.2) * sample_rate)), dtype=np.float32)
    harmonic_weights = ((1.0, 1.0), (2.0, 0.38), (3.0, 0.17), (4.0, 0.08))
    for note in notes:
        start_index = int(round(float(note["time"]) * sample_rate))
        sounding_seconds = float(note["duration"]) + release_seconds
        frame_count = max(1, int(round(sounding_seconds * sample_rate)))
        end_index = min(len(samples), start_index + frame_count)
        frame_count = end_index - start_index
        if frame_count <= 0:
            continue
        time = np.arange(frame_count, dtype=np.float32) / sample_rate
        attack = np.minimum(1.0, time / 0.008)
        key_end = float(note["duration"])
        release = np.ones_like(time)
        release_mask = time > key_end
        release[release_mask] = np.maximum(
            0.0,
            1.0 - (time[release_mask] - key_end) / max(0.01, release_seconds),
        )
        decay = 0.72 + 0.28 * np.exp(-time * 2.1)
        envelope = attack * release * decay
        frequency = 440.0 * 2.0 ** ((int(note["midi"]) - 69) / 12.0)
        tone = np.zeros(frame_count, dtype=np.float32)
        for harmonic, weight in harmonic_weights:
            if frequency * harmonic < sample_rate * 0.47:
                tone += weight * np.sin(2.0 * math.pi * frequency * harmonic * time)
        samples[start_index:end_index] += (
            tone * envelope * float(note["velocity"]) * 0.16
        ).astype(np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples *= min(1.0, 0.92 / peak)
    return samples


def write_pcm_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm.tobytes())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    if not 8000 <= args.sample_rate <= 48000:
        raise ValueError("sample rate must be between 8000 and 48000")
    payload = json.loads(args.notes.read_text(encoding="utf-8-sig"))
    notes = normalized_notes(payload)
    samples = render_notes(notes, sample_rate=args.sample_rate)
    write_pcm_wav(args.out, samples, args.sample_rate)
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "notes": len(notes),
                "durationSeconds": round(len(samples) / args.sample_rate, 4),
                "sampleRate": args.sample_rate,
            }
        )
    )


if __name__ == "__main__":
    main()
