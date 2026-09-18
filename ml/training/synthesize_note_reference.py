"""Render note JSON into a deterministic alignment-only piano reference.

The result is deliberately not a consumer-facing piano sound.  It preserves
pitch classes, onsets, sustains, rests, and the source timeline so the chroma
aligner can compare an authored MIDI/JSON target with a full recording even
when no clean target performance was recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TAIL_SECONDS = 0.35
MINIMUM_NOTE_SECONDS = 0.03
MAXIMUM_NOTE_SECONDS = 12.0
PARTIALS = ((1.0, 1.0), (2.0, 0.34), (3.0, 0.16), (4.0, 0.08))


def normalize_notes(values: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    notes: list[dict[str, float]] = []
    for value in values:
        try:
            midi = int(round(float(value.get("midi"))))
            onset = max(0.0, float(value.get("time")))
            duration = min(
                MAXIMUM_NOTE_SECONDS,
                max(MINIMUM_NOTE_SECONDS, float(value.get("duration") or 0.25)),
            )
            velocity = min(1.0, max(0.08, float(value.get("velocity") or 0.72)))
        except (TypeError, ValueError):
            continue
        if 21 <= midi <= 108 and all(math.isfinite(item) for item in (onset, duration, velocity)):
            notes.append({
                "midi": float(midi),
                "time": onset,
                "duration": duration,
                "velocity": velocity,
            })
    return sorted(notes, key=lambda note: (note["time"], note["midi"]))


def synthesize_notes(
    notes: Iterable[dict[str, Any]],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tail_seconds: float = DEFAULT_TAIL_SECONDS,
) -> np.ndarray:
    """Return a mono float32 waveform suitable for harmonic alignment."""

    normalized = normalize_notes(notes)
    if not normalized:
        raise ValueError("The note payload contains no playable piano notes")
    if sample_rate < 4_000:
        raise ValueError("sample_rate must be at least 4000 Hz")
    ending = max(note["time"] + note["duration"] for note in normalized)
    sample_count = max(1, int(math.ceil((ending + max(0.0, tail_seconds)) * sample_rate)))
    waveform = np.zeros(sample_count, dtype=np.float32)
    nyquist = sample_rate / 2.0

    for note in normalized:
        start = int(round(note["time"] * sample_rate))
        length = max(1, int(round(note["duration"] * sample_rate)))
        end = min(sample_count, start + length)
        if end <= start:
            continue
        seconds = np.arange(end - start, dtype=np.float32) / sample_rate
        frequency = 440.0 * (2.0 ** ((note["midi"] - 69.0) / 12.0))
        tone = np.zeros_like(seconds)
        partial_weight = 0.0
        for multiplier, amplitude in PARTIALS:
            partial_frequency = frequency * multiplier
            if partial_frequency >= nyquist * 0.96:
                continue
            tone += amplitude * np.sin(2.0 * np.pi * partial_frequency * seconds)
            partial_weight += amplitude
        if partial_weight <= 0.0:
            continue
        tone /= partial_weight

        attack_seconds = min(0.018, note["duration"] * 0.24)
        release_seconds = min(0.10, note["duration"] * 0.40)
        attack = np.minimum(1.0, seconds / max(1.0 / sample_rate, attack_seconds))
        remaining = np.maximum(0.0, note["duration"] - seconds)
        release = np.minimum(1.0, remaining / max(1.0 / sample_rate, release_seconds))
        decay = np.exp(-seconds / max(0.22, note["duration"] * 1.6))
        envelope = attack * release * (0.36 + 0.64 * decay)
        waveform[start:end] += (0.19 + 0.31 * note["velocity"]) * tone * envelope

    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        raise ValueError("The synthesized reference is silent")
    waveform *= min(1.0, 0.86 / peak)
    return waveform.astype(np.float32, copy=False)


def write_pcm16_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2")
    temporary = path.with_name(f"{path.name}.tmp")
    with wave.open(str(temporary), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm.tobytes())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render note JSON to a deterministic alignment-only piano WAV."
    )
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    payload = json.loads(args.notes.read_text(encoding="utf-8-sig"))
    notes = normalize_notes(payload.get("notes") or [])
    waveform = synthesize_notes(notes, sample_rate=args.sample_rate)
    write_pcm16_wav(args.output, waveform, args.sample_rate)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "notes": len(notes),
        "sampleRate": args.sample_rate,
        "durationSeconds": round(len(waveform) / args.sample_rate, 6),
        "purpose": "alignment-only; not consumer playback",
    }))


if __name__ == "__main__":
    main()
