"""Render arranged piano JSON with one fixed, non-normalizing signal chain.

The renderer is for controlled research listening, not final mastering.  Every
candidate receives the same Iowa 88-key samples, velocity curve, envelope,
master gain, and soft limiter.  Crucially, files are never peak-normalized one
by one; a denser or harsher arrangement cannot gain an artificial loudness
advantage after rendering.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 44_100
RELEASE_SECONDS = 0.72
ATTACK_SECONDS = 0.006
MASTER_GAIN = 0.82
LIMITER_DRIVE = 0.88
NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
PEDAL_RETRIGGER_GAP_SECONDS = 0.012
TARGET_RMS_POINTS = (
    (21, 0.078),
    (33, 0.076),
    (45, 0.073),
    (60, 0.069),
    (72, 0.064),
    (84, 0.057),
    (96, 0.049),
    (108, 0.041),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midi_to_sample_name(midi: int) -> str:
    if midi < 21 or midi > 108:
        raise ValueError(f"MIDI {midi} is outside the 88-key piano range")
    return f"{NOTE_NAMES_FLAT[midi % 12]}{midi // 12 - 1}.wav"


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def normalize_render_notes(
    payload: dict[str, Any], *, training_eligible_only: bool = False
) -> list[dict[str, float | int]]:
    notes: list[dict[str, float | int]] = []
    for item in payload.get("notes", []):
        if not isinstance(item, dict):
            continue
        if training_eligible_only and not bool(item.get("trainingEligible")):
            continue
        midi = int(round(_finite(item.get("midi", item.get("pitch")), -1)))
        time = _finite(item.get("time", item.get("startTime", item.get("start"))), -1)
        duration = _finite(item.get("audioDuration", item.get("duration")), 0.2)
        velocity = _finite(item.get("velocity"), 0.75)
        performance_gain = _finite(item.get("performanceGain"), 1.0)
        if not 21 <= midi <= 108 or time < 0 or duration <= 0:
            continue
        notes.append(
            {
                "midi": midi,
                "time": time,
                "duration": min(60.0, max(0.01, duration)),
                "velocity": min(1.0, max(0.01, velocity)),
                "performanceGain": min(1.5, max(0.25, performance_gain)),
            }
        )
    return sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))


def apply_sustain_pedal_to_render_notes(
    notes: list[dict[str, float | int]], payload: dict[str, Any]
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Extend a released key to pedal-up, like the browser piano engine.

    ``audioDuration`` is the physical key hold. If the damper pedal is down at
    key-up, the string continues until pedal-up or a same-key retrigger. The
    listening renderer must model that or sustained endings sound falsely
    weak compared with the actual website and source performance.
    """

    raw_events = payload.get("pedals")
    if not isinstance(raw_events, list) or not raw_events:
        raw_events = payload.get("pedalEvents")
    events: list[tuple[float, bool]] = []
    for event in raw_events or []:
        if not isinstance(event, dict):
            continue
        time = _finite(event.get("time"), -1.0)
        if time >= 0:
            events.append((time, bool(event.get("down"))))
    events.sort(key=lambda event: event[0])
    if not events:
        return [dict(note) for note in notes], {
            "pedalAware": False,
            "pedalEvents": 0,
            "pedalExtendedNotes": 0,
            "maximumPedalExtensionSeconds": 0.0,
        }

    event_times = [event[0] for event in events]
    next_up_after: list[float | None] = [None] * len(events)
    next_up: float | None = None
    for index in range(len(events) - 1, -1, -1):
        if not events[index][1]:
            next_up = events[index][0]
        next_up_after[index] = next_up

    pitch_times: dict[int, list[float]] = {}
    for note in notes:
        pitch_times.setdefault(int(note["midi"]), []).append(float(note["time"]))

    output: list[dict[str, float | int]] = []
    extended = 0
    maximum_extension = 0.0
    for source in notes:
        note = dict(source)
        start = float(note["time"])
        key_up = start + float(note["duration"])
        event_index = bisect.bisect_right(event_times, key_up) - 1
        if event_index >= 0 and events[event_index][1]:
            pedal_up = next_up_after[event_index]
            if pedal_up is not None and pedal_up > key_up:
                same_pitch_times = pitch_times[int(note["midi"])]
                next_note_index = bisect.bisect_right(same_pitch_times, start + 1e-9)
                audible_end = pedal_up
                if next_note_index < len(same_pitch_times):
                    audible_end = min(
                        audible_end,
                        max(
                            key_up,
                            same_pitch_times[next_note_index]
                            - PEDAL_RETRIGGER_GAP_SECONDS,
                        ),
                    )
                if audible_end > key_up + 1e-9:
                    extension = audible_end - key_up
                    note["duration"] = min(60.0, audible_end - start)
                    extended += 1
                    maximum_extension = max(maximum_extension, extension)
        output.append(note)

    return output, {
        "pedalAware": True,
        "pedalEvents": len(events),
        "pedalExtendedNotes": extended,
        "maximumPedalExtensionSeconds": round(maximum_extension, 6),
    }


def read_pcm16_stereo(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        compression = handle.getcomptype()
        raw = handle.readframes(frames)
    if sample_width != 2 or compression != "NONE":
        raise ValueError(f"Unsupported sample encoding in {path}")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz sample, got {sample_rate}: {path}")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 1:
        return np.repeat(values[:, None], 2, axis=1)
    if channels == 2:
        return values.reshape(-1, 2)
    raise ValueError(f"Expected mono or stereo sample, got {channels} channels: {path}")


def target_rms_for_midi(midi: int) -> float:
    """Mirror the browser piano engine's register-shaped RMS target."""

    if midi <= TARGET_RMS_POINTS[0][0]:
        return TARGET_RMS_POINTS[0][1]
    if midi >= TARGET_RMS_POINTS[-1][0]:
        return TARGET_RMS_POINTS[-1][1]
    for (left_midi, left_rms), (right_midi, right_rms) in zip(
        TARGET_RMS_POINTS, TARGET_RMS_POINTS[1:]
    ):
        if left_midi <= midi <= right_midi:
            fraction = (midi - left_midi) / (right_midi - left_midi)
            return left_rms + (right_rms - left_rms) * fraction
    return TARGET_RMS_POINTS[-1][1]


def browser_sample_gain(sample: np.ndarray, midi: int) -> float:
    """Apply the same bounded per-key normalization used by the website."""

    start = min(len(sample), round(0.02 * SAMPLE_RATE))
    end = min(len(sample), round(1.2 * SAMPLE_RATE))
    window = sample[start:end]
    if not window.size:
        rms = 0.05
        peak = 0.05
    else:
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
        peak = float(np.max(np.abs(window)))
    rms_gain = target_rms_for_midi(midi) / max(rms, 0.0001)
    peak_gain_limit = 0.92 / max(peak, 0.0001)
    return min(1.95, max(0.35, min(rms_gain, peak_gain_limit)))


def write_pcm16_stereo(path: Path, audio: np.ndarray) -> None:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("audio must have shape (frames, 2)")
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)


def note_envelope(frames: int, hold_frames: int, release_frames: int) -> np.ndarray:
    envelope = np.ones(frames, dtype=np.float32)
    attack_frames = min(frames, max(1, round(ATTACK_SECONDS * SAMPLE_RATE)))
    envelope[:attack_frames] = np.linspace(0.0, 1.0, attack_frames, dtype=np.float32)
    release_start = min(frames, max(attack_frames, hold_frames))
    release_end = min(frames, release_start + release_frames)
    if release_end > release_start:
        phase = np.linspace(0.0, math.pi / 2.0, release_end - release_start, dtype=np.float32)
        envelope[release_start:release_end] *= np.square(np.cos(phase))
    if release_end < frames:
        envelope[release_end:] = 0.0
    return envelope


def render_payload(
    payload: dict[str, Any],
    samples_directory: Path,
    output_path: Path,
    *,
    duration_seconds: float | None = None,
    hard_stop_seconds: float | None = None,
    training_eligible_only: bool = False,
) -> dict[str, Any]:
    notes = normalize_render_notes(
        payload, training_eligible_only=training_eligible_only
    )
    if not notes:
        raise ValueError("Input JSON contains no playable 88-key piano notes")
    notes, pedal_summary = apply_sustain_pedal_to_render_notes(notes, payload)
    natural_end = max(float(note["time"]) + float(note["duration"]) + RELEASE_SECONDS for note in notes)
    if hard_stop_seconds is not None:
        render_seconds = _finite(hard_stop_seconds, -1.0)
        if render_seconds <= 0:
            raise ValueError("hard_stop_seconds must be positive")
    else:
        render_seconds = max(natural_end, _finite(duration_seconds, natural_end))
    total_frames = max(1, math.ceil(render_seconds * SAMPLE_RATE))
    mix = np.zeros((total_frames, 2), dtype=np.float32)
    cache: dict[int, tuple[np.ndarray, float]] = {}
    release_frames = max(1, round(RELEASE_SECONDS * SAMPLE_RATE))

    for note in notes:
        midi = int(note["midi"])
        if midi not in cache:
            sample_path = samples_directory / midi_to_sample_name(midi)
            if not sample_path.is_file():
                raise FileNotFoundError(f"Missing piano sample: {sample_path}")
            sample = read_pcm16_stereo(sample_path)
            cache[midi] = (sample, browser_sample_gain(sample, midi))
        sample, sample_gain = cache[midi]
        start = max(0, round(float(note["time"]) * SAMPLE_RATE))
        if start >= total_frames:
            continue
        hold_frames = max(1, round(float(note["duration"]) * SAMPLE_RATE))
        audible_frames = min(len(sample), hold_frames + release_frames, total_frames - start)
        if audible_frames <= 0:
            continue
        envelope = note_envelope(audible_frames, hold_frames, release_frames)
        velocity = float(note["velocity"])
        velocity_gain = 0.12 + 0.88 * (velocity**1.7)
        performance_gain = float(note.get("performanceGain", 1.0))
        mix[start : start + audible_frames] += (
            sample[:audible_frames]
            * envelope[:, None]
            * velocity_gain
            * performance_gain
            * sample_gain
        )

    pre_limiter_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    # A fixed soft limiter protects the WAV container from clipping.  It is
    # deliberately identical for every file and never uses that file's peak.
    rendered = np.tanh(mix * LIMITER_DRIVE) * MASTER_GAIN
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(rendered), dtype=np.float64))) if rendered.size else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_stereo(output_path, rendered)
    return {
        "file": str(output_path),
        "sha256": sha256_file(output_path),
        "notes": len(notes),
        "seconds": round(total_frames / SAMPLE_RATE, 4),
        "sampleRate": SAMPLE_RATE,
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "preLimiterPeak": round(pre_limiter_peak, 6),
        "perFileNormalization": False,
        "releaseSeconds": RELEASE_SECONDS,
        "sampleGainCalibration": "browser-target-rms-v1",
        **pedal_summary,
        "hardStopSeconds": (
            round(float(hard_stop_seconds), 6)
            if hard_stop_seconds is not None
            else None
        ),
        "trainingEligibleOnly": training_eligible_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--hard-stop-seconds",
        type=float,
        help="Clip the render at this exact duration, including note releases.",
    )
    parser.add_argument(
        "--training-eligible-only",
        action="store_true",
        help="Render only notes explicitly approved for training.",
    )
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    summary = render_payload(
        json.loads(input_path.read_text(encoding="utf-8-sig")),
        Path(args.samples).resolve(),
        Path(args.output).resolve(),
        duration_seconds=args.duration_seconds,
        hard_stop_seconds=args.hard_stop_seconds,
        training_eligible_only=args.training_eligible_only,
    )
    summary["sourceJson"] = str(input_path)
    summary["sourceJsonSha256"] = sha256_file(input_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
