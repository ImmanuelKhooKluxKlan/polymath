#!/usr/bin/env python3
"""Audit and derive the 88-key perceptual loudness calibration.

This utility models the linear part of the browser piano signal chain:
sample analysis, register gains, per-voice EQ, master EQ, phase-safe compact
output, A-weighting, and a conservative compact-speaker response. Reverb and
dynamic compressors are intentionally excluded because this audit measures one
isolated reference strike at a time.

The Iowa recordings are stereo microphone captures, not level-matched stereo
masters. Many keys have large channel imbalance and 50/88 have negative
left/right correlation. A normal mono sum therefore makes some keys disappear
on phones. Compact output deliberately chooses the more useful microphone
channel for each source recording before producing dual-mono output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = REPO_ROOT / "public" / "samples" / "iowa-mf"
PITCH_CLASSES = {
    "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
    "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
}
EXPECTED_KEYS = 88

TARGET_RMS = (
    (21, 0.078), (33, 0.076), (45, 0.073), (60, 0.069),
    (72, 0.064), (84, 0.057), (96, 0.049), (108, 0.041),
)
LOW_COMPENSATION = (
    (21, 3.2), (23, 3.1), (24, 2.5), (28, 2.25),
    (33, 1.9), (36, 1.65), (40, 1.3), (48, 1.0),
)
COMPACT_LOW_COMPENSATION = (
    (21, 1.72), (24, 1.66), (28, 1.56), (33, 1.46),
    (36, 1.36), (40, 1.22), (48, 1.0),
)
REGISTER_GAIN = (
    (21, 0.82), (36, 0.86), (60, 0.9),
    (72, 0.92), (96, 0.88), (108, 0.82),
)
COMPACT_REGISTER_GAIN = (
    (21, 1.26), (24, 1.28), (33, 1.24), (36, 1.2),
    (48, 1.08), (55, 1.0), (60, 0.9), (72, 0.72),
    (84, 0.61), (96, 0.54), (108, 0.5),
)


def interpolate(points: tuple[tuple[float, float], ...], value: float) -> float:
    if value <= points[0][0]:
        return points[0][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if value <= right_x:
            progress = (value - left_x) / (right_x - left_x)
            return left_y + (right_y - left_y) * progress
    return points[-1][1]


def note_to_midi(name: str) -> int:
    match = re.fullmatch(r"([A-G](?:b)?)(-?\d+)", name)
    if not match:
        raise ValueError(f"Invalid sample name: {name}")
    return 12 * (int(match.group(2)) + 1) + PITCH_CLASSES[match.group(1)]


def read_wave(path: Path, maximum_seconds: float = 1.2) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM: {path}")
        rate = source.getframerate()
        frames = min(source.getnframes(), round(rate * maximum_seconds))
        audio = np.frombuffer(source.readframes(frames), dtype="<i2")
        audio = audio.reshape(-1, source.getnchannels()).astype(np.float64) / 32768.0
    return rate, audio


def a_weighting(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.maximum(frequencies, 1e-9)
    squared = frequencies * frequencies
    ratio = (
        (12200.0**2) * squared * squared
        / (
            (squared + 20.6**2)
            * np.sqrt((squared + 107.7**2) * (squared + 737.9**2))
            * (squared + 12200.0**2)
        )
    )
    decibels = 20 * np.log10(np.maximum(ratio, 1e-30)) + 2.0
    return 10 ** (decibels / 20)


def biquad_coefficients(
    kind: str,
    frequency: float,
    q: float,
    gain_db: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10 ** (gain_db / 40)
    omega = 2 * math.pi * frequency / sample_rate
    cosine = math.cos(omega)
    sine = math.sin(omega)
    alpha = sine / (2 * q)
    root_amplitude = math.sqrt(amplitude)

    if kind == "highpass":
        b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "peaking":
        b0, b1, b2 = 1 + alpha * amplitude, -2 * cosine, 1 - alpha * amplitude
        a0, a1, a2 = 1 + alpha / amplitude, -2 * cosine, 1 - alpha / amplitude
    elif kind in {"lowshelf", "highshelf"}:
        # Web Audio shelf filters use the cookbook's S=1 form; Q is ignored.
        shelf_alpha = sine / 2 * math.sqrt(2)
        if kind == "lowshelf":
            b0 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine + 2 * root_amplitude * shelf_alpha)
            b1 = 2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine - 2 * root_amplitude * shelf_alpha)
            a0 = (amplitude + 1) + (amplitude - 1) * cosine + 2 * root_amplitude * shelf_alpha
            a1 = -2 * ((amplitude - 1) + (amplitude + 1) * cosine)
            a2 = (amplitude + 1) + (amplitude - 1) * cosine - 2 * root_amplitude * shelf_alpha
        else:
            b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + 2 * root_amplitude * shelf_alpha)
            b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - 2 * root_amplitude * shelf_alpha)
            a0 = (amplitude + 1) - (amplitude - 1) * cosine + 2 * root_amplitude * shelf_alpha
            a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cosine)
            a2 = (amplitude + 1) - (amplitude - 1) * cosine - 2 * root_amplitude * shelf_alpha
    else:
        raise ValueError(f"Unsupported biquad: {kind}")

    return (
        np.asarray((b0, b1, b2), dtype=np.float64) / a0,
        np.asarray((1.0, a1 / a0, a2 / a0), dtype=np.float64),
    )


def biquad_response(
    frequencies: np.ndarray,
    sample_rate: int,
    kind: str,
    frequency: float,
    q: float,
    gain_db: float,
) -> np.ndarray:
    numerator, denominator = biquad_coefficients(kind, frequency, q, gain_db, sample_rate)
    delay = np.exp(-2j * math.pi * frequencies / sample_rate)
    return (
        numerator[0] + numerator[1] * delay + numerator[2] * delay * delay
    ) / (
        denominator[0] + denominator[1] * delay + denominator[2] * delay * delay
    )


def velocity_hammer_gain(velocity: float = 0.8) -> float:
    return interpolate(
        ((0.02, -0.55), (0.45, -0.55 * 0.32), (0.75, 0.52 * 0.36), (1.18, 0.52)),
        velocity,
    )


def voice_filters(midi: int, compact: bool) -> list[tuple[str, float, float, float]]:
    bass_amount = max(0.0, min(1.0, (60 - midi) / 24))
    treble_amount = max(0.0, min(1.0, (midi - 60) / 36))
    fundamental = 440 * (2 ** ((midi - 69) / 12))
    harmonic_number = 1
    while fundamental * harmonic_number < 110:
        harmonic_number += 2
    harmonic = fundamental * harmonic_number
    while harmonic > 240 and harmonic_number > 1:
        harmonic_number = max(1, harmonic_number - 2)
        harmonic = fundamental * harmonic_number

    highpass = (45 if midi < 36 else 38 if midi < 48 else 30) if compact else (14 if midi < 36 else 18 if midi < 48 else 26)
    body_type = "peaking" if compact and midi < 60 else "lowshelf"
    body_frequency = harmonic if compact and midi < 60 else (128 if not compact and midi < 48 else 170)
    body_q = 0.82 if compact and midi < 60 else 0.7
    body_gain = (0.72 if midi < 40 else 0.24 if midi < 58 else -0.12) + 0.22
    hammer_offset = 0.0
    air_offset = 0.0
    if compact:
        body_gain += 1.4 + 4.4 * bass_amount if midi < 60 else -0.5 * treble_amount
        hammer_offset = -0.65 - 3.7 * treble_amount
        air_offset = -0.5 - 3.0 * treble_amount

    return [
        ("highpass", highpass, 0.6, 0.0),
        (body_type, body_frequency, body_q, body_gain),
        ("peaking", 1750 if midi < 60 else 2450 if midi < 78 else 3150, 0.9, velocity_hammer_gain() + hammer_offset),
        ("highshelf", 7200, 0.707, (-0.12 if midi > 84 else 0.12) + air_offset),
    ]


def master_filters(compact: bool) -> list[tuple[str, float, float, float]]:
    if compact:
        return [
            ("highpass", 42, 0.7, 0.0),
            ("lowshelf", 180, 0.707, 2.05),
            ("peaking", 420, 0.72, 0.1),
            ("peaking", 2500, 0.82, -3.92),
            ("highshelf", 7800, 0.707, -3.72),
        ]
    return [
        ("highpass", 25, 0.7, 0.0),
        ("lowshelf", 135, 0.707, 0.85),
        ("peaking", 315, 0.95, -1.05),
        ("peaking", 2500, 0.82, 0.68),
        ("highshelf", 7800, 0.707, 0.28),
    ]


def weighted_level_db(audio: np.ndarray, sample_rate: int, midi: int, compact: bool) -> float:
    if audio.ndim == 1:
        audio = audio[:, None]
    spectrum = np.fft.rfft(audio * np.hanning(len(audio))[:, None], axis=0)
    frequencies = np.maximum(np.fft.rfftfreq(len(audio), 1 / sample_rate), 1e-9)
    response = a_weighting(frequencies).astype(np.complex128)
    for filter_spec in [*voice_filters(midi, compact), *master_filters(compact)]:
        response *= biquad_response(frequencies, sample_rate, *filter_spec)
    if compact:
        response /= np.sqrt(1 + (155 / frequencies) ** 8)
        response /= np.sqrt(1 + (frequencies / 14000) ** 8)
    ratio = math.sqrt(
        np.sum(np.abs(spectrum * response[:, None]) ** 2)
        / max(np.sum(np.abs(spectrum) ** 2), 1e-30)
    )
    rms = math.sqrt(np.mean(audio * audio))
    return 20 * math.log10(max(rms * ratio, 1e-30))


def compact_channel_levels(
    attack: np.ndarray,
    body: np.ndarray,
    sample_rate: int,
    midi: int,
    source_gain: float,
) -> list[tuple[float, float, float]]:
    """Return attack/body/combined phone-band level for every microphone."""
    levels: list[tuple[float, float, float]] = []
    harmonic_amount = max(0.0, min(1.0, (52 - midi) / 24))
    drive = 1.15 + 0.85 * harmonic_amount if harmonic_amount > 0 else 1.0

    def prepare(values: np.ndarray) -> np.ndarray:
        amplified = values * source_gain
        if drive <= 1:
            return amplified
        return np.tanh(drive * np.clip(amplified, -1, 1)) / drive

    for channel in range(attack.shape[1]):
        attack_db = weighted_level_db(prepare(attack[:, channel]), sample_rate, midi, True)
        body_db = weighted_level_db(prepare(body[:, channel]), sample_rate, midi, True)
        # Attacks carry note intelligibility on a phone, while the body still
        # needs enough weight to prevent the key sounding clipped or broken.
        combined_db = (0.62 * attack_db) + (0.38 * body_db)
        levels.append((attack_db, body_db, combined_db))
    return levels


def stereo_diagnostics(audio: np.ndarray) -> dict[str, float]:
    """Measure what a conventional L+R mono fold-down would lose."""
    if audio.shape[1] < 2:
        return {
            "correlation": 1.0,
            "channelImbalanceDb": 0.0,
            "monoCollapseLossDb": 0.0,
        }
    left = audio[:, 0]
    right = audio[:, 1]
    left_rms = math.sqrt(np.mean(left * left))
    right_rms = math.sqrt(np.mean(right * right))
    stereo_rms = math.sqrt((np.mean(left * left) + np.mean(right * right)) / 2)
    mono_rms = math.sqrt(np.mean(((left + right) / 2) ** 2))
    correlation = float(np.corrcoef(left, right)[0, 1])
    return {
        "correlation": correlation,
        "channelImbalanceDb": abs(20 * math.log10(max(left_rms, 1e-30) / max(right_rms, 1e-30))),
        "monoCollapseLossDb": 20 * math.log10(max(mono_rms, 1e-30) / max(stereo_rms, 1e-30)),
    }


def smooth_target(levels: np.ndarray) -> np.ndarray:
    median = np.asarray([
        np.median(levels[max(0, index - 6):min(len(levels), index + 7)])
        for index in range(len(levels))
    ])
    kernel = np.asarray((1, 4, 6, 4, 1), dtype=np.float64) / 16
    for _ in range(5):
        median = np.convolve(np.pad(median, (2, 2), mode="edge"), kernel, mode="valid")
    # A real keyboard changes timbre across the scale, but no neighboring key
    # should create a sudden loudness cliff at the same strike velocity.
    # The short upper strings genuinely lose body faster than bass strings.
    # Keep that physical decay, but remove discontinuities large enough to make
    # one neighboring key sound like a different instrument.
    maximum_step_db = 1.5
    for _ in range(8):
        for index in range(1, len(median)):
            median[index] = np.clip(
                median[index],
                median[index - 1] - maximum_step_db,
                median[index - 1] + maximum_step_db,
            )
        for index in range(len(median) - 2, -1, -1):
            median[index] = np.clip(
                median[index],
                median[index + 1] - maximum_step_db,
                median[index + 1] + maximum_step_db,
            )
    return median


def sample_set_checksum(paths: list[Path]) -> str:
    individual = "".join(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    return hashlib.sha256(individual.encode()).hexdigest()


def audit() -> dict[str, object]:
    paths = sorted(SAMPLE_ROOT.glob("*.wav"), key=lambda path: note_to_midi(path.stem))
    if len(paths) != EXPECTED_KEYS:
        raise RuntimeError(f"Expected {EXPECTED_KEYS} samples, found {len(paths)}")

    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        midi = note_to_midi(path.stem)
        sample_rate, audio = read_wave(path)
        analysis = audio[round(sample_rate * 0.02):]
        rms = math.sqrt(np.mean(analysis * analysis))
        peak = float(np.max(np.abs(analysis)))
        analysis_gain = np.clip(
            min(interpolate(TARGET_RMS, midi) / max(rms, 0.0001), 0.92 / max(peak, 0.0001)),
            0.35,
            1.95,
        )
        base_gain = analysis_gain * interpolate(LOW_COMPENSATION, midi) * interpolate(REGISTER_GAIN, midi)
        compact_source_gain = analysis_gain * interpolate(COMPACT_LOW_COMPENSATION, midi)
        attack = audio[round(sample_rate * 0.02):round(sample_rate * 0.25)]
        body = audio[round(sample_rate * 0.25):round(sample_rate * 1.2)]
        compact_levels = compact_channel_levels(
            attack, body, sample_rate, midi, compact_source_gain,
        )
        compact_channel = max(
            range(len(compact_levels)),
            key=lambda channel: compact_levels[channel][2],
        )
        diagnostics = stereo_diagnostics(analysis)
        compact_gain = interpolate(REGISTER_GAIN, midi) * interpolate(COMPACT_REGISTER_GAIN, midi)
        harmonic_amount = max(0.0, min(1.0, (52 - midi) / 24))
        harmonic_drive = 1.15 + 0.85 * harmonic_amount if harmonic_amount > 0 else 1.0
        compact_channel_audio = analysis[:, compact_channel] * compact_source_gain
        if harmonic_drive > 1:
            compact_channel_audio = np.tanh(
                harmonic_drive * np.clip(compact_channel_audio, -1, 1)
            ) / harmonic_drive
        rows.append({
            "midi": midi,
            "note": path.stem,
            "peak": peak * base_gain,
            "full_attack": weighted_level_db(attack, sample_rate, midi, False) + 20 * math.log10(base_gain),
            "full_body": weighted_level_db(body, sample_rate, midi, False) + 20 * math.log10(base_gain),
            "compact_channel": compact_channel,
            "compact_attack": compact_levels[compact_channel][0] + 20 * math.log10(compact_gain),
            "compact_body": compact_levels[compact_channel][1] + 20 * math.log10(compact_gain),
            "compact_peak": float(np.max(np.abs(compact_channel_audio))) * compact_gain,
            **diagnostics,
        })

    result: dict[str, object] = {
        "sampleSetSha256": sample_set_checksum(paths),
        "keys": len(rows),
        "notes": [row["note"] for row in rows],
        "compactSourceChannel": [row["compact_channel"] for row in rows],
        "stereoDiagnostics": {
            "negativeCorrelationKeys": sum(row["correlation"] < 0 for row in rows),
            "worstMonoCollapseLossDb": round(min(row["monoCollapseLossDb"] for row in rows), 3),
            "maximumChannelImbalanceDb": round(max(row["channelImbalanceDb"] for row in rows), 3),
        },
    }
    for field in ("full_attack", "full_body", "compact_attack", "compact_body"):
        levels = np.asarray([row[field] for row in rows], dtype=np.float64)
        correction = np.clip(smooth_target(levels) - levels, -12, 12)
        corrected = levels + correction
        result[field] = {
            "referenceDb": [round(float(value), 3) for value in levels],
            "correctionDb": [round(float(value), 3) for value in correction],
            "maximumAdjacentJumpDb": round(float(np.max(np.abs(np.diff(corrected)))), 3),
            "p95AdjacentJumpDb": round(float(np.percentile(np.abs(np.diff(corrected)), 95)), 3),
        }
    result["maximumPreMasterPeak"] = round(max(
        max(row["peak"] * 10 ** (correction / 20) for row, correction in zip(rows, result["full_attack"]["correctionDb"])),
        max(row["compact_peak"] * 10 ** (correction / 20) for row, correction in zip(rows, result["compact_attack"]["correctionDb"])),
    ), 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable audit")
    args = parser.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"keys={result['keys']} source={result['sampleSetSha256']}")
    stereo = result["stereoDiagnostics"]
    print(
        f"stereo: negative-correlation={stereo['negativeCorrelationKeys']} "
        f"worst-mono-loss={stereo['worstMonoCollapseLossDb']:.3f} dB "
        f"maximum-channel-imbalance={stereo['maximumChannelImbalanceDb']:.3f} dB"
    )
    print(f"COMPACT_SOURCE_CHANNEL = {json.dumps(result['compactSourceChannel'])}")
    for field in ("full_attack", "full_body", "compact_attack", "compact_body"):
        metrics = result[field]
        print(
            f"{field}: max-adjacent={metrics['maximumAdjacentJumpDb']:.3f} dB "
            f"p95={metrics['p95AdjacentJumpDb']:.3f} dB"
        )
        print(f"{field.upper()}_DB = {json.dumps(metrics['correctionDb'])}")
    print(f"maximum-pre-master-peak={result['maximumPreMasterPeak']:.6f}")


if __name__ == "__main__":
    main()
