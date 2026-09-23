#!/usr/bin/env python3
"""Audit the 88-key full-range and weak-device piano renderers.

This utility models the linear part of the browser piano signal chain: sample
normalisation, register gain, per-voice EQ, master EQ, A-weighting, and a
conservative small-speaker response.  The weak-device path keeps the accepted
stereo Iowa recording and adds a quiet, pitch-locked missing-fundamental layer
below A2.  Reverb and dynamic compressors are intentionally excluded because
this audit measures one isolated reference strike at a time.

The Iowa files are stereo microphone captures, not level-matched masters. Many
keys have large channel imbalance and 50/88 have negative left/right
correlation, so the audit also reports the damage a conventional mono fold-down
would cause. Production deliberately preserves stereo instead of choosing one
microphone or summing phase-opposed channels.
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
REGISTER_GAIN = (
    (21, 0.82), (36, 0.86), (60, 0.9),
    (72, 0.92), (96, 0.88), (108, 0.82),
)
COMPACT_REGISTER_GAIN = (
    (21, 1.08), (33, 1.08), (48, 1.04), (60, 1.0),
    (72, 1.0), (84, 0.96), (96, 0.92), (108, 0.9),
)
VIRTUAL_BASS_GAIN = (
    (21, 0.052), (33, 0.044), (40, 0.032), (45, 0.02),
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
        ((0.02, -0.55), (0.45, -0.55 * 0.32), (0.75, 0.82 * 0.36), (1.18, 0.82)),
        velocity,
    )


def voice_filters(midi: int, compact: bool) -> list[tuple[str, float, float, float]]:
    bass_amount = max(0.0, min(1.0, (60 - midi) / 39))
    treble_amount = max(0.0, min(1.0, (midi - 72) / 36))
    fundamental = 440 * (2 ** ((midi - 69) / 12))
    harmonic_number = 1
    while fundamental * harmonic_number < 130:
        harmonic_number += 1
    harmonic = fundamental * harmonic_number
    while harmonic > 260 and harmonic_number > 1:
        harmonic_number = max(1, harmonic_number - 1)
        harmonic = fundamental * harmonic_number

    highpass = 14 if midi < 36 else 18 if midi < 48 else 26
    body_type = "peaking" if compact and midi < 60 else "lowshelf"
    body_frequency = harmonic if compact and midi < 60 else (128 if not compact and midi < 48 else 170)
    body_q = 0.9 if compact and midi < 60 else 0.7
    body_gain = (0.72 if midi < 40 else 0.24 if midi < 58 else -0.12) + 0.22
    hammer_offset = 0.0
    air_offset = 0.0
    if compact:
        body_gain += 1.2 + 3.0 * bass_amount if midi < 60 else -0.2 * treble_amount
        hammer_offset = -0.15 - 1.2 * treble_amount
        air_offset = -0.1 - 0.9 * treble_amount

    return [
        ("highpass", highpass, 0.6, 0.0),
        (body_type, body_frequency, body_q, body_gain),
        ("peaking", 1750 if midi < 60 else 2450 if midi < 78 else 3150, 0.9, velocity_hammer_gain() + hammer_offset),
        ("highshelf", 7200, 0.707, (-0.12 if midi > 84 else 0.12) + air_offset),
    ]


def master_filters(compact: bool) -> list[tuple[str, float, float, float]]:
    if compact:
        return [
            ("highpass", 25, 0.7, 0.0),
            ("lowshelf", 180, 0.707, 2.25),
            ("peaking", 360, 0.82, -0.55),
            ("peaking", 2500, 0.82, -0.30),
            ("highshelf", 7800, 0.707, -0.72),
        ]
    return [
        ("highpass", 25, 0.7, 0.0),
        ("lowshelf", 135, 0.707, 0.85),
        ("peaking", 315, 0.95, -1.05),
        ("peaking", 2500, 0.82, 0.68),
        ("highshelf", 7800, 0.707, 0.28),
    ]


def weighted_level_db(
    audio: np.ndarray,
    sample_rate: int,
    midi: int,
    compact: bool,
    speaker_corner_hz: float = 155,
) -> float:
    if audio.ndim == 1:
        audio = audio[:, None]
    spectrum = np.fft.rfft(audio * np.hanning(len(audio))[:, None], axis=0)
    frequencies = np.maximum(np.fft.rfftfreq(len(audio), 1 / sample_rate), 1e-9)
    response = a_weighting(frequencies).astype(np.complex128)
    for filter_spec in [*voice_filters(midi, compact), *master_filters(compact)]:
        response *= biquad_response(frequencies, sample_rate, *filter_spec)
    if compact:
        response /= np.sqrt(1 + (speaker_corner_hz / frequencies) ** 8)
        response /= np.sqrt(1 + (frequencies / 14000) ** 8)
    ratio = math.sqrt(
        np.sum(np.abs(spectrum * response[:, None]) ** 2)
        / max(np.sum(np.abs(spectrum) ** 2), 1e-30)
    )
    rms = math.sqrt(np.mean(audio * audio))
    return 20 * math.log10(max(rms * ratio, 1e-30))


def virtual_bass_audio(midi: int, sample_rate: int, frames: int) -> np.ndarray:
    """Render the browser's normalized three-partial missing-fundamental layer."""
    if midi > 45 or frames <= 0:
        return np.zeros(frames, dtype=np.float64)

    fundamental = 440 * (2 ** ((midi - 69) / 12))
    first_harmonic = max(2, math.ceil(120 / fundamental))
    harmonics = (first_harmonic, first_harmonic + 1, first_harmonic + 2)
    weights = (1.0, 0.58, 0.32)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    wave = sum(
        weight * np.sin(2 * math.pi * fundamental * harmonic * time)
        for harmonic, weight in zip(harmonics, weights)
    )
    wave /= max(float(np.max(np.abs(wave))), 1e-9)

    amount = interpolate(VIRTUAL_BASS_GAIN, midi) * (0.72 + 0.28 * 0.8)
    envelope = np.full(frames, amount * 0.42, dtype=np.float64)
    attack_frames = max(1, round(sample_rate * 0.012))
    decay_frames = max(attack_frames + 1, round(sample_rate * 0.34))
    envelope[:attack_frames] = np.linspace(1e-5, amount, attack_frames)
    envelope[attack_frames:decay_frames] = np.linspace(
        amount, amount * 0.42, decay_frames - attack_frames,
    )
    return wave * envelope


def compact_render(audio: np.ndarray, sample_rate: int, midi: int, source_gain: float) -> np.ndarray:
    """Preserve stereo samples and up-mix the quiet virtual-bass layer."""
    if audio.ndim == 1:
        audio = audio[:, None]
    rendered = audio * source_gain
    bass = virtual_bass_audio(midi, sample_rate, len(audio))[:, None]
    return rendered + np.repeat(bass, rendered.shape[1], axis=1)


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
        compact_source_gain = analysis_gain * interpolate(LOW_COMPENSATION, midi)
        attack_start = round(sample_rate * 0.02)
        attack_end = round(sample_rate * 0.25)
        body_end = round(sample_rate * 1.2)
        attack = audio[attack_start:attack_end]
        body = audio[attack_end:body_end]
        compact_audio = compact_render(audio, sample_rate, midi, compact_source_gain)
        compact_without_virtual = audio * compact_source_gain
        compact_attack = compact_audio[attack_start:attack_end]
        compact_body = compact_audio[attack_end:body_end]
        compact_attack_without_virtual = compact_without_virtual[attack_start:attack_end]
        compact_body_without_virtual = compact_without_virtual[attack_end:body_end]
        diagnostics = stereo_diagnostics(analysis)
        compact_gain = interpolate(REGISTER_GAIN, midi) * interpolate(COMPACT_REGISTER_GAIN, midi)
        rows.append({
            "midi": midi,
            "note": path.stem,
            "peak": peak * base_gain,
            "full_attack": weighted_level_db(attack, sample_rate, midi, False) + 20 * math.log10(base_gain),
            "full_body": weighted_level_db(body, sample_rate, midi, False) + 20 * math.log10(base_gain),
            "compact_attack": weighted_level_db(
                compact_attack, sample_rate, midi, True,
            ) + 20 * math.log10(compact_gain),
            "compact_body": weighted_level_db(
                compact_body, sample_rate, midi, True,
            ) + 20 * math.log10(compact_gain),
            "compact_attack_without_virtual": weighted_level_db(
                compact_attack_without_virtual, sample_rate, midi, True,
            ) + 20 * math.log10(compact_gain),
            "compact_body_without_virtual": weighted_level_db(
                compact_body_without_virtual, sample_rate, midi, True,
            ) + 20 * math.log10(compact_gain),
            "compact_peak": float(np.max(np.abs(compact_audio))) * compact_gain,
            **diagnostics,
        })

    result: dict[str, object] = {
        "sampleSetSha256": sample_set_checksum(paths),
        "keys": len(rows),
        "notes": [row["note"] for row in rows],
        "compactRender": "stereo-samples-plus-pitch-locked-virtual-bass-v2",
        "stereoDiagnostics": {
            "negativeCorrelationKeys": sum(row["correlation"] < 0 for row in rows),
            "worstMonoCollapseLossDb": round(min(row["monoCollapseLossDb"] for row in rows), 3),
            "maximumChannelImbalanceDb": round(max(row["channelImbalanceDb"] for row in rows), 3),
        },
    }
    low_rows = [row for row in rows if row["midi"] <= 45]
    result["virtualBassLiftDb"] = {
        "attackMedian": round(float(np.median([
            row["compact_attack"] - row["compact_attack_without_virtual"]
            for row in low_rows
        ])), 3),
        "bodyMedian": round(float(np.median([
            row["compact_body"] - row["compact_body_without_virtual"]
            for row in low_rows
        ])), 3),
        "minimumAttack": round(float(min(
            row["compact_attack"] - row["compact_attack_without_virtual"]
            for row in low_rows
        )), 3),
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
    virtual = result["virtualBassLiftDb"]
    print(
        f"virtual-bass: median-attack-lift={virtual['attackMedian']:.3f} dB "
        f"median-body-lift={virtual['bodyMedian']:.3f} dB "
        f"minimum-attack-lift={virtual['minimumAttack']:.3f} dB"
    )
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
