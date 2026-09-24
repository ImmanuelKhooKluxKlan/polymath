"""Snap predicted piano gestures to nearby waveform attacks.

This operator is deliberately inference-safe: it reads only the candidate
score and its source WAV.  It never reads a reference MIDI.  Parameter search
belongs in a separate song-held-out experiment; this module only applies one
already-frozen policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .audio_chroma_alignment import extract_chroma_features, load_mono_pcm_wav


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_peak_indices(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.arange(len(values), dtype=np.int64)
    middle = np.arange(1, len(values) - 1, dtype=np.int64)
    peaks = middle[
        (values[middle] >= values[middle - 1])
        & (values[middle] > values[middle + 1])
    ]
    endpoints: list[int] = []
    if values[0] > values[1]:
        endpoints.append(0)
    if values[-1] >= values[-2]:
        endpoints.append(len(values) - 1)
    return np.asarray([*endpoints[:1], *peaks.tolist(), *endpoints[1:]], dtype=np.int64)


def onset_envelope(
    audio_path: Path,
    *,
    hop_seconds: float = 0.01,
    fft_seconds: float = 0.064,
) -> tuple[np.ndarray, np.ndarray, float]:
    samples, sample_rate = load_mono_pcm_wav(audio_path)
    features = extract_chroma_features(
        samples,
        sample_rate,
        hop_seconds=hop_seconds,
        fft_seconds=fft_seconds,
    )
    # A light triangular smoother suppresses single-bin numerical spikes while
    # retaining a piano hammer attack at 10 ms resolution.
    smoothed = np.convolve(
        features.onset.astype(np.float64),
        np.asarray([0.25, 0.50, 0.25], dtype=np.float64),
        mode="same",
    )
    return features.times, smoothed, features.duration_seconds


def candidate_groups(
    notes: list[dict[str, Any]], window_seconds: float
) -> list[list[int]]:
    ordered: list[tuple[float, int]] = []
    for index, note in enumerate(notes):
        try:
            onset = float(note.get("time", note.get("startTime")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(onset) and onset >= 0:
            ordered.append((onset, index))
    ordered.sort()
    groups: list[list[int]] = []
    starts: list[float] = []
    for onset, index in ordered:
        if not groups or onset - starts[-1] > window_seconds:
            groups.append([index])
            starts.append(onset)
        else:
            groups[-1].append(index)
    return groups


def snap_payload(
    payload: dict[str, Any],
    feature_times: np.ndarray,
    onset_strength: np.ndarray,
    *,
    radius_seconds: float,
    minimum_peak_strength: float,
    distance_weight: float,
    peak_time_offset_seconds: float,
    minimum_strength_gain: float,
    group_window_seconds: float = 0.035,
    audio_duration_seconds: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if radius_seconds <= 0:
        raise ValueError("radius_seconds must be positive")
    if not 0 <= minimum_peak_strength <= 1:
        raise ValueError("minimum_peak_strength must be between zero and one")
    if distance_weight < 0 or minimum_strength_gain < 0:
        raise ValueError("distance_weight and minimum_strength_gain cannot be negative")
    if len(feature_times) != len(onset_strength) or not len(feature_times):
        raise ValueError("feature time and onset arrays must be non-empty and equal length")

    output = copy.deepcopy(payload)
    notes = output.get("notes")
    if not isinstance(notes, list):
        notes = []
        output["notes"] = notes
    peaks = local_peak_indices(onset_strength)
    peak_times = feature_times[peaks] + float(peak_time_offset_seconds)
    peak_strengths = onset_strength[peaks]
    snapped_groups = 0
    snapped_notes = 0
    shifts: list[float] = []

    for group in candidate_groups(notes, group_window_seconds):
        onsets = [float(notes[index].get("time", notes[index].get("startTime"))) for index in group]
        anchor = min(onsets)
        left = int(np.searchsorted(peak_times, anchor - radius_seconds, side="left"))
        right = int(np.searchsorted(peak_times, anchor + radius_seconds, side="right"))
        if left >= right:
            continue
        original_index = int(np.argmin(np.abs(feature_times - anchor)))
        original_strength = float(onset_strength[original_index])
        options: list[tuple[float, float, float]] = []
        for index in range(left, right):
            strength = float(peak_strengths[index])
            if strength < minimum_peak_strength:
                continue
            shift = float(peak_times[index]) - anchor
            score = strength - distance_weight * abs(shift) / radius_seconds
            options.append((score, -abs(shift), shift))
        if not options:
            continue
        _score, _negative_distance, shift = max(options)
        chosen_strength = max(
            float(peak_strengths[index])
            for index in range(left, right)
            if abs(float(peak_times[index]) - anchor - shift) < 1e-9
        )
        if chosen_strength < original_strength + minimum_strength_gain:
            continue
        if abs(shift) < 0.002:
            continue
        if anchor + shift < 0:
            continue
        if audio_duration_seconds is not None and anchor + shift >= audio_duration_seconds:
            continue
        for index in group:
            original = float(notes[index].get("time", notes[index].get("startTime")))
            notes[index]["timeBeforeAudioOnsetSnap"] = round(original, 6)
            notes[index]["time"] = round(max(0.0, original + shift), 6)
        snapped_groups += 1
        snapped_notes += len(group)
        shifts.append(shift)

    notes.sort(
        key=lambda note: (
            float(note.get("time", 0.0)),
            int(round(float(note.get("midi", note.get("pitch", 0))))),
        )
    )
    diagnostics = {
        "schema": "polymath-audio-onset-snap-v1",
        "inferenceSafe": True,
        "referenceMidiUsed": False,
        "radiusSeconds": round(radius_seconds, 6),
        "minimumPeakStrength": round(minimum_peak_strength, 6),
        "distanceWeight": round(distance_weight, 6),
        "peakTimeOffsetSeconds": round(peak_time_offset_seconds, 6),
        "minimumStrengthGain": round(minimum_strength_gain, 6),
        "groupWindowSeconds": round(group_window_seconds, 6),
        "candidateGroups": len(candidate_groups(notes, group_window_seconds)),
        "snappedGroups": snapped_groups,
        "snappedNotes": snapped_notes,
        "medianShiftMilliseconds": (
            round(median_value(shifts) * 1000.0, 3) if shifts else 0.0
        ),
        "maximumAbsoluteShiftMilliseconds": (
            round(max(abs(value) for value in shifts) * 1000.0, 3)
            if shifts
            else 0.0
        ),
        "pitchesFrozen": True,
        "durationsFrozen": True,
        "velocitiesFrozen": True,
    }
    output.setdefault("transcriptionTiming", {})["audioOnsetSnap"] = diagnostics
    return output, diagnostics


def median_value(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--radius-seconds", type=float, default=0.08)
    parser.add_argument("--minimum-peak-strength", type=float, default=0.45)
    parser.add_argument("--distance-weight", type=float, default=0.30)
    parser.add_argument("--peak-time-offset-seconds", type=float, default=-0.03)
    parser.add_argument("--minimum-strength-gain", type=float, default=0.05)
    parser.add_argument("--group-window-seconds", type=float, default=0.035)
    parser.add_argument("--hop-seconds", type=float, default=0.01)
    parser.add_argument("--fft-seconds", type=float, default=0.064)
    args = parser.parse_args()

    policy = {}
    if args.profile:
        policy = json.loads(args.profile.resolve().read_text(encoding="utf-8-sig"))
    times, strengths, duration = onset_envelope(
        args.audio.resolve(),
        hop_seconds=float(policy.get("hopSeconds", args.hop_seconds)),
        fft_seconds=float(policy.get("fftSeconds", args.fft_seconds)),
    )
    output, diagnostics = snap_payload(
        json.loads(args.input.resolve().read_text(encoding="utf-8-sig")),
        times,
        strengths,
        radius_seconds=float(policy.get("radiusSeconds", args.radius_seconds)),
        minimum_peak_strength=float(
            policy.get("minimumPeakStrength", args.minimum_peak_strength)
        ),
        distance_weight=float(policy.get("distanceWeight", args.distance_weight)),
        peak_time_offset_seconds=float(
            policy.get("peakTimeOffsetSeconds", args.peak_time_offset_seconds)
        ),
        minimum_strength_gain=float(
            policy.get("minimumStrengthGain", args.minimum_strength_gain)
        ),
        group_window_seconds=float(
            policy.get("groupWindowSeconds", args.group_window_seconds)
        ),
        audio_duration_seconds=duration,
    )
    atomic_json(args.output.resolve(), output)
    print(json.dumps({"output": str(args.output.resolve()), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
