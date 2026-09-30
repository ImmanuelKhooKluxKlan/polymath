"""Attach target-blind monophonic vocal pitch evidence to audit cells.

The separated vocal stem carries the clearest proxy for the right-hand melody,
but running a polyphonic note transcriber on that stem creates many spurious
notes.  This utility uses a deterministic YIN-style difference function to
estimate one fundamental frequency per frame, then summarizes the confident
pitch classes around each frozen arranger onset.  No MIDI reference or target
field is consulted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

try:
    from .augment_texture_audit_audio_chroma import read_mono
except ImportError:  # pragma: no cover
    from augment_texture_audit_audio_chroma import read_mono  # type: ignore


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def stabilize_pitch_classes(
    frequencies: np.ndarray,
    confidences: np.ndarray,
    *,
    window_frames: int,
) -> np.ndarray:
    """Suppress isolated semitone jitter without consulting score labels."""

    valid = np.isfinite(frequencies) & (frequencies > 0.0)
    pitch_classes = np.full(len(frequencies), -1, dtype=np.int8)
    if np.any(valid):
        midi = 69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)
        pitch_classes[valid] = np.rint(midi).astype(int) % 12
    width = max(1, int(window_frames))
    if width <= 1:
        return pitch_classes
    if width % 2 == 0:
        width += 1
    radius = width // 2
    stabilized = pitch_classes.copy()
    for index in np.flatnonzero(valid):
        left = max(0, int(index) - radius)
        right = min(len(pitch_classes), int(index) + radius + 1)
        local = pitch_classes[left:right]
        local_valid = local >= 0
        if not np.any(local_valid):
            continue
        weights = np.maximum(0.0, confidences[left:right][local_valid])
        votes = np.bincount(
            local[local_valid], weights=weights, minlength=12
        )
        winners = np.flatnonzero(np.isclose(votes, np.max(votes)))
        current = int(pitch_classes[index])
        stabilized[index] = current if current in winners else int(winners[0])
    return stabilized


def yin_f0(
    path: Path,
    *,
    frame_seconds: float = 0.08,
    hop_seconds: float = 0.01,
    minimum_hz: float = 75.0,
    maximum_hz: float = 1000.0,
    trough_threshold: float = 0.18,
    pitch_smoothing_frames: int = 1,
) -> dict[str, Any]:
    sample_rate, signal = read_mono(path)
    frame_length = max(512, int(round(sample_rate * frame_seconds)))
    hop = max(64, int(round(sample_rate * hop_seconds)))
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    frame_count = 1 + max(0, (len(signal) - frame_length) // hop)
    minimum_lag = max(2, int(math.floor(sample_rate / maximum_hz)))
    maximum_lag = min(frame_length // 2, int(math.ceil(sample_rate / minimum_hz)))
    fft_size = _next_power_of_two(frame_length * 2)
    window = np.hanning(frame_length).astype(np.float64)
    offsets = np.arange(frame_length)
    times = np.zeros(frame_count, dtype=np.float64)
    frequencies = np.full(frame_count, np.nan, dtype=np.float64)
    confidences = np.zeros(frame_count, dtype=np.float64)
    rms_values = np.zeros(frame_count, dtype=np.float64)
    batch_size = 256
    lag_values = np.arange(1, maximum_lag + 1, dtype=np.int64)

    for start in range(0, frame_count, batch_size):
        stop = min(frame_count, start + batch_size)
        starts = np.arange(start, stop) * hop
        raw = signal[starts[:, None] + offsets[None, :]]
        raw = raw - np.mean(raw, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(raw**2, axis=1))
        frames = raw * window[None, :]
        spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
        autocorrelation = np.fft.irfft(
            np.abs(spectrum) ** 2, n=fft_size, axis=1
        )[:, : maximum_lag + 1]
        squared = frames**2
        prefix = np.concatenate(
            [np.zeros((len(frames), 1)), np.cumsum(squared, axis=1)], axis=1
        )
        total = prefix[:, -1]
        first_energy = prefix[:, frame_length - lag_values]
        second_energy = total[:, None] - prefix[:, lag_values]
        difference = np.maximum(
            0.0,
            first_energy + second_energy - 2.0 * autocorrelation[:, 1 : maximum_lag + 1],
        )
        cumulative = np.cumsum(difference, axis=1)
        cmnd = difference * lag_values[None, :] / np.maximum(cumulative, 1e-12)
        cmnd[:, 0] = 1.0

        for local_index in range(len(frames)):
            row = cmnd[local_index]
            search = row[minimum_lag - 1 : maximum_lag]
            below = np.flatnonzero(search < trough_threshold)
            if len(below):
                selected = int(below[0] + minimum_lag - 1)
                while selected + 1 < len(row) and row[selected + 1] < row[selected]:
                    selected += 1
            else:
                selected = int(np.argmin(search) + minimum_lag - 1)
            lag = float(selected + 1)
            if 0 < selected < len(row) - 1:
                left, center, right = row[selected - 1 : selected + 2]
                denominator = left - 2.0 * center + right
                if abs(denominator) > 1e-12:
                    lag += 0.5 * (left - right) / denominator
            confidence = max(0.0, min(1.0, 1.0 - float(row[selected])))
            output_index = start + local_index
            if rms[local_index] >= 1e-4 and confidence > 0.0:
                frequencies[output_index] = sample_rate / max(1.0, lag)
                confidences[output_index] = confidence
            rms_values[output_index] = rms[local_index]
        times[start:stop] = (starts + frame_length / 2.0) / sample_rate

    pitch_classes = stabilize_pitch_classes(
        frequencies,
        confidences,
        window_frames=pitch_smoothing_frames,
    )
    return {
        "sampleRate": sample_rate,
        "frameLength": frame_length,
        "hopSamples": hop,
        "times": times,
        "frequencies": frequencies,
        "pitchClasses": pitch_classes,
        "confidences": confidences,
        "rms": rms_values,
        "durationSeconds": len(signal) / sample_rate,
        "minimumHz": minimum_hz,
        "maximumHz": maximum_hz,
        "troughThreshold": trough_threshold,
        "pitchSmoothingFrames": max(1, int(pitch_smoothing_frames)),
    }


def local_vocal_evidence(
    features: dict[str, Any],
    time_seconds: float,
    *,
    radius_seconds: float,
    minimum_confidence: float,
) -> dict[str, Any]:
    times = features["times"]
    left = int(np.searchsorted(times, time_seconds - radius_seconds, side="left"))
    right = int(np.searchsorted(times, time_seconds + radius_seconds, side="right"))
    if right <= left:
        nearest = int(np.argmin(np.abs(times - time_seconds)))
        left, right = nearest, nearest + 1
    frequencies = features["frequencies"][left:right]
    stabilized_pitch_classes = features.get("pitchClasses")
    local_pitch_classes = (
        stabilized_pitch_classes[left:right]
        if isinstance(stabilized_pitch_classes, np.ndarray)
        else None
    )
    confidences = features["confidences"][left:right]
    valid = np.isfinite(frequencies) & (confidences >= minimum_confidence)
    histogram = np.zeros(12, dtype=np.float64)
    frame_counts = np.zeros(12, dtype=np.int64)
    confidence_sums = np.zeros(12, dtype=np.float64)
    if np.any(valid):
        if local_pitch_classes is None:
            midi = 69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)
            valid_pitch_classes = np.rint(midi).astype(int) % 12
        else:
            valid_pitch_classes = local_pitch_classes[valid]
        for pitch_class, confidence in zip(valid_pitch_classes, confidences[valid]):
            pitch_class = int(pitch_class) % 12
            histogram[pitch_class] += float(confidence)
            frame_counts[pitch_class] += 1
            confidence_sums[pitch_class] += float(confidence)
    total = float(np.sum(histogram))
    probabilities = histogram / total if total > 1e-12 else histogram
    valid_count = int(np.sum(valid))
    window_count = max(1, right - left)
    return {
        "available": True,
        "voicedShare": round(valid_count / window_count, 8),
        "meanConfidence": round(
            float(np.mean(confidences[valid])) if valid_count else 0.0, 8
        ),
        "dominantPitchClass": int(np.argmax(histogram)) if total > 1e-12 else None,
        "pitchClasses": [
            {
                "pitchClass": pitch_class,
                "probability": round(float(probabilities[pitch_class]), 8),
                "frameShare": round(float(frame_counts[pitch_class]) / window_count, 8),
                "meanConfidence": round(
                    float(confidence_sums[pitch_class])
                    / max(1, int(frame_counts[pitch_class])),
                    8,
                ),
            }
            for pitch_class in range(12)
        ],
    }


def parse_audio_rows(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --vocal-audio value must be SONG_ID=PATH")
        song_id, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not song_id.strip() or not path.is_file():
            raise FileNotFoundError(f"Invalid vocal audio mapping: {value}")
        output[song_id.strip()] = path
    return output


def augment_audit(
    audit: dict[str, Any],
    audio_paths: dict[str, Path],
    *,
    radius_seconds: float,
    minimum_confidence: float,
    pitch_smoothing_frames: int = 1,
    time_offset_seconds: float = 0.0,
    comparison_time_offset_seconds: float | None = None,
) -> dict[str, Any]:
    output = dict(audit)
    songs: list[dict[str, Any]] = []
    for song in audit.get("songs") or []:
        copied = dict(song)
        song_id = str(song.get("id") or "")
        path = audio_paths.get(song_id)
        if path is None:
            copied["vocalF0"] = {"available": False}
            songs.append(copied)
            continue
        features = yin_f0(path, pitch_smoothing_frames=pitch_smoothing_frames)
        cells: list[dict[str, Any]] = []
        for cell in song.get("cells") or []:
            row = dict(cell)
            row["vocalF0Evidence"] = local_vocal_evidence(
                features,
                float(cell.get("sourceTime") or 0.0) + time_offset_seconds,
                radius_seconds=radius_seconds,
                minimum_confidence=minimum_confidence,
            )
            if comparison_time_offset_seconds is not None:
                row["vocalF0DelayedEvidence"] = local_vocal_evidence(
                    features,
                    float(cell.get("sourceTime") or 0.0)
                    + comparison_time_offset_seconds,
                    radius_seconds=radius_seconds,
                    minimum_confidence=minimum_confidence,
                )
            cells.append(row)
        copied["cells"] = cells
        copied["vocalF0"] = {
            "available": True,
            "source": str(path),
            "sampleRate": features["sampleRate"],
            "frameLength": features["frameLength"],
            "hopSamples": features["hopSamples"],
            "durationSeconds": round(float(features["durationSeconds"]), 6),
            "cellRadiusSeconds": radius_seconds,
            "minimumConfidence": minimum_confidence,
            "pitchSmoothingFrames": features["pitchSmoothingFrames"],
            "timeOffsetSeconds": time_offset_seconds,
            "comparisonTimeOffsetSeconds": comparison_time_offset_seconds,
            "minimumHz": features["minimumHz"],
            "maximumHz": features["maximumHz"],
            "troughThreshold": features["troughThreshold"],
        }
        songs.append(copied)
    output["songs"] = songs
    output["vocalF0EvidenceBoundary"] = (
        "Separated vocal audio only; reference MIDI/audio and target fields are never read."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--vocal-audio", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-confidence", type=float, default=0.72)
    parser.add_argument("--pitch-smoothing-frames", type=int, default=1)
    parser.add_argument("--time-offset-seconds", type=float, default=0.0)
    parser.add_argument("--comparison-time-offset-seconds", type=float)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8-sig"))
    mappings = parse_audio_rows(args.vocal_audio)
    radius = max(0.03, min(0.30, float(args.radius_seconds)))
    confidence = max(0.0, min(1.0, float(args.minimum_confidence)))
    smoothing_frames = max(1, min(21, int(args.pitch_smoothing_frames)))
    time_offset = max(-0.5, min(0.5, float(args.time_offset_seconds)))
    comparison_offset = (
        None
        if args.comparison_time_offset_seconds is None
        else max(-0.5, min(0.5, float(args.comparison_time_offset_seconds)))
    )
    payload = augment_audit(
        audit,
        mappings,
        radius_seconds=radius,
        minimum_confidence=confidence,
        pitch_smoothing_frames=smoothing_frames,
        time_offset_seconds=time_offset,
        comparison_time_offset_seconds=comparison_offset,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "songsWithVocalF0": [
                    song["id"]
                    for song in payload["songs"]
                    if song.get("vocalF0", {}).get("available")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
