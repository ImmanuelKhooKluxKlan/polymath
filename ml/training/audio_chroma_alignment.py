"""Align a piano-cover timeline to a full-mix timeline from the audio itself.

The note-coordinate aligner is useful for a coarse estimate, but it can become
uncertain when the full mix contains thousands of percussion and guitar events.
This module compares instrument-invariant pitch-class energy (chroma), then uses
a constrained monotonic dynamic-time-warping path to model local tempo drift.

It intentionally has only one runtime dependency: NumPy, which is already part
of the Polymath training environment.  Generated reports use the same
``referenceTime`` -> ``observedTime`` anchor contract as the route evaluator.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-9
MAXIMUM_NOTE_SUPPORT_SECONDS = 0.25
PERCUSSION_INSTRUMENTS = {"drums", "timpani"}
MINIMUM_TRANSPOSITION_EVIDENCE_MARGIN = 0.03


@dataclass(frozen=True)
class AudioFeatures:
    times: np.ndarray
    chroma: np.ndarray
    onset: np.ndarray
    energy: np.ndarray
    duration_seconds: float
    sample_rate: int
    hop_seconds: float


def _finite_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def load_mono_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        raw = source.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit: {path}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size < sample_rate:
        raise ValueError(f"Audio is too short for alignment: {path}")
    return samples, sample_rate


def extract_chroma_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_seconds: float = 0.10,
    fft_seconds: float = 0.256,
    minimum_frequency: float = 55.0,
    maximum_frequency: float = 3520.0,
) -> AudioFeatures:
    """Create pitch-class and onset features without loading a DSP framework."""
    hop = max(1, int(round(sample_rate * hop_seconds)))
    requested_fft = max(1024, int(round(sample_rate * fft_seconds)))
    fft_size = 1 << int(math.ceil(math.log2(requested_fft)))
    if samples.size < fft_size:
        samples = np.pad(samples, (0, fft_size - samples.size))
    frame_count = 1 + int(math.ceil((samples.size - fft_size) / hop))
    required = (frame_count - 1) * hop + fft_size
    if required > samples.size:
        samples = np.pad(samples, (0, required - samples.size))

    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    usable = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    usable_indices = np.flatnonzero(usable)
    usable_frequencies = frequencies[usable_indices]
    midi_bins = np.rint(69.0 + 12.0 * np.log2(usable_frequencies / 440.0)).astype(int)
    pitch_classes = np.mod(midi_bins, 12)
    # Equal-loudness-like emphasis prevents high-frequency broadband percussion
    # from overpowering the harmonic fundamental and low overtones.
    frequency_weights = np.clip((440.0 / usable_frequencies) ** 0.22, 0.55, 1.8)
    window = np.hanning(fft_size).astype(np.float32)

    chroma = np.zeros((frame_count, 12), dtype=np.float32)
    energy = np.zeros(frame_count, dtype=np.float32)
    spectral_flux = np.zeros(frame_count, dtype=np.float32)
    previous_spectrum: np.ndarray | None = None
    batch_size = 192
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(frame_count, batch_start + batch_size)
        starts = np.arange(batch_start, batch_end) * hop
        frames = np.stack([samples[start : start + fft_size] for start in starts])
        energy[batch_start:batch_end] = np.sqrt(np.mean(np.square(frames), axis=1))
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))[:, usable_indices]
        spectrum = np.log1p(spectrum * frequency_weights[None, :]).astype(np.float32)
        for pitch_class in range(12):
            chroma[batch_start:batch_end, pitch_class] = spectrum[:, pitch_classes == pitch_class].sum(axis=1)

        for local_index, current in enumerate(spectrum):
            global_index = batch_start + local_index
            if previous_spectrum is not None:
                spectral_flux[global_index] = np.maximum(current - previous_spectrum, 0.0).mean()
            previous_spectrum = current

    # Broadband energy is shared by most pitch classes. Removing each frame's
    # lower quartile makes harmonic peaks dominate while retaining soft chords.
    floor = np.quantile(chroma, 0.25, axis=1, keepdims=True)
    chroma = np.maximum(chroma - floor, 0.0)
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    chroma = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > EPSILON)

    def robust_scale(values: np.ndarray) -> np.ndarray:
        low, high = np.quantile(values, [0.10, 0.90])
        return np.clip((values - low) / max(EPSILON, high - low), 0.0, 1.0).astype(np.float32)

    onset = robust_scale(spectral_flux)
    normalized_energy = robust_scale(energy)
    times = (np.arange(frame_count) * hop + fft_size / 2) / sample_rate
    return AudioFeatures(
        times=times.astype(np.float64),
        chroma=chroma,
        onset=onset,
        energy=normalized_energy,
        duration_seconds=float(samples.size / sample_rate),
        sample_rate=sample_rate,
        hop_seconds=float(hop / sample_rate),
    )


def feature_file(path: Path, *, hop_seconds: float = 0.10) -> AudioFeatures:
    samples, sample_rate = load_mono_pcm_wav(path)
    return extract_chroma_features(samples, sample_rate, hop_seconds=hop_seconds)


def load_seed_mapping(path: Path | None, reference_duration: float, observed_duration: float) -> tuple[float, float]:
    if path and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        coarse = payload.get("coarse") or {}
        scale = _finite_number(coarse.get("scale"), 1.0)
        offset = _finite_number(coarse.get("offset"), 0.0)
        if 0.70 <= scale <= 1.30 and abs(offset) <= 30.0:
            return scale, offset
    # A duration ratio is only a seed; DTW remains free to model local drift.
    return observed_duration / max(EPSILON, reference_duration), 0.0


def estimate_pitch_class_shift(
    reference: AudioFeatures,
    observed: AudioFeatures,
    scale: float,
    offset: float,
) -> tuple[int, list[float]]:
    similarities: list[float] = []
    reference_indices = np.arange(0, len(reference.times), max(1, int(round(0.5 / reference.hop_seconds))))
    mapped_times = reference.times[reference_indices] * scale + offset
    observed_indices = np.rint(mapped_times / observed.hop_seconds).astype(int)
    valid = (
        (observed_indices >= 0)
        & (observed_indices < len(observed.times))
        & (reference.energy[reference_indices] >= 0.08)
    )
    reference_rows = reference.chroma[reference_indices[valid]]
    observed_rows = observed.chroma[observed_indices[valid]]
    if not len(reference_rows):
        return 0, [0.0] * 12
    for shift in range(12):
        shifted = np.roll(observed_rows, shift, axis=1)
        similarities.append(float(np.mean(np.sum(reference_rows * shifted, axis=1))))
    best = int(np.argmax(similarities))
    # Pop progressions are highly symmetric: a dominant/subdominant can beat
    # the true key by a few thousandths when the seed clock is imperfect.  A
    # transposition changes every training label, so fail closed to no shift
    # unless it beats the unshifted evidence by a meaningful margin.
    if best and similarities[best] - similarities[0] < MINIMUM_TRANSPOSITION_EVIDENCE_MARGIN:
        best = 0
    return best, similarities


def _frame_cost(reference: AudioFeatures, observed: AudioFeatures, i: int, j: int, shift: int) -> float:
    observed_chroma = np.roll(observed.chroma[j], shift)
    harmonic = 1.0 - float(np.dot(reference.chroma[i], observed_chroma))
    onset = abs(float(reference.onset[i]) - float(observed.onset[j]))
    energy = abs(float(reference.energy[i]) - float(observed.energy[j]))
    if reference.energy[i] < 0.025 and observed.energy[j] < 0.025:
        harmonic = min(harmonic, 0.20)
    elif min(reference.energy[i], observed.energy[j]) < 0.025:
        harmonic += 0.18
    return harmonic + 0.16 * onset + 0.035 * energy


def constrained_dtw(
    reference: AudioFeatures,
    observed: AudioFeatures,
    *,
    coarse_scale: float,
    coarse_offset: float,
    band_seconds: float = 6.0,
    gap_penalty: float = 0.055,
    pitch_class_shift: int = 0,
) -> tuple[list[tuple[int, int]], dict[str, float]]:
    """Return a monotonic frame path inside a band around the coarse line."""
    n = len(reference.times)
    m = len(observed.times)
    band_frames = max(3, int(round(band_seconds / observed.hop_seconds)))
    back = np.full((n, m), 255, dtype=np.uint8)
    previous = np.full(m, np.inf, dtype=np.float64)
    first_i: int | None = None
    last_i: int | None = None

    for i, reference_time in enumerate(reference.times):
        predicted_time = reference_time * coarse_scale + coarse_offset
        predicted = int(round(predicted_time / observed.hop_seconds))
        low = max(0, predicted - band_frames)
        high = min(m - 1, predicted + band_frames)
        if low > high:
            continue
        current = np.full(m, np.inf, dtype=np.float64)
        if first_i is None:
            first_i = i
            for j in range(low, high + 1):
                start_penalty = abs(j - predicted) * gap_penalty * 0.35
                current[j] = _frame_cost(reference, observed, i, j, pitch_class_shift) + start_penalty
                back[i, j] = 3
        else:
            for j in range(low, high + 1):
                diagonal = previous[j - 1] if j > 0 else np.inf
                vertical = previous[j] + gap_penalty
                horizontal = current[j - 1] + gap_penalty if j > low else np.inf
                options = (diagonal, vertical, horizontal)
                direction = int(np.argmin(options))
                best = options[direction]
                if math.isfinite(best):
                    current[j] = best + _frame_cost(reference, observed, i, j, pitch_class_shift)
                    back[i, j] = direction
        if np.isfinite(current).any():
            previous = current
            last_i = i

    if first_i is None or last_i is None:
        raise ValueError("The coarse map does not overlap the two recordings")
    end_prediction = int(round((reference.times[last_i] * coarse_scale + coarse_offset) / observed.hop_seconds))
    valid = np.flatnonzero(np.isfinite(previous))
    if not len(valid):
        raise ValueError("Dynamic time warping found no valid path")
    end_j = min(
        (int(index) for index in valid),
        key=lambda index: previous[index] + abs(index - end_prediction) * gap_penalty * 0.35,
    )
    total_cost = float(previous[end_j])
    path: list[tuple[int, int]] = []
    i, j = last_i, end_j
    while i >= first_i and j >= 0:
        path.append((i, j))
        direction = int(back[i, j])
        if direction == 3:
            break
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise ValueError("Dynamic time warping path is incomplete")
    path.reverse()
    if len(path) < min(n, m) * 0.25:
        raise ValueError("Dynamic time warping path covers too little audio")
    return path, {
        "totalCost": total_cost,
        "meanCost": total_cost / max(1, len(path)),
        "pathFrames": float(len(path)),
        "referenceStartFrame": float(path[0][0]),
        "referenceEndFrame": float(path[-1][0]),
        "observedStartFrame": float(path[0][1]),
        "observedEndFrame": float(path[-1][1]),
    }


def path_to_anchors(
    path: list[tuple[int, int]],
    reference: AudioFeatures,
    observed: AudioFeatures,
    *,
    interval_seconds: float = 2.0,
    pitch_class_shift: int = 0,
) -> list[dict[str, Any]]:
    by_reference: dict[int, list[int]] = {}
    for i, j in path:
        by_reference.setdefault(i, []).append(j)
    reference_indices = np.asarray(sorted(by_reference), dtype=int)
    observed_indices = np.asarray(
        [int(round(float(np.median(by_reference[index])))) for index in reference_indices],
        dtype=int,
    )
    if len(observed_indices) >= 5:
        radius = 2
        smoothed = observed_indices.copy()
        for index in range(len(observed_indices)):
            low = max(0, index - radius)
            high = min(len(observed_indices), index + radius + 1)
            smoothed[index] = int(round(float(np.median(observed_indices[low:high]))))
        observed_indices = np.maximum.accumulate(smoothed)

    every = max(1, int(round(interval_seconds / reference.hop_seconds)))
    chosen_positions = list(range(0, len(reference_indices), every))
    if chosen_positions[-1] != len(reference_indices) - 1:
        chosen_positions.append(len(reference_indices) - 1)
    anchors: list[dict[str, Any]] = []
    for position in chosen_positions:
        i = int(reference_indices[position])
        j = int(observed_indices[position])
        similarity = float(
            np.dot(reference.chroma[i], np.roll(observed.chroma[j], pitch_class_shift))
        )
        candidate = {
            "referenceTime": round(float(reference.times[i]), 6),
            "observedTime": round(float(observed.times[j]), 6),
            "support": 1,
            "structuralSimilarity": round(similarity, 6),
            "kind": "audio-chroma-dtw",
        }
        # A long silence can produce horizontal/vertical DTW runs.  Flat time
        # anchors make duration mapping collapse to zero and are not useful as
        # musical coordinates, so keep the path but omit those duplicate knots.
        if anchors and candidate["observedTime"] <= anchors[-1]["observedTime"] + 0.02:
            continue
        anchors.append(candidate)
    if len(anchors) < 2:
        raise ValueError("Audio alignment produced fewer than two monotonic anchors")
    return anchors


def map_time(value: float, anchors: list[dict[str, Any]]) -> float:
    references = [float(anchor["referenceTime"]) for anchor in anchors]
    position = bisect.bisect_right(references, value)
    if position <= 0:
        left, right = anchors[0], anchors[1]
    elif position >= len(anchors):
        left, right = anchors[-2], anchors[-1]
    else:
        left, right = anchors[position - 1], anchors[position]
    left_reference = float(left["referenceTime"])
    right_reference = float(right["referenceTime"])
    scale = (float(right["observedTime"]) - float(left["observedTime"])) / max(
        EPSILON, right_reference - left_reference
    )
    return float(left["observedTime"]) + (value - left_reference) * scale


def normalized_notes(payload: dict[str, Any], *, exclude_percussion: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_index, item in enumerate(payload.get("notes", [])):
        instrument = str(item.get("instrument") or "").strip().lower()
        if exclude_percussion and instrument in PERCUSSION_INSTRUMENTS:
            continue
        midi = int(round(_finite_number(item.get("midi", item.get("pitch")), -1)))
        time = _finite_number(item.get("time", item.get("startTime")), -1)
        duration = max(0.01, _finite_number(item.get("duration"), 0.2))
        if 0 <= midi <= 127 and time >= 0:
            result.append(
                {
                    **item,
                    "sourceIndex": int(item.get("sourceIndex", source_index)),
                    "midi": midi,
                    "pitchClass": midi % 12,
                    "time": time,
                    "duration": duration,
                }
            )
    return sorted(result, key=lambda note: (note["time"], note["midi"]))


def coordinate_matches(
    reference_notes: list[dict[str, Any]],
    observed_notes: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    *,
    maximum_distance_seconds: float = MAXIMUM_NOTE_SUPPORT_SECONDS,
) -> list[dict[str, Any]]:
    """Match desired notes to detected source events on the frozen audio warp.

    The audio supplies the clock.  Notes only provide supervised selection and
    octave evidence after that clock has been frozen, preventing the model's own
    predictions from bending the target timeline in its favour.  A pair outside
    the evaluator's broad 250 ms onset tolerance is not evidence of alignment;
    keeping it unmatched also prevents a false far-away pair from poisoning a
    window's timing-tail statistic.
    """
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {pitch_class: [] for pitch_class in range(12)}
    for index, note in enumerate(observed_notes):
        grouped[note["midi"] % 12].append((index, note))
    grouped_times = {
        pitch_class: [item[1]["time"] for item in items]
        for pitch_class, items in grouped.items()
    }
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    last_observed_time = -math.inf
    for reference_index, reference in enumerate(reference_notes):
        predicted = map_time(reference["time"], anchors)
        pitch_class = reference["midi"] % 12
        items = grouped[pitch_class]
        times = grouped_times[pitch_class]
        position = bisect.bisect_left(times, predicted)
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for candidate_position in range(max(0, position - 8), min(len(items), position + 9)):
            observed_index, note = items[candidate_position]
            distance = abs(note["time"] - predicted)
            if (
                observed_index not in used
                and distance <= maximum_distance_seconds
                and note["time"] >= last_observed_time - 0.10
            ):
                candidates.append((distance, observed_index, note))
        if not candidates:
            continue
        distance, observed_index, chosen = min(
            candidates,
            key=lambda item: (
                item[0] + (0.0 if item[2]["midi"] == reference["midi"] else 0.035),
                abs(item[2]["midi"] - reference["midi"]),
            ),
        )
        used.add(observed_index)
        last_observed_time = max(last_observed_time, float(chosen["time"]))
        matches.append(
            {
                "referenceIndex": reference_index,
                "observedIndex": int(chosen["sourceIndex"]),
                "reference": reference,
                "observed": chosen,
                "expectedTime": round(predicted, 6),
                "coarseResidual": round(float(chosen["time"]) - predicted, 6),
                "exactPitch": chosen["midi"] == reference["midi"],
                "octaveDifference": int(round((chosen["midi"] - reference["midi"]) / 12)),
                "cost": round(distance / max(EPSILON, maximum_distance_seconds), 6),
            }
        )
    return matches


def timing_support_metrics(
    reference_notes: list[dict[str, Any]],
    observed_notes: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    residuals = [abs(float(match["coarseResidual"])) for match in matches]
    exact = sum(bool(match["exactPitch"]) for match in matches)
    ordered = sorted(residuals)
    return {
        "matchedNotes": len(matches),
        "unmatchedReferenceNotes": len(reference_notes) - len(matches),
        "unmatchedObservedNotes": len(observed_notes) - len({match["observedIndex"] for match in matches}),
        "matchedReferencePercent": round(100.0 * len(matches) / max(1, len(reference_notes)), 3),
        "exactPitchMatches": exact,
        "exactPitchPercent": round(100.0 * exact / max(1, len(matches)), 3),
        "medianTimingResidualMs": round(1000.0 * median(ordered), 3) if ordered else None,
        "p95TimingResidualMs": round(1000.0 * ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3) if ordered else None,
        "within100msPercent": round(100.0 * sum(value <= 0.10 for value in residuals) / max(1, len(residuals)), 3),
        "within250msPercent": round(100.0 * sum(value <= 0.25 for value in residuals) / max(1, len(residuals)), 3),
    }


def alignment_windows(
    path: list[tuple[int, int]],
    reference: AudioFeatures,
    observed: AudioFeatures,
    anchors: list[dict[str, Any]],
    reference_notes: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    *,
    seconds: float = 5.0,
    pitch_class_shift: int = 0,
) -> list[dict[str, Any]]:
    similarities: dict[int, list[float]] = {}
    for i, j in path:
        window = int(reference.times[i] // seconds)
        similarities.setdefault(window, []).append(
            float(np.dot(reference.chroma[i], np.roll(observed.chroma[j], pitch_class_shift)))
        )
    output: list[dict[str, Any]] = []
    match_by_reference = {int(match["referenceIndex"]): match for match in matches}
    for window, values in sorted(similarities.items()):
        reference_start = window * seconds
        reference_end = min(reference.duration_seconds, reference_start + seconds)
        score = float(np.median(values))
        entries = [
            (index, note)
            for index, note in enumerate(reference_notes)
            if reference_start <= float(note["time"]) < reference_end
        ]
        window_matches = [match_by_reference[index] for index, _note in entries if index in match_by_reference]
        residuals = sorted(abs(float(match["coarseResidual"])) for match in window_matches)
        coverage = len(window_matches) / max(1, len(entries))
        exact_share = sum(bool(match["exactPitch"]) for match in window_matches) / max(1, len(window_matches))
        median_residual = median(residuals) if residuals else math.inf
        p95_residual = residuals[min(len(residuals) - 1, int(len(residuals) * 0.95))] if residuals else math.inf
        source_start = max(0.0, map_time(reference_start, anchors))
        source_end = min(observed.duration_seconds, map_time(reference_end, anchors))
        local_scale = (source_end - source_start) / max(EPSILON, reference_end - reference_start)
        enough_notes = len(entries) >= 4
        trusted = (
            enough_notes
            and score >= 0.42
            and coverage >= 0.55
            and median_residual <= 0.12
            and p95_residual <= 0.30
            and 0.65 <= local_scale <= 1.45
        )
        review = (
            enough_notes
            and score >= 0.28
            and coverage >= 0.30
            and median_residual <= 0.25
            and 0.40 <= local_scale <= 2.0
        )
        status = "trusted" if trusted else "review" if review else "unsafe"
        flags: list[str] = []
        if not enough_notes:
            flags.append("reference-silence-or-too-few-notes")
        if score < 0.42:
            flags.append("weak-audio-structure")
        if coverage < 0.55:
            flags.append("low-note-support")
        if not residuals or median_residual > 0.12 or p95_residual > 0.30:
            flags.append("timing-residual")
        if not 0.65 <= local_scale <= 1.45:
            flags.append("suspicious-local-tempo")
        output.append(
            {
                "id": f"w{window + 1:04d}",
                "referenceStartSeconds": round(reference_start, 4),
                "referenceEndSeconds": round(reference_end, 4),
                "sourceStartSeconds": round(source_start, 4),
                "sourceEndSeconds": round(source_end, 4),
                # Compatibility fields consumed by dataset_builder.py.
                "referenceStart": round(reference_start, 4),
                "referenceEnd": round(reference_end, 4),
                "sourceStart": round(source_start, 4),
                "sourceEnd": round(source_end, 4),
                "structuralSimilarity": round(score, 6),
                "referenceNotes": len(entries),
                "matchedNotes": len(window_matches),
                "matchPercent": round(100.0 * coverage, 3),
                "matchedPercent": round(100.0 * coverage, 3),
                "exactPitchPercent": round(100.0 * exact_share, 3),
                "medianResidualMs": round(1000.0 * median_residual, 3) if residuals else None,
                "p95ResidualMs": round(1000.0 * p95_residual, 3) if residuals else None,
                "localScale": round(local_scale, 6),
                "localTempoDifferencePercent": round((local_scale - 1.0) * 100.0, 3),
                "automaticStatus": status,
                "decision": "auto",
                "status": status,
                "trainingEligible": status == "trusted",
                "flags": flags,
            }
        )
    return output


def supervision_package(
    payload: dict[str, Any],
    report: dict[str, Any],
    source_duration: float,
    reference_duration: float,
) -> dict[str, Any]:
    anchors = report["anchors"]
    windows = report["qualityWindows"]
    mapped: list[dict[str, Any]] = []
    for note in normalized_notes(payload):
        start = map_time(note["time"], anchors)
        end = map_time(note["time"] + note["duration"], anchors)
        if start > source_duration or end < 0:
            continue
        window = next(
            (
                item
                for item in windows
                if float(item["referenceStart"]) <= float(note["time"]) < float(item["referenceEnd"])
            ),
            None,
        )
        status = str((window or {}).get("status") or "unsafe")
        mapped.append(
            {
                **note,
                "originalTime": round(note["time"], 6),
                "time": round(max(0.0, start), 6),
                "duration": round(max(0.01, min(source_duration, end) - max(0.0, start)), 6),
                "instrument": "acoustic_piano",
                "qualityWindowId": (window or {}).get("id"),
                "qualityStatus": status,
                "trainingEligible": status == "trusted",
            }
        )
    tempo_segments = [
        {
            "referenceStart": left["referenceTime"],
            "referenceEnd": right["referenceTime"],
            "sourceStart": left["observedTime"],
            "sourceEnd": right["observedTime"],
            "localScale": round(
                (float(right["observedTime"]) - float(left["observedTime"]))
                / max(EPSILON, float(right["referenceTime"]) - float(left["referenceTime"])),
                6,
            ),
        }
        for left, right in zip(anchors, anchors[1:])
    ]
    return {
        "schema": "polymath-supervision-package-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "timeline": {
            "sourceDurationSeconds": round(source_duration, 6),
            "referenceDurationSeconds": round(reference_duration, 6),
            "rule": (
                "Reference piano notes are warped onto the source audio timeline by "
                "frozen audio-chroma DTW; only trusted windows may create gradients."
            ),
        },
        "alignment": {
            "method": report["method"],
            "metrics": report["metrics"],
            "coarse": report["coarse"],
            "anchors": anchors,
            "manualAnchors": [],
            "tempoSegments": tempo_segments,
            "qualityWindows": windows,
        },
        "review": {
            "status": "automatic-preflight-complete",
            "manualReviewRequired": any(window["status"] != "trusted" for window in windows),
            "trainingPolicy": "trusted-windows-only",
        },
        "notes": sorted(mapped, key=lambda note: (note["time"], note["midi"])),
    }


def align_files(
    reference_audio_path: Path,
    observed_audio_path: Path,
    reference_notes_path: Path,
    observed_notes_path: Path,
    output_directory: Path,
    *,
    seed_report_path: Path | None = None,
    hop_seconds: float = 0.10,
    band_seconds: float = 6.0,
) -> dict[str, Any]:
    reference = feature_file(reference_audio_path, hop_seconds=hop_seconds)
    observed = feature_file(observed_audio_path, hop_seconds=hop_seconds)
    coarse_scale, coarse_offset = load_seed_mapping(
        seed_report_path,
        reference.duration_seconds,
        observed.duration_seconds,
    )
    pitch_class_shift, shift_scores = estimate_pitch_class_shift(
        reference, observed, coarse_scale, coarse_offset
    )
    path, path_metrics = constrained_dtw(
        reference,
        observed,
        coarse_scale=coarse_scale,
        coarse_offset=coarse_offset,
        band_seconds=band_seconds,
        pitch_class_shift=pitch_class_shift,
    )
    anchors = path_to_anchors(
        path,
        reference,
        observed,
        pitch_class_shift=pitch_class_shift,
    )
    reference_payload = json.loads(reference_notes_path.read_text(encoding="utf-8"))
    observed_payload = json.loads(observed_notes_path.read_text(encoding="utf-8"))
    reference_notes = normalized_notes(reference_payload)
    observed_notes = normalized_notes(observed_payload, exclude_percussion=True)
    matches = coordinate_matches(reference_notes, observed_notes, anchors)
    support = timing_support_metrics(reference_notes, observed_notes, matches)
    windows = alignment_windows(
        path,
        reference,
        observed,
        anchors,
        reference_notes,
        matches,
        pitch_class_shift=pitch_class_shift,
    )
    slopes = [
        (float(right["observedTime"]) - float(left["observedTime"]))
        / max(EPSILON, float(right["referenceTime"]) - float(left["referenceTime"]))
        for left, right in zip(anchors, anchors[1:])
    ]
    trusted = sum(window["status"] == "trusted" for window in windows)
    report = {
        "schema": "polymath-audio-chroma-alignment-v1",
        "method": "harmonic-chroma-constrained-dtw",
        "referenceAudio": str(reference_audio_path.resolve()),
        "observedAudio": str(observed_audio_path.resolve()),
        "referenceNotes": str(reference_notes_path.resolve()),
        "observedNotes": str(observed_notes_path.resolve()),
        "coarse": {"scale": coarse_scale, "offset": coarse_offset},
        "anchors": anchors,
        "matches": matches,
        "qualityWindows": windows,
        "metrics": {
            **support,
            "referenceDurationSeconds": round(reference.duration_seconds, 6),
            "observedDurationSeconds": round(observed.duration_seconds, 6),
            "anchorCount": len(anchors),
            "pathFrames": int(path_metrics["pathFrames"]),
            "pathMeanCost": round(path_metrics["meanCost"], 6),
            "pitchClassShiftForAlignment": pitch_class_shift,
            "pitchClassShiftEvidenceMargin": round(
                shift_scores[pitch_class_shift] - shift_scores[0], 6
            ),
            "pitchClassShiftScores": [round(value, 6) for value in shift_scores],
            "medianLocalScale": round(median(slopes), 6) if slopes else None,
            "minimumLocalScale": round(min(slopes), 6) if slopes else None,
            "maximumLocalScale": round(max(slopes), 6) if slopes else None,
            "qualityWindowCount": len(windows),
            "trustedWindowCount": trusted,
            "reviewWindowCount": sum(window["status"] == "review" for window in windows),
            "unsafeWindowCount": sum(window["status"] == "unsafe" for window in windows),
            "trustedTimelinePercent": round(
                100.0
                * sum(
                    max(0.0, float(window["sourceEndSeconds"]) - float(window["sourceStartSeconds"]))
                    for window in windows
                    if window["status"] == "trusted"
                )
                / max(EPSILON, observed.duration_seconds),
                3,
            ),
        },
    }
    confidence = (
        0.45 * min(1.0, float(support["within100msPercent"]) / 75.0)
        + 0.35 * min(1.0, float(support["matchedReferencePercent"]) / 80.0)
        + 0.20 * trusted / max(1, len(windows))
    )
    report["metrics"]["confidence"] = round(confidence, 4)
    report["metrics"]["verdict"] = (
        "review-likely-pass"
        if confidence >= 0.82 and report["metrics"]["unsafeWindowCount"] <= 1
        else "manual-review-required"
        if confidence >= 0.60
        else "reject-or-add-manual-anchors"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "alignment-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    mapped = supervision_package(
        reference_payload,
        report,
        observed.duration_seconds,
        reference.duration_seconds,
    )
    (output_directory / "aligned-training-labels.json").write_text(
        json.dumps(mapped, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--observed-audio", type=Path, required=True)
    parser.add_argument("--reference-notes", type=Path, required=True)
    parser.add_argument("--observed-notes", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hop-seconds", type=float, default=0.10)
    parser.add_argument("--band-seconds", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    report = align_files(
        args.reference_audio,
        args.observed_audio,
        args.reference_notes,
        args.observed_notes,
        args.out,
        seed_report_path=args.seed_report,
        hop_seconds=args.hop_seconds,
        band_seconds=args.band_seconds,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
