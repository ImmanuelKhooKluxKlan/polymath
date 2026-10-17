"""Read-only, pitch-conditioned timing check of visual labels against piano audio.

This does not transcribe music, refit the frozen source/reference warp, or
measure model accuracy. Its lag search is exploratory evidence for manual
review, never permission to silently move answer-key coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .reference_region_policy import matching_policy


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_end(composition_id: str, requested: float | None) -> float | None:
    if requested is not None and (not np.isfinite(requested) or requested <= 0):
        raise ValueError("Reference cutoff must be positive and finite")
    policy = matching_policy({"compositionId": composition_id})
    if policy is None:
        return requested
    owner_end = float(policy["referenceEndSeconds"])
    return owner_end if requested is None else min(owner_end, requested)


def pitch_onset_features(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import stft

    if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) < 4096:
        raise ValueError("Need a finite mono piano signal of at least one FFT window")
    hop = max(1, int(round(sample_rate * 0.01)))
    frequencies, times, spectrum = stft(
        audio.astype(np.float32), fs=sample_rate, window="hann",
        nperseg=4096, noverlap=4096 - hop, nfft=4096,
        boundary="zeros", padded=True,
    )
    magnitude = np.abs(spectrum)
    features = []
    for midi in range(21, 109):
        fundamental = 440.0 * 2 ** ((midi - 69) / 12)
        energy = np.zeros(len(times), dtype=np.float64)
        for harmonic in range(1, 5):
            frequency = fundamental * harmonic
            if frequency >= sample_rate / 2:
                continue
            center = int(np.argmin(np.abs(frequencies - frequency)))
            band = magnitude[max(0, center - 1):center + 2]
            energy += np.max(band, axis=0) / harmonic
        scale = max(float(np.quantile(energy, 0.8)), 1e-7)
        # Use linear magnitude: log compression can overweight tiny energy
        # leaking into a centered FFT window before an isolated attack.
        scaled = energy / scale
        positive = np.maximum(0.0, scaled - np.r_[np.repeat(scaled[0], 3), scaled[:-3]])
        positive = gaussian_filter1d(positive, sigma=1)
        # Include only active flux when normalizing. A percentile over a long
        # mostly-silent signal would clip the entire attack into a plateau and
        # make argmax select its early edge instead of its physical peak.
        active = positive[positive > max(float(np.max(positive)) * 1e-3, 1e-12)]
        normalizer = max(float(np.quantile(active, 0.95)), 1e-5) if len(active) else 1e-5
        normalized = np.minimum(positive / normalizer, 4.0)
        features.append(normalized - np.mean(normalized))
    return times, np.asarray(features)


def lag_curve(frame_times, features, note_times, note_midis, lags):
    frame_times, features = np.asarray(frame_times), np.asarray(features)
    note_times, note_midis, lags = np.asarray(note_times), np.asarray(note_midis), np.asarray(lags)
    if (features.shape != (88, len(frame_times)) or len(frame_times) < 2
            or not np.all(np.diff(frame_times) > 0) or not len(lags)
            or not len(note_times) or note_times.shape != note_midis.shape
            or not np.isfinite(note_times).all() or not np.isfinite(features).all()
            or not np.isfinite(lags).all() or not np.isfinite(frame_times).all()
            or not np.isfinite(note_midis).all() or np.any(note_midis != np.floor(note_midis))
            or np.any(note_midis < 21) or np.any(note_midis > 108)):
        raise ValueError("Invalid feature grid, note pitches, or lag range")
    # Fix the eligible notes across all lags: no edge-dependent sample count.
    keep = (note_times + np.min(lags) >= frame_times[0]) & (note_times + np.max(lags) <= frame_times[-1])
    note_times, note_midis = note_times[keep], note_midis[keep].astype(int)
    if not len(note_times):
        raise ValueError("No notes have audio support across the full lag search")
    queries = note_times[None, :] + lags[:, None]
    left = np.clip(np.searchsorted(frame_times, queries, side="right") - 1, 0, len(frame_times) - 2)
    fraction = (queries - frame_times[left]) / (frame_times[left + 1] - frame_times[left])
    pitch = note_midis[None, :] - 21
    values = features[pitch, left] * (1 - fraction) + features[pitch, left + 1] * fraction
    return values.mean(axis=1), int(len(note_times))


def summarize_curve(lags, curve, count):
    best = int(np.argmax(curve))
    distant = np.abs(lags - lags[best]) > 0.15
    return {
        "notesChecked": count,
        "bestLagSeconds": round(float(lags[best]), 6),
        "bestScore": round(float(curve[best]), 6),
        "zeroLagScore": round(float(curve[int(np.argmin(np.abs(lags)))]), 6),
        "marginOverLagsMoreThan150msAway": round(float(curve[best] - np.max(curve[distant])), 6) if distant.any() else None,
        "bestAtSearchBoundary": best in {0, len(lags) - 1},
    }


def main():
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--composition-id", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-end-seconds", type=float)
    parser.add_argument("--maximum-lag-seconds", type=float, default=2.0)
    parser.add_argument("--null-permutations", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve() in {args.audio.resolve(), args.reference.resolve()}:
        raise FileExistsError("Append-only diagnostic output must be new")
    if not 0.15 < args.maximum_lag_seconds <= 5 or args.null_permutations < 1:
        raise ValueError("Invalid bounded lag search or null sample count")
    reference_end = scope_end(args.composition_id, args.reference_end_seconds)
    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    if reference.get("sourceType") != "piano-roll-video-geometry":
        raise ValueError("This audit requires unwarped visual labels in the reference video's native clock")
    notes = [n for n in reference["notes"] if reference_end is None or float(n["time"]) < reference_end]
    note_times = np.asarray([float(n["time"]) for n in notes])
    note_midis = np.asarray([int(n["midi"]) for n in notes])
    audio, sample_rate = sf.read(args.audio, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    times, features = pitch_onset_features(mono, sample_rate)
    steps = int(round(args.maximum_lag_seconds / 0.01))
    lags = np.arange(-steps, steps + 1) * 0.01
    curve, count = lag_curve(times, features, note_times, note_midis, lags)
    rng = np.random.default_rng(20261017)
    null_best = []
    for _ in range(args.null_permutations):
        null_curve, _count = lag_curve(times, features, note_times, rng.permutation(note_midis), lags)
        null_best.append(float(np.max(null_curve)))
    blocks = []
    duration = len(mono) / sample_rate
    for start in np.arange(0, duration, 60.0):
        selected = (note_times >= start) & (note_times < start + 60)
        if selected.sum() < 10:
            continue
        block_curve, block_count = lag_curve(times, features, note_times[selected], note_midis[selected], lags)
        blocks.append({"start": float(start), "end": min(float(start + 60), duration), **summarize_curve(lags, block_curve, block_count)})
    result = {
        "schema": "polymath-visual-audio-timing-diagnostic-v1",
        "implementationSha256": sha256(Path(__file__)),
        "referenceMutated": False, "weightsMutated": False, "productionChanged": False,
        "audio": {"path": str(args.audio.resolve()), "sha256": sha256(args.audio), "sampleRate": sample_rate, "seconds": duration},
        "reference": {"compositionId": args.composition_id, "path": str(args.reference.resolve()), "sha256": sha256(args.reference), "endSecondsExclusive": reference_end, "visualTimingMetadata": reference.get("visualExtraction", {}).get("timing")},
        "method": "Centered 4096-point harmonic linear-magnitude STFT positive flux, 10ms hop, pitch-conditioned linear sampling; fixed eligible notes across lags",
        "lagConvention": "Positive means audio evidence occurs later than the existing label coordinate",
        "global": summarize_curve(lags, curve, count), "sixtySecondBlocks": blocks,
        "pitchShuffleControl": {"seed": 20261017, "permutations": args.null_permutations, "maximumBestScore": float(max(null_best)), "medianBestScore": float(np.median(null_best)), "observedExceedsAllShuffles": float(np.max(curve)) > max(null_best)},
        "curve": [{"lagSeconds": round(float(lag), 6), "score": round(float(score), 6)} for lag, score in zip(lags, curve)],
        "interpretation": "Exploratory timing self-consistency only, not model accuracy or independent timing certification. The front-end window/flux and real piano attack envelope can shift peaks by tens of milliseconds. Harmonics, dense chords, recording edits and repeated rhythms can confound the peak. Never silently rewrite the frozen benchmark from this result.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"global": result["global"], "blocks": blocks, "shuffleControl": result["pitchShuffleControl"], "outSha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
