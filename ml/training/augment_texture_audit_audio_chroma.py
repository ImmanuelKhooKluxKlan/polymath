"""Attach target-blind spectral pitch evidence to texture-audit cells.

MuScriptor JSON collapses audio into discrete notes and therefore loses the
relative spectral evidence needed to reject weak sustained false positives.
This utility reads the original full-mix WAV, measures normalized chroma near
each already-frozen candidate onset, and stores only inference-safe features.
No reference MIDI or target label is consulted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.maximum(0.0, np.asarray(vector, dtype=float))
    scale = float(np.sum(vector))
    return vector / scale if scale > 1e-12 else np.zeros_like(vector)


def read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path)
    data = np.asarray(samples)
    if data.ndim == 2:
        data = np.mean(data.astype(np.float64), axis=1)
    elif data.ndim != 1:
        raise ValueError(f"Unsupported WAV shape {data.shape} for {path}")
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        signal = data.astype(np.float64) / max(1, scale)
    else:
        signal = data.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 1.0:
        signal /= peak
    return int(sample_rate), signal


def projection_matrices(sample_rate: int, fft_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    valid = (frequencies >= 27.5) & (frequencies <= min(5000.0, sample_rate / 2.0))
    midi = np.full(len(frequencies), -1, dtype=int)
    midi[valid] = np.rint(69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)).astype(int)
    valid &= (midi >= 21) & (midi <= 108)
    counts = {
        value: max(1, int(np.sum(valid & (midi == value))))
        for value in np.unique(midi[valid])
    }
    all_projection = np.zeros((len(frequencies), 12), dtype=np.float32)
    bass_projection = np.zeros_like(all_projection)
    treble_projection = np.zeros_like(all_projection)
    for index in np.flatnonzero(valid):
        note = int(midi[index])
        weight = 1.0 / counts[note]
        pitch_class = note % 12
        all_projection[index, pitch_class] = weight
        if note < 60:
            bass_projection[index, pitch_class] = weight
        else:
            treble_projection[index, pitch_class] = weight
    return all_projection, bass_projection, treble_projection


def audio_chroma(
    path: Path, *, frame_seconds: float = 0.128, hop_seconds: float = 0.016
) -> dict[str, Any]:
    sample_rate, signal = read_mono(path)
    fft_size = 1
    desired = max(512, int(round(sample_rate * frame_seconds)))
    while fft_size < desired:
        fft_size *= 2
    hop = max(64, int(round(sample_rate * hop_seconds)))
    if len(signal) < fft_size:
        signal = np.pad(signal, (0, fft_size - len(signal)))
    frames = 1 + max(0, (len(signal) - fft_size) // hop)
    window = np.hanning(fft_size).astype(np.float64)
    all_projection, bass_projection, treble_projection = projection_matrices(
        sample_rate, fft_size
    )
    chroma = np.zeros((frames, 12), dtype=np.float32)
    bass = np.zeros_like(chroma)
    treble = np.zeros_like(chroma)
    batch_size = 256
    offsets = np.arange(fft_size)
    for start in range(0, frames, batch_size):
        stop = min(frames, start + batch_size)
        starts = np.arange(start, stop) * hop
        block = signal[starts[:, None] + offsets[None, :]] * window[None, :]
        spectrum = np.fft.rfft(block, axis=1)
        # Log compression keeps a loud bass harmonic from erasing quieter vocal
        # fundamentals while retaining relative pitch evidence.
        power = np.log1p(80.0 * np.abs(spectrum) ** 2).astype(np.float32)
        chroma[start:stop] = power @ all_projection
        bass[start:stop] = power @ bass_projection
        treble[start:stop] = power @ treble_projection
    attack = np.maximum(0.0, chroma - np.vstack([chroma[:1], chroma[:-1]]))
    times = (np.arange(frames) * hop + fft_size / 2.0) / sample_rate
    return {
        "sampleRate": sample_rate,
        "fftSize": fft_size,
        "hopSamples": hop,
        "times": times,
        "chroma": chroma,
        "attack": attack,
        "bass": bass,
        "treble": treble,
        "durationSeconds": len(signal) / sample_rate,
    }


def local_evidence(features: dict[str, Any], time: float, radius: float) -> list[dict[str, float | int]]:
    times = features["times"]
    left = int(np.searchsorted(times, time - radius, side="left"))
    right = int(np.searchsorted(times, time + radius, side="right"))
    if right <= left:
        nearest = int(np.argmin(np.abs(times - time)))
        left, right = nearest, nearest + 1
    chroma_block = features["chroma"][left:right]
    attack_block = features["attack"][left:right]
    bass_block = features["bass"][left:right]
    treble_block = features["treble"][left:right]
    mean_chroma = normalized(np.mean(chroma_block, axis=0))
    peak_chroma = normalized(np.max(chroma_block, axis=0))
    attack = normalized(np.max(attack_block, axis=0))
    bass = normalized(np.mean(bass_block, axis=0))
    treble = normalized(np.mean(treble_block, axis=0))
    return [
        {
            "pitchClass": pitch_class,
            "meanChroma": round(float(mean_chroma[pitch_class]), 8),
            "peakChroma": round(float(peak_chroma[pitch_class]), 8),
            "attackChroma": round(float(attack[pitch_class]), 8),
            "bassChroma": round(float(bass[pitch_class]), 8),
            "trebleChroma": round(float(treble[pitch_class]), 8),
        }
        for pitch_class in range(12)
    ]


def parse_audio_rows(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --audio value must be SONG_ID=PATH")
        song_id, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not song_id.strip() or not path.is_file():
            raise FileNotFoundError(f"Invalid audio mapping: {value}")
        output[song_id.strip()] = path
    return output


def augment_audit(
    audit: dict[str, Any], audio_paths: dict[str, Path], *, radius: float
) -> dict[str, Any]:
    output = dict(audit)
    songs: list[dict[str, Any]] = []
    for song in audit.get("songs") or []:
        copied = dict(song)
        song_id = str(song.get("id") or "")
        path = audio_paths.get(song_id)
        if path is None:
            copied["audioChroma"] = {"available": False}
            songs.append(copied)
            continue
        features = audio_chroma(path)
        cells: list[dict[str, Any]] = []
        for cell in song.get("cells") or []:
            row = dict(cell)
            time = float(cell.get("sourceTime") or 0.0)
            row["audioPitchClassEvidence"] = local_evidence(features, time, radius)
            cells.append(row)
        copied["cells"] = cells
        copied["audioChroma"] = {
            "available": True,
            "source": str(path),
            "sampleRate": features["sampleRate"],
            "fftSize": features["fftSize"],
            "hopSamples": features["hopSamples"],
            "durationSeconds": round(float(features["durationSeconds"]), 6),
            "cellRadiusSeconds": radius,
        }
        songs.append(copied)
    output["songs"] = songs
    output["audioEvidenceBoundary"] = (
        "Full-mix audio only. Target/reference audio and MIDI are never read."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius-seconds", type=float, default=0.08)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8-sig"))
    mappings = parse_audio_rows(args.audio)
    radius = max(0.02, min(0.25, float(args.radius_seconds)))
    output_payload = augment_audit(audit, mappings, radius=radius)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "songsWithAudio": [
                    song["id"]
                    for song in output_payload["songs"]
                    if song.get("audioChroma", {}).get("available")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
