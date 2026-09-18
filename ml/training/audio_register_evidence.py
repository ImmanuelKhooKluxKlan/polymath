"""Audio-side evidence for conservative octave routing.

The symbolic decoder often identifies the correct pitch class but chooses the
wrong octave.  This module measures the source waveform at the decoded onset
without reading an answer sheet.  It deliberately exposes evidence only for
octave-equivalent alternatives, so it cannot invent a different note name or
change timing, duration, density, or velocity.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .fit_pianist_register_router import SHIFT_CLASSES


FFT_SIZE = 4096
MINIMUM_MIDI = 33
MAXIMUM_MIDI = 108
ODD_HARMONIC_WEIGHTS = (1.0, 0.48, 0.32)


def _shift_name(shift: int) -> str:
    return f"m{abs(shift)}" if shift < 0 else f"p{shift}"


AUDIO_FEATURE_NAMES = (
    *(f"audio_fundamental_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    *(f"audio_odd_harmonic_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    *(f"audio_fundamental_delta_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    *(f"audio_odd_harmonic_delta_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    *(f"audio_fundamental_winner_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    *(f"audio_odd_harmonic_winner_{_shift_name(shift)}" for shift in SHIFT_CLASSES),
    "audio_fundamental_winner_gain",
    "audio_odd_harmonic_winner_gain",
)


def midi_frequency(midi: int) -> float:
    return 440.0 * 2.0 ** ((int(midi) - 69) / 12.0)


def load_pcm_wave(path: Path) -> tuple[np.ndarray, int]:
    """Load an ordinary PCM WAV and return normalized mono float32 samples."""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Audio register evidence requires 16-bit PCM WAV: {path}")
    if channels < 1:
        raise ValueError(f"WAV has no channels: {path}")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if values.size % channels:
        raise ValueError(f"WAV frame data is truncated: {path}")
    values = values.reshape(-1, channels).mean(axis=1) / 32768.0
    return values.astype(np.float32, copy=False), int(sample_rate)


class AudioRegisterEvidence:
    """Cache short onset spectra and score octave-equivalent fundamentals."""

    def __init__(self, samples: np.ndarray, sample_rate: int, fft_size: int = FFT_SIZE):
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("Audio samples must be a non-empty mono vector")
        if sample_rate < 8_000:
            raise ValueError("Audio sample rate is too low for piano register evidence")
        self.samples = samples.astype(np.float32, copy=False)
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.frequencies = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        self._cache: dict[int, np.ndarray] = {}

    @classmethod
    def from_wav(cls, path: Path) -> "AudioRegisterEvidence":
        samples, sample_rate = load_pcm_wave(path)
        return cls(samples, sample_rate)

    def spectrum(self, time_seconds: float) -> np.ndarray:
        # A 15 ms lead includes the piano hammer transient while the 256 ms
        # window is long enough to resolve bass fundamentals at 16 kHz.
        start = int(round((float(time_seconds) - 0.015) * self.sample_rate))
        if start in self._cache:
            return self._cache[start]
        frame = np.zeros(self.fft_size, dtype=np.float32)
        source_start = max(0, start)
        source_end = min(self.samples.size, start + self.fft_size)
        if source_end > source_start:
            frame[source_start - start : source_end - start] = self.samples[
                source_start:source_end
            ]
        spectrum = np.abs(np.fft.rfft(frame * self.window)) ** 2
        self._cache[start] = spectrum
        return spectrum

    def local_ratio(self, spectrum: np.ndarray, frequency: float) -> float:
        nyquist = self.sample_rate / 2.0
        if frequency <= 20.0 or frequency >= nyquist * 0.96:
            return 1e-8
        narrow_start = int(np.searchsorted(self.frequencies, frequency * 0.982))
        narrow_end = int(np.searchsorted(self.frequencies, frequency * 1.018)) + 1
        signal = float(
            np.mean(spectrum[max(0, narrow_start) : min(spectrum.size, narrow_end)])
        ) + 1e-12
        local_start = int(np.searchsorted(self.frequencies, frequency / 1.35))
        local_end = int(np.searchsorted(self.frequencies, frequency * 1.35))
        local = spectrum[max(1, local_start) : min(spectrum.size, local_end)]
        noise = float(np.median(local)) + 1e-12 if local.size else 1e-12
        return max(1e-8, signal / noise)

    def score(self, time_seconds: float, candidate_midi: int) -> dict[str, Any]:
        spectrum = self.spectrum(time_seconds)
        fundamental: dict[int, float] = {}
        odd: dict[int, float] = {}
        valid: dict[int, bool] = {}
        for shift in SHIFT_CLASSES:
            midi = int(candidate_midi) + int(shift)
            valid[shift] = MINIMUM_MIDI <= midi <= MAXIMUM_MIDI
            if not valid[shift]:
                fundamental[shift] = -8.0
                odd[shift] = -8.0
                continue
            frequency = midi_frequency(midi)
            ratios = [
                self.local_ratio(spectrum, frequency * harmonic)
                for harmonic in (1, 3, 5)
            ]
            fundamental[shift] = float(np.clip(math.log(ratios[0]), -8.0, 8.0))
            odd[shift] = float(
                np.clip(
                    sum(
                        weight * math.log(ratio)
                        for weight, ratio in zip(ODD_HARMONIC_WEIGHTS, ratios)
                    ),
                    -12.0,
                    12.0,
                )
            )

        valid_shifts = [shift for shift in SHIFT_CLASSES if valid[shift]]
        fundamental_winner = max(valid_shifts, key=lambda shift: fundamental[shift])
        odd_winner = max(valid_shifts, key=lambda shift: odd[shift])
        current_fundamental = fundamental.get(0, -8.0)
        current_odd = odd.get(0, -8.0)
        return {
            "fundamental": fundamental,
            "odd": odd,
            "fundamentalWinnerShift": int(fundamental_winner),
            "oddWinnerShift": int(odd_winner),
            "fundamentalGainByShift": {
                shift: float(fundamental[shift] - current_fundamental)
                for shift in SHIFT_CLASSES
            },
            "oddGainByShift": {
                shift: float(odd[shift] - current_odd) for shift in SHIFT_CLASSES
            },
            "fundamentalWinnerGain": float(
                fundamental[fundamental_winner] - current_fundamental
            ),
            "oddWinnerGain": float(odd[odd_winner] - current_odd),
        }


def audio_feature_values(evidence: dict[str, Any]) -> list[float]:
    fundamental = evidence["fundamental"]
    odd = evidence["odd"]
    fundamental_gain = evidence["fundamentalGainByShift"]
    odd_gain = evidence["oddGainByShift"]
    fundamental_winner = int(evidence["fundamentalWinnerShift"])
    odd_winner = int(evidence["oddWinnerShift"])
    return [
        *(float(fundamental[shift]) for shift in SHIFT_CLASSES),
        *(float(odd[shift]) for shift in SHIFT_CLASSES),
        *(float(fundamental_gain[shift]) for shift in SHIFT_CLASSES),
        *(float(odd_gain[shift]) for shift in SHIFT_CLASSES),
        *(float(shift == fundamental_winner) for shift in SHIFT_CLASSES),
        *(float(shift == odd_winner) for shift in SHIFT_CLASSES),
        float(evidence["fundamentalWinnerGain"]),
        float(evidence["oddWinnerGain"]),
    ]


def audio_gate(
    prediction: int,
    evidence: dict[str, Any],
    *,
    agreement: str,
    minimum_gain: float,
) -> bool:
    """Return whether source evidence supports one proposed octave change."""

    prediction = int(prediction)
    if prediction == 0:
        return False
    fundamental = (
        prediction == int(evidence["fundamentalWinnerShift"])
        and float(evidence["fundamentalGainByShift"][prediction]) >= minimum_gain
    )
    odd = (
        prediction == int(evidence["oddWinnerShift"])
        and float(evidence["oddGainByShift"][prediction]) >= minimum_gain
    )
    if agreement == "none":
        return True
    if agreement == "fundamental":
        return fundamental
    if agreement == "odd":
        return odd
    if agreement == "either":
        return fundamental or odd
    if agreement == "both":
        return fundamental and odd
    raise ValueError(f"Unknown audio agreement policy: {agreement}")
