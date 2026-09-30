from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ml.training.augment_texture_audit_vocal_f0 import (
    local_vocal_evidence,
    stabilize_pitch_classes,
    yin_f0,
)


def test_yin_tracks_a_440_sine_without_reference_data(tmp_path: Path):
    sample_rate = 16_000
    time = np.arange(sample_rate * 2) / sample_rate
    signal = (0.4 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    path = tmp_path / "a4.wav"
    wavfile.write(path, sample_rate, signal)
    features = yin_f0(path)
    voiced = features["frequencies"][features["confidences"] >= 0.72]
    assert len(voiced) > 20
    assert abs(float(np.median(voiced)) - 440.0) < 3.0
    evidence = local_vocal_evidence(
        features, 1.0, radius_seconds=0.1, minimum_confidence=0.72
    )
    assert evidence["dominantPitchClass"] == 9
    assert evidence["pitchClasses"][9]["probability"] > 0.9


def test_pitch_class_stabilization_removes_one_frame_semitone_jitter():
    frequencies = np.asarray([440.0, 440.0, 466.1638, 440.0, 440.0])
    confidences = np.asarray([0.9, 0.9, 0.9, 0.9, 0.9])
    raw = stabilize_pitch_classes(
        frequencies, confidences, window_frames=1
    )
    smoothed = stabilize_pitch_classes(
        frequencies, confidences, window_frames=5
    )
    assert list(raw) == [9, 9, 10, 9, 9]
    assert list(smoothed) == [9, 9, 9, 9, 9]
