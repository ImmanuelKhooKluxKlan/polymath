import numpy as np

from ml.training.audio_register_evidence import (
    AUDIO_FEATURE_NAMES,
    AudioRegisterEvidence,
    audio_feature_values,
    audio_gate,
)


def test_pure_tone_prefers_its_actual_octave() -> None:
    sample_rate = 16_000
    seconds = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.7 * np.sin(2.0 * np.pi * 440.0 * seconds)).astype(np.float32)
    reader = AudioRegisterEvidence(samples, sample_rate)

    evidence = reader.score(0.2, 57)  # Candidate A3; waveform is A4.

    assert evidence["fundamentalWinnerShift"] == 12
    assert evidence["fundamentalGainByShift"][12] > 0
    assert len(audio_feature_values(evidence)) == len(AUDIO_FEATURE_NAMES)


def test_audio_gate_requires_declared_agreement_and_gain() -> None:
    evidence = {
        "fundamentalWinnerShift": 12,
        "oddWinnerShift": 12,
        "fundamentalGainByShift": {-24: 0.0, -12: 0.0, 0: 0.0, 12: 2.5, 24: 0.0},
        "oddGainByShift": {-24: 0.0, -12: 0.0, 0: 0.0, 12: 1.5, 24: 0.0},
    }

    assert audio_gate(12, evidence, agreement="both", minimum_gain=1.0)
    assert not audio_gate(12, evidence, agreement="both", minimum_gain=2.0)
    assert audio_gate(12, evidence, agreement="fundamental", minimum_gain=2.0)
    assert not audio_gate(-12, evidence, agreement="either", minimum_gain=0.0)
    assert not audio_gate(0, evidence, agreement="none", minimum_gain=0.0)
