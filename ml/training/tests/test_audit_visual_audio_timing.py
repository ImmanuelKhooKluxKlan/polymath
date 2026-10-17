import numpy as np
import pytest

from ml.training.audit_visual_audio_timing import lag_curve, pitch_onset_features, scope_end, summarize_curve


def test_owner_scope_always_caps_kiss_me_but_not_wane():
    assert scope_end("kiss-me-pianella", None) == 150
    assert scope_end("kiss-me-pianella", 200) == 150
    assert scope_end("kiss-me-pianella", 120) == 120
    assert scope_end("wane-gbt-pianella", None) is None
    with pytest.raises(ValueError, match="cutoff"):
        scope_end("kiss-me", float("nan"))


def test_pitch_conditioned_lag_recovers_known_later_audio_without_edge_selection():
    frames = np.arange(0, 5.01, 0.01)
    features = np.zeros((88, len(frames)))
    notes = np.array([1.0, 2.0, 3.0])
    pitches = np.array([60, 64, 67])
    for time, pitch in zip(notes, pitches):
        features[pitch - 21] = np.exp(-0.5 * ((frames - time - 0.2) / 0.015) ** 2)
    lags = np.arange(-0.4, 0.401, 0.01)
    curve, count = lag_curve(frames, features, notes, pitches, lags)
    summary = summarize_curve(lags, curve, count)
    assert summary["bestLagSeconds"] == 0.2
    assert count == 3
    assert not summary["bestAtSearchBoundary"]
    assert summary["marginOverLagsMoreThan150msAway"] > 0.9


def test_edge_notes_are_excluded_once_for_every_lag():
    frames = np.arange(0, 5.01, 0.01)
    features = np.ones((88, len(frames)))
    curve, count = lag_curve(frames, features, [0.1, 2, 4.9], [60, 60, 60], [-0.2, 0, 0.2])
    assert count == 1
    assert np.allclose(curve, 1)


def test_unknown_pitch_or_empty_audio_support_fails_closed():
    frames = np.arange(0, 1.01, 0.01)
    features = np.zeros((88, len(frames)))
    with pytest.raises(ValueError, match="Invalid"):
        lag_curve(frames, features, [0.5], [109], [-0.2, 0.2])
    with pytest.raises(ValueError, match="Invalid"):
        lag_curve(frames, features, [0.5], [60.5], [-0.2, 0.2])
    with pytest.raises(ValueError, match="No notes"):
        lag_curve(frames, features, [0.01], [60], [-0.2, 0.2])


def test_audio_frontend_has_bounded_but_not_zero_latency_on_known_attacks():
    pytest.importorskip("scipy")
    sample_rate = 16000
    time = np.arange(sample_rate * 5) / sample_rate
    notes, pitches = np.array([1.0, 2.1, 3.2]), np.array([60, 64, 67])
    audio = np.zeros_like(time)
    for start, pitch in zip(notes, pitches):
        elapsed = np.maximum(0, time - start)
        envelope = np.minimum(elapsed / 0.01, 1) * np.exp(-elapsed / 0.15) * (time >= start)
        audio += envelope * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * elapsed)
    frames, features = pitch_onset_features(audio, sample_rate)
    lags = np.arange(-0.3, 0.301, 0.01)
    curve, count = lag_curve(frames, features, notes, pitches, lags)
    assert count == 3
    assert abs(lags[np.argmax(curve)]) <= 0.08
