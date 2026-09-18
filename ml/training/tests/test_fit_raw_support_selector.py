import pytest

from ml.training.fit_raw_support_selector import nearest_distance, source_family_allowed


def test_nearest_distance_handles_edges_and_middle():
    assert nearest_distance([], 1.0) == float("inf")
    assert nearest_distance([1.0, 2.0, 4.0], 0.9) == pytest.approx(0.1)
    assert nearest_distance([1.0, 2.0, 4.0], 2.2) == pytest.approx(0.2)
    assert nearest_distance([1.0, 2.0, 4.0], 4.3) == pytest.approx(0.3)


def test_source_family_filter_always_excludes_voice_and_honors_allowlist():
    assert source_family_allowed("acoustic_guitar", None)
    assert not source_family_allowed("voice", None)
    assert source_family_allowed("electric_bass", {"guitar", "bass"})
    assert not source_family_allowed("synth_pad", {"guitar", "bass"})
    assert not source_family_allowed("voice", {"voice"})
