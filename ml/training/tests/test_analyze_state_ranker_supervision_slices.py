from ml.training.analyze_state_ranker_supervision_slices import numeric_band


def test_numeric_band_has_stable_boundaries():
    assert numeric_band(0.0, (0.5, 0.75)) == "0.00-0.50"
    assert numeric_band(0.5, (0.5, 0.75)) == "0.50-0.75"
    assert numeric_band(0.75, (0.5, 0.75)) == "0.75+"
