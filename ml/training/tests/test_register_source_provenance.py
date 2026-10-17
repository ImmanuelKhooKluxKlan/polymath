import pytest

from ml.training.repair_register_source_provenance import attach_source_indices, repair


def source(midi=48):
    return {"notes": [{"midi": midi, "time": 1, "duration": .3, "instrument": "acoustic_guitar"}]}


def candidate(midi=72):
    return {"notes": [{"midi": midi, "time": 1, "duration": .4, "velocity": .7, "sourceIndex": 0, "sourceInstrument": "acoustic_guitar", "arrangementRole": "harmony"}]}


def test_source_pitch_is_recovered_without_changing_the_played_note():
    before = candidate()
    output, report = repair(before, source())
    assert output["notes"][0]["sourceMidiBeforeArrangement"] == 48
    assert output["notes"][0]["midi"] == 72
    assert "sourceMidiBeforeArrangement" not in before["notes"][0]
    assert report["changedInferredSourcePitchFeatures"] == 1
    assert report["playedNoteValuesUnchanged"] is True


def test_provenance_index_join_rejects_instrument_and_pitch_class_mismatch():
    with pytest.raises(ValueError, match="contradicts"):
        repair(candidate(), source(49))
    invalid = source()
    invalid["notes"][0]["instrument"] = "voice"
    with pytest.raises(ValueError, match="contradicts"):
        repair(candidate(), invalid)


def test_missing_and_ambiguous_source_indices_fail_closed():
    with pytest.raises(ValueError, match="no verified"):
        repair(candidate(), {"notes": []})
    duplicated = source()
    duplicated["notes"] = [{**duplicated["notes"][0], "sourceIndex": 0}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        repair(candidate(), duplicated)


def test_provenance_copy_must_reproduce_actual_playback_values():
    shifted = candidate()
    shifted["notes"][0]["time"] = 2
    with pytest.raises(ValueError, match="same played notes"):
        attach_source_indices(candidate(), shifted)
