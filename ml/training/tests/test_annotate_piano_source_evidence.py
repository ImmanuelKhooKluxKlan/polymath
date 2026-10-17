import pytest

from ml.training.annotate_piano_source_evidence import annotate_source_evidence


def note(midi, time, instrument):
    return {"midi": midi, "time": time, "duration": 0.3, "velocity": 0.7, "instrument": instrument}


def test_source_tags_preserve_physical_notes_and_do_not_mutate_inputs():
    piano = {"notes": [note(69, 1.03, "acoustic_piano"), note(72, 2.0, "acoustic_piano")]}
    broad = {"notes": [note(69, 1.0, "voice"), note(72, 2.0, "drums")]}
    output, report = annotate_source_evidence(piano, broad, known_shared_audio=True)
    assert output["notes"][0]["instrument"] == "voice"
    assert output["notes"][1]["instrument"] == "acoustic_piano"
    for before, after in zip(piano["notes"], output["notes"]):
        assert [after[key] for key in ("midi", "time", "duration", "velocity")] == [before[key] for key in ("midi", "time", "duration", "velocity")]
    assert piano["notes"][0]["instrument"] == "acoustic_piano"
    assert report["taggedNotes"] == 1


def test_single_voice_event_is_not_reused_for_multiple_piano_strikes():
    piano = {"notes": [note(69, 1.06, "acoustic_piano"), note(69, 1.01, "acoustic_piano")]}
    broad = {"notes": [note(69, 1.0, "voice")]}
    output, report = annotate_source_evidence(piano, broad, known_shared_audio=True)
    assert report["taggedNotes"] == 1
    assert output["notes"][0]["instrument"] == "acoustic_piano"
    assert output["notes"][1]["instrument"] == "voice"


def test_octave_equivalent_tagging_is_explicit_and_never_moves_a_key():
    piano = {"notes": [note(81, 1.01, "acoustic_piano")]}
    broad = {"notes": [note(69, 1.0, "voice")]}
    unchanged, _ = annotate_source_evidence(piano, broad, known_shared_audio=True)
    tagged, _ = annotate_source_evidence(piano, broad, known_shared_audio=True, maximum_octave_distance=12)
    assert unchanged["notes"][0]["instrument"] == "acoustic_piano"
    assert tagged["notes"][0]["instrument"] == "voice"
    assert tagged["notes"][0]["midi"] == 81


def test_source_routing_fails_without_verified_shared_audio():
    with pytest.raises(ValueError, match="shared audio"):
        annotate_source_evidence({"notes": []}, {"notes": []})
