import pytest

from ml.training.run_piano_task_vector_experiment import audit_separation


def row(start=10, song="train-song", clip="wane-test"):
    return {"songId": song, "clipId": clip, "sourceStart": start, "durationSeconds": 5}


def test_wane_windows_are_audited_in_every_audio_variant():
    result = audit_separation([row()], [row(song="validation-song", clip="validation")], [{"start": 15, "end": 20}])
    assert result["overlappingTimeWindows"] == 0
    assert result["waneClipsChecked"] == 1
    with pytest.raises(ValueError, match="overlaps a held-out"):
        audit_separation([row(12.5)], [], [{"start": 15, "end": 20}])


def test_composition_overlap_is_rejected_even_with_different_clip_ids():
    with pytest.raises(ValueError, match="compositions overlap"):
        audit_separation([row()], [row(20, clip="different")], [])
