import copy

import pytest

from ml.training.probe_arrangement_role_coupling import splice_roles


def test_splicing_changes_only_selected_roles_and_does_not_mutate_inputs():
    baseline = {"notes": [{"time": 1, "midi": 72, "arrangementRole": "melody", "velocity": 0.7}, {"time": 1, "midi": 36, "arrangementRole": "bass", "velocity": 0.5}]}
    ablation = {"notes": [{"time": 1, "midi": 60, "arrangementRole": "melody", "velocity": 0.9}, {"time": 1, "midi": 38, "arrangementRole": "bass", "velocity": 0.6}]}
    before = copy.deepcopy((baseline, ablation))
    result = splice_roles(baseline, ablation, ["bass"])
    assert sorted(note["midi"] for note in result["notes"]) == [38, 72]
    assert next(n for n in result["notes"] if n["arrangementRole"] == "melody") == baseline["notes"][0]
    assert (baseline, ablation) == before
    assert result["deployableCandidate"] is False


def test_no_replacements_is_a_note_identical_control():
    notes = [{"time": 1, "midi": 60, "arrangementRole": "melody"}]
    assert splice_roles({"notes": notes}, {"notes": []}, [])["notes"] == notes
    with pytest.raises(ValueError, match="Unknown"):
        splice_roles({"notes": notes}, {"notes": []}, ["invented"])
