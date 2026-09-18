from ml.training.apply_raw_support_recovery import (
    deduplicate_proposals,
    fold_midi,
    nearby,
    selector_model,
    within_recovery_register,
)


def test_nearby_checks_only_adjacent_sorted_positions():
    assert nearby([1.0, 2.0, 3.0], 2.08, 0.1)
    assert not nearby([1.0, 2.0, 3.0], 2.2, 0.1)


def test_fold_midi_keeps_pitch_class_inside_output_range():
    assert fold_midi(23, 33, 71, 12) == 35
    assert fold_midi(65, 33, 71, 12) == 65
    assert fold_midi(58, 33, 71, 12) == 70


def test_fold_midi_preserves_native_register_when_it_is_playable():
    assert fold_midi(48, 33, 71, 0) == 48
    assert fold_midi(57, 33, 71, 0) == 57
    assert fold_midi(28, 33, 71, 0) == 40


def test_piano_recovery_uses_a_stricter_low_support_register():
    assert within_recovery_register("piano", 48, 59, 50)
    assert not within_recovery_register("piano", 53, 59, 50)
    assert within_recovery_register("guitar", 58, 59, 50)
    assert not within_recovery_register("guitar", 60, 59, 50)


def test_deduplicate_proposals_keeps_strongest_nearby_pitch_class():
    proposals = [
        {"midi": 48, "time": 1.0, "score": 0.91},
        {"midi": 60, "time": 1.04, "score": 0.97},
        {"midi": 48, "time": 1.20, "score": 0.94},
        {"midi": 49, "time": 1.02, "score": 0.92},
    ]
    selected = deduplicate_proposals(proposals, 0.08)
    assert [(item["midi"], item["score"]) for item in selected] == [
        (49, 0.92),
        (60, 0.97),
        (48, 0.94),
    ]


def test_selector_model_accepts_nested_left_hand_profile():
    model = {"type": "example"}
    profile = {"decoder": {"leftHandAccompaniment": {"selectionModel": model}}}
    assert selector_model(profile) is model


def test_selector_model_prefers_task_specific_nested_model():
    direct = {"type": "general"}
    nested = {"type": "left-hand"}
    profile = {
        "selectionModel": direct,
        "decoder": {"leftHandAccompaniment": {"selectionModel": nested}},
    }
    assert selector_model(profile) is nested
