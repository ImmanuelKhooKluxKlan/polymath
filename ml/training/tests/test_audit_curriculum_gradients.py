import copy

import numpy as np
import pytest

from ml.training.audit_curriculum_gradients import PREFIXES, cosine, gradient_summary, paired_groups, teacher_forced_program_summary
from ml.training.muscriptor_tokens import EOS_ID, INITIAL_TOKEN_ID, PITCH_BASE, PROGRAM_BASE, TIE_ID


def test_program_audit_distinguishes_teacher_forcing_from_free_running_notes():
    tokens = [TIE_ID, PROGRAM_BASE, PITCH_BASE + 60, EOS_ID]
    logits = np.zeros((4, INITIAL_TOKEN_ID))
    logits[1, PROGRAM_BASE + 4] = 2
    report = teacher_forced_program_summary(tokens, logits, [1, 1, 1, 1])
    assert report["programTokens"] == 1
    assert report["programPredictions"][0]["predictedToken"] == PROGRAM_BASE + 4
    assert report["programPredictions"][0]["targetRank"] == 2
    assert not report["programPredictions"][0]["top1Correct"]
    assert 0 < report["programWeightedLossShare"] < 1
    assert "not autoregressive note accuracy" in report["interpretation"]


def test_negative_examples_do_not_invent_a_program_loss():
    report = teacher_forced_program_summary([TIE_ID, EOS_ID], np.zeros((2, INITIAL_TOKEN_ID)), [1, 1])
    assert report["programTokens"] == 0
    assert report["programWeightedLossShare"] == 0


def records():
    return [{"clipId": prefix + str(index), "sourceStart": float(index * 5), "durationSeconds": 5.0, "instrumentFocus": "acoustic_piano", "notes": [{"midi": 60, "time": 1.0, "duration": 0.5, "velocity": 0.7}]} for index in range(5) for prefix in PREFIXES.values()]


def test_complete_pairs_are_evenly_sampled_before_reading_gradients():
    assert [key[0] for key, _variants in paired_groups(records(), 3)] == [0, 10, 20]


def test_short_or_sparse_tails_can_be_excluded_by_predeclared_selection():
    rows = records()
    for row in rows:
        if row["sourceStart"] == 20:
            row["durationSeconds"] = 1.5
    assert [key[0] for key, _variants in paired_groups(rows, 3, minimum_duration=4)] == [0, 10, 15]
    with pytest.raises(ValueError, match="Insufficient"):
        paired_groups(rows, 3, minimum_notes=5)


def test_missing_or_duplicate_variants_are_not_silently_dropped():
    rows = records()
    with pytest.raises(ValueError, match="incomplete"):
        paired_groups(rows[:-1], 3)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_groups(rows + [rows[0]], 3)


def test_different_targets_cannot_be_called_a_paired_audio_comparison():
    rows = copy.deepcopy(records())
    rows[1]["notes"][0]["midi"] = 61
    with pytest.raises(ValueError, match="same target tokens"):
        paired_groups(rows, 3)


def test_opposing_gradient_is_identified_without_claiming_accuracy():
    result = gradient_summary([1, 0], [-2, 0])
    assert result["cosineOriginalVersusSynthetic"] == -1
    assert result["firstOrderDirectionalConflict"]
    assert result["combinedProjectionOntoOriginal"] == -1
    assert gradient_summary([1, 0], [2, 0])["combinedProjectionOntoOriginal"] == 3


def test_undefined_or_nonfinite_directions_fail_closed():
    with pytest.raises(ValueError, match="Zero gradient"):
        cosine([0, 0], [1, 0])
    with pytest.raises(ValueError, match="Invalid"):
        cosine([np.nan, 0], [1, 0])
