import pytest

from ml.training.muscriptor_tokens import DRUM_BASE, EOS_ID, PITCH_BASE, PROGRAM_BASE, TIE_ID, VELOCITY_BASE
from ml.training.train_muscriptor_piano import TrainingError, target_token_weights


def test_program_weight_is_relative_and_mean_one_without_changing_default():
    tokens = [TIE_ID, PROGRAM_BASE, VELOCITY_BASE + 1, PITCH_BASE + 60, EOS_ID]
    default = target_token_weights(tokens)
    assert default == target_token_weights(tokens, program_weight=1)
    weighted = target_token_weights(tokens, program_weight=8)
    assert sum(weighted) / len(weighted) == pytest.approx(1)
    assert weighted[1] / weighted[3] == pytest.approx(8)
    assert default[1] / default[3] == pytest.approx(1)


def test_program_range_does_not_accidentally_weight_drums_or_silence():
    tokens = [PROGRAM_BASE, DRUM_BASE + 38, EOS_ID]
    weighted = target_token_weights(tokens, program_weight=8)
    assert weighted[0] / weighted[1] == pytest.approx(8)
    assert target_token_weights([TIE_ID, EOS_ID], program_weight=8) == target_token_weights([TIE_ID, EOS_ID])


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), 33])
def test_invalid_program_weight_fails_before_optimizer_work(invalid):
    with pytest.raises(TrainingError, match="Program-token loss weight"):
        target_token_weights([PROGRAM_BASE, EOS_ID], program_weight=invalid)
