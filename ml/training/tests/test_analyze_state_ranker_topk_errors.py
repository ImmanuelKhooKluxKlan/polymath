from ml.training.analyze_state_ranker_topk_errors import (
    edit_type,
    jaccard,
    reward_band,
)


def test_edit_type_separates_add_remove_and_replace():
    assert edit_type(frozenset({1, 2}), frozenset({1})) == "remove-only"
    assert edit_type(frozenset({1}), frozenset({1, 2})) == "add-only"
    assert edit_type(frozenset({1, 2}), frozenset({2, 3})) == "replace-mixed"


def test_jaccard_and_reward_bands_are_stable_at_boundaries():
    assert jaccard(frozenset({1, 2}), frozenset({2, 3})) == 1 / 3
    assert reward_band(0.24) == "00-24"
    assert reward_band(0.25) == "25-49"
    assert reward_band(0.75) == "75-99"
    assert reward_band(1.0) == "100"
