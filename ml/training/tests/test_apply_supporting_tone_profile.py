import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from apply_supporting_tone_profile import apply_supporting_tone_ranker  # noqa: E402


def test_installs_ranker_inside_left_hand_without_replacing_top_selector():
    base_model = {
        "featureNames": ["bias"],
        "weights": [1.0],
        "means": [0.0],
        "scales": [1.0],
    }
    alternative_model = {
        "featureNames": ["bias"],
        "weights": [-1.0],
        "means": [0.0],
        "scales": [1.0],
    }
    result = apply_supporting_tone_ranker(
        {
            "selectionModel": base_model,
            "decoder": {"leftHandAccompaniment": {"enabled": True}},
        },
        {"id": "swap", "selectionModel": alternative_model},
        profile_id="candidate",
        alternative_path="swap.json",
        alternative_share=0.5,
        chord_completion_share=0.25,
        source_families=["guitar"],
        minimum_source_midi=48,
        maximum_source_midi=76,
        created_at="2026-09-14T00:00:00+00:00",
    )

    assert result["selectionModel"] == base_model
    left = result["decoder"]["leftHandAccompaniment"]
    assert left["supportingToneSelectionModel"] == alternative_model
    assert left["supportingToneAlternativeShare"] == 0.5
    assert left["supportingToneChordCompletionShare"] == 0.25
    assert result["training"]["supportingToneUpgrade"]["rightHandFrozen"]
