from __future__ import annotations

import copy

import numpy as np

from ml.training.fit_raw_support_swap_benefit_ranker import label_proposals
from ml.training.raw_support_swap_benefit import (
    SWAP_BENEFIT_FEATURE_NAMES,
    apply_benefit_swaps,
)
from server.piano_arranger_adapter import HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES


def selector_profile() -> dict:
    size = len(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
    return {
        "id": "selector-test",
        "selectionModel": {
            "type": "standardized-logistic-left-hand-ranker-v1",
            "featureNames": list(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES),
            "means": [0.0] * size,
            "scales": [1.0] * size,
            "weights": [0.0] * size,
            "threshold": 0.5,
        },
    }


def benefit_profile() -> dict:
    return {
        "id": "benefit-test",
        "model": {
            "type": "standardized-logistic-swap-benefit-v1",
            "featureNames": list(SWAP_BENEFIT_FEATURE_NAMES),
            "means": [0.0] * len(SWAP_BENEFIT_FEATURE_NAMES),
            "scales": [1.0] * len(SWAP_BENEFIT_FEATURE_NAMES),
            "weights": [0.0] * len(SWAP_BENEFIT_FEATURE_NAMES),
            "bias": 10.0,
        },
    }


def test_runtime_feature_contract_contains_no_reference_or_label_fields() -> None:
    forbidden = ("reference", "label", "utility", "ground_truth", "desired")
    assert not any(
        token in name.lower()
        for name in SWAP_BENEFIT_FEATURE_NAMES
        for token in forbidden
    )


def test_reference_labels_reward_exact_recovery_and_reject_exact_loss() -> None:
    proposal = {"incumbentIndex": 0, "targetMidi": 48, "groupTime": 1.0}
    candidate = [
        {
            "candidateIndex": 0,
            "midi": 49,
            "time": 1.0,
            "duration": 0.4,
            "velocity": 0.7,
        }
    ]
    recovered, utilities, details = label_proposals(
        [proposal],
        [
            {
                "referenceIndex": 0,
                "midi": 48,
                "time": 1.0,
                "duration": 0.4,
                "velocity": 0.7,
            }
        ],
        candidate,
    )
    assert np.array_equal(recovered, np.asarray([1.0]))
    assert utilities[0] > 0
    assert details[0]["deltaExact250"] == 1

    lost, utilities, details = label_proposals(
        [proposal],
        [
            {
                "referenceIndex": 0,
                "midi": 49,
                "time": 1.0,
                "duration": 0.4,
                "velocity": 0.7,
            }
        ],
        candidate,
    )
    assert np.array_equal(lost, np.asarray([0.0]))
    assert utilities[0] < 0
    assert details[0]["deltaExact250"] == -1


def test_application_changes_only_pitch_and_provenance() -> None:
    candidate = {
        "notes": [
            {
                "midi": 49,
                "time": 1.0,
                "duration": 0.8,
                "audioDuration": 1.0,
                "velocity": 0.41,
                "hand": "left",
                "arrangementRole": "bass",
            },
            {
                "midi": 67,
                "time": 1.0,
                "duration": 0.6,
                "audioDuration": 0.8,
                "velocity": 0.72,
                "hand": "right",
                "arrangementRole": "melody",
            },
        ]
    }
    source = {
        "notes": [
            {
                "midi": 40,
                "time": 1.01,
                "duration": 0.5,
                "velocity": 0.8,
                "instrument": "acoustic_guitar",
            }
        ]
    }
    original = copy.deepcopy(candidate)
    output, diagnostics = apply_benefit_swaps(
        candidate,
        source,
        selector_profile(),
        benefit_profile(),
        threshold=0.5,
        source_radius=0.18,
        minimum_selector_probability=0.0,
        minimum_probability_gain=-1.0,
        maximum_replacements_per_gesture=1,
        minimum_source_midi=0,
        maximum_source_midi=59,
        minimum_output_midi=33,
        maximum_output_midi=71,
        register_shifts=(12,),
    )
    assert diagnostics["replacements"] == 1
    assert len(output["notes"]) == len(original["notes"])
    assert output["notes"][0]["midi"] == 52
    for index in range(len(original["notes"])):
        for field in ("time", "duration", "audioDuration", "velocity"):
            assert output["notes"][index][field] == original["notes"][index][field]
    assert output["notes"][1] == original["notes"][1]
