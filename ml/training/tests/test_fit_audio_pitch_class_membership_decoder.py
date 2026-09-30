import numpy as np
import pytest

pytest.importorskip("sklearn")

from ml.training.fit_audio_pitch_class_membership_decoder import (
    FEATURE_NAMES,
    feature_vector,
    included_logit_score,
    lattice_state_score,
    relative_bits,
)


def base_cell(candidate, source):
    return {
        "candidatePitchClasses": candidate,
        "sourcePitchClasses": source,
        "candidateRoleSignature": "melody+harmony",
        "sourcePitchClassEvidence": [],
        "audioPitchClassEvidence": [
            {
                "pitchClass": pc,
                "meanChroma": 0.8 if pc == candidate[0] else 0.01,
                "peakChroma": 0.8 if pc == candidate[0] else 0.01,
                "attackChroma": 0.8 if pc == candidate[0] else 0.01,
                "bassChroma": 0.05,
                "trebleChroma": 0.1,
            }
            for pc in range(12)
        ],
    }


def test_relative_bits_are_transposition_invariant():
    assert relative_bits([0, 4, 7], 0) == relative_bits([3, 7, 10], 3)


def test_feature_contract_is_transposition_invariant():
    left = [base_cell([0], [0, 4, 7])]
    right = [base_cell([3], [3, 7, 10])]
    left_values = feature_vector(left, 0, 0)
    right_values = feature_vector(right, 0, 3)
    assert len(left_values) == len(FEATURE_NAMES)
    assert np.allclose(left_values, right_values)


def test_multiview_evidence_is_transposition_invariant_and_changes_features():
    left = [base_cell([0], [0, 4, 7])]
    right = [base_cell([3], [3, 7, 10])]
    for rows, pitch_class in ((left, 0), (right, 3)):
        rows[0]["multiViewAvailable"] = True
        rows[0]["multiViewPitchClassEvidence"] = [
            {
                "pitchClass": pitch_class,
                "viewAgreementCount": 2,
                "attackAgreementCount": 1,
                "views": {
                    "fullMix": {
                        "available": True,
                        "nearCount": 2,
                        "attackCount": 1,
                        "nearestOnsetDistanceSeconds": 0.02,
                        "maximumVelocity": 0.8,
                    },
                    "vocals": {
                        "available": True,
                        "nearCount": 1,
                        "attackCount": 0,
                        "nearestOnsetDistanceSeconds": 0.04,
                        "maximumVelocity": 0.6,
                    },
                    "backing": {
                        "available": True,
                        "nearCount": 0,
                        "attackCount": 0,
                        "nearestOnsetDistanceSeconds": 0.25,
                        "maximumVelocity": 0.0,
                    },
                },
            }
        ]
    left_values = feature_vector(left, 0, 0)
    right_values = feature_vector(right, 0, 3)
    assert np.allclose(left_values, right_values)
    assert not np.allclose(left_values, feature_vector([base_cell([0], [0, 4, 7])], 0, 0))


def test_joint_likelihood_prefers_high_probability_state():
    probabilities = np.full(12, 0.05)
    probabilities[[0, 4, 7]] = 0.9
    assert lattice_state_score(frozenset([0, 4, 7]), probabilities) > lattice_state_score(
        frozenset([0, 7]), probabilities
    )


def test_included_logit_prefers_better_same_size_state():
    probabilities = np.full(12, 0.1)
    probabilities[[0, 4]] = [0.9, 0.8]
    probabilities[7] = 0.2
    assert included_logit_score(
        frozenset([0, 4]), probabilities
    ) > included_logit_score(frozenset([0, 7]), probabilities)
