import numpy as np
import pytest
from copy import deepcopy

from ml.training.fit_pitch_class_state_ranker import (
    FEATURE_NAMES,
    canonical_song_id,
    estimate_key,
    evaluate_song,
    fit_ridge,
    fit_ridge_from_group_statistics,
    normalize_sample_weights,
    parse_song_weight_overrides,
    prepare_song,
    ridge_group_statistics,
    select_margin_threshold_cross_validated,
    state_features,
)


def cell(candidate, source, target, *, voice=0.0):
    return {
        "candidatePitchClasses": candidate,
        "sourcePitchClasses": source,
        "referencePitchClasses": [target],
        "candidateRoleSignature": "melody+harmony",
        "candidateChordSize": len(candidate),
        "candidateDuration": 0.4,
        "candidateVelocity": 0.7,
        "previousGapSeconds": 0.3,
        "nextGapSeconds": 0.3,
        "localCandidateOnsetsPerSecond": 3.0,
        "sourceVoiceShare": voice,
        "sourcePitchClassEvidence": [
            {
                "pitchClass": pc,
                "noteCount": 1,
                "attackCount": 1,
                "maximumVelocity": 0.8,
                "medianDurationSeconds": 0.4,
                "nearestOnsetOffsetSeconds": 0.01,
                "minimumRegisterDistanceSemitones": 2,
                "familyShares": {"piano": 1.0},
            }
            for pc in source
        ],
    }


def song(song_id, transpose=0):
    shift = lambda values: [(value + transpose) % 12 for value in values]
    return {
        "id": song_id,
        "weight": 1.0,
        "cells": [
            cell(shift([0, 7]), shift([0, 4, 7]), shift([0, 4, 7])),
            cell(shift([5, 0]), shift([0, 5, 9]), shift([0, 5, 9])),
            cell(shift([7]), shift([2, 7, 11]), shift([7, 11])),
        ],
    }


def test_feature_contract_and_transposition_invariance():
    left = song("left", 0)["cells"]
    right = song("right", 3)["cells"]
    left_state = frozenset([0, 4, 7])
    right_state = frozenset([3, 7, 10])
    assert len(state_features(left, 0, left_state)) == len(FEATURE_NAMES)
    assert np.allclose(
        state_features(left, 0, left_state), state_features(right, 0, right_state)
    )


def test_song_weight_overrides_require_explicit_nonnegative_values():
    assert parse_song_weight_overrides(["mean-auxiliary=0.7", "calm=0.1"]) == {
        "mean-auxiliary": 0.7,
        "calm": 0.1,
    }
    with pytest.raises(ValueError, match="id=weight"):
        parse_song_weight_overrides(["missing-separator"])
    with pytest.raises(ValueError, match="non-negative"):
        parse_song_weight_overrides(["mean=-1"])


def test_ridge_ranker_is_finite_and_evaluable():
    songs = [prepare_song(song(f"song-{shift}", shift), maximum_states=32) for shift in range(3)]
    model = fit_ridge(songs[:2], ridge=0.1, best_state_weight=3.0)
    assert np.all(np.isfinite(model["coefficients"]))
    result = evaluate_song(songs[2], model)
    assert 0.0 <= result["ranker"]["f1"] <= 1.0
    assert result["cells"] == 3


def test_cached_group_statistics_reproduce_direct_ridge_fit():
    songs = [
        prepare_song(song(f"song-{shift}", shift), maximum_states=32)
        for shift in range(3)
    ]
    direct = fit_ridge(songs[:2], ridge=0.1, best_state_weight=3.0)
    statistics = ridge_group_statistics(songs, best_state_weight=3.0)
    cached = fit_ridge_from_group_statistics(
        statistics,
        included_groups=[songs[0]["groupId"], songs[1]["groupId"]],
        ridge=0.1,
        best_state_weight=3.0,
    )

    assert np.allclose(cached["means"], direct["means"], atol=1e-10)
    assert np.allclose(cached["scales"], direct["scales"], atol=1e-10)
    assert np.allclose(cached["coefficients"], direct["coefficients"], atol=1e-8)




def test_large_margin_falls_back_to_the_unchanged_candidate():
    songs = [prepare_song(song(f"song-{shift}", shift), maximum_states=32) for shift in range(3)]
    model = fit_ridge(songs[:2], ridge=0.1, best_state_weight=3.0)
    result = evaluate_song(songs[2], model, minimum_margin=1_000_000.0)
    assert result["changes"] == 0
    assert result["ranker"] == result["baseline"]


def test_margin_calibration_holds_out_complete_song_groups():
    songs = [prepare_song(song(f"song-{shift}", shift), maximum_states=32) for shift in range(4)]
    result = select_margin_threshold_cross_validated(
        songs,
        ridge=0.1,
        best_state_weight=3.0,
        thresholds=[0.0, 0.05, 1_000_000.0],
    )
    assert result["method"] == "nested-whole-song-cross-validation"
    assert result["selected"]["heldoutGroups"] == 4
    assert result["selected"]["threshold"] in {0.0, 0.05, 1_000_000.0}


def test_duplicate_song_variants_share_a_fold_group():
    assert canonical_song_id("mean-auxiliary") == "mean"
    assert canonical_song_id("the-way-i-loved-you") == canonical_song_id(
        "the-way-i-loved-you-auxiliary"
    )


def test_duplicate_song_variant_does_not_double_its_training_influence():
    first = prepare_song(song("first", 0), maximum_states=32)
    second = prepare_song(song("second", 3), maximum_states=32)
    duplicate = deepcopy(first)
    duplicate["id"] = "first-auxiliary"
    duplicate["groupId"] = first["groupId"]
    without_duplicate = fit_ridge(
        [first, second], ridge=0.1, best_state_weight=3.0
    )
    with_duplicate = fit_ridge(
        [first, duplicate, second], ridge=0.1, best_state_weight=3.0
    )
    assert np.allclose(
        without_duplicate["coefficients"], with_duplicate["coefficients"]
    )


def test_audio_evidence_changes_state_features_without_reading_target():
    rows = song("audio", 0)["cells"]
    before = state_features(rows, 0, frozenset([0, 4, 7]))
    rows[0]["audioPitchClassEvidence"] = [
        {
            "pitchClass": pc,
            "meanChroma": 0.8 if pc == 4 else 0.02,
            "peakChroma": 0.8 if pc == 4 else 0.02,
            "attackChroma": 0.9 if pc == 4 else 0.01,
            "bassChroma": 0.1,
            "trebleChroma": 0.2,
        }
        for pc in range(12)
    ]
    after = state_features(rows, 0, frozenset([0, 4, 7]))
    assert before != after


def test_sample_weight_normalization_preserves_ratios_and_has_unit_mean():
    normalized = normalize_sample_weights(np.asarray([0.1, 0.2, 0.7]))
    assert normalized.mean() == pytest.approx(1.0)
    assert normalized[1] / normalized[0] == pytest.approx(2.0)
    assert normalized[2] / normalized[0] == pytest.approx(7.0)


def test_harmonic_context_is_transposition_invariant_after_preparation():
    left = prepare_song(
        song("left-context", 0), maximum_states=32, include_harmonic_context=True
    )
    right = prepare_song(
        song("right-context", 3), maximum_states=32, include_harmonic_context=True
    )
    left_index = next(
        index for index, state in enumerate(left["cells"][0]["states"])
        if state == frozenset([0, 4, 7])
    )
    right_index = next(
        index for index, state in enumerate(right["cells"][0]["states"])
        if state == frozenset([3, 7, 10])
    )
    assert np.allclose(
        left["cells"][0]["features"][left_index],
        right["cells"][0]["features"][right_index],
    )


def test_key_estimation_uses_source_profile_without_reference_data():
    tonic, mode, profile = estimate_key([10.0, 0.0, 2.0, 0.0, 6.0, 2.0, 0.0, 8.0, 0.0, 3.0, 0.0, 2.0])
    assert tonic == 0
    assert mode == "major"
    assert len(profile) == 12


def test_vocal_f0_changes_state_features_without_reading_reference():
    rows = song("vocal-f0", 0)["cells"]
    before = state_features(rows, 0, frozenset([0, 4, 7]))
    rows[0]["vocalF0Evidence"] = {
        "available": True,
        "voicedShare": 0.8,
        "meanConfidence": 0.9,
        "dominantPitchClass": 4,
        "pitchClasses": [
            {
                "pitchClass": pc,
                "probability": 0.9 if pc == 4 else 0.1 / 11.0,
                "frameShare": 0.7 if pc == 4 else 0.0,
                "meanConfidence": 0.9 if pc == 4 else 0.0,
            }
            for pc in range(12)
        ],
    }
    after = state_features(rows, 0, frozenset([0, 4, 7]))
    assert before != after
