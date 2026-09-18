from __future__ import annotations

import numpy as np
import pytest

from ml.training.analyze_pianist_texture_patterns import cell_example
from ml.training.pianist_pitch_class_context import FEATURE_NAMES
from ml.training.pianist_pitch_class_decoder_origin_context import (
    FEATURE_NAMES as ORIGIN_FEATURE_NAMES,
)
from ml.training.score_pianist_pitch_class_addition_ranker import (
    score_song,
    training_ids,
)


class ConstantModel:
    n_features_in_ = len(FEATURE_NAMES)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(features.shape[0], 0.73, dtype=float)
        return np.column_stack((1.0 - positive, positive))


class OriginConstantModel(ConstantModel):
    n_features_in_ = len(ORIGIN_FEATURE_NAMES)


def inference_song() -> dict:
    return {
        "id": "blind-song",
        "cells": [
            {
                "sourceTime": 0.5,
                "candidatePitchClasses": [0],
                "sourcePitchClasses": [0, 4, 7],
                "candidateChordSize": 1,
                "sourcePitchClassCount": 3,
            }
        ],
    }


def test_score_song_only_scores_source_supported_missing_pitch_classes() -> None:
    payload = score_song(
        inference_song(),
        ConstantModel(),
        model_id="frozen-model",
        model_sha256="abc",
        training_song_ids=["development-a"],
    )
    probabilities = payload["cells"][0]["probabilities"]
    assert probabilities[0] == 1.0
    assert probabilities[4] == 0.73
    assert probabilities[7] == 0.73
    assert probabilities[1] == 0.0
    assert payload["referenceFieldsRead"] is False


def test_score_song_rejects_reference_leakage() -> None:
    song = inference_song()
    song["cells"][0]["referencePitchClasses"] = [[0, 4]]
    with pytest.raises(ValueError, match="reference-derived"):
        score_song(
            song,
            ConstantModel(),
            model_id="frozen-model",
            model_sha256="abc",
            training_song_ids=["development-a"],
        )


def test_membership_scope_scores_existing_candidate_pitch_classes() -> None:
    payload = score_song(
        inference_song(),
        ConstantModel(),
        model_id="membership-model",
        model_sha256="abc",
        training_song_ids=["development-a"],
        decision_scope="membership",
    )

    probabilities = payload["cells"][0]["probabilities"]
    assert probabilities[0] == 0.73
    assert probabilities[4] == 0.73
    assert probabilities[1] == 0.0
    assert payload["decisionScope"] == "membership"


def test_decoder_origin_scope_scores_generated_notes_and_trusts_foundation() -> None:
    foundation = {
        "midi": 60,
        "time": 1.0,
        "duration": 0.4,
        "scoreDuration": 0.4,
        "velocity": 0.7,
        "instrument": "acoustic_piano",
        "arrangementRole": "harmony",
    }
    generated = {
        **foundation,
        "midi": 64,
        "generatedBy": "pianist-pitch-class-decoder-v1",
        "pitchClassProbability": 0.61,
        "sourceInstrument": "clean_electric_guitar",
    }
    cell = cell_example(
        "blind-song", 0, [foundation, generated], [], [foundation, generated], 0.3, 2.0
    )
    payload = score_song(
        {"id": "blind-song", "cells": [cell]},
        OriginConstantModel(),
        model_id="origin-model",
        model_sha256="abc",
        training_song_ids=["development-a"],
        feature_contract="origin274",
        decision_scope="decoder-generated-removals-only",
    )

    probabilities = payload["cells"][0]["probabilities"]
    assert probabilities[0] == 1.0
    assert probabilities[4] == 0.73
    assert probabilities[1] == 0.0


def test_training_ids_verifies_a_whole_song_excluded_fold() -> None:
    report = {
        "folds": [
            {
                "heldOutSong": "blind-song",
                "trainingSongIds": ["development-a", "development-b"],
                "model": {"sha256": "fold-hash"},
            }
        ],
        "finalModel": {
            "sha256": "final-hash",
            "trainingSongIds": ["blind-song", "development-a"],
        },
    }
    assert training_ids(
        report, "fold-hash", held_out_song="blind-song"
    ) == ["development-a", "development-b"]


def test_training_ids_rejects_a_fold_hash_mismatch() -> None:
    report = {
        "folds": [
            {
                "heldOutSong": "blind-song",
                "trainingSongIds": ["development-a"],
                "model": {"sha256": "expected"},
            }
        ]
    }
    with pytest.raises(ValueError, match="fold and model"):
        training_ids(report, "wrong", held_out_song="blind-song")
