from copy import deepcopy

from ml.training.pianist_pitch_class_context import (
    FEATURE_NAMES,
    context_examples_from_song,
)


def test_long_context_features_are_inference_safe():
    song = {
        "id": "fixture",
        "cells": [
            {
                "candidatePitchClasses": [0, 4],
                "sourcePitchClasses": [0, 4, 7],
                "referencePitchClasses": [[0, 4, 7]],
            },
            {
                "candidatePitchClasses": [2, 7],
                "sourcePitchClasses": [2, 7, 11],
                "referencePitchClasses": [[2, 7]],
            },
        ],
    }
    changed_reference = deepcopy(song)
    changed_reference["cells"][0]["referencePitchClasses"] = [[1, 6, 10]]
    original = context_examples_from_song(song)
    changed = context_examples_from_song(changed_reference)

    assert len(original) == 24
    assert len(original[0]["features"]) == len(FEATURE_NAMES) == 237
    assert [row["features"] for row in original] == [
        row["features"] for row in changed
    ]
    assert [row["label"] for row in original] != [row["label"] for row in changed]
