from __future__ import annotations

import copy

from ml.training.analyze_pianist_texture_patterns import cell_example
from ml.training.pianist_pitch_class_sequence_context import (
    FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    sequence_examples_from_song,
)


def note(midi: int, time: float, *, instrument: str = "acoustic_piano") -> dict:
    return {
        "midi": midi,
        "time": time,
        "duration": 0.3,
        "scoreDuration": 0.3,
        "velocity": 0.7,
        "instrument": instrument,
        "arrangementRole": "harmony",
    }


def song(reference_middle: bool = False) -> dict:
    rows = []
    for index, midi in enumerate((60, 64, 60)):
        candidate = [note(48, float(index))]
        source = [note(midi, float(index))]
        reference = [[note(64, float(index))]] if reference_middle and index == 1 else []
        rows.append(
            cell_example(
                "song", index, candidate, reference, source, 1.0, 1.0
            )
        )
    return {"id": "song", "cells": rows}


def test_sequence_features_capture_neighbor_support() -> None:
    examples = sequence_examples_from_song(song())
    middle_c = examples[12]
    assert len(middle_c["features"]) == len(FEATURE_NAMES)
    assert len(SEQUENCE_FEATURE_NAMES) == 74
    sequence = middle_c["features"][-len(SEQUENCE_FEATURE_NAMES) :]
    assert sequence[5] == 1.0  # previous candidate/source-relative slot is populated
    assert sequence[17] == 1.0  # C is present in both neighboring source cells
    assert sequence[-2] == 1.0  # source present on both sides


def test_reference_changes_labels_but_not_sequence_features() -> None:
    left = song(reference_middle=False)
    right = song(reference_middle=True)
    before = sequence_examples_from_song(copy.deepcopy(left))[12 + 4]
    after = sequence_examples_from_song(copy.deepcopy(right))[12 + 4]
    assert before["features"] == after["features"]
    assert before["label"] == 0.0
    assert after["label"] == 1.0
