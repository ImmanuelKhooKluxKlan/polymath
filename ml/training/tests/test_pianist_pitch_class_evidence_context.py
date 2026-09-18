from __future__ import annotations

import copy

from ml.training.analyze_pianist_texture_patterns import cell_example
from ml.training.pianist_pitch_class_evidence_context import (
    FEATURE_NAMES,
    evidence_examples_from_song,
)


def note(midi: int, time: float, *, duration: float, instrument: str, role: str):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "scoreDuration": duration,
        "velocity": 0.7,
        "instrument": instrument,
        "arrangementRole": role,
    }


def test_evidence_features_capture_sustain_family_and_register_without_reference() -> None:
    candidate = [note(43, 4.0, duration=0.3, instrument="electric_bass", role="bass")]
    source = [
        note(74, 1.0, duration=4.0, instrument="clean_electric_guitar", role="harmony"),
        note(43, 4.0, duration=0.3, instrument="electric_bass", role="bass"),
    ]
    cell = cell_example("song", 0, candidate, [], source, 0.3, 2.0)
    song = {"id": "song", "cells": [cell]}

    examples = evidence_examples_from_song(song)
    d_row = examples[2]

    assert len(d_row["features"]) == len(FEATURE_NAMES) == 259
    assert d_row["source"] == 1.0
    assert d_row["candidate"] == 0.0
    assert d_row["features"][-21] == 0.0  # no fresh D attack
    assert d_row["features"][-14] == 31.0 / 48.0
    assert d_row["features"][-10] == 1.0  # guitar share
    assert d_row["features"][-5] == 1.0  # guitar x bass-role interaction


def test_reference_changes_labels_but_never_evidence_features() -> None:
    candidate = [note(60, 1.0, duration=0.3, instrument="piano", role="harmony")]
    source = [note(64, 1.0, duration=0.4, instrument="piano", role="harmony")]
    without = cell_example("song", 0, candidate, [], source, 0.3, 2.0)
    with_reference = copy.deepcopy(without)
    with_reference["referencePitchClasses"] = [[4]]
    left = evidence_examples_from_song({"id": "song", "cells": [without]})[4]
    right = evidence_examples_from_song(
        {"id": "song", "cells": [with_reference]}
    )[4]

    assert left["features"] == right["features"]
    assert left["label"] == 0.0
    assert right["label"] == 1.0
