from __future__ import annotations

from ml.training.analyze_pianist_texture_patterns import cell_example
from ml.training.pianist_pitch_class_decoder_origin_context import (
    FEATURE_NAMES,
    decoder_origin_examples_from_song,
)


def note(midi: int, time: float, *, generated: bool = False) -> dict:
    value = {
        "midi": midi,
        "time": time,
        "duration": 0.4,
        "scoreDuration": 0.4,
        "velocity": 0.7,
        "instrument": "acoustic_piano",
        "sourceInstrument": "clean_electric_guitar",
        "arrangementRole": "harmony",
    }
    if generated:
        value.update(
            {
                "generatedBy": "pianist-pitch-class-decoder-v1",
                "pitchClassProbability": 0.72,
            }
        )
    return value


def test_decoder_origin_contract_marks_only_generated_pitch_class() -> None:
    candidate = [note(60, 1.0), note(64, 1.0, generated=True)]
    cell = cell_example("song", 0, candidate, [], candidate, 0.3, 2.0)

    examples = decoder_origin_examples_from_song({"id": "song", "cells": [cell]})

    assert len(examples[0]["features"]) == len(FEATURE_NAMES) == 274
    assert examples[0]["decoderGenerated"] is False
    assert examples[4]["decoderGenerated"] is True
    assert examples[4]["features"][-14] == 1.0
    assert examples[4]["features"][-12] == 0.72
