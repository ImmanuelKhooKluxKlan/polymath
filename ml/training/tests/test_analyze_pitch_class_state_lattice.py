from ml.training.analyze_pitch_class_state_lattice import (
    generate_states,
    pitch_classes,
    score_song,
)


def cell(candidate, source, target):
    return {
        "candidatePitchClasses": candidate,
        "sourcePitchClasses": source,
        "referencePitchClasses": [target],
        "sourcePitchClassEvidence": [
            {
                "pitchClass": pc,
                "noteCount": 1,
                "attackCount": 1,
                "maximumVelocity": 0.8,
                "minimumRegisterDistanceSemitones": 2,
                "familyShares": {"piano": 1.0},
            }
            for pc in source
        ],
    }


def test_state_generation_is_reference_independent():
    left = [cell([0, 7], [0, 4, 7], [0, 4, 7])]
    right = [cell([0, 7], [0, 4, 7], [1, 6, 10])]
    assert generate_states(left, 0) == generate_states(right, 0)


def test_state_generation_keeps_candidate_and_supported_triad():
    rows = [cell([0, 7], [0, 4, 7], [0, 4, 7])]
    states = generate_states(rows, 0)
    assert pitch_classes([0, 7]) in states
    assert pitch_classes([0, 4, 7]) in states


def test_harmonic_completion_is_explicit_and_reference_independent():
    rows = [cell([0, 7], [0, 7], [1, 6, 10])]

    frozen = generate_states(rows, 0)
    expanded = generate_states(rows, 0, allow_harmonic_completion=True)

    assert pitch_classes([0, 4, 7]) not in frozen
    assert pitch_classes([0, 4, 7]) in expanded


def test_compact_oracle_can_select_coherent_whole_set():
    song = {
        "id": "song",
        "cells": [
            cell([0, 7], [0, 4, 7], [0, 4, 7]),
            cell([5, 0], [0, 5, 9], [0, 5, 9]),
        ],
    }
    result = score_song(song, maximum_states=32)
    assert result["compactLatticeOracle"]["f1"] == 1.0
    assert result["compactLatticeOracle"]["exactSetRate"] == 1.0
    assert result["baseline"]["f1"] < 1.0
