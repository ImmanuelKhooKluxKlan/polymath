"""Feature contract for auditing pitch classes created by the prior decoder."""

from __future__ import annotations

from typing import Any

from .pianist_pitch_class_evidence_context import (
    FEATURE_NAMES as EVIDENCE_FEATURE_NAMES,
    clamp,
    evidence_examples_from_song,
    finite,
)


ORIGIN_FEATURE_NAMES = (
    "pc_candidate_note_count",
    "pc_candidate_decoder_generated_share",
    "pc_candidate_minimum_decoder_probability",
    "pc_candidate_median_decoder_probability",
    "pc_candidate_melody_share",
    "pc_candidate_bass_share",
    "pc_candidate_harmony_share",
    "pc_candidate_left_hand_share",
    "pc_candidate_mean_midi_centered",
    "pc_candidate_source_voice_share",
    "pc_candidate_source_guitar_share",
    "pc_candidate_source_bass_share",
    "pc_candidate_source_piano_share",
    "pc_candidate_source_strings_share",
    "pc_candidate_source_other_share",
)
FEATURE_NAMES = EVIDENCE_FEATURE_NAMES + ORIGIN_FEATURE_NAMES


def candidate_evidence_by_pitch_class(
    row: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    values = row.get("candidatePitchClassEvidence")
    if not isinstance(values, list) or len(values) != 12:
        raise ValueError(
            "Every cell needs twelve inference-safe candidatePitchClassEvidence rows"
        )
    parsed: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Candidate pitch-class evidence rows must be objects")
        pitch_class = int(item.get("pitchClass", -1))
        if pitch_class in parsed or not 0 <= pitch_class < 12:
            raise ValueError(
                "Candidate pitch-class evidence indexes must be unique from 0 to 11"
            )
        parsed[pitch_class] = item
    if set(parsed) != set(range(12)):
        raise ValueError("Candidate pitch-class evidence must cover 0 through 11")
    return parsed


def origin_features(row: dict[str, Any], pitch_class: int) -> list[float]:
    item = candidate_evidence_by_pitch_class(row)[pitch_class]
    families = item.get("sourceFamilyShares") or {}
    return [
        clamp(finite(item.get("noteCount")) / 4.0, 0.0, 1.0),
        clamp(finite(item.get("decoderGeneratedShare")), 0.0, 1.0),
        clamp(finite(item.get("minimumDecoderProbability")), 0.0, 1.0),
        clamp(finite(item.get("medianDecoderProbability")), 0.0, 1.0),
        clamp(finite(item.get("melodyShare")), 0.0, 1.0),
        clamp(finite(item.get("bassShare")), 0.0, 1.0),
        clamp(finite(item.get("harmonyShare")), 0.0, 1.0),
        clamp(finite(item.get("leftHandShare")), 0.0, 1.0),
        clamp(finite(item.get("meanMidiCentered")), -1.5, 1.5),
        clamp(finite(families.get("voice")), 0.0, 1.0),
        clamp(finite(families.get("guitar")), 0.0, 1.0),
        clamp(finite(families.get("bass")), 0.0, 1.0),
        clamp(finite(families.get("piano")), 0.0, 1.0),
        clamp(finite(families.get("strings")), 0.0, 1.0),
        clamp(finite(families.get("other")), 0.0, 1.0),
    ]


def decoder_origin_examples_from_song(song: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(song.get("cells") or [])
    base = evidence_examples_from_song(song)
    if len(base) != len(rows) * 12:
        raise ValueError("Evidence pitch-class example grid is malformed")
    output: list[dict[str, Any]] = []
    for cell_index, row in enumerate(rows):
        evidence = candidate_evidence_by_pitch_class(row)
        for pitch_class in range(12):
            example = base[cell_index * 12 + pitch_class]
            extra = origin_features(row, pitch_class)
            if len(extra) != len(ORIGIN_FEATURE_NAMES):
                raise AssertionError("Decoder-origin feature contract drifted")
            output.append(
                {
                    **example,
                    "features": list(example["features"]) + extra,
                    "decoderGenerated": bool(
                        finite(evidence[pitch_class].get("decoderGeneratedCount"))
                        > 0
                    ),
                }
            )
    return output
