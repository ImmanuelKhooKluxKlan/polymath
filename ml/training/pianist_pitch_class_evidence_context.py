"""Per-pitch source-evidence features for conservative chord additions."""

from __future__ import annotations

import math
from typing import Any

from .pianist_pitch_class_context import (
    FEATURE_NAMES as CONTEXT_FEATURE_NAMES,
    context_examples_from_song,
)


EVIDENCE_FEATURE_NAMES = (
    "pc_source_note_count",
    "pc_source_attack_count",
    "pc_source_sustain_only_count",
    "pc_source_attack_share",
    "pc_source_onset_closeness",
    "pc_source_onset_offset",
    "pc_source_median_duration",
    "pc_source_maximum_velocity",
    "pc_source_register_distance",
    "pc_source_register_offset",
    "pc_source_exact_candidate_midi",
    "pc_source_voice_share",
    "pc_source_guitar_share",
    "pc_source_bass_share",
    "pc_source_piano_share",
    "pc_source_strings_share",
    "pc_source_other_share",
    "pc_guitar_x_candidate_bass",
    "pc_bass_x_candidate_bass",
    "pc_voice_x_candidate_melody",
    "pc_piano_x_candidate_harmony",
    "pc_strings_x_candidate_harmony",
)
FEATURE_NAMES = CONTEXT_FEATURE_NAMES + EVIDENCE_FEATURE_NAMES


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def evidence_by_pitch_class(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = row.get("sourcePitchClassEvidence")
    if not isinstance(values, list) or len(values) != 12:
        raise ValueError(
            "Every cell needs twelve inference-safe sourcePitchClassEvidence rows"
        )
    parsed: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Pitch-class evidence rows must be objects")
        pitch_class = int(item.get("pitchClass", -1))
        if pitch_class in parsed or not 0 <= pitch_class < 12:
            raise ValueError("Pitch-class evidence indexes must be unique from 0 to 11")
        parsed[pitch_class] = item
    if set(parsed) != set(range(12)):
        raise ValueError("Pitch-class evidence indexes must cover 0 through 11")
    return parsed


def evidence_features(row: dict[str, Any], pitch_class: int) -> list[float]:
    item = evidence_by_pitch_class(row)[pitch_class]
    families = item.get("familyShares") or {}
    count = max(0.0, finite(item.get("noteCount")))
    attacks = max(0.0, finite(item.get("attackCount")))
    sustain = max(0.0, finite(item.get("sustainOnlyCount")))
    onset_offset = finite(item.get("nearestOnsetOffsetSeconds"))
    voice = clamp(finite(families.get("voice")), 0.0, 1.0)
    guitar = clamp(finite(families.get("guitar")), 0.0, 1.0)
    bass = clamp(finite(families.get("bass")), 0.0, 1.0)
    piano = clamp(finite(families.get("piano")), 0.0, 1.0)
    strings = clamp(finite(families.get("strings")), 0.0, 1.0)
    other = clamp(finite(families.get("other")), 0.0, 1.0)
    candidate_bass = clamp(finite(row.get("candidateBassShare")), 0.0, 1.0)
    candidate_melody = clamp(
        finite(row.get("candidateMelodyShare")), 0.0, 1.0
    )
    candidate_harmony = clamp(1.0 - candidate_bass - candidate_melody, 0.0, 1.0)
    return [
        clamp(count / 6.0, 0.0, 1.0),
        clamp(attacks / 4.0, 0.0, 1.0),
        clamp(sustain / 6.0, 0.0, 1.0),
        attacks / max(1.0, count),
        max(0.0, 1.0 - abs(onset_offset) / 0.5),
        clamp(onset_offset / 2.0, -1.0, 1.0),
        clamp(math.log1p(max(0.0, finite(item.get("medianDurationSeconds")))) / math.log(5.0), 0.0, 1.5),
        clamp(finite(item.get("maximumVelocity")), 0.0, 1.0),
        clamp(finite(item.get("minimumRegisterDistanceSemitones")) / 48.0, 0.0, 1.5),
        clamp(finite(item.get("closestRegisterOffsetSemitones")) / 48.0, -1.5, 1.5),
        float(bool(item.get("exactCandidateMidiMatch"))),
        voice,
        guitar,
        bass,
        piano,
        strings,
        other,
        guitar * candidate_bass,
        bass * candidate_bass,
        voice * candidate_melody,
        piano * candidate_harmony,
        strings * candidate_harmony,
    ]


def evidence_examples_from_song(song: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(song.get("cells") or [])
    base = context_examples_from_song(song)
    if len(base) != len(rows) * 12:
        raise ValueError("Context pitch-class example grid is malformed")
    output: list[dict[str, Any]] = []
    for cell_index, row in enumerate(rows):
        for pitch_class in range(12):
            example = base[cell_index * 12 + pitch_class]
            extra = evidence_features(row, pitch_class)
            if len(extra) != len(EVIDENCE_FEATURE_NAMES):
                raise AssertionError("Per-pitch evidence feature contract drifted")
            output.append(
                {**example, "features": list(example["features"]) + extra}
            )
    return output
