"""Inference-safe phrase context for source-supported chord additions.

The evidence259 contract describes one arranger gesture well, but a pianist's
choice is also constrained by the gestures immediately before and after it.
This module adds only candidate/source-derived sequence evidence.  It never
reads the aligned reference; the reference continues to supply labels only
during training.
"""

from __future__ import annotations

from typing import Any

from .pianist_pitch_class_evidence_context import (
    FEATURE_NAMES as EVIDENCE_FEATURE_NAMES,
    evidence_by_pitch_class,
    evidence_examples_from_song,
)


NEIGHBOR_OFFSETS = (-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8)
FAMILY_OFFSETS = (-2, -1, 1, 2)
FAMILIES = ("voice", "guitar", "bass", "piano")
PERSISTENCE_RADII = (1, 2, 4, 8)

SEQUENCE_FEATURE_NAMES = (
    tuple(f"neighbor_{offset:+d}_candidate" for offset in NEIGHBOR_OFFSETS)
    + tuple(f"neighbor_{offset:+d}_source" for offset in NEIGHBOR_OFFSETS)
    + tuple(f"neighbor_{offset:+d}_attack" for offset in NEIGHBOR_OFFSETS)
    + tuple(
        f"neighbor_{offset:+d}_{family}_share"
        for offset in FAMILY_OFFSETS
        for family in FAMILIES
    )
    + tuple(
        f"radius_{radius}_{kind}_persistence"
        for radius in PERSISTENCE_RADII
        for kind in ("candidate", "source", "attack")
    )
    + (
        "centered_source_run_length",
        "minimum_candidate_pitch_class_distance",
        "candidate_consonant_interval_share",
        "candidate_close_dissonance_share",
        "completed_major_minor_triad_count",
        "source_missing_pitch_class_share",
        "current_source_attack_share",
        "neighbor_source_attack_mean",
        "source_present_on_both_sides",
        "candidate_present_on_both_sides",
    )
)
FEATURE_NAMES = EVIDENCE_FEATURE_NAMES + SEQUENCE_FEATURE_NAMES


def _pitch_classes(row: dict[str, Any], field: str) -> set[int]:
    return {
        int(value) % 12
        for value in row.get(field) or []
        if isinstance(value, (int, float))
    }


def _evidence(rows: list[dict[str, Any]], index: int, pitch_class: int) -> dict[str, Any]:
    if not 0 <= index < len(rows):
        return {}
    return evidence_by_pitch_class(rows[index]).get(pitch_class, {})


def _present(rows: list[dict[str, Any]], index: int, field: str, pitch_class: int) -> float:
    if not 0 <= index < len(rows):
        return 0.0
    return float(pitch_class in _pitch_classes(rows[index], field))


def _attack(rows: list[dict[str, Any]], index: int, pitch_class: int) -> float:
    return float(float(_evidence(rows, index, pitch_class).get("attackCount") or 0.0) > 0.0)


def _family_share(
    rows: list[dict[str, Any]], index: int, pitch_class: int, family: str
) -> float:
    families = _evidence(rows, index, pitch_class).get("familyShares") or {}
    return max(0.0, min(1.0, float(families.get(family) or 0.0)))


def _persistence(
    rows: list[dict[str, Any]],
    index: int,
    pitch_class: int,
    radius: int,
    kind: str,
) -> float:
    values: list[float] = []
    for offset in range(-radius, radius + 1):
        if not offset:
            continue
        position = index + offset
        if not 0 <= position < len(rows):
            continue
        if kind == "candidate":
            values.append(_present(rows, position, "candidatePitchClasses", pitch_class))
        elif kind == "source":
            values.append(_present(rows, position, "sourcePitchClasses", pitch_class))
        else:
            values.append(_attack(rows, position, pitch_class))
    return sum(values) / max(1, len(values))


def _centered_run(rows: list[dict[str, Any]], index: int, pitch_class: int) -> float:
    run = 1 if _present(rows, index, "sourcePitchClasses", pitch_class) else 0
    if not run:
        return 0.0
    for direction in (-1, 1):
        for distance in range(1, 9):
            if not _present(
                rows, index + direction * distance, "sourcePitchClasses", pitch_class
            ):
                break
            run += 1
    return min(1.0, run / 9.0)


def _harmonic_features(row: dict[str, Any], pitch_class: int) -> list[float]:
    candidate = _pitch_classes(row, "candidatePitchClasses")
    source = _pitch_classes(row, "sourcePitchClasses")
    intervals = {(value - pitch_class) % 12 for value in candidate}
    circular = [min(value, 12 - value) for value in intervals]
    consonant = {0, 3, 4, 5, 7, 8, 9}
    close_dissonance = {1, 2, 6, 10, 11}
    combined = candidate | {pitch_class}
    triads = 0
    for root in range(12):
        for third in (3, 4):
            triad = {root, (root + third) % 12, (root + 7) % 12}
            if pitch_class in triad and triad <= combined:
                triads += 1
    return [
        min(circular, default=6) / 6.0,
        len(intervals & consonant) / max(1, len(intervals)),
        len(intervals & close_dissonance) / max(1, len(intervals)),
        min(1.0, triads / 3.0),
        len(source - candidate) / max(1, len(source)),
    ]


def sequence_features(
    rows: list[dict[str, Any]], index: int, pitch_class: int
) -> list[float]:
    values: list[float] = []
    values.extend(
        _present(rows, index + offset, "candidatePitchClasses", pitch_class)
        for offset in NEIGHBOR_OFFSETS
    )
    values.extend(
        _present(rows, index + offset, "sourcePitchClasses", pitch_class)
        for offset in NEIGHBOR_OFFSETS
    )
    values.extend(_attack(rows, index + offset, pitch_class) for offset in NEIGHBOR_OFFSETS)
    values.extend(
        _family_share(rows, index + offset, pitch_class, family)
        for offset in FAMILY_OFFSETS
        for family in FAMILIES
    )
    values.extend(
        _persistence(rows, index, pitch_class, radius, kind)
        for radius in PERSISTENCE_RADII
        for kind in ("candidate", "source", "attack")
    )
    current = _evidence(rows, index, pitch_class)
    note_count = max(0.0, float(current.get("noteCount") or 0.0))
    attack_count = max(0.0, float(current.get("attackCount") or 0.0))
    neighboring_attacks = [
        _attack(rows, index + offset, pitch_class) for offset in (-2, -1, 1, 2)
    ]
    values.extend(
        [
            _centered_run(rows, index, pitch_class),
            *_harmonic_features(rows[index], pitch_class),
            attack_count / max(1.0, note_count),
            sum(neighboring_attacks) / len(neighboring_attacks),
            _present(rows, index - 1, "sourcePitchClasses", pitch_class)
            * _present(rows, index + 1, "sourcePitchClasses", pitch_class),
            _present(rows, index - 1, "candidatePitchClasses", pitch_class)
            * _present(rows, index + 1, "candidatePitchClasses", pitch_class),
        ]
    )
    if len(values) != len(SEQUENCE_FEATURE_NAMES):
        raise AssertionError("Sequence feature contract drifted")
    return values


def sequence_examples_from_song(song: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(song.get("cells") or [])
    base = evidence_examples_from_song(song)
    if len(base) != len(rows) * 12:
        raise ValueError("Evidence pitch-class example grid is malformed")
    output: list[dict[str, Any]] = []
    for cell_index, _row in enumerate(rows):
        for pitch_class in range(12):
            example = base[cell_index * 12 + pitch_class]
            extra = sequence_features(rows, cell_index, pitch_class)
            output.append({**example, "features": list(example["features"]) + extra})
    return output
