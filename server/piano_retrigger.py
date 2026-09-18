"""Classify same-key piano events without deleting legitimate fast rhythm.

Transcription noise and musical repetitions can both look like two nearby
onsets on the same MIDI pitch.  A single minimum-gap threshold cannot tell
them apart: it removes machine-gun fragments, but it also erases tremolos,
repeated melody syllables, and rhythmic accompaniment.

This module keeps the decision small and deterministic so both the learned
arranger and its final range-fold cleanup use the same rules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


DEFAULT_DUPLICATE_ONSET_SECONDS = 0.065
DEFAULT_COLLISION_WINDOW_SECONDS = 0.14
FAST_MUSICAL_RETRIGGER_SECONDS = 0.18


def _text(note: dict[str, Any], *names: str) -> str:
    for name in names:
        value = note.get(name)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _number(note: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = note.get(name)
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return None


def source_pitch(note: dict[str, Any]) -> int | None:
    """Return the pitch before arranger range shifts whenever it is known."""

    return _number(
        note,
        "sourceMidiBeforeArrangement",
        "originalMidiBeforeRangeShift",
        "originalMidiBeforeRangeFold",
        "midi",
    )


def classify_same_key_retrigger(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    duplicate_seconds: float = DEFAULT_DUPLICATE_ONSET_SECONDS,
    collision_seconds: float = DEFAULT_COLLISION_WINDOW_SECONDS,
) -> str:
    """Return ``duplicate``, ``collision``, or ``musical-repeat``.

    A repeat from the same source pitch, part, and instrument is retained once
    it clears the short duplicate window.  A nearby event created by two roles,
    stems, source pitches, or copies of one source event landing on the same
    final key is an arranger collision and is merged.
    """

    gap = float(current.get("time", 0)) - float(previous.get("time", 0))
    duplicate_limit = max(0.035, min(0.075, float(duplicate_seconds)))
    collision_limit = max(
        duplicate_limit,
        min(FAST_MUSICAL_RETRIGGER_SECONDS, float(collision_seconds)),
    )
    if gap < duplicate_limit:
        return "duplicate"
    if gap >= collision_limit:
        return "musical-repeat"

    previous_role = _text(previous, "arrangementRole", "role")
    current_role = _text(current, "arrangementRole", "role")
    previous_instrument = _text(previous, "sourceInstrument", "instrument")
    current_instrument = _text(current, "sourceInstrument", "instrument")
    previous_source_index = _number(previous, "sourceIndex")
    current_source_index = _number(current, "sourceIndex")
    previous_pitch = source_pitch(previous)
    current_pitch = source_pitch(current)

    same_source_event = (
        previous_source_index is not None
        and current_source_index is not None
        and previous_source_index == current_source_index
    )
    different_roles = bool(
        previous_role and current_role and previous_role != current_role
    )
    different_stems = bool(
        previous_instrument
        and current_instrument
        and previous_instrument != current_instrument
    )
    range_fold_collision = bool(
        previous_pitch is not None
        and current_pitch is not None
        and previous_pitch != current_pitch
    )
    generated_collision = bool(
        _text(previous, "generatedBy") or _text(current, "generatedBy")
    ) and same_source_event

    if (
        same_source_event
        or different_roles
        or different_stems
        or range_fold_collision
        or generated_collision
    ):
        return "collision"
    return "musical-repeat"


def summarize_same_key_retriggers(
    notes: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Describe what survived cleanup without labelling all fast notes bad."""

    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        try:
            by_pitch[int(round(float(note.get("midi"))))].append(note)
        except (TypeError, ValueError):
            continue
    under_duplicate = 0
    fast_musical = 0
    minimum_gap: float | None = None
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda item: float(item.get("time", 0)))
        for previous, current in zip(pitch_notes, pitch_notes[1:]):
            gap = float(current.get("time", 0)) - float(previous.get("time", 0))
            if gap <= 0:
                under_duplicate += 1
                minimum_gap = 0.0 if minimum_gap is None else min(minimum_gap, 0.0)
                continue
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            if gap < DEFAULT_DUPLICATE_ONSET_SECONDS:
                under_duplicate += 1
            elif gap < FAST_MUSICAL_RETRIGGER_SECONDS:
                fast_musical += 1
    return {
        "unresolvedDuplicateRetriggersUnder65ms": under_duplicate,
        "preservedFastMusicalRetriggers65To180ms": fast_musical,
        "minimumSameKeyGapSeconds": (
            None if minimum_gap is None else round(minimum_gap, 6)
        ),
    }


__all__ = [
    "DEFAULT_COLLISION_WINDOW_SECONDS",
    "DEFAULT_DUPLICATE_ONSET_SECONDS",
    "FAST_MUSICAL_RETRIGGER_SECONDS",
    "classify_same_key_retrigger",
    "source_pitch",
    "summarize_same_key_retriggers",
]
