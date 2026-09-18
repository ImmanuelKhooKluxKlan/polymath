"""Reduce MuScriptor events into an idiomatic, playable 88-key piano arrangement.

The transcription model reports what it hears. A piano cover needs a second
stage that decides what a pianist should actually play. This module preserves
genuine acoustic-piano sources, but reduces full mixes to melody, bass, and
compact harmony while removing percussion and excessive event density.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from omr.polymath_omr.performance import shape_piano_performance
from piano_mix_balance import adaptive_melody_mix_balance
from piano_gesture_calibration import (
    calibrated_gesture_duration,
    calibrated_gesture_velocity,
    gesture_sequence_contexts,
)
from piano_retrigger import (
    DEFAULT_COLLISION_WINDOW_SECONDS,
    DEFAULT_DUPLICATE_ONSET_SECONDS,
    FAST_MUSICAL_RETRIGGER_SECONDS,
    classify_same_key_retrigger,
    summarize_same_key_retriggers,
)


PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108
COMPACT_PIANO_MIN_MIDI = 33  # A1
COMPACT_PIANO_MAX_MIDI = 108  # C8
PIANELLA_PREFERRED_GLOBAL_SHIFT = 0
PIANELLA_RANGE_PROFILE = "pianella-range-aware-a1-c8-v3"
MELODY_MIN_MIDI = 55
MELODY_MAX_MIDI = 88
BASS_MIN_MIDI = 28
BASS_MAX_MIDI = 52
HARMONY_MIN_MIDI = 48
HARMONY_MAX_MIDI = 76

MIN_NOTE_SECONDS = 0.05
MAX_NOTE_SECONDS = 6.0
MIN_RETRIGGER_SECONDS = 0.10
SAME_KEY_RELEASE_GAP_SECONDS = 0.018
MAX_ONSET_CLUSTER = 6
MAX_ARRANGED_NOTES_PER_SECOND = 12
MAX_DIRECT_PIANO_NOTES_PER_SECOND = 12
MAX_DIRECT_PURE_PIANO_NOTES_PER_SECOND = 48
MAX_HARMONY_PITCH_CLASSES = 4
MAX_DIRECT_CLEANUP_PRESSURE = 0.18
MAX_DIRECT_PURE_PIANO_CLEANUP_PRESSURE = 0.25
DIRECT_PIANO_MINIMUM_SCORE_HOLD_SECONDS = 0.10
DIRECT_PIANO_SHORT_BASS_MAXIMUM_MIDI = 59
DIRECT_PIANO_SHORT_BASS_THRESHOLD_SECONDS = 0.14
DIRECT_PIANO_SHORT_BASS_EXTENSION_SECONDS = 0.01
DIRECT_PIANO_SHORT_HOLD_SCALE = 1.45
DIRECT_PIANO_SHORT_HOLD_SCALE_MAX_SECONDS = 0.25

# A piano reduction needs orchestration dynamics, not only the model's raw
# confidence velocity.  Full mixes frequently arrive with every event at the
# same (or almost the same) velocity, and pushing the melody to 1.0 only makes
# the browser compressor flatten the whole arrangement.  These deliberately
# separated bands leave headroom while keeping the sung/top line in front.
MELODY_VELOCITY_RANGE = (0.84, 0.96)
RIGHT_HAND_VELOCITY_RANGE = (0.60, 0.76)
LEFT_HAND_VELOCITY_RANGE = (0.40, 0.57)
BASS_VELOCITY_RANGE = (0.34, 0.50)
SOURCE_RIGHT_VELOCITY_RANGE = (0.64, 0.86)
SOURCE_LEFT_VELOCITY_RANGE = (0.42, 0.59)
EXPRESSION_ONSET_WINDOW_SECONDS = 0.055
GESTURE_DYNAMIC_ONSET_WINDOW_SECONDS = 0.035

DEFAULT_ROLE_LEGATO_BRIDGE_SECONDS = {
    "melody": 0.90,
    "bass": 1.80,
    "harmony": 2.40,
}
DEFAULT_ROLE_PHYSICAL_HOLD_SECONDS = {
    "melody": 1.35,
    "bass": 2.20,
    "harmony": 2.60,
}

PIANO_INSTRUMENTS = {"acoustic_piano", "electric_piano"}
PERCUSSION_INSTRUMENTS = {"drums", "timpani"}
BASS_INSTRUMENTS = {"acoustic_bass", "electric_bass", "contrabass"}
VOICE_INSTRUMENTS = {"voice"}
LEAD_INSTRUMENTS = {
    "synth_lead",
    "violin",
    "viola",
    "cello",
    "flutes",
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
    "trumpet",
}

ROLE_PRIORITY = {
    "melody": 0,
    "bass": 1,
    "source_piano": 2,
    "harmony": 3,
}


def learned_physical_performance_limits(
    style_profile: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    """Return bounded per-role key-hold limits from a learned profile.

    Older profiles have no physical-performance block and therefore retain the
    exact historic values.  New candidates can shorten accompaniment key holds
    while the pedal/release layer continues the string, avoiding a sparse
    arrangement turning every gap into a physically held note.
    """

    decoder = (style_profile or {}).get("decoder") or {}
    configured = decoder.get("physicalPerformance") or {}
    configured_bridges = configured.get("maximumLegatoBridgeSeconds") or {}
    configured_holds = configured.get("maximumPhysicalHoldSeconds") or {}

    def bounded_values(
        defaults: dict[str, float],
        values: dict[str, Any],
        minimum: float,
        maximum: float,
    ) -> dict[str, float]:
        if not isinstance(values, dict):
            values = {}
        return {
            role: round_number(
                clamp(float(values.get(role, default)), minimum, maximum), 4
            )
            for role, default in defaults.items()
        }

    return {
        "maximumLegatoBridgeSeconds": bounded_values(
            DEFAULT_ROLE_LEGATO_BRIDGE_SECONDS,
            configured_bridges,
            0.05,
            4.0,
        ),
        "maximumPhysicalHoldSeconds": bounded_values(
            DEFAULT_ROLE_PHYSICAL_HOLD_SECONDS,
            configured_holds,
            0.055,
            4.0,
        ),
    }


def adapt_physical_performance_to_source(
    style_profile: dict[str, Any],
    factual_source_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Interpolate role hold limits only for a sparse, vocal, non-piano mix."""

    decoder = style_profile.get("decoder") or {}
    physical = decoder.get("physicalPerformance") or {}
    policy = physical.get("adaptiveSourceDensity") or {}
    if not policy.get("enabled"):
        return style_profile, {}

    low_nps = float(policy.get("lowSourceNotesPerSecond", 5.0))
    high_nps = float(policy.get("highSourceNotesPerSecond", 10.0))
    if high_nps <= low_nps:
        raise ValueError(
            "Adaptive physical-performance high threshold must exceed its low threshold."
        )
    source_nps = max(
        0.0, float(factual_source_profile.get("sourceNotesPerSecond", 0.0))
    )
    source_voice_ratio = max(
        0.0, float(factual_source_profile.get("voiceRatio", 0.0))
    )
    source_piano_ratio = max(
        0.0, float(factual_source_profile.get("pianoRatio", 0.0))
    )
    density_ratio = clamp((source_nps - low_nps) / (high_nps - low_nps), 0.0, 1.0)
    ratio = density_ratio
    minimum_voice_ratio = clamp(
        float(policy.get("minimumVoiceRatio", 0.05)), 0.0, 1.0
    )
    maximum_piano_ratio = clamp(
        float(policy.get("maximumPianoRatio", 0.05)), 0.0, 1.0
    )
    voice_gate_applied = (
        source_voice_ratio < minimum_voice_ratio
        or source_piano_ratio > maximum_piano_ratio
    )
    if voice_gate_applied:
        ratio = 1.0

    static_limits = learned_physical_performance_limits(style_profile)
    effective = copy.deepcopy(style_profile)
    effective_physical = effective.setdefault("decoder", {}).setdefault(
        "physicalPerformance", {}
    )

    def interpolate_roles(
        name: str,
        defaults: dict[str, float],
        minimum: float,
    ) -> dict[str, float]:
        low_values = policy.get(f"low{name}") or {}
        high_values = policy.get(f"high{name}") or {}
        if not isinstance(low_values, dict) or not isinstance(high_values, dict):
            raise ValueError("Adaptive physical-performance role limits must be objects.")
        values: dict[str, float] = {}
        for role, default in defaults.items():
            low = clamp(float(low_values.get(role, default)), minimum, 4.0)
            high = clamp(float(high_values.get(role, default)), minimum, 4.0)
            values[role] = round_number(low + ratio * (high - low), 4)
        effective_physical[name[0].lower() + name[1:]] = values
        return values

    bridges = interpolate_roles(
        "MaximumLegatoBridgeSeconds",
        static_limits["maximumLegatoBridgeSeconds"],
        0.05,
    )
    holds = interpolate_roles(
        "MaximumPhysicalHoldSeconds",
        static_limits["maximumPhysicalHoldSeconds"],
        0.055,
    )
    return effective, {
        "physicalPerformancePolicy": "sparse-vocal-source-density-v1",
        "physicalPerformanceDensityRatio": round_number(density_ratio, 4),
        "physicalPerformanceInterpolationRatio": round_number(ratio, 4),
        "physicalPerformanceVoiceGateApplied": voice_gate_applied,
        "physicalPerformanceSourceNotesPerSecond": round_number(source_nps, 3),
        "physicalPerformanceSourceVoiceRatio": round_number(source_voice_ratio, 4),
        "physicalPerformanceSourcePianoRatio": round_number(source_piano_ratio, 4),
        "effectiveMaximumLegatoBridgeSeconds": bridges,
        "effectiveMaximumPhysicalHoldSeconds": holds,
    }

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_number(value: float, places: int = 4) -> float:
    return round(float(value), places)


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def safe_number(value: Any, fallback: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def map_octave_to_range(midi: int, minimum: int, maximum: int) -> int:
    value = int(round(midi))
    while value < minimum:
        value += 12
    while value > maximum:
        value -= 12
    return int(clamp(value, minimum, maximum))


def compact_pianella_register(
    notes: list[dict[str, Any]],
    preferred_global_shift_semitones: int = PIANELLA_PREFERRED_GLOBAL_SHIFT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the authored register, then octave-fold only unavoidable edges.

    Pianella-style playback no longer applies a blanket upward transposition:
    A1 stays A1 and C4 stays C4. Notes outside A1-C8 are octave-folded by the
    minimum required amount, never dropped or pitch-class-clamped.
    """
    preferred_global_shift = int(
        clamp(
            round(float(preferred_global_shift_semitones) / 12.0) * 12,
            -84,
            0,
        )
    )
    source_midis = [int(note["midi"]) for note in notes]
    global_shift = 0
    source_minimum = min(source_midis) if source_midis else None
    source_maximum = max(source_midis) if source_midis else None
    global_inside = 0
    if source_midis:
        candidates: list[tuple[int, int]] = []
        for shift in range(preferred_global_shift, -85, -12):
            inside = sum(
                COMPACT_PIANO_MIN_MIDI <= midi + shift <= COMPACT_PIANO_MAX_MIDI
                for midi in source_midis
            )
            candidates.append((shift, inside))
        best_coverage = max(inside for _, inside in candidates)
        tolerated_misses = math.floor(len(source_midis) * 0.01)
        global_shift, global_inside = next(
            candidate
            for candidate in candidates
            if candidate[1] >= best_coverage - tolerated_misses
        )

    output: list[dict[str, Any]] = []
    shifted = 0
    edge_folded = 0
    shift_counts: Counter[int] = Counter()
    for source in notes:
        note = dict(source)
        original_midi = int(note["midi"])
        midi = original_midi + global_shift
        before_edge_fold = original_midi + global_shift
        while midi < COMPACT_PIANO_MIN_MIDI:
            midi += 12
        while midi > COMPACT_PIANO_MAX_MIDI:
            midi -= 12
        midi = int(clamp(midi, COMPACT_PIANO_MIN_MIDI, COMPACT_PIANO_MAX_MIDI))
        edge_fold = midi - before_edge_fold
        shift = midi - original_midi
        if shift:
            shifted += 1
            shift_counts[shift] += 1
            note["originalMidiBeforeRangeShift"] = original_midi
            note["originalNoteBeforeRangeShift"] = note.get("note") or midi_to_note(original_midi)
            # Backwards-compatible names retained for previously downloaded
            # Polymath JSON files and older analytics views.
            note["originalMidiBeforeRangeFold"] = original_midi
            note["originalNoteBeforeRangeFold"] = note.get("note") or midi_to_note(original_midi)
        if edge_fold:
            edge_folded += 1
        note["globalRegisterShiftSemitones"] = global_shift
        note["edgeOctaveFoldSemitones"] = edge_fold
        note["octaveShiftSemitones"] = shift
        note["wasOctaveFolded"] = edge_fold != 0
        note["wasRegisterShifted"] = shift != 0
        note["midi"] = midi
        note["note"] = midi_to_note(midi)
        source_hand = str(source.get("hand") or "").lower()
        note["hand"] = (
            source_hand
            if source_hand in {"left", "right"}
            else "left" if original_midi < 60 else "right"
        )
        output.append(note)
    return output, {
        "profile": PIANELLA_RANGE_PROFILE,
        "mode": "authored-register-then-edge-fold-a1-c8",
        "applied": True,
        "minimumMidi": COMPACT_PIANO_MIN_MIDI,
        "maximumMidi": COMPACT_PIANO_MAX_MIDI,
        "minimumNote": "A1",
        "maximumNote": "C8",
        "preferredShiftSemitones": preferred_global_shift,
        "globalShiftSemitones": global_shift,
        "sourceNoteCount": len(source_midis),
        "sourceMinimumMidi": source_minimum,
        "sourceMaximumMidi": source_maximum,
        "notesInsideAfterGlobalShift": global_inside,
        "notesOutsideAfterGlobalShift": len(source_midis) - global_inside,
        "coverageRatio": round_number(global_inside / len(source_midis), 4)
        if source_midis else 1.0,
        "shiftedNotes": shifted,
        "edgeFoldedNotes": edge_folded,
        "semitoneShiftCounts": {
            str(shift): count for shift, count in sorted(shift_counts.items())
        },
    }


def adapt_melody_register_separation(
    notes: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Lift one whole melody only when it repeatedly collides with harmony.

    A singer-derived melody sometimes arrives in the same octave as the guitar
    reduction.  Moving isolated notes creates distracting octave jumps, while a
    blanket song-wide lift damages arrangements whose melody is already clear.
    This inference-safe gate measures only the current arrangement and either
    moves the complete melody by one octave or leaves every note untouched.
    """

    policy = config or {}
    enabled = bool(policy.get("enabled", False))
    diagnostics: dict[str, Any] = {
        "profile": "adaptive-melody-register-separation-v1",
        "enabled": enabled,
        "applied": False,
        "reason": "disabled" if not enabled else "not-evaluated",
    }
    if not enabled:
        return list(notes), diagnostics

    melody = [
        note for note in notes if str(note.get("arrangementRole") or "") == "melody"
    ]
    harmony = sorted(
        (
            note
            for note in notes
            if str(note.get("arrangementRole") or "") == "harmony"
        ),
        key=lambda note: float(note["time"]),
    )
    minimum_melody_notes = max(1, int(policy.get("minimumMelodyNotes", 24)))
    minimum_compared_notes = max(1, int(policy.get("minimumComparedNotes", 12)))
    radius = clamp(float(policy.get("nearbyHarmonyRadiusSeconds", 0.12)), 0.02, 0.40)
    clearance = int(clamp(int(policy.get("collisionClearanceSemitones", 2)), 0, 12))
    minimum_collision_share = clamp(
        float(policy.get("minimumCollisionShare", 0.20)), 0.0, 1.0
    )
    shift = int(
        clamp(
            round(float(policy.get("octaveShiftSemitones", 12)) / 12.0) * 12,
            0,
            24,
        )
    )
    onset_delay = clamp(float(policy.get("onsetDelaySeconds", 0.0)), -0.08, 0.08)
    diagnostics.update(
        {
            "melodyNotes": len(melody),
            "harmonyNotes": len(harmony),
            "minimumMelodyNotes": minimum_melody_notes,
            "minimumComparedNotes": minimum_compared_notes,
            "nearbyHarmonyRadiusSeconds": round_number(radius, 4),
            "collisionClearanceSemitones": clearance,
            "minimumCollisionShare": round_number(minimum_collision_share, 4),
            "requestedShiftSemitones": shift,
            "requestedOnsetDelaySeconds": round_number(onset_delay, 6),
        }
    )
    if len(melody) < minimum_melody_notes:
        diagnostics["reason"] = "insufficient-melody-notes"
        return list(notes), diagnostics
    if not harmony or shift <= 0:
        diagnostics["reason"] = "missing-harmony-or-shift"
        return list(notes), diagnostics

    harmony_times = [float(note["time"]) for note in harmony]
    compared = 0
    collisions = 0
    separations: list[int] = []
    for melody_note in melody:
        onset = float(melody_note["time"])
        left = bisect.bisect_left(harmony_times, onset - radius)
        right = bisect.bisect_right(harmony_times, onset + radius)
        nearby = harmony[left:right]
        if not nearby:
            continue
        compared += 1
        separation = int(melody_note["midi"]) - max(
            int(note["midi"]) for note in nearby
        )
        separations.append(separation)
        collisions += int(separation <= clearance)
    collision_share = collisions / max(1, compared)
    diagnostics.update(
        {
            "comparedMelodyNotes": compared,
            "collisionNotes": collisions,
            "collisionShare": round_number(collision_share, 6),
            "medianMelodyAboveHarmonySemitones": (
                round_number(float(median(separations)), 3) if separations else None
            ),
        }
    )
    if compared < minimum_compared_notes:
        diagnostics["reason"] = "insufficient-nearby-harmony"
        return list(notes), diagnostics
    if collision_share < minimum_collision_share:
        diagnostics["reason"] = "melody-register-already-separated"
        return list(notes), diagnostics
    if any(int(note["midi"]) + shift > COMPACT_PIANO_MAX_MIDI for note in melody):
        diagnostics["reason"] = "whole-melody-shift-exceeds-piano-range"
        return list(notes), diagnostics

    output: list[dict[str, Any]] = []
    shifted_notes = 0
    for source in notes:
        note = dict(source)
        if str(note.get("arrangementRole") or "") == "melody":
            original_midi = int(note["midi"])
            original_time = float(note["time"])
            note["originalMidiBeforeAdaptiveMelodyShift"] = original_midi
            note["originalNoteBeforeAdaptiveMelodyShift"] = (
                note.get("note") or midi_to_note(original_midi)
            )
            note["adaptiveMelodyRegisterShiftSemitones"] = shift
            note["originalTimeBeforeAdaptiveMelodyDelay"] = round_number(
                original_time, 6
            )
            note["adaptiveMelodyOnsetDelaySeconds"] = round_number(
                onset_delay, 6
            )
            note["midi"] = original_midi + shift
            note["note"] = midi_to_note(int(note["midi"]))
            note["time"] = round_number(max(0.0, original_time + onset_delay), 6)
            note["hand"] = "right"
            shifted_notes += 1
        output.append(note)
    diagnostics.update(
        {
            "applied": True,
            "reason": "repeated-melody-harmony-register-collisions",
            "shiftedNotes": shifted_notes,
            "appliedShiftSemitones": shift,
            "appliedOnsetDelaySeconds": round_number(onset_delay, 6),
        }
    )
    return output, diagnostics


def _map_octave_with_preference(
    midi: int,
    minimum: int,
    maximum: int,
    preferred_shift_semitones: int,
) -> int:
    """Move a pitch by octaves only, preserving its pitch class."""

    value = int(midi) + int(round(preferred_shift_semitones / 12.0)) * 12
    while value < minimum:
        value += 12
    while value > maximum:
        value -= 12
    if minimum <= value <= maximum:
        return value
    candidates = [
        candidate
        for candidate in range(minimum, maximum + 1)
        if candidate % 12 == int(midi) % 12
    ]
    if not candidates:
        return int(clamp(value, minimum, maximum))
    center = (minimum + maximum) / 2.0
    return min(candidates, key=lambda candidate: abs(candidate - center))


def compact_authored_two_hand_register(
    notes: list[dict[str, Any]],
    style: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Voice a reduction like an authored two-staff piano arrangement.

    A full-mix transcription contains pitch classes from several instruments.
    A single global octave shift cannot turn that orchestration into two human
    hands.  This policy keeps melody in the authored upper band, bass in the
    authored lower band, and divides harmony between them to reproduce the
    reference staff balance.  Every move is an octave, so note identity and
    key remain unchanged.
    """

    lower = style.get("lower") or {}
    upper = style.get("upper") or {}
    lower_minimum = int(clamp(float(lower.get("minimumMidi", 34)), 21, 108))
    lower_maximum = int(clamp(float(lower.get("maximumMidi", 58)), 21, 108))
    upper_minimum = int(clamp(float(upper.get("minimumMidi", 60)), 21, 108))
    upper_maximum = int(clamp(float(upper.get("maximumMidi", 84)), 21, 108))
    if lower_maximum < lower_minimum or upper_maximum < upper_minimum:
        raise ValueError("Authored two-hand register ranges are invalid.")

    target_upper_share = clamp(float(style.get("targetUpperNoteShare", 0.54)), 0.25, 0.80)
    harmony_hand_assignment = str(
        style.get("harmonyHandAssignment") or "target-upper-share"
    ).strip().lower()
    melody = [note for note in notes if note.get("arrangementRole") == "melody"]
    harmony = [note for note in notes if note.get("arrangementRole") == "harmony"]
    target_upper_notes = int(round(len(notes) * target_upper_share))
    if harmony_hand_assignment == "preserve-native-register":
        required_upper_harmony = sum(
            int(note.get("midi", 60)) >= upper_minimum for note in harmony
        )
    else:
        required_upper_harmony = int(
            clamp(target_upper_notes - len(melody), 0, len(harmony))
        )

    # Higher harmony tones make the most stable right-hand candidates.  The
    # remaining chord tones support the bass in the left hand.  Deterministic
    # tie breakers avoid a score changing between identical runs.
    if harmony_hand_assignment == "preserve-native-register":
        upper_harmony_ids = {
            id(note) for note in harmony if int(note.get("midi", 60)) >= upper_minimum
        }
    else:
        ranked_harmony = sorted(
            harmony,
            key=lambda note: (
                int(note.get("midi", 60)),
                float(note.get("selectionProbability", 0.0)),
                float(note.get("velocity", 0.0)),
                float(note.get("duration", 0.0)),
                -float(note.get("time", 0.0)),
                -int(note.get("sourceIndex", 0)),
            ),
            reverse=True,
        )
        upper_harmony_ids = {
            id(note) for note in ranked_harmony[:required_upper_harmony]
        }
    shifts = style.get("preferredOctaveShiftsSemitones") or {}
    preferred_shifts = {
        "melodyUpper": int(shifts.get("melodyUpper", 0)),
        "bassLower": int(shifts.get("bassLower", 0)),
        "harmonyUpper": int(shifts.get("harmonyUpper", 0)),
        "harmonyLower": int(shifts.get("harmonyLower", 0)),
    }

    output: list[dict[str, Any]] = []
    shift_counts: Counter[int] = Counter()
    hand_counts: Counter[str] = Counter()
    for source in notes:
        note = dict(source)
        original_midi = int(note["midi"])
        role = str(note.get("arrangementRole") or "harmony")
        if role == "melody":
            hand = "right"
            minimum, maximum = upper_minimum, upper_maximum
            preferred_shift = preferred_shifts["melodyUpper"]
        elif role == "bass":
            hand = "left"
            minimum, maximum = lower_minimum, lower_maximum
            preferred_shift = preferred_shifts["bassLower"]
        elif id(source) in upper_harmony_ids:
            hand = "right"
            minimum, maximum = upper_minimum, upper_maximum
            preferred_shift = preferred_shifts["harmonyUpper"]
        else:
            hand = "left"
            minimum, maximum = lower_minimum, lower_maximum
            preferred_shift = preferred_shifts["harmonyLower"]
        midi = _map_octave_with_preference(
            original_midi, minimum, maximum, preferred_shift
        )
        shift = midi - original_midi
        if shift:
            note["originalMidiBeforeRangeShift"] = original_midi
            note["originalNoteBeforeRangeShift"] = note.get("note") or midi_to_note(original_midi)
            note["originalMidiBeforeRangeFold"] = original_midi
            note["originalNoteBeforeRangeFold"] = note.get("note") or midi_to_note(original_midi)
        note["globalRegisterShiftSemitones"] = 0
        note["edgeOctaveFoldSemitones"] = shift
        note["octaveShiftSemitones"] = shift
        note["wasOctaveFolded"] = shift != 0
        note["wasRegisterShifted"] = shift != 0
        note["midi"] = midi
        note["note"] = midi_to_note(midi)
        note["hand"] = hand
        output.append(note)
        shift_counts[shift] += 1
        hand_counts[hand] += 1

    return output, {
        "profile": "authored-two-hand-register-v1",
        "mode": "role-aware-octave-voicing",
        "applied": True,
        "pitchClassPreserved": True,
        "lower": {
            "minimumMidi": lower_minimum,
            "maximumMidi": lower_maximum,
            "notes": hand_counts["left"],
        },
        "upper": {
            "minimumMidi": upper_minimum,
            "maximumMidi": upper_maximum,
            "notes": hand_counts["right"],
        },
        "targetUpperNoteShare": round_number(target_upper_share, 4),
        "harmonyHandAssignment": harmony_hand_assignment,
        "actualUpperNoteShare": round_number(
            hand_counts["right"] / max(1, len(output)), 4
        ),
        "upperHarmonyNotes": required_upper_harmony,
        "preferredOctaveShiftsSemitones": preferred_shifts,
        "semitoneShiftCounts": {
            str(shift): count for shift, count in sorted(shift_counts.items())
        },
    }


def normalize_source_note(note: dict[str, Any]) -> dict[str, Any] | None:
    midi_value = safe_number(note.get("midi"))
    time = safe_number(note.get("time"))
    duration = safe_number(note.get("duration"), 0.4)
    if midi_value is None or time is None or duration is None:
        return None
    midi = int(round(midi_value))
    if midi < PIANO_MIN_MIDI or midi > PIANO_MAX_MIDI or time < 0:
        return None
    instrument = str(note.get("instrument") or "acoustic_piano").strip().lower()
    normalized = dict(note)
    normalized.update(
        {
            "midi": midi,
            "note": midi_to_note(midi),
            "time": round_number(time),
            "duration": round_number(clamp(duration, MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)),
            "velocity": round_number(
                clamp(safe_number(note.get("velocity"), 0.72) or 0.72, 0.05, 1.0),
                3,
            ),
            "instrument": instrument or "acoustic_piano",
        }
    )
    return normalized


def notes_per_second(notes: list[dict[str, Any]]) -> float:
    if not notes:
        return 0.0
    duration = max(note["time"] + note["duration"] for note in notes)
    return len(notes) / max(1.0, duration)


def maximum_onset_cluster(notes: Iterable[dict[str, Any]], window: float = 0.04) -> int:
    buckets: Counter[int] = Counter(round(note["time"] / window) for note in notes)
    return max(buckets.values(), default=0)


def source_profile(
    notes: list[dict[str, Any]],
    cleanup: dict[str, Any] | None = None,
    *,
    allow_pure_piano_density_override: bool = True,
) -> dict[str, Any]:
    counts = Counter(note["instrument"] for note in notes)
    total = max(1, len(notes))
    acoustic_ratio = counts["acoustic_piano"] / total
    piano_ratio = sum(counts[name] for name in PIANO_INSTRUMENTS) / total
    voice_ratio = sum(counts[name] for name in VOICE_INSTRUMENTS) / total
    bass_ratio = sum(counts[name] for name in BASS_INSTRUMENTS) / total
    density = notes_per_second(notes)
    non_percussive_durations = [
        float(note["duration"])
        for note in notes
        if note["instrument"] not in PERCUSSION_INSTRUMENTS
    ]
    onset_cluster = maximum_onset_cluster(notes)
    cleanup = cleanup or {}
    cleanup_input = max(1, int(safe_number(cleanup.get("inputNotes"), len(notes)) or len(notes)))
    cleanup_artifacts = sum(
        int(safe_number(cleanup.get(field), 0) or 0)
        for field in ("removedDuplicateNotes", "shortenedSameKeyOverlaps")
    )
    cleanup_pressure = cleanup_artifacts / cleanup_input
    pure_piano_source = (
        allow_pure_piano_density_override
        and sum(counts[name] for name in PIANO_INSTRUMENTS) >= 24
        and piano_ratio >= 0.98
        and voice_ratio == 0.0
        and bass_ratio == 0.0
    )
    direct_density_limit = (
        MAX_DIRECT_PURE_PIANO_NOTES_PER_SECOND
        if pure_piano_source
        else MAX_DIRECT_PIANO_NOTES_PER_SECOND
    )
    direct_cleanup_limit = (
        MAX_DIRECT_PURE_PIANO_CLEANUP_PRESSURE
        if pure_piano_source
        else MAX_DIRECT_CLEANUP_PRESSURE
    )
    direct_acoustic_piano = (
        sum(counts[name] for name in PIANO_INSTRUMENTS) >= 24
        and (pure_piano_source or acoustic_ratio >= 0.70)
        and piano_ratio >= 0.90
        and density <= direct_density_limit
        and onset_cluster <= 8
        and cleanup_pressure <= direct_cleanup_limit
    )
    return {
        "instrumentCounts": dict(sorted(counts.items())),
        "acousticPianoRatio": round_number(acoustic_ratio, 3),
        "pianoRatio": round_number(piano_ratio, 3),
        "voiceRatio": round_number(voice_ratio, 3),
        "bassRatio": round_number(bass_ratio, 3),
        "detectedVoiceNotes": sum(counts[name] for name in VOICE_INSTRUMENTS),
        "sourceNotesPerSecond": round_number(density, 3),
        "sourceMedianDurationSeconds": round_number(
            quantile(non_percussive_durations, 0.50), 4
        ),
        "sourceDurationP10Seconds": round_number(
            quantile(non_percussive_durations, 0.10), 4
        ),
        "sourceDurationP90Seconds": round_number(
            quantile(non_percussive_durations, 0.90), 4
        ),
        "sourceMaximumOnsetCluster": onset_cluster,
        "cleanupArtifactPressure": round_number(cleanup_pressure, 3),
        "cleanupArtifactsDetected": cleanup_artifacts,
        "maximumDirectCleanupPressure": direct_cleanup_limit,
        "purePianoSource": pure_piano_source,
        # Retain the original field for old research readers. Its meaning is
        # now piano-family purity because the model may call the same grand
        # piano acoustic in one passage and electric in another.
        "pureAcousticPianoDensityOverride": pure_piano_source,
        "directPianoDensityLimitNotesPerSecond": direct_density_limit,
        "detectedAcousticPianoPerformance": direct_acoustic_piano,
    }


def grouped_by_window(
    notes: Iterable[dict[str, Any]],
    window_seconds: float,
) -> list[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        groups[round(note["time"] / window_seconds)].append(note)
    return [groups[index] for index in sorted(groups)]


def collapse_focused_melody_collisions(
    notes: list[dict[str, Any]],
    window_seconds: float = EXPRESSION_ONSET_WINDOW_SECONDS,
) -> tuple[list[dict[str, Any]], int]:
    """Keep one singer note when focused decoding creates an onset collision.

    The focused pass is intentionally sensitive, so a vocal onset can retain a
    nearby guitar/harmony candidate as a second ``melody`` note.  The physical
    performance layer interprets that collision as an immediate new melody
    onset and releases the real singer note after only a few milliseconds.

    This repair is deliberately narrow: a cluster is collapsed only when it
    contains provenance from the focused melody decoder.  Ordinary polyphonic
    melody material and every non-melody note pass through unchanged.
    """
    if not notes:
        return notes, 0

    if window_seconds <= 0:
        raise ValueError("Focused melody collision window must be positive.")

    copied = [dict(note) for note in notes]
    melody = sorted(
        (note for note in copied if note.get("arrangementRole") == "melody"),
        key=lambda note: (float(note["time"]), int(note["midi"])),
    )
    non_melody = [
        note for note in copied if note.get("arrangementRole") != "melody"
    ]

    clusters: list[list[dict[str, Any]]] = []
    for note in melody:
        if (
            not clusters
            or float(note["time"]) - float(clusters[-1][0]["time"])
            > window_seconds
        ):
            clusters.append([note])
        else:
            clusters[-1].append(note)

    def has_focused_provenance(note: dict[str, Any]) -> bool:
        return bool(
            note.get("focusedMelodyDecoded")
            or note.get("focusedMelodyAnchorRecovered")
        )

    def priority(note: dict[str, Any]) -> tuple[float, ...]:
        source_instrument = str(note.get("sourceInstrument") or "").lower()
        selection_probability = safe_number(note.get("selectionProbability"), 0.0)
        velocity = safe_number(note.get("velocity"), 0.0)
        duration = safe_number(note.get("duration"), 0.0)
        return (
            float(bool(note.get("focusedMelodyAnchorRecovered"))),
            float(source_instrument in VOICE_INSTRUMENTS),
            float(bool(note.get("focusedMelodyDecoded"))),
            float(selection_probability or 0.0),
            float(velocity or 0.0),
            float(duration or 0.0),
            -float(note["time"]),
            -float(note["midi"]),
        )

    retained_melody: list[dict[str, Any]] = []
    removed = 0
    for cluster in clusters:
        if len(cluster) > 1 and any(has_focused_provenance(note) for note in cluster):
            retained_melody.append(max(cluster, key=priority))
            removed += len(cluster) - 1
        else:
            retained_melody.extend(cluster)

    return (
        sorted(
            non_melody + retained_melody,
            key=lambda note: (float(note["time"]), int(note["midi"])),
        ),
        removed,
    )


def collapse_exact_left_hand_duplicates(
    notes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Collapse impossible simultaneous strikes of one left-hand piano key.

    Hybrid reconstruction can retain the same pitch from two source stems at
    one timestamp. A physical piano has only one such key, so layering both
    samples adds false loudness. Melody and upper-hand notes stay outside this
    repair to preserve the accepted vocal layer exactly.
    """

    protected: list[dict[str, Any]] = []
    protected_keys: set[tuple[int, int]] = set()
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        key = (
            int(round(float(note["time"]) * 1000.0)),
            int(note["midi"]),
        )
        if (
            int(note.get("midi", PIANO_MAX_MIDI + 1)) >= 72
            or note.get("arrangementRole") == "melody"
            or str(note.get("sourceInstrument") or "").lower()
            in VOICE_INSTRUMENTS
        ):
            protected.append(note)
            protected_keys.add(key)
            continue
        groups[key].append(note)

    retained: list[dict[str, Any]] = []
    removed = 0
    for key, duplicates in groups.items():
        # Keep the accepted melody/right-hand event byte-for-byte unchanged.
        # A simultaneous accompaniment copy cannot create another physical
        # strike of the same key, so it is safely discarded.
        if key in protected_keys:
            removed += len(duplicates)
            continue
        if len(duplicates) == 1:
            retained.append(duplicates[0])
            continue
        winner = max(
            duplicates,
            key=lambda note: (
                -ROLE_PRIORITY.get(
                    str(note.get("arrangementRole") or "harmony"), 9
                ),
                float(note.get("leftHandRankingScore", 0.0)),
                float(note.get("selectionProbability", 0.0)),
                float(note.get("duration", 0.0)),
                float(note.get("velocity", 0.0)),
            ),
        )
        merged = dict(winner)
        merged["duration"] = round_number(
            max(float(note.get("duration", 0.0)) for note in duplicates)
        )
        merged["scoreDuration"] = merged["duration"]
        merged["visualDuration"] = merged["duration"]
        merged["velocity"] = round_number(
            max(float(note.get("velocity", 0.0)) for note in duplicates)
        )
        if any("audioDuration" in note for note in duplicates):
            merged["audioDuration"] = round_number(
                max(float(note.get("audioDuration", 0.0)) for note in duplicates)
            )
        if any("releaseSeconds" in note for note in duplicates):
            merged["releaseSeconds"] = round_number(
                max(float(note.get("releaseSeconds", 0.0)) for note in duplicates)
            )
        merged["collapsedSourceInstruments"] = sorted(
            {
                str(note.get("sourceInstrument") or note.get("instrument") or "unknown")
                for note in duplicates
            }
        )
        merged["collapsedSimultaneousSourceStrikes"] = len(duplicates)
        retained.append(merged)
        removed += len(duplicates) - 1
    return (
        sorted(
            protected + retained,
            key=lambda note: (float(note["time"]), int(note["midi"])),
        ),
        removed,
    )


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = int(round((len(ordered) - 1) * clamp(fraction, 0.0, 1.0)))
    return ordered[position]


def average(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _estimate_score_subdivision_seconds(notes: list[dict[str, Any]]) -> float:
    """Estimate the shortest stable rhythmic pulse without changing tempo."""

    onsets: list[float] = []
    for note in sorted(notes, key=lambda item: float(item.get("time", 0.0))):
        onset = float(note.get("time", 0.0))
        if not onsets or onset - onsets[-1] > 0.035:
            onsets.append(onset)
    gaps = [
        following - previous
        for previous, following in zip(onsets, onsets[1:])
        if 0.065 <= following - previous <= 0.55
    ]
    if not gaps:
        return 0.20
    # The lower half contains the recurring eighth/sixteenth-note pulse while
    # excluding longer phrase gaps. A median is robust to separator jitter.
    cutoff = quantile(gaps, 0.50)
    short_gaps = [gap for gap in gaps if gap <= cutoff + 0.012]
    return clamp(quantile(short_gaps or gaps, 0.50), 0.07, 0.35)


def _duration_units_at_percentile(
    histogram: list[dict[str, Any]], percentile: float
) -> float:
    entries: list[tuple[float, float]] = []
    for item in histogram:
        if not isinstance(item, dict):
            continue
        units = clamp(float(item.get("units", 1.0)), 0.25, 32.0)
        share = max(0.0, float(item.get("share", 0.0)))
        if share > 0:
            entries.append((units, share))
    if not entries:
        return 1.0
    entries.sort()
    total = sum(share for _, share in entries)
    target = clamp(percentile, 0.0, 1.0) * total
    cumulative = 0.0
    for units, share in entries:
        cumulative += share
        if cumulative >= target:
            return units
    return entries[-1][0]


def apply_authored_duration_style(
    notes: list[dict[str, Any]],
    style: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transfer an authored score's hold vocabulary at the source tempo.

    The reference may be faster or shorter than the uploaded video. We learn
    durations in rhythmic units, estimate the uploaded song's own subdivision,
    and transfer only the distribution of short notes and long holds.
    """

    config = style.get("durationGrid") or {}
    if not config.get("enabled") or not notes:
        return notes, {"applied": False}
    subdivision = _estimate_score_subdivision_seconds(notes)
    release_ratio = clamp(float(config.get("writtenReleaseRatio", 0.95)), 0.70, 1.10)
    maximum_units = clamp(float(config.get("maximumUnits", 12.0)), 1.0, 32.0)
    output = [dict(note) for note in notes]
    hand_diagnostics: dict[str, Any] = {}

    for hand, histogram_name in (("left", "lowerHistogram"), ("right", "upperHistogram")):
        indices = [
            index
            for index, note in enumerate(output)
            if str(note.get("hand") or "right") == hand
        ]
        histogram = config.get(histogram_name) or []
        ranked = sorted(
            indices,
            key=lambda index: (
                float(output[index].get("duration", 0.0)),
                float(output[index].get("selectionProbability", 0.0)),
                float(output[index].get("time", 0.0)),
                int(output[index].get("midi", 60)),
                int(output[index].get("sourceIndex", index)),
            ),
        )
        assigned_units: list[float] = []
        for rank, index in enumerate(ranked):
            percentile = (rank + 0.5) / max(1, len(ranked))
            units = min(
                maximum_units,
                _duration_units_at_percentile(histogram, percentile),
            )
            duration = clamp(
                units * subdivision * release_ratio,
                MIN_NOTE_SECONDS,
                MAX_NOTE_SECONDS,
            )
            output[index]["duration"] = round_number(duration, 6)
            output[index]["authoredDurationUnits"] = round_number(units, 3)
            assigned_units.append(units)
        hand_diagnostics[hand] = {
            "notes": len(ranked),
            "medianUnits": round_number(quantile(assigned_units, 0.50), 3)
            if assigned_units
            else 0.0,
            "p90Units": round_number(quantile(assigned_units, 0.90), 3)
            if assigned_units
            else 0.0,
        }

    return sorted(output, key=lambda note: (note["time"], note["midi"])), {
        "applied": True,
        "profile": "tempo-relative-authored-duration-distribution-v1",
        "estimatedSubdivisionSeconds": round_number(subdivision, 6),
        "writtenReleaseRatio": round_number(release_ratio, 4),
        "maximumUnits": round_number(maximum_units, 3),
        "hands": hand_diagnostics,
    }


def stabilize_direct_piano_score_holds(
    notes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repair implausibly short written holds on a preserved piano source.

    MuScriptor can locate a clean piano attack accurately while ending its
    event a frame or two too early.  A pianist still depresses the key long
    enough for the hammer/key mechanism to speak; low notes need a tiny extra
    settling interval.  This rule is deliberately restricted to notes already
    routed as ``source_piano`` and never changes pitch, onset, velocity, or the
    number of attacks.  The physical-performance pass still enforces a release
    gap before a same-key restrike.
    """

    output: list[dict[str, Any]] = []
    minimum_hold_applied = 0
    short_bass_extension_applied = 0
    calibrated_short_holds = 0
    total_extension = 0.0
    maximum_extension = 0.0
    for source in notes:
        note = dict(source)
        if str(note.get("arrangementRole") or "") != "source_piano":
            output.append(note)
            continue
        original = max(MIN_NOTE_SECONDS, float(note.get("duration", 0.2)))
        stabilized = max(DIRECT_PIANO_MINIMUM_SCORE_HOLD_SECONDS, original)
        if stabilized > original + 1e-9:
            minimum_hold_applied += 1
        if (
            int(note.get("midi", 60)) <= DIRECT_PIANO_SHORT_BASS_MAXIMUM_MIDI
            and original <= DIRECT_PIANO_SHORT_BASS_THRESHOLD_SECONDS
        ):
            stabilized += DIRECT_PIANO_SHORT_BASS_EXTENSION_SECONDS
            short_bass_extension_applied += 1
        # Clean-piano inference commonly returns quantized 50--250 ms events
        # whose attacks are reliable but whose key-up estimates are early.
        # Calibrate only that short-event regime.  Longer written holds are
        # already informative and must not be stretched.  The later physical
        # pianist pass remains responsible for opening a safe same-key
        # restrike gap, so this cannot merge two genuine repeated attacks.
        if stabilized <= DIRECT_PIANO_SHORT_HOLD_SCALE_MAX_SECONDS + 1e-9:
            stabilized *= DIRECT_PIANO_SHORT_HOLD_SCALE
            calibrated_short_holds += 1
        extension = max(0.0, stabilized - original)
        total_extension += extension
        maximum_extension = max(maximum_extension, extension)
        note["duration"] = round_number(stabilized, 6)
        output.append(note)
    return output, {
        "applied": bool(
            minimum_hold_applied
            or short_bass_extension_applied
            or calibrated_short_holds
        ),
        "profile": "direct-piano-short-hold-stabilizer-v2",
        "minimumScoreHoldSeconds": DIRECT_PIANO_MINIMUM_SCORE_HOLD_SECONDS,
        "shortBassMaximumMidi": DIRECT_PIANO_SHORT_BASS_MAXIMUM_MIDI,
        "shortBassThresholdSeconds": DIRECT_PIANO_SHORT_BASS_THRESHOLD_SECONDS,
        "shortBassExtensionSeconds": DIRECT_PIANO_SHORT_BASS_EXTENSION_SECONDS,
        "shortHoldScale": DIRECT_PIANO_SHORT_HOLD_SCALE,
        "shortHoldScaleMaximumSeconds": DIRECT_PIANO_SHORT_HOLD_SCALE_MAX_SECONDS,
        "minimumHoldAdjustedNotes": minimum_hold_applied,
        "shortBassAdjustedNotes": short_bass_extension_applied,
        "calibratedShortHoldNotes": calibrated_short_holds,
        "meanExtensionSeconds": round_number(
            total_extension / max(1, len(output)), 6
        ),
        "maximumExtensionSeconds": round_number(maximum_extension, 6),
    }


def route_conditional_learned_profile(
    style_profile: dict[str, Any],
    factual_source_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route a narrow source failure mode to a learned selector.

    A broad transcriber can occasionally collapse a full mix into almost only
    bass events.  A learned selector helps that domain, but applying it to a
    source with healthy voice evidence regresses timing and cutoffs.  The gate
    therefore uses only factual, pre-arrangement source ratios and otherwise
    falls back to a frozen default-pipeline decoder embedded in the profile.
    """

    decoder = style_profile.get("decoder") or {}
    policy = decoder.get("conditionalLearnedRoute") or {}
    if not policy.get("enabled"):
        return style_profile, {}

    maximum_voice_ratio = clamp(
        float(policy.get("maximumVoiceRatio", 0.02)), 0.0, 1.0
    )
    minimum_bass_ratio = clamp(
        float(policy.get("minimumBassRatio", 0.50)), 0.0, 1.0
    )
    maximum_piano_ratio = clamp(
        float(policy.get("maximumPianoRatio", 0.08)), 0.0, 1.0
    )
    minimum_source_notes = max(8, int(policy.get("minimumSourceNotes", 64)))
    instrument_counts = factual_source_profile.get("instrumentCounts") or {}
    source_notes = sum(max(0, int(value)) for value in instrument_counts.values())
    voice_ratio = float(factual_source_profile.get("voiceRatio", 0.0))
    piano_ratio = float(factual_source_profile.get("pianoRatio", 0.0))
    bass_ratio = float(factual_source_profile.get("bassRatio", 0.0))
    if "bassRatio" not in factual_source_profile and source_notes:
        bass_notes = sum(
            max(0, int(instrument_counts.get(name, 0))) for name in BASS_INSTRUMENTS
        )
        bass_ratio = bass_notes / source_notes

    monophonic_policy = policy.get("monophonicVocalRoute") or {}
    monophonic_checks: dict[str, bool] = {}
    if monophonic_policy.get("enabled"):
        minimum_monophonic_voice_ratio = clamp(
            float(monophonic_policy.get("minimumVoiceRatio", 0.95)), 0.0, 1.0
        )
        maximum_monophonic_bass_ratio = clamp(
            float(monophonic_policy.get("maximumBassRatio", 0.02)), 0.0, 1.0
        )
        maximum_monophonic_piano_ratio = clamp(
            float(monophonic_policy.get("maximumPianoRatio", 0.02)), 0.0, 1.0
        )
        minimum_monophonic_source_notes = max(
            8, int(monophonic_policy.get("minimumSourceNotes", 32))
        )
        monophonic_checks = {
            "voiceRatioMeetsMinimum": voice_ratio >= minimum_monophonic_voice_ratio,
            "bassRatioWithinMaximum": bass_ratio <= maximum_monophonic_bass_ratio,
            "pianoRatioWithinMaximum": piano_ratio <= maximum_monophonic_piano_ratio,
            "sourceNoteCountMeetsMinimum": (
                source_notes >= minimum_monophonic_source_notes
            ),
        }
        if all(monophonic_checks.values()):
            fallback_decoder = monophonic_policy.get("fallbackDecoder")
            if not isinstance(fallback_decoder, dict) or not fallback_decoder.get(
                "defaultPipeline"
            ):
                raise ValueError(
                    "conditionalLearnedRoute.monophonicVocalRoute requires "
                    "fallbackDecoder.defaultPipeline=true"
                )
            fallback = {
                "schema": "polymath-piano-arranger-profile-v1",
                "id": str(
                    monophonic_policy.get("fallbackProfileId")
                    or f"{style_profile.get('id', 'conditional')}-monophonic-vocal"
                ),
                "decoder": copy.deepcopy(fallback_decoder),
            }
            return fallback, {
                "profile": "source-factual-conditional-learned-route-v2",
                "enabled": True,
                "selectedRoute": "monophonic-vocal-default",
                "sourceVoiceRatio": round_number(voice_ratio, 4),
                "sourceBassRatio": round_number(bass_ratio, 4),
                "sourcePianoRatio": round_number(piano_ratio, 4),
                "sourceNoteCount": source_notes,
                "monophonicChecks": monophonic_checks,
                "fallbackProfileId": fallback["id"],
            }

    checks = {
        "voiceRatioWithinMaximum": voice_ratio <= maximum_voice_ratio,
        "bassRatioMeetsMinimum": bass_ratio >= minimum_bass_ratio,
        "pianoRatioWithinMaximum": piano_ratio <= maximum_piano_ratio,
        "sourceNoteCountMeetsMinimum": source_notes >= minimum_source_notes,
    }
    learned_route = all(checks.values())
    diagnostics = {
        "profile": "source-factual-conditional-learned-route-v2",
        "enabled": True,
        "selectedRoute": (
            "learned-selector" if learned_route else "default-pipeline-fallback"
        ),
        "sourceVoiceRatio": round_number(voice_ratio, 4),
        "sourceBassRatio": round_number(bass_ratio, 4),
        "sourcePianoRatio": round_number(piano_ratio, 4),
        "sourceNoteCount": source_notes,
        "maximumVoiceRatio": round_number(maximum_voice_ratio, 4),
        "minimumBassRatio": round_number(minimum_bass_ratio, 4),
        "maximumPianoRatio": round_number(maximum_piano_ratio, 4),
        "minimumSourceNotes": minimum_source_notes,
        "checks": checks,
    }
    if monophonic_checks:
        diagnostics["monophonicChecks"] = monophonic_checks
    if learned_route:
        return style_profile, diagnostics

    fallback_decoder = policy.get("fallbackDecoder")
    if not isinstance(fallback_decoder, dict) or not fallback_decoder.get(
        "defaultPipeline"
    ):
        raise ValueError(
            "conditionalLearnedRoute requires fallbackDecoder.defaultPipeline=true"
        )
    fallback = {
        "schema": "polymath-piano-arranger-profile-v1",
        "id": str(
            policy.get("fallbackProfileId")
            or f"{style_profile.get('id', 'conditional')}-default-fallback"
        ),
        "decoder": copy.deepcopy(fallback_decoder),
    }
    diagnostics["fallbackProfileId"] = fallback["id"]
    return fallback, diagnostics


def suppress_monophonic_vocal_floor_runs(
    notes: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove long floor-energy vocal hallucinations and calibrate onset latency.

    A real held-out vocal recording exposed a characteristic failure: during a
    vocal gap, the transcriber emitted a long metronomic phrase whose dynamics
    were clamped to the postprocessor's noise floor.  Quiet real syllables also
    touch that floor, so individual quiet notes must never be removed.  The
    suppressor acts only on a consecutive run, and only on the separately
    gated monophonic-vocal route.
    """

    policy = config or {}
    if not policy.get("enabled") or not notes:
        return notes, {
            "applied": False,
            "profile": "monophonic-vocal-floor-run-suppression-v1",
            "reason": "disabled" if not policy.get("enabled") else "no-notes",
        }

    maximum_floor_velocity = clamp(
        float(policy.get("maximumFloorVelocity", 0.46)), 0.05, 1.0
    )
    minimum_run_notes = max(3, int(policy.get("minimumRunNotes", 6)))
    maximum_run_gap_seconds = clamp(
        float(policy.get("maximumRunGapSeconds", 0.35)), 0.08, 1.0
    )
    onset_delay_seconds = clamp(
        float(policy.get("onsetDelaySeconds", 0.02)), -0.08, 0.08
    )
    ordered = sorted(
        (dict(note) for note in notes),
        key=lambda note: (float(note["time"]), int(note["midi"])),
    )
    floor_runs: list[list[int]] = []
    current: list[int] = []
    for index, note in enumerate(ordered):
        is_voice = str(note.get("instrument") or "").lower() in VOICE_INSTRUMENTS
        is_floor = is_voice and float(note.get("velocity", 0.72)) <= maximum_floor_velocity
        contiguous = bool(
            current
            and float(note["time"]) - float(ordered[current[-1]]["time"])
            <= maximum_run_gap_seconds
        )
        if is_floor and (not current or contiguous):
            current.append(index)
            continue
        if len(current) >= minimum_run_notes:
            floor_runs.append(current)
        current = [index] if is_floor else []
    if len(current) >= minimum_run_notes:
        floor_runs.append(current)

    removed_indices = {index for run in floor_runs for index in run}
    output: list[dict[str, Any]] = []
    for index, note in enumerate(ordered):
        if index in removed_indices:
            continue
        shifted = dict(note)
        shifted["time"] = round_number(
            max(0.0, float(shifted["time"]) + onset_delay_seconds), 6
        )
        if onset_delay_seconds:
            shifted["monophonicVocalOnsetDelaySeconds"] = round_number(
                onset_delay_seconds, 6
            )
        output.append(shifted)

    return output, {
        "applied": bool(removed_indices or onset_delay_seconds),
        "profile": "monophonic-vocal-floor-run-suppression-v1",
        "maximumFloorVelocity": round_number(maximum_floor_velocity, 4),
        "minimumRunNotes": minimum_run_notes,
        "maximumRunGapSeconds": round_number(maximum_run_gap_seconds, 4),
        "onsetDelaySeconds": round_number(onset_delay_seconds, 6),
        "detectedRuns": len(floor_runs),
        "removedNotes": len(removed_indices),
        "retainedNotes": len(output),
    }


def expression_role(note: dict[str, Any]) -> str:
    role = note.get("arrangementRole")
    if role == "melody":
        return "melody"
    if role == "bass":
        return "bass"
    hand = str(note.get("hand") or "").lower()
    is_left = hand == "left" if hand in {"left", "right"} else note["midi"] < 60
    if is_left:
        return "source_left" if role == "source_piano" else "left_hand"
    return "source_right" if role == "source_piano" else "right_hand"


def melody_forward_balance_config(
    style_profile: dict[str, Any] | None,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    defaults = {
        "melody": MELODY_VELOCITY_RANGE,
        "right_hand": RIGHT_HAND_VELOCITY_RANGE,
        "left_hand": LEFT_HAND_VELOCITY_RANGE,
        "bass": BASS_VELOCITY_RANGE,
        "source_right": SOURCE_RIGHT_VELOCITY_RANGE,
        "source_left": SOURCE_LEFT_VELOCITY_RANGE,
    }
    configured = ((style_profile or {}).get("decoder") or {}).get(
        "melodyForwardBalance"
    ) or {}
    if not isinstance(configured, dict) or not configured.get("enabled"):
        return defaults, {"enabled": False}

    range_fields = {
        "melody": "melodyVelocityRange",
        "right_hand": "rightHandVelocityRange",
        "left_hand": "leftHandVelocityRange",
        "bass": "bassVelocityRange",
        "source_right": "sourceRightVelocityRange",
        "source_left": "sourceLeftVelocityRange",
    }
    ranges = dict(defaults)
    for role, field in range_fields.items():
        value = configured.get(field)
        if not isinstance(value, list) or len(value) != 2:
            continue
        try:
            minimum = clamp(float(value[0]), 0.05, 1.0)
            maximum = clamp(float(value[1]), minimum, 1.0)
        except (TypeError, ValueError):
            continue
        ranges[role] = (minimum, maximum)

    def bounded(field: str, fallback: float, minimum: float, maximum: float) -> float:
        value = safe_number(configured.get(field), fallback)
        return clamp(float(fallback if value is None else value), minimum, maximum)

    return ranges, {
        "enabled": True,
        "melodyGain": bounded("melodyGain", 1.0, 0.8, 1.2),
        "leftGainDuringMelody": bounded(
            "leftGainDuringMelody", 1.0, 0.25, 1.0
        ),
        "rightGainDuringMelody": bounded(
            "rightGainDuringMelody", 1.0, 0.25, 1.0
        ),
        "lookBehindSeconds": bounded("lookBehindSeconds", 0.04, 0.0, 0.30),
        "lookAheadSeconds": bounded("lookAheadSeconds", 0.08, 0.0, 0.30),
    }


def shape_melody_forward_expression(
    notes: list[dict[str, Any]],
    style_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make the sung/top line clear without clipping or flattening dynamics.

    MuScriptor's velocities are useful confidence/evidence, but a separated
    full mix can give nearly every event the same value.  We retain any real
    source contour, mix in a small duration accent, and place each musical
    role into a non-overlapping performance band.  The highest right-hand note
    at an onset receives a subtle top-voice accent; explicit vocal melody gets
    the strongest band automatically.
    """
    if not notes:
        return notes, {
            "profile": "melody-forward-v1",
            "shapedNotes": 0,
            "rightHandMeanVelocity": 0.0,
            "leftHandMeanVelocity": 0.0,
            "rightToLeftVelocityRatio": 0.0,
        }

    shaped = [dict(note) for note in notes]
    source_velocities = [note["velocity"] for note in shaped]
    quiet = quantile(source_velocities, 0.10)
    loud = quantile(source_velocities, 0.90)
    useful_source_range = loud - quiet >= 0.06

    top_voice_ids: set[int] = set()
    for group in grouped_by_window(shaped, EXPRESSION_ONSET_WINDOW_SECONDS):
        right_hand = [note for note in group if note["midi"] >= 60]
        if right_hand:
            top = max(
                right_hand,
                key=lambda note: (
                    note.get("arrangementRole") == "melody",
                    note["midi"],
                    note["duration"],
                ),
            )
            top_voice_ids.add(id(top))

    velocity_ranges, balance = melody_forward_balance_config(style_profile)
    melody_spans: list[tuple[float, float]] = []
    if balance["enabled"]:
        for note in shaped:
            if expression_role(note) != "melody":
                continue
            melody_spans.append(
                (
                    float(note["time"]) - balance["lookBehindSeconds"],
                    float(note["time"])
                    + float(note["duration"])
                    + balance["lookAheadSeconds"],
                )
            )
        melody_spans.sort()

    ducked_left = 0
    ducked_right = 0
    role_velocities: dict[str, list[float]] = defaultdict(list)
    original_velocities: list[float] = []

    for note in shaped:
        original = note["velocity"]
        original_velocities.append(original)
        if useful_source_range:
            source_contour = clamp((original - quiet) / (loud - quiet), 0.0, 1.0)
        else:
            source_contour = 0.50

        # Longer notes normally carry more lyrical weight.  This is a small
        # accent only; it cannot override the melody/accompaniment hierarchy.
        duration_accent = clamp(
            (math.sqrt(note["duration"]) - math.sqrt(MIN_NOTE_SECONDS))
            / (math.sqrt(1.6) - math.sqrt(MIN_NOTE_SECONDS)),
            0.0,
            1.0,
        )
        expression = 0.72 * source_contour + 0.28 * duration_accent
        role = expression_role(note)
        minimum, maximum = velocity_ranges[role]
        velocity = minimum + (maximum - minimum) * expression

        # Bring out a chord's upper voice without treating every high harmony
        # tone as if it were the singer.
        if id(note) in top_voice_ids and role in {"right_hand", "source_right"}:
            velocity += 0.035

        if balance["enabled"] and role == "melody":
            velocity *= balance["melodyGain"]
        elif balance["enabled"] and any(
            start <= float(note["time"]) + float(note["duration"])
            and end >= float(note["time"])
            for start, end in melody_spans
        ):
            if role in {"left_hand", "source_left", "bass"}:
                velocity *= balance["leftGainDuringMelody"]
                ducked_left += 1
            else:
                velocity *= balance["rightGainDuringMelody"]
                ducked_right += 1

        note["velocity"] = round_number(clamp(velocity, 0.05, 1.0), 3)
        if str(note.get("hand") or "").lower() not in {"left", "right"}:
            note["hand"] = "left" if note["midi"] < 60 else "right"
        role_velocities[role].append(note["velocity"])

    right_velocities = [
        note["velocity"] for note in shaped if note.get("hand") != "left"
    ]
    left_velocities = [
        note["velocity"] for note in shaped if note.get("hand") == "left"
    ]
    right_mean = average(right_velocities)
    left_mean = average(left_velocities)
    return shaped, {
        "profile": "melody-forward-v1",
        "shapedNotes": len(shaped),
        "sourceVelocityRangeWasUsable": useful_source_range,
        "sourceVelocityP10": round_number(quiet, 3),
        "sourceVelocityP90": round_number(loud, 3),
        "inputMeanVelocity": round_number(average(original_velocities), 3),
        "rightHandMeanVelocity": round_number(right_mean, 3),
        "leftHandMeanVelocity": round_number(left_mean, 3),
        "rightToLeftVelocityRatio": round_number(
            right_mean / left_mean if left_mean else 0.0,
            3,
        ),
        "roleMeanVelocities": {
            role: round_number(average(values), 3)
            for role, values in sorted(role_velocities.items())
        },
        "velocityBands": {
            role: [minimum, maximum]
            for role, (minimum, maximum) in velocity_ranges.items()
        },
        "melodyBalance": {
            **balance,
            "duckedLeftNotes": ducked_left,
            "duckedRightNotes": ducked_right,
        },
    }


def shape_gesture_coherent_expression(
    notes: list[dict[str, Any]],
    style_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn separately arranged stems into one pianist's physical gestures.

    The transcription service does not expose MIDI strike velocity. The web
    server reconstructs a useful loudness contour from the source WAV, while
    the learned arranger historically replaced it with separate role bands.
    Notes struck at one instant could therefore use wildly different sample
    timbres -- effectively several players colliding on one piano.

    Profiles must explicitly enable this pass. Every near-simultaneous onset
    then receives one hammer velocity. Melody focus becomes a post-sample
    ``performanceGain`` so balance can change without changing hammer timbre.
    """

    config = ((style_profile or {}).get("decoder") or {}).get(
        "gestureDynamics"
    ) or {}
    if not config.get("enabled") or not notes:
        return notes, {
            "applied": False,
            "profile": "gesture-coherent-piano-dynamics-v1",
            "reason": "profile-disabled" if notes else "no-notes",
        }

    onset_window = clamp(
        float(
            config.get(
                "onsetWindowSeconds", GESTURE_DYNAMIC_ONSET_WINDOW_SECONDS
            )
        ),
        0.005,
        0.080,
    )
    minimum_velocity = clamp(float(config.get("minimumVelocity", 0.38)), 0.05, 0.95)
    maximum_velocity = clamp(
        float(config.get("maximumVelocity", 0.94)), minimum_velocity, 1.0
    )
    minimum_source_range = clamp(
        float(config.get("minimumUsableSourceRange", 0.06)), 0.0, 0.50
    )
    source_blend = clamp(float(config.get("sourceBlend", 0.82)), 0.0, 1.0)
    preserve_input_gesture_velocity = bool(
        config.get("preserveInputGestureVelocity", False)
    )
    source_gamma = clamp(float(config.get("sourceGamma", 1.0)), 0.35, 2.5)
    duration_accent_strength = clamp(
        float(config.get("durationAccentStrength", 0.04)), 0.0, 0.20
    )
    melody_performance_gain = clamp(
        float(config.get("melodyPerformanceGain", 1.08)), 0.5, 1.5
    )
    accompaniment_gain_during_melody = clamp(
        float(config.get("accompanimentGainDuringMelody", 0.94)), 0.25, 1.25
    )
    neutral_performance_gain = clamp(
        float(config.get("neutralPerformanceGain", 1.0)), 0.5, 1.25
    )
    duck_overlapping_accompaniment = bool(
        config.get("duckOverlappingAccompaniment", False)
    )
    performance_gain_look_behind = clamp(
        float(config.get("performanceGainLookBehindSeconds", 0.04)),
        0.0,
        0.30,
    )
    performance_gain_look_ahead = clamp(
        float(config.get("performanceGainLookAheadSeconds", 0.08)),
        0.0,
        0.30,
    )
    size_defaults = {
        1: 0.68,
        2: 0.67,
        3: 0.71,
        4: 0.88,
        5: 0.90,
        6: 0.86,
    }
    configured_sizes = config.get("velocityByChordSize") or {}
    size_velocity = {
        size: clamp(
            float(configured_sizes.get(str(size), fallback)),
            minimum_velocity,
            maximum_velocity,
        )
        for size, fallback in size_defaults.items()
    }

    shaped = [dict(note) for note in notes]
    shaped.sort(key=lambda note: (float(note["time"]), int(note["midi"])))
    groups: list[list[dict[str, Any]]] = []
    for note in shaped:
        if (
            not groups
            or float(note["time"]) - float(groups[-1][0]["time"])
            > onset_window
        ):
            groups.append([note])
        else:
            groups[-1].append(note)

    def factual_velocity(note: dict[str, Any]) -> float:
        value = safe_number(
            note.get("sourceVelocityBeforeArrangement"),
            safe_number(note.get("velocity"), 0.72),
        )
        return clamp(float(value if value is not None else 0.72), 0.05, 1.0)

    factual_values = [factual_velocity(note) for note in shaped]
    source_quiet = quantile(factual_values, 0.10)
    source_loud = quantile(factual_values, 0.90)
    useful_source_range = source_loud - source_quiet >= minimum_source_range
    group_velocities: list[float] = []
    original_spreads: list[float] = []
    melody_groups = 0
    multi_note_groups = 0
    overlapping_accompaniment_notes = 0

    melody_spans = sorted(
        (
            float(note["time"]) - performance_gain_look_behind,
            float(note["time"])
            + float(note.get("duration", MIN_NOTE_SECONDS))
            + performance_gain_look_ahead,
        )
        for note in shaped
        if expression_role(note) == "melody"
    )

    def overlaps_active_melody(note: dict[str, Any]) -> bool:
        if not duck_overlapping_accompaniment or not melody_spans:
            return False
        start = float(note["time"])
        end = start + float(note.get("duration", MIN_NOTE_SECONDS))
        return any(span_start <= end and span_end >= start for span_start, span_end in melody_spans)

    for group_index, group in enumerate(groups):
        original = [float(note.get("velocity", 0.72)) for note in group]
        original_spreads.append(max(original) - min(original))
        chord_size = min(6, max(1, len({int(note["midi"]) for note in group})))
        if preserve_input_gesture_velocity:
            # A performance-only profile is trained against the already
            # accepted default arrangement. Preserve that arrangement's median
            # hammer strength as the model baseline instead of rebuilding it
            # from source stems and accidentally changing two variables at once.
            gesture_velocity = median(original)
        else:
            structural_velocity = size_velocity[chord_size]
            median_duration = quantile(
                [float(note.get("duration", MIN_NOTE_SECONDS)) for note in group],
                0.50,
            )
            # Longer written gestures receive only a subtle accent. Chord size and
            # the source-audio contour remain the dominant signals.
            duration_expression = clamp(
                (math.sqrt(max(MIN_NOTE_SECONDS, median_duration)) - math.sqrt(0.12))
                / (math.sqrt(0.85) - math.sqrt(0.12)),
                0.0,
                1.0,
            )
            structural_velocity = clamp(
                structural_velocity
                + duration_accent_strength * (duration_expression - 0.45),
                minimum_velocity,
                maximum_velocity,
            )

            factual_group_velocity = quantile(
                [factual_velocity(note) for note in group], 0.50
            )
            if useful_source_range:
                normalized = clamp(
                    (factual_group_velocity - source_quiet)
                    / max(1e-9, source_loud - source_quiet),
                    0.0,
                    1.0,
                )
                source_velocity = minimum_velocity + (
                    maximum_velocity - minimum_velocity
                ) * (normalized**source_gamma)
                gesture_velocity = (
                    source_blend * source_velocity
                    + (1.0 - source_blend) * structural_velocity
                )
            else:
                # A deterministic, slow phrase contour avoids a flat mechanical
                # fallback without inventing random accents inside a chord.
                phrase_contour = 0.025 * math.sin(group_index * 0.31)
                gesture_velocity = structural_velocity + phrase_contour
        gesture_velocity = round_number(
            clamp(gesture_velocity, minimum_velocity, maximum_velocity), 3
        )
        group_velocities.append(gesture_velocity)

        has_melody = any(expression_role(note) == "melody" for note in group)
        overlapping_melody = has_melody or any(
            overlaps_active_melody(note)
            for note in group
            if expression_role(note) != "melody"
        )
        if has_melody:
            melody_groups += 1
        if len(group) > 1:
            multi_note_groups += 1
        for note in group:
            note["velocityBeforeGestureCoherence"] = round_number(
                float(note.get("velocity", 0.72)), 4
            )
            note["velocity"] = gesture_velocity
            role = expression_role(note)
            note["performanceGain"] = round_number(
                melody_performance_gain
                if role == "melody"
                else accompaniment_gain_during_melody
                if overlapping_melody
                else neutral_performance_gain,
                4,
            )
            if role != "melody" and overlapping_melody:
                overlapping_accompaniment_notes += 1
            note["gestureDynamicGroup"] = group_index

    configured_quantile_map = config.get("velocityQuantileMap") or {}
    quantile_calibration_blend = clamp(
        float(config.get("quantileCalibrationBlend", 1.0)), 0.0, 1.0
    )
    quantile_points: list[tuple[float, float]] = []
    if isinstance(configured_quantile_map, dict):
        for raw_fraction, raw_velocity in configured_quantile_map.items():
            try:
                fraction = clamp(float(raw_fraction), 0.0, 1.0)
                velocity = clamp(
                    float(raw_velocity), minimum_velocity, maximum_velocity
                )
            except (TypeError, ValueError):
                continue
            quantile_points.append((fraction, velocity))
    quantile_points = sorted(dict(quantile_points).items())
    quantile_calibration_changes: list[float] = []
    if len(quantile_points) >= 2 and group_velocities:
        def mapped_quantile(fraction: float) -> float:
            if fraction <= quantile_points[0][0]:
                return quantile_points[0][1]
            if fraction >= quantile_points[-1][0]:
                return quantile_points[-1][1]
            for (left_fraction, left_velocity), (
                right_fraction,
                right_velocity,
            ) in zip(quantile_points, quantile_points[1:]):
                if left_fraction <= fraction <= right_fraction:
                    ratio = (fraction - left_fraction) / max(
                        1e-9, right_fraction - left_fraction
                    )
                    return left_velocity + ratio * (
                        right_velocity - left_velocity
                    )
            return quantile_points[-1][1]

        by_velocity: dict[float, list[int]] = defaultdict(list)
        for index, velocity in enumerate(group_velocities):
            by_velocity[velocity].append(index)
        calibrated = list(group_velocities)
        rank_start = 0
        denominator = max(1, len(group_velocities) - 1)
        for velocity, indices in sorted(by_velocity.items()):
            # Tied gestures receive their shared mid-rank. This prevents an
            # entirely flat source from being mistaken for the loudest event.
            rank_end = rank_start + len(indices) - 1
            percentile = ((rank_start + rank_end) / 2.0) / denominator
            target_velocity = mapped_quantile(percentile)
            value = round_number(
                clamp(
                    velocity * (1.0 - quantile_calibration_blend)
                    + target_velocity * quantile_calibration_blend,
                    minimum_velocity,
                    maximum_velocity,
                ),
                3,
            )
            for index in indices:
                calibrated[index] = value
                quantile_calibration_changes.append(value - velocity)
            rank_start = rank_end + 1
        group_velocities = calibrated
        for group_index, group in enumerate(groups):
            for note in group:
                note["velocity"] = group_velocities[group_index]

    pre_paired_velocities = list(group_velocities)
    paired_config = config.get("pairedCalibration") or {}
    paired_changes: list[float] = []
    if paired_config.get("enabled") and group_velocities:
        calibrated_values: list[float] = []
        velocity_contexts = gesture_sequence_contexts(groups, group_velocities)
        for group, velocity, sequence_context in zip(
            groups, group_velocities, velocity_contexts
        ):
            calibrated_velocity, change = calibrated_gesture_velocity(
                group,
                velocity,
                paired_config,
                minimum_velocity=minimum_velocity,
                maximum_velocity=maximum_velocity,
                context=sequence_context,
            )
            value = round_number(calibrated_velocity, 3)
            calibrated_values.append(value)
            paired_changes.append(value - velocity)
        group_velocities = calibrated_values
        for group_index, group in enumerate(groups):
            for note in group:
                note["velocity"] = group_velocities[group_index]

    paired_duration_config = config.get("pairedDurationCalibration") or {}
    paired_duration_changes: list[float] = []
    paired_duration_ratios: list[float] = []
    if paired_duration_config.get("enabled") and groups:
        duration_contexts = gesture_sequence_contexts(groups, group_velocities)
        minimum_duration = clamp(
            float(paired_duration_config.get("minimumDurationSeconds", 0.05)),
            MIN_NOTE_SECONDS,
            1.0,
        )
        maximum_duration = clamp(
            float(paired_duration_config.get("maximumDurationSeconds", 4.0)),
            minimum_duration,
            MAX_NOTE_SECONDS,
        )
        for group, sequence_context in zip(groups, duration_contexts):
            base_duration = quantile(
                [
                    float(note.get("scoreDuration", note.get("duration", 0.2)))
                    for note in group
                ],
                0.50,
            )
            calibrated_duration, change = calibrated_gesture_duration(
                group,
                base_duration,
                paired_duration_config,
                minimum_duration=minimum_duration,
                maximum_duration=maximum_duration,
                context=sequence_context,
            )
            ratio = calibrated_duration / max(MIN_NOTE_SECONDS, base_duration)
            paired_duration_changes.append(change)
            paired_duration_ratios.append(ratio)
            for note in group:
                original_duration = float(
                    note.get("scoreDuration", note.get("duration", 0.2))
                )
                value = round_number(
                    clamp(
                        original_duration * ratio,
                        minimum_duration,
                        maximum_duration,
                    ),
                    6,
                )
                note["durationBeforeGestureCalibration"] = round_number(
                    original_duration, 6
                )
                note["duration"] = value
                note["scoreDuration"] = value
                note["visualDuration"] = value

    return shaped, {
        "applied": True,
        "profile": "gesture-coherent-piano-dynamics-v1",
        "groups": len(groups),
        "multiNoteGroups": multi_note_groups,
        "melodyGroups": melody_groups,
        "duckOverlappingAccompaniment": duck_overlapping_accompaniment,
        "overlappingAccompanimentNotes": overlapping_accompaniment_notes,
        "performanceGainLookBehindSeconds": round_number(
            performance_gain_look_behind, 4
        ),
        "performanceGainLookAheadSeconds": round_number(
            performance_gain_look_ahead, 4
        ),
        "onsetWindowSeconds": round_number(onset_window, 4),
        "sourceVelocityRangeWasUsable": useful_source_range,
        "preservedInputGestureVelocity": preserve_input_gesture_velocity,
        "sourceVelocityP10": round_number(source_quiet, 4),
        "sourceVelocityP90": round_number(source_loud, 4),
        "sourceBlend": round_number(source_blend, 4),
        "outputVelocityP10": round_number(quantile(group_velocities, 0.10), 4),
        "outputVelocityMedian": round_number(quantile(group_velocities, 0.50), 4),
        "outputVelocityP90": round_number(quantile(group_velocities, 0.90), 4),
        "meanOriginalWithinGestureSpread": round_number(
            average(original_spreads), 4
        ),
        "maximumOutputWithinGestureSpread": 0.0,
        "velocityQuantileCalibrationApplied": len(quantile_points) >= 2,
        "velocityQuantileMap": {
            str(fraction): round_number(velocity, 4)
            for fraction, velocity in quantile_points
        },
        "quantileCalibrationBlend": round_number(
            quantile_calibration_blend, 4
        ),
        "meanAbsoluteQuantileCalibrationChange": round_number(
            average([abs(value) for value in quantile_calibration_changes]),
            4,
        ),
        "pairedCalibrationApplied": bool(
            paired_config.get("enabled") and paired_changes
        ),
        "pairedCalibrationProfile": paired_config.get("id"),
        "prePairedVelocityP10": round_number(
            quantile(pre_paired_velocities, 0.10), 4
        ),
        "prePairedVelocityMedian": round_number(
            quantile(pre_paired_velocities, 0.50), 4
        ),
        "prePairedVelocityP90": round_number(
            quantile(pre_paired_velocities, 0.90), 4
        ),
        "meanAbsolutePairedCalibrationChange": round_number(
            average([abs(value) for value in paired_changes]), 4
        ),
        "maximumAbsolutePairedCalibrationChange": round_number(
            max((abs(value) for value in paired_changes), default=0.0), 4
        ),
        "pairedDurationCalibrationApplied": bool(
            paired_duration_config.get("enabled") and paired_duration_changes
        ),
        "pairedDurationCalibrationProfile": paired_duration_config.get("id"),
        "meanAbsolutePairedDurationChangeSeconds": round_number(
            average([abs(value) for value in paired_duration_changes]), 4
        ),
        "maximumAbsolutePairedDurationChangeSeconds": round_number(
            max(
                (abs(value) for value in paired_duration_changes),
                default=0.0,
            ),
            4,
        ),
        "medianPairedDurationScale": round_number(
            quantile(paired_duration_ratios, 0.50), 4
        )
        if paired_duration_ratios
        else 1.0,
        "melodyPerformanceGain": round_number(melody_performance_gain, 4),
        "accompanimentGainDuringMelody": round_number(
            accompaniment_gain_during_melody, 4
        ),
        "velocityByChordSize": {
            str(size): round_number(value, 4)
            for size, value in sorted(size_velocity.items())
        },
    }


def synchronize_generated_visual_holds(payload: dict[str, Any]) -> dict[str, Any]:
    """Make falling bars show the physical key hold of a generated reduction."""
    synchronized = 0
    for note in payload.get("notes", []):
        try:
            physical = float(note.get("audioDuration", note.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        note["visualDuration"] = round_number(max(MIN_NOTE_SECONDS, physical), 6)
        synchronized += 1
    payload.setdefault("performance", {})["visualDurationPolicy"] = "physical-key-hold"
    payload.setdefault("pianoPerformance", {})["visualHoldsSynchronized"] = synchronized
    return payload


def arranged_note(
    source: dict[str, Any],
    *,
    midi: int,
    role: str,
    velocity: float,
    duration_minimum: float,
    duration_maximum: float,
) -> dict[str, Any]:
    value = int(clamp(midi, PIANO_MIN_MIDI, PIANO_MAX_MIDI))
    note = dict(source)
    note.update(
        {
            "sourceVelocityBeforeArrangement": round_number(
                clamp(float(source.get("velocity", velocity)), 0.05, 1.0), 4
            ),
            "midi": value,
            "note": midi_to_note(value),
            "instrument": "acoustic_piano",
            "sourceInstrument": source["instrument"],
            "arrangementRole": role,
            "hand": "left" if value < 60 else "right",
            "duration": round_number(
                clamp(source["duration"], duration_minimum, duration_maximum)
            ),
            "velocity": round_number(clamp(velocity, 0.05, 1.0), 3),
        }
    )
    return note


def select_lead(
    notes: list[dict[str, Any]],
    *,
    role: str = "melody",
    window_seconds: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    previous_midi: int | None = None
    for group in grouped_by_window(notes, window_seconds):
        def score(note: dict[str, Any]) -> tuple[float, float, int]:
            continuity = 0.0 if previous_midi is None else -abs(note["midi"] - previous_midi) * 0.035
            voice_bonus = 2.0 if note["instrument"] == "voice" else 0.0
            lead_bonus = 0.5 if note["instrument"] in LEAD_INSTRUMENTS else 0.0
            return (voice_bonus + lead_bonus + note["velocity"] + continuity, note["time"], note["midi"])

        source = max(group, key=score)
        midi = map_octave_to_range(source["midi"], MELODY_MIN_MIDI, MELODY_MAX_MIDI)
        velocity = max(0.84, source["velocity"] * (1.12 if source["instrument"] == "voice" else 1.04))
        selected.append(
            arranged_note(
                source,
                midi=midi,
                role=role,
                velocity=velocity,
                duration_minimum=0.11,
                duration_maximum=2.8,
            )
        )
        previous_midi = midi
    return selected


def select_bass(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in grouped_by_window(notes, 0.20):
        source = min(group, key=lambda note: (note["midi"], -note["velocity"], note["time"]))
        midi = map_octave_to_range(source["midi"], BASS_MIN_MIDI, BASS_MAX_MIDI)
        selected.append(
            arranged_note(
                source,
                midi=midi,
                role="bass",
                velocity=clamp(source["velocity"] * 0.82, 0.56, 0.76),
                duration_minimum=0.18,
                duration_maximum=2.6,
            )
        )
    return selected


def compact_harmony(
    notes: list[dict[str, Any]],
    *,
    window_seconds: float = 0.30,
    maximum_pitch_classes: int = MAX_HARMONY_PITCH_CLASSES,
) -> list[dict[str, Any]]:
    arranged: list[dict[str, Any]] = []
    previous_voicing: list[int] = []
    window_seconds = clamp(float(window_seconds), 0.06, 0.40)
    maximum_pitch_classes = int(clamp(int(maximum_pitch_classes), 1, 4))
    for group in grouped_by_window(notes, window_seconds):
        by_pitch_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for note in group:
            by_pitch_class[note["midi"] % 12].append(note)
        ranked_pitch_classes = sorted(
            by_pitch_class,
            key=lambda pitch_class: (
                max(note["velocity"] for note in by_pitch_class[pitch_class])
                + min(0.18, len(by_pitch_class[pitch_class]) * 0.025)
            ),
            reverse=True,
        )[:maximum_pitch_classes]
        if not ranked_pitch_classes:
            continue

        chord: list[tuple[int, dict[str, Any]]] = []
        for index, pitch_class in enumerate(ranked_pitch_classes):
            source = max(
                by_pitch_class[pitch_class],
                key=lambda note: (note["velocity"], note["duration"]),
            )
            candidates = [
                midi
                for midi in range(HARMONY_MIN_MIDI, HARMONY_MAX_MIDI + 1)
                if midi % 12 == pitch_class
            ]
            source_target = map_octave_to_range(
                source["midi"], HARMONY_MIN_MIDI, HARMONY_MAX_MIDI
            )
            previous_target = (
                previous_voicing[min(index, len(previous_voicing) - 1)]
                if previous_voicing
                else source_target
            )
            midi = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate - previous_target) * 0.65
                    + abs(candidate - source_target) * 0.35
                ),
            )
            chord.append((midi, source))

        chord.sort(key=lambda item: item[0])
        previous_voicing = [midi for midi, _source in chord]
        onset = min(source["time"] for _midi, source in chord)
        for midi, source in chord:
            voiced_source = dict(source)
            voiced_source["time"] = round_number(onset)
            arranged.append(
                arranged_note(
                    voiced_source,
                    midi=midi,
                    role="harmony",
                    velocity=clamp(source["velocity"] * 0.72, 0.42, 0.67),
                    duration_minimum=0.14,
                    duration_maximum=2.4,
                )
            )
    return arranged


def _tight_onset_groups(
    notes: Iterable[dict[str, Any]], tolerance_seconds: float = 0.035
) -> list[list[dict[str, Any]]]:
    ordered = sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))
    groups: list[list[dict[str, Any]]] = []
    for note in ordered:
        if (
            not groups
            or float(note["time"]) - float(groups[-1][0]["time"]) > tolerance_seconds
        ):
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _estimate_short_harmony_pulse(
    groups: list[list[dict[str, Any]]],
) -> tuple[float, float]:
    """Return the source pulse and its integer-grid coverage.

    MuScriptor's exact attacks preserve tempo even when the old 300 ms reducer
    removes every other gesture.  Searching short integer subdivisions makes
    the estimate tempo-relative and does not require a trustworthy BPM tag.
    """

    times = [float(group[0]["time"]) for group in groups]
    gaps = [
        right - left
        for left, right in zip(times, times[1:])
        if 0.055 <= right - left <= 0.90
    ]
    if not gaps:
        return 0.30, 0.0
    trials: list[tuple[float, float, float]] = []
    for milliseconds in range(80, 321):
        pulse = milliseconds / 1000.0
        residuals: list[float] = []
        close = 0
        for gap in gaps:
            multiple = max(1, min(8, round(gap / pulse)))
            residual = abs(gap - multiple * pulse) / pulse
            residuals.append(min(1.0, residual))
            close += int(residual <= 0.12)
        coverage = close / len(gaps)
        score = (
            sum(residuals) / len(residuals)
            + 0.18 * (1.0 - coverage)
            + 0.025 * pulse
        )
        trials.append((score, pulse, coverage))
    _score, pulse, coverage = min(trials)
    return round_number(pulse, 6), round_number(coverage, 6)


def _source_family(note: dict[str, Any]) -> str:
    instrument = str(note.get("instrument") or "").lower()
    if instrument in BASS_INSTRUMENTS or "bass" in instrument:
        return "bass"
    if "guitar" in instrument:
        return "guitar"
    if instrument in PIANO_INSTRUMENTS or "piano" in instrument or "keyboard" in instrument:
        return "piano"
    if instrument in VOICE_INSTRUMENTS or "voice" in instrument or "vocal" in instrument:
        return "voice"
    return "other"


def _pitch_class_similarity(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def source_supported_cyclic_harmony(
    notes: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn a fast, stable guitar texture into a playable two-hand cycle.

    This decoder is deliberately opt-in.  Every emitted pitch class must exist
    in the nearby transcription; only octave placement and which voices play on
    each phase are authored.  Unqualified passages use ``compact_harmony``
    unchanged, preventing a Kiss-Me-specific experiment from rewriting other
    kinds of songs.
    """

    policy = config or {}
    onset_window = clamp(float(policy.get("onsetWindowSeconds", 0.035)), 0.01, 0.08)
    source_groups = _tight_onset_groups(notes, onset_window)
    if not source_groups:
        return [], {
            "applied": False,
            "profile": "source-supported-cyclic-harmony-v1",
            "reason": "no-harmony-source-events",
        }
    pulse, pulse_coverage = _estimate_short_harmony_pulse(source_groups)
    minimum_pulse = clamp(float(policy.get("minimumPulseSeconds", 0.125)), 0.08, 0.30)
    maximum_pulse = clamp(float(policy.get("maximumPulseSeconds", 0.17)), minimum_pulse, 0.32)
    minimum_guitar_share = clamp(float(policy.get("minimumGuitarShare", 0.70)), 0.0, 1.0)
    minimum_grid_coverage = clamp(float(policy.get("minimumGridCoverage", 0.82)), 0.0, 1.0)
    minimum_run_groups = max(4, int(policy.get("minimumRunGroups", 8)))
    context_radius = clamp(float(policy.get("contextRadiusSeconds", 0.26)), 0.08, 0.60)
    maximum_pitch_classes = int(
        clamp(int(policy.get("fallbackMaximumHarmonyPitchClasses", MAX_HARMONY_PITCH_CLASSES)), 1, 4)
    )
    fallback_window = clamp(float(policy.get("fallbackHarmonyWindowSeconds", 0.30)), 0.06, 0.40)
    source_piano_ratio = clamp(float(policy.get("_sourcePianoRatio", 0.0)), 0.0, 1.0)
    maximum_source_piano_ratio = clamp(
        float(policy.get("maximumSourcePianoRatio", 0.08)), 0.0, 1.0
    )
    if source_piano_ratio > maximum_source_piano_ratio:
        # A meaningful detected-piano layer is stronger evidence than a guitar
        # texture prior.  Re-authoring that material as a cyclic guitar pattern
        # discarded real piano notes on mixed recordings such as the locked 22
        # holdout.  Keep the ordinary source-supported compact reduction instead.
        return compact_harmony(
            notes,
            window_seconds=fallback_window,
            maximum_pitch_classes=maximum_pitch_classes,
        ), {
            "applied": False,
            "profile": "source-supported-cyclic-harmony-v1",
            "reason": "piano-source-preservation-gate",
            "sourcePianoRatio": round_number(source_piano_ratio, 4),
            "maximumSourcePianoRatio": round_number(maximum_source_piano_ratio, 4),
        }
    preferred_onsets = sorted(
        float(value)
        for value in policy.get("_preferredOnsets", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )
    snap_to_role_onsets = clamp(
        float(policy.get("snapToRoleOnsetsSeconds", 0.0)), 0.0, 0.12
    )

    times = [float(group[0]["time"]) for group in source_groups]
    group_pitch_classes = [
        {int(note["midi"]) % 12 for note in group} for group in source_groups
    ]
    runs: list[list[int]] = []
    for position in range(len(source_groups)):
        if not runs or times[position] - times[runs[-1][-1]] >= max(0.75, 4.5 * pulse):
            runs.append([position])
        else:
            runs[-1].append(position)

    eligible_positions: set[int] = set()
    eligible_runs = 0
    run_diagnostics: list[dict[str, Any]] = []
    for run in runs:
        run_notes = [note for position in run for note in source_groups[position]]
        guitar_share = sum(_source_family(note) == "guitar" for note in run_notes) / max(
            1, len(run_notes)
        )
        adjacent_similarities = [
            _pitch_class_similarity(group_pitch_classes[left], group_pitch_classes[right])
            for left, right in zip(run, run[1:])
        ]
        stable_share = sum(value >= 0.40 for value in adjacent_similarities) / max(
            1, len(adjacent_similarities)
        )
        qualifies = (
            len(run) >= minimum_run_groups
            and minimum_pulse <= pulse <= maximum_pulse
            and pulse_coverage >= minimum_grid_coverage
            and guitar_share >= minimum_guitar_share
            and stable_share >= float(policy.get("minimumStableTransitionShare", 0.50))
        )
        if qualifies:
            eligible_positions.update(run)
            eligible_runs += 1
        run_diagnostics.append(
            {
                "groups": len(run),
                "guitarShare": round_number(guitar_share, 4),
                "stableTransitionShare": round_number(stable_share, 4),
                "qualified": qualifies,
            }
        )

    if not eligible_positions:
        fallback = compact_harmony(
            notes,
            window_seconds=fallback_window,
            maximum_pitch_classes=maximum_pitch_classes,
        )
        return fallback, {
            "applied": False,
            "profile": "source-supported-cyclic-harmony-v1",
            "reason": "source-texture-gate-not-satisfied",
            "estimatedPulseSeconds": pulse,
            "pulseGridCoverage": pulse_coverage,
            "runs": run_diagnostics,
            "fallbackNotes": len(fallback),
        }

    # The learned eight-phase texture mirrors the structural statistics of the
    # approved fast acoustic reference.  Names describe physical roles, not
    # target pitches, so every actual note still comes from current source PCs.
    phase_pattern = tuple(
        str(value)
        for value in policy.get(
            "phasePattern",
            (
                "low-root+upper",
                "inner",
                "low-inner+upper",
                "root+upper",
                "inner",
                "upper",
                "upper",
                "inner",
            ),
        )
    )
    if not phase_pattern:
        raise ValueError("Cyclic harmony phasePattern cannot be empty.")
    supplemental_phase_pattern = tuple(
        str(value)
        for value in policy.get("supplementalPhasePattern", ())
    )
    melody_onsets = sorted(
        float(value)
        for value in policy.get("_melodyOnsets", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )
    supplemental_minimum_onset_distance = clamp(
        float(policy.get("supplementalMinimumOnsetDistanceSeconds", 0.08)),
        0.04,
        0.20,
    )
    supplemental_melody_quiet_radius = clamp(
        float(policy.get("supplementalMelodyQuietRadiusSeconds", 0.45)),
        0.10,
        2.0,
    )
    context_action_map = {
        str(key): str(value)
        for key, value in (policy.get("contextActionMap") or {}).items()
    }
    context_melody_radius = clamp(
        float(policy.get("contextMelodyRadiusSeconds", 0.18)),
        0.04,
        0.60,
    )
    context_harmony_change_threshold = clamp(
        float(policy.get("contextHarmonyChangeThreshold", 0.52)),
        0.05,
        0.95,
    )
    use_bass_context_for_root = bool(policy.get("useBassContextForRoot", False))
    bass_context_radius = clamp(
        float(policy.get("bassContextRadiusSeconds", 1.20)), 0.10, 3.0
    )
    bass_context_notes = sorted(
        (
            note
            for note in policy.get("_bassContextNotes", [])
            if isinstance(note, dict)
            and isinstance(note.get("time"), (int, float))
            and isinstance(note.get("midi"), (int, float))
        ),
        key=lambda note: float(note["time"]),
    )
    bass_context_times = [float(note["time"]) for note in bass_context_notes]

    onset_policy = str(policy.get("onsetPolicy", "raw-source")).strip().lower()
    if onset_policy not in {"raw-source", "compact", "pulse-grid"}:
        raise ValueError(
            "Cyclic harmony onsetPolicy must be raw-source, compact, or pulse-grid."
        )
    phase_clock = str(policy.get("phaseClock", "event-index")).strip().lower()
    if phase_clock not in {
        "event-index",
        "pulse-grid",
        "harmonic-root-event-index",
    }:
        raise ValueError(
            "Cyclic harmony phaseClock must be event-index, pulse-grid, or "
            "harmonic-root-event-index."
        )
    supplemental_grid_gestures = 0
    suppressed_supplemental_grid_gestures = 0
    if onset_policy == "compact":
        render_groups = _tight_onset_groups(
            compact_harmony(
                notes,
                window_seconds=fallback_window,
                maximum_pitch_classes=maximum_pitch_classes,
            ),
            onset_window,
        )
        render_entries: list[
            tuple[list[dict[str, Any]], int | None, str | None]
        ] = [
            (group, None, None) for group in render_groups
        ]
        if supplemental_phase_pattern:
            compact_times = [float(group[0]["time"]) for group in render_groups]
            for run in runs:
                if not run or run[0] not in eligible_positions:
                    continue
                start = times[run[0]]
                end = times[run[-1]]
                number_of_steps = max(
                    0, int(round((end - start) / max(0.08, pulse)))
                )
                run_times = [times[position] for position in run]
                for step in range(number_of_steps + 1):
                    grid_time = start + step * pulse
                    if grid_time > end + pulse * 0.35:
                        break
                    phase = step % len(supplemental_phase_pattern)
                    action = supplemental_phase_pattern[phase]
                    if action == "rest":
                        suppressed_supplemental_grid_gestures += 1
                        continue
                    compact_insertion = bisect.bisect_left(compact_times, grid_time)
                    compact_candidates = [
                        candidate
                        for candidate in (compact_insertion - 1, compact_insertion)
                        if 0 <= candidate < len(compact_times)
                    ]
                    if compact_candidates and min(
                        abs(compact_times[candidate] - grid_time)
                        for candidate in compact_candidates
                    ) <= supplemental_minimum_onset_distance:
                        continue
                    melody_insertion = bisect.bisect_left(melody_onsets, grid_time)
                    melody_candidates = [
                        candidate
                        for candidate in (melody_insertion - 1, melody_insertion)
                        if 0 <= candidate < len(melody_onsets)
                    ]
                    if melody_candidates and min(
                        abs(melody_onsets[candidate] - grid_time)
                        for candidate in melody_candidates
                    ) <= supplemental_melody_quiet_radius:
                        continue
                    insertion = bisect.bisect_left(run_times, grid_time)
                    local_candidates = [
                        candidate
                        for candidate in (insertion - 1, insertion)
                        if 0 <= candidate < len(run)
                    ]
                    if not local_candidates:
                        continue
                    nearest_in_run = min(
                        local_candidates,
                        key=lambda candidate: abs(run_times[candidate] - grid_time),
                    )
                    synthetic_group = []
                    for source in source_groups[run[nearest_in_run]]:
                        synthetic = dict(source)
                        synthetic["time"] = round_number(grid_time, 6)
                        synthetic_group.append(synthetic)
                    render_entries.append(
                        (synthetic_group, phase, action)
                    )
                    supplemental_grid_gestures += 1
    elif onset_policy == "pulse-grid":
        # The old 300 ms harmony reducer can erase a real 125--230 ms
        # accompaniment pulse.  Rebuild only that clock here.  The nearest raw
        # group supplies pitch-class evidence; its timestamp is never copied,
        # and later code still refuses to emit an unsupported pitch class.
        render_entries = []
        snapped_grid_gestures = 0
        for run in runs:
            if not run or run[0] not in eligible_positions:
                continue
            start = times[run[0]]
            end = times[run[-1]]
            number_of_steps = max(0, int(round((end - start) / max(0.08, pulse))))
            run_times = [times[position] for position in run]
            for step in range(number_of_steps + 1):
                grid_time = start + step * pulse
                if grid_time > end + pulse * 0.35:
                    break
                if preferred_onsets and snap_to_role_onsets > 0:
                    preferred_insertion = bisect.bisect_left(
                        preferred_onsets, grid_time
                    )
                    preferred_candidates = [
                        candidate
                        for candidate in (
                            preferred_insertion - 1,
                            preferred_insertion,
                        )
                        if 0 <= candidate < len(preferred_onsets)
                    ]
                    if preferred_candidates:
                        nearest_preferred = min(
                            preferred_candidates,
                            key=lambda candidate: abs(
                                preferred_onsets[candidate] - grid_time
                            ),
                        )
                        preferred_time = preferred_onsets[nearest_preferred]
                        if abs(preferred_time - grid_time) <= snap_to_role_onsets:
                            grid_time = preferred_time
                            snapped_grid_gestures += 1
                insertion = bisect.bisect_left(run_times, grid_time)
                local_candidates = [
                    candidate
                    for candidate in (insertion - 1, insertion)
                    if 0 <= candidate < len(run)
                ]
                if not local_candidates:
                    continue
                nearest_in_run = min(
                    local_candidates,
                    key=lambda candidate: abs(run_times[candidate] - grid_time),
                )
                source_group = source_groups[run[nearest_in_run]]
                synthetic_group = []
                for source in source_group:
                    synthetic = dict(source)
                    synthetic["time"] = round_number(grid_time, 6)
                    synthetic_group.append(synthetic)
                render_entries.append(
                    (synthetic_group, step % len(phase_pattern), None)
                )
    else:
        render_groups = source_groups
        render_entries = [(group, None, None) for group in render_groups]
        snapped_grid_gestures = 0

    if onset_policy == "compact":
        snapped_grid_gestures = 0

    arranged: list[dict[str, Any]] = []
    fallback_source: list[dict[str, Any]] = (
        [
            note
            for position, group in enumerate(source_groups)
            if position not in eligible_positions
            for note in group
        ]
        if onset_policy == "pulse-grid"
        else []
    )
    fallback_arranged: list[dict[str, Any]] = []
    phase_counts: Counter[int] = Counter()
    context_counts: Counter[str] = Counter()
    context_action_counts: Counter[str] = Counter()
    octave_shifted = 0
    retained_octave_equivalents = 0
    source_supported_notes = 0
    suppressed_grid_gestures = 0
    run_start_by_position: dict[int, float] = {}
    for run in runs:
        start = times[run[0]]
        for position in run:
            run_start_by_position[position] = start

    render_run_index = 0
    previous_render_time: float | None = None
    previous_phase_root: int | None = None
    harmonic_root_resets = 0
    harmonic_root_counts: Counter[int] = Counter()
    for render_position, (
        group,
        explicit_phase,
        explicit_action,
    ) in enumerate(render_entries):
        time = float(group[0]["time"])
        insertion = bisect.bisect_left(times, time)
        candidates = [
            source_position
            for source_position in (insertion - 1, insertion)
            if 0 <= source_position < len(source_groups)
        ]
        source_position = min(
            candidates,
            key=lambda candidate: abs(times[candidate] - time),
        )
        source_group = source_groups[source_position]
        if source_position not in eligible_positions:
            if onset_policy == "compact":
                fallback_arranged.extend(group)
            elif onset_policy == "raw-source":
                fallback_source.extend(group)
            continue
        local_left = bisect.bisect_left(times, time - context_radius)
        local_right = bisect.bisect_right(times, time + context_radius)
        local = [
            note
            for other_group in source_groups[local_left:local_right]
            for note in other_group
        ]
        salience: dict[int, float] = defaultdict(float)
        anchors_by_pc: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for source in local:
            distance = abs(float(source["time"]) - time)
            family_weight = {
                "bass": 1.25,
                "piano": 1.15,
                "guitar": 1.0,
                "other": 0.72,
            }.get(_source_family(source), 0.72)
            midi = int(source["midi"])
            salience[midi] += math.exp(-distance / max(0.04, context_radius * 0.55)) * family_weight
            anchors_by_pc[midi % 12].append(source)
        ranked_midis = sorted(salience, key=lambda midi: (-salience[midi], midi))
        # Keep a compact four-voice skeleton while retaining exact octave
        # equivalents.  The previous implementation collapsed these by PC.
        skeleton = sorted(ranked_midis[:6])
        current_midis = sorted({int(source["midi"]) for source in source_group})
        if current_midis:
            for midi in current_midis:
                if midi not in skeleton:
                    skeleton.append(midi)
            skeleton = sorted(skeleton)
        if not skeleton:
            if onset_policy == "compact":
                fallback_arranged.extend(group)
            else:
                fallback_source.extend(group)
            continue

        local_bass: list[dict[str, Any]] = []
        if use_bass_context_for_root and bass_context_notes:
            bass_left = bisect.bisect_left(
                bass_context_times, time - bass_context_radius
            )
            bass_right = bisect.bisect_right(
                bass_context_times, time + bass_context_radius
            )
            local_bass = bass_context_notes[bass_left:bass_right]
        if local_bass:
            bass_scores: dict[int, float] = defaultdict(float)
            for source in local_bass:
                distance = abs(float(source["time"]) - time)
                bass_scores[int(source["midi"]) % 12] += (
                    math.exp(
                        -distance / max(0.10, bass_context_radius * 0.55)
                    )
                    * (0.30 + clamp(float(source.get("velocity", 0.7)), 0.0, 1.0))
                    * (
                        0.30
                        + min(1.40, max(0.0, float(source.get("duration", 0.2))))
                    )
                )
            root_pc = max(
                bass_scores,
                key=lambda pitch_class: (bass_scores[pitch_class], -pitch_class),
            )
            root_source = min(
                local_bass,
                key=lambda source: (
                    int(source["midi"]) % 12 != root_pc,
                    abs(float(source["time"]) - time),
                    int(source["midi"]),
                ),
            )
            # Bass is legitimate source support for a chord root even when a
            # separated guitar stem omits that pitch at the exact attack.
            for source in local_bass:
                if int(source["midi"]) % 12 == root_pc:
                    anchors_by_pc[root_pc].append(source)
        else:
            root_source = min(
                (
                    source
                    for source in local
                    if _source_family(source) == "bass"
                ),
                key=lambda source: int(source["midi"]),
                default=min(source_group, key=lambda source: int(source["midi"])),
            )
            root_pc = int(root_source["midi"]) % 12
        harmonic_root_counts[root_pc] += 1
        root_candidates = [midi for midi in skeleton if midi % 12 == root_pc]
        root_midi = min(root_candidates or skeleton)
        while root_midi < 45:
            root_midi += 12
        while root_midi > 59:
            root_midi -= 12
        low_root = root_midi - 12 if root_midi - 12 >= COMPACT_PIANO_MIN_MIDI else root_midi

        left_candidates = sorted(midi for midi in skeleton if midi < 60 and midi % 12 != root_pc)
        inner = left_candidates[-1] if left_candidates else root_midi
        while inner < 45:
            inner += 12
        while inner > 59:
            inner -= 12
        low_inner = inner - 12 if inner - 12 >= COMPACT_PIANO_MIN_MIDI else inner

        upper_candidates = sorted({midi for midi in skeleton if midi >= 60})
        if len(upper_candidates) < 2:
            for midi in reversed(skeleton):
                lifted = midi
                while lifted < 60:
                    lifted += 12
                while lifted > HARMONY_MAX_MIDI:
                    lifted -= 12
                if 60 <= lifted <= HARMONY_MAX_MIDI and lifted not in upper_candidates:
                    upper_candidates.append(lifted)
                if len(upper_candidates) >= 2:
                    break
        upper = sorted(upper_candidates)[-2:]

        if explicit_phase is not None:
            phase = explicit_phase
        elif onset_policy == "compact":
            if phase_clock == "pulse-grid":
                pulse_units = round(
                    (time - run_start_by_position[source_position])
                    / max(0.08, pulse)
                )
                phase = int(pulse_units) % len(phase_pattern)
            elif phase_clock == "harmonic-root-event-index":
                new_run = (
                    previous_render_time is None
                    or time - previous_render_time >= max(0.75, 4.5 * pulse)
                )
                if new_run or previous_phase_root != root_pc:
                    render_run_index = 0
                    harmonic_root_resets += 1
                else:
                    render_run_index += 1
                phase = render_run_index % len(phase_pattern)
                previous_phase_root = root_pc
                previous_render_time = time
            else:
                if (
                    previous_render_time is None
                    or time - previous_render_time >= max(0.75, 4.5 * pulse)
                ):
                    render_run_index = 0
                else:
                    render_run_index += 1
                phase = render_run_index % len(phase_pattern)
                previous_render_time = time
        else:
            pulse_units = round(
                (time - run_start_by_position[source_position]) / max(0.08, pulse)
            )
            phase = int(pulse_units) % len(phase_pattern)
        phase_counts[phase] += 1
        compact_midis = sorted({int(note["midi"]) for note in group})
        context_key: str | None = None
        if onset_policy == "compact" and explicit_action is None and context_action_map:
            compact_has_left = any(midi < 60 for midi in compact_midis)
            compact_has_right = any(midi >= 60 for midi in compact_midis)
            compact_occupancy = (
                "both"
                if compact_has_left and compact_has_right
                else "left"
                if compact_has_left
                else "right"
            )
            melody_insertion = bisect.bisect_left(melody_onsets, time)
            melody_candidates = [
                candidate
                for candidate in (melody_insertion - 1, melody_insertion)
                if 0 <= candidate < len(melody_onsets)
            ]
            melody_state = (
                "melody"
                if melody_candidates
                and min(
                    abs(melody_onsets[candidate] - time)
                    for candidate in melody_candidates
                )
                <= context_melody_radius
                else "quiet"
            )
            previous_position = source_position - 1
            is_run_start = (
                previous_position not in eligible_positions
                or times[source_position] - times[previous_position]
                >= max(0.75, 4.5 * pulse)
            )
            previous_similarity = (
                0.0
                if is_run_start
                else _pitch_class_similarity(
                    group_pitch_classes[previous_position],
                    group_pitch_classes[source_position],
                )
            )
            harmony_state = (
                "change"
                if is_run_start
                or previous_similarity < context_harmony_change_threshold
                else "stable"
            )
            context_key = f"{melody_state}:{compact_occupancy}:{harmony_state}"
            context_counts[context_key] += 1
        action = (
            explicit_action
            or (context_action_map.get(context_key) if context_key is not None else None)
            or phase_pattern[phase]
        )
        if context_key is not None:
            context_action_counts[f"{context_key}={action}"] += 1
        selected_midis: list[int]
        if action == "rest":
            suppressed_grid_gestures += 1
            continue
        if action == "compact-preserve":
            selected_midis = compact_midis
        elif action == "compact-octave-root":
            selected_midis = list(compact_midis)
            upper_root_candidates = sorted(
                midi
                for midi in skeleton
                if midi % 12 == root_pc and 60 <= midi <= HARMONY_MAX_MIDI
            )
            missing_root = next(
                (midi for midi in upper_root_candidates if midi not in selected_midis),
                None,
            )
            if missing_root is not None:
                replaceable = [
                    midi
                    for midi in selected_midis
                    if midi % 12 != root_pc and midi < 60
                ]
                if not replaceable:
                    replaceable = [
                        midi
                        for midi in selected_midis
                        if midi % 12 != root_pc and midi != max(selected_midis)
                    ]
                if replaceable:
                    selected_midis.remove(max(replaceable))
                selected_midis.append(missing_root)
        elif action == "compact-upper":
            selected_midis = [midi for midi in compact_midis if midi >= 60] or list(upper)
        elif action == "compact-inner":
            left = [midi for midi in compact_midis if midi < 60]
            selected_midis = [max(left)] if left else [inner]
        elif action == "compact-root+upper":
            selected_midis = [
                *(midi for midi in compact_midis if midi % 12 == root_pc),
                *upper,
            ]
        elif action == "low-root+upper":
            selected_midis = [low_root, *upper]
        elif action == "inner":
            selected_midis = [inner]
        elif action == "low-inner+upper":
            selected_midis = [low_inner, *upper]
        elif action == "root+upper":
            selected_midis = [root_midi, *upper]
        elif action == "upper":
            selected_midis = list(upper)
        elif action == "root":
            selected_midis = [root_midi]
        else:
            raise ValueError(f"Unsupported cyclic harmony action: {action}")
        selected_midis = list(dict.fromkeys(selected_midis))
        if not selected_midis:
            if onset_policy == "compact":
                fallback_arranged.extend(group)
            else:
                fallback_source.extend(group)
            continue

        for midi in selected_midis:
            incumbent = next(
                (item for item in group if int(item["midi"]) == midi),
                None,
            )
            if (
                onset_policy == "compact"
                and explicit_action is None
                and incumbent is not None
            ):
                rendered = dict(incumbent)
                rendered["generatedBy"] = "source-supported-cyclic-harmony-v1"
                rendered["pianistTexturePhase"] = phase
                rendered["pianistTextureAction"] = action
                arranged.append(rendered)
                source_supported_notes += 1
                continue
            candidates = anchors_by_pc.get(midi % 12) or list(source_group)
            source = min(
                candidates,
                key=lambda item: (
                    abs(int(item["midi"]) - midi),
                    abs(float(item["time"]) - time),
                ),
            )
            if int(source["midi"]) % 12 != midi % 12:
                continue
            voiced_source = dict(source)
            voiced_source["time"] = round_number(time, 6)
            voiced_source["duration"] = round_number(
                clamp(max(float(source.get("duration", pulse)), pulse * 0.88), 0.08, 1.20),
                6,
            )
            role = (
                "harmony"
                if onset_policy == "compact" and action.startswith("compact-")
                else "bass"
                if midi < 60
                else "harmony"
            )
            velocity = (
                clamp(float(source.get("velocity", 0.6)) * 0.80, 0.40, 0.64)
                if role == "bass"
                else clamp(float(source.get("velocity", 0.6)) * 0.86, 0.46, 0.72)
            )
            rendered = arranged_note(
                voiced_source,
                midi=midi,
                role=role,
                velocity=velocity,
                duration_minimum=0.08,
                duration_maximum=1.20,
            )
            rendered["generatedBy"] = "source-supported-cyclic-harmony-v1"
            rendered["pianistTexturePhase"] = phase
            rendered["pianistTextureAction"] = action
            arranged.append(rendered)
            source_supported_notes += 1
            octave_shifted += int(int(source["midi"]) != midi)
        pcs = Counter(midi % 12 for midi in selected_midis)
        retained_octave_equivalents += int(any(count > 1 for count in pcs.values()))

    fallback = (
        fallback_arranged
        if onset_policy == "compact"
        else compact_harmony(
            fallback_source,
            window_seconds=fallback_window,
            maximum_pitch_classes=maximum_pitch_classes,
        )
    )
    return sorted(arranged + fallback, key=lambda note: (note["time"], note["midi"])), {
        "applied": bool(arranged),
        "profile": "source-supported-cyclic-harmony-v1",
        "estimatedPulseSeconds": pulse,
        "pulseGridCoverage": pulse_coverage,
        "eligibleRuns": eligible_runs,
        "eligibleSourceGroups": len(eligible_positions),
        "patternedOutputNotes": len(arranged),
        "sourceSupportedPatternNotes": source_supported_notes,
        "generatedGridGestures": (
            len(render_entries) - suppressed_grid_gestures
            if onset_policy == "pulse-grid"
            else 0
        ),
        "suppressedGridGestures": suppressed_grid_gestures,
        "snappedGridGestures": snapped_grid_gestures,
        "supplementalGridGestures": supplemental_grid_gestures,
        "suppressedSupplementalGridGestures": (
            suppressed_supplemental_grid_gestures
        ),
        "octaveShiftedPatternNotes": octave_shifted,
        "gesturesWithRetainedOctaveEquivalents": retained_octave_equivalents,
        "fallbackNotes": len(fallback),
        "phasePattern": list(phase_pattern),
        "phaseClock": phase_clock,
        "bassContextForRoot": use_bass_context_for_root,
        "bassContextNotes": len(bass_context_notes),
        "bassContextRadiusSeconds": bass_context_radius,
        "harmonicRootResets": harmonic_root_resets,
        "harmonicRootCounts": {
            str(key): value for key, value in sorted(harmonic_root_counts.items())
        },
        "onsetPolicy": onset_policy,
        "phaseCounts": {str(key): value for key, value in sorted(phase_counts.items())},
        "contextCounts": dict(sorted(context_counts.items())),
        "contextActionCounts": dict(sorted(context_action_counts.items())),
        "runs": run_diagnostics,
    }


def limit_onset_clusters(notes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for group in grouped_by_window(notes, 0.04):
        ranked = sorted(
            group,
            key=lambda note: (
                ROLE_PRIORITY.get(note.get("arrangementRole", "harmony"), 9),
                -note["velocity"],
                note["midi"],
            ),
        )
        kept.extend(ranked[:MAX_ONSET_CLUSTER])
        removed += max(0, len(ranked) - MAX_ONSET_CLUSTER)
    return kept, removed


def nearby_count(sorted_times: list[float], time: float, radius: float = 0.5) -> int:
    '''Return the fullest one-second window that would contain the candidate.'''
    del radius
    first_relevant = bisect.bisect_left(sorted_times, time - 1.0)
    insertion = bisect.bisect_right(sorted_times, time)
    fullest = 0
    for start_index in range(first_relevant, insertion):
        start_time = sorted_times[start_index]
        if time <= start_time + 1.0:
            existing = bisect.bisect_right(
                sorted_times, start_time + 1.0
            ) - start_index
            fullest = max(fullest, existing)
    future = bisect.bisect_right(sorted_times, time + 1.0) - insertion
    return max(fullest, future)


def limit_density(
    role_notes: dict[str, list[dict[str, Any]]],
    maximum: int,
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    accepted_times: list[float] = []
    removed = 0
    for role in ("melody", "bass", "source_piano", "harmony"):
        for note in sorted(role_notes.get(role, []), key=lambda item: (item["time"], item["midi"])):
            if nearby_count(accepted_times, note["time"]) >= maximum:
                removed += 1
                continue
            bisect.insort(accepted_times, note["time"])
            accepted.append(note)
    return accepted, removed


def shape_legato(notes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    '''Carry arranged parts toward their next onset without bridging real rests.'''
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_role[note['arrangementRole']].append(dict(note))
    maximum_bridge = {'melody': 1.25, 'bass': 1.8, 'harmony': 1.0}
    maximum_duration = {'melody': 2.8, 'bass': 2.6, 'harmony': 2.4}
    legato_overlap = {'melody': 0.08, 'bass': 0.12, 'harmony': 0.14}
    extended = 0
    output: list[dict[str, Any]] = []
    for role, role_notes in by_role.items():
        onset_times = sorted({note['time'] for note in role_notes})
        for note in role_notes:
            next_index = bisect.bisect_right(onset_times, note['time'] + 0.04)
            if next_index < len(onset_times) and role in maximum_bridge:
                gap = onset_times[next_index] - note['time']
                if gap <= maximum_bridge[role]:
                    target = min(
                        maximum_duration[role],
                        max(MIN_NOTE_SECONDS, gap + legato_overlap[role]),
                    )
                    if target > note['duration']:
                        note['duration'] = round_number(target)
                        extended += 1
            output.append(note)
    return output, extended


def suppress_rapid_retriggers(
    notes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_pitch[note["midi"]].append(note)
    output: list[dict[str, Any]] = []
    removed = 0
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda note: (note["time"], ROLE_PRIORITY.get(note["arrangementRole"], 9)))
        merged: list[dict[str, Any]] = []
        for note in pitch_notes:
            if not merged or note["time"] - merged[-1]["time"] >= MIN_RETRIGGER_SECONDS:
                merged.append(note)
                continue
            previous = merged[-1]
            previous_end = previous["time"] + previous["duration"]
            note_end = note["time"] + note["duration"]
            preferred = min(
                (previous, note),
                key=lambda item: (
                    ROLE_PRIORITY.get(item["arrangementRole"], 9),
                    -item["velocity"],
                ),
            )
            preferred = dict(preferred)
            preferred["time"] = min(previous["time"], note["time"])
            preferred["duration"] = round_number(
                clamp(max(previous_end, note_end) - preferred["time"], MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)
            )
            preferred["velocity"] = max(previous["velocity"], note["velocity"])
            merged[-1] = preferred
            removed += 1

        for index, note in enumerate(merged[:-1]):
            following = merged[index + 1]
            latest_end = following["time"] - SAME_KEY_RELEASE_GAP_SECONDS
            if note["time"] + note["duration"] > latest_end:
                note["duration"] = round_number(
                    max(MIN_NOTE_SECONDS, latest_end - note["time"])
                )
        output.extend(merged)
    return sorted(output, key=lambda note: (note["time"], note["midi"])), removed


def annotate_musical_retriggers(
    notes: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Give deliberate fast restrikes a physical release and human touch."""

    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(note)
    shaped = 0
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda item: float(item["time"]))
        run_position = 0
        for previous, current in zip(pitch_notes, pitch_notes[1:]):
            gap = float(current["time"]) - float(previous["time"])
            if not (
                DEFAULT_DUPLICATE_ONSET_SECONDS
                <= gap
                < FAST_MUSICAL_RETRIGGER_SECONDS
            ):
                run_position = 0
                continue
            run_position += 1
            previous["articulation"] = "repeated-note"
            previous["fastRetriggerGapSeconds"] = round_number(gap, 6)
            current["retriggerReleaseSeconds"] = round_number(
                clamp(0.18 + gap * 0.35, 0.20, 0.25), 4
            )
            role = str(current.get("arrangementRole") or "harmony")
            contours = {
                # Keep the singer clear; merely avoid identical consecutive
                # hammer attacks. Accompaniment receives a little more wrist
                # relaxation so a rapid ostinato does not sound typewritten.
                "melody": (0.98, 1.00, 0.97, 1.01),
                "bass": (0.94, 0.98, 0.92, 0.99),
                "harmony": (0.95, 0.99, 0.93, 1.00),
                "source_piano": (0.97, 1.00, 0.95, 1.00),
            }
            contour = contours.get(role, contours["harmony"])
            velocity_scale = contour[(run_position - 1) % len(contour)]
            current["velocity"] = round_number(
                clamp(float(current.get("velocity", 0.72)) * velocity_scale, 0.04, 1.0),
                4,
            )
            current["retriggerVelocityScale"] = velocity_scale
            current["retriggerRunPosition"] = run_position
            shaped += 1
    return {
        **summarize_same_key_retriggers(notes),
        "humanizedFastRetriggers": shaped,
        "timingJitterApplied": False,
    }


def interpolate_raw_selection_models(
    low_source_model: dict[str, Any],
    high_source_model: dict[str, Any],
    ratio: float,
) -> dict[str, Any]:
    """Interpolate two compatible raw-feature selector logits."""

    names = [str(value) for value in low_source_model.get("featureNames") or []]
    high_names = [
        str(value) for value in high_source_model.get("featureNames") or []
    ]
    low_weights = [float(value) for value in low_source_model.get("weights") or []]
    high_weights = [float(value) for value in high_source_model.get("weights") or []]
    low_means = [float(value) for value in low_source_model.get("means") or []]
    high_means = [float(value) for value in high_source_model.get("means") or []]
    low_scales = [float(value) for value in low_source_model.get("scales") or []]
    high_scales = [float(value) for value in high_source_model.get("scales") or []]
    if not names or not high_names:
        raise ValueError("Adaptive selection endpoints have no feature contract.")
    if names != high_names:
        # Context selectors append features to the frozen v1 contract.  A v1
        # sparse endpoint is exactly representable in that larger space with
        # zero coefficients, so permit only this strict prefix relationship.
        if len(names) < len(high_names) and high_names[: len(names)] == names:
            extension = len(high_names) - len(names)
            names = list(high_names)
            low_weights += [0.0] * extension
            low_means += [0.0] * extension
            low_scales += [1.0] * extension
        elif len(high_names) < len(names) and names[: len(high_names)] == high_names:
            extension = len(names) - len(high_names)
            high_names = list(names)
            high_weights += [0.0] * extension
            high_means += [0.0] * extension
            high_scales += [1.0] * extension
        else:
            raise ValueError(
                "Adaptive selection endpoints use incompatible feature contracts."
            )
    if not (
        len(names)
        == len(low_weights)
        == len(high_weights)
        == len(low_means)
        == len(high_means)
        == len(low_scales)
        == len(high_scales)
    ):
        raise ValueError("Adaptive selection endpoint fields have different lengths.")
    if any(abs(value) > 1e-12 for value in [*low_means, *high_means]) or any(
        abs(value - 1.0) > 1e-12 for value in [*low_scales, *high_scales]
    ):
        raise ValueError("Adaptive selection endpoints must use raw-feature coefficients.")
    mix = clamp(float(ratio), 0.0, 1.0)
    low_threshold = float(low_source_model.get("threshold", 0.5))
    high_threshold = float(high_source_model.get("threshold", 0.5))
    return {
        "type": "raw-feature-adaptive-residual-logit-blend-v1",
        "featureNames": names,
        "weights": [
            low + mix * (high - low)
            for low, high in zip(low_weights, high_weights)
        ],
        "means": [0.0] * len(names),
        "scales": [1.0] * len(names),
        "threshold": low_threshold + mix * (high_threshold - low_threshold),
    }


def adapt_profile_to_source_density(
    style_profile: dict[str, Any],
    factual_source_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Interpolate conservative and dense/vocal-mix decoder settings.

    Every signal is measured from the immutable transcription before arranging.
    The optional voice gates keep non-vocal instrumental material on the frozen
    density/expansion policy while allowing a singer-led mix to avoid generated
    octave doublings that can turn syllables into machine-gun accompaniment.
    Older profiles remain byte-for-byte compatible at inference.
    """

    decoder = style_profile.get("decoder") or {}
    selection_policy = decoder.get("adaptiveSelectionBlend") or {}
    selection_enabled = bool(selection_policy.get("enabled"))
    physical_policy = (
        (decoder.get("physicalPerformance") or {}).get("adaptiveSourceDensity")
        or {}
    )
    physical_enabled = bool(physical_policy.get("enabled"))
    policy = decoder.get("adaptiveSourceDensity") or {}
    density_enabled = bool(policy.get("enabled"))
    sparse_voice_threshold = policy.get(
        "minimumVoiceRatioForDisablingSparseExpansion"
    )
    if (
        not density_enabled
        and not selection_enabled
        and not physical_enabled
        and sparse_voice_threshold is None
    ):
        return style_profile, None
    source_nps = max(
        0.0, float(factual_source_profile.get("sourceNotesPerSecond", 0.0))
    )
    source_voice_ratio = max(
        0.0, float(factual_source_profile.get("voiceRatio", 0.0))
    )
    source_piano_ratio = max(
        0.0, float(factual_source_profile.get("pianoRatio", 0.0))
    )
    maximum_piano_ratio_for_vocal_adaptation = policy.get(
        "maximumPianoRatioForVocalAdaptation"
    )
    if maximum_piano_ratio_for_vocal_adaptation is not None:
        maximum_piano_ratio_for_vocal_adaptation = clamp(
            float(maximum_piano_ratio_for_vocal_adaptation), 0.0, 1.0
        )
    effective_style_profile = style_profile
    selection_diagnostics: dict[str, Any] = {}
    if selection_enabled:
        selection_low_nps = float(
            selection_policy.get("lowSourceNotesPerSecond", 20.0)
        )
        selection_high_nps = float(
            selection_policy.get("highSourceNotesPerSecond", 30.0)
        )
        if selection_high_nps <= selection_low_nps:
            raise ValueError(
                "Adaptive selection high threshold must exceed its low threshold."
            )
        selection_density_ratio = clamp(
            (source_nps - selection_low_nps)
            / (selection_high_nps - selection_low_nps),
            0.0,
            1.0,
        )
        selection_ratio = selection_density_ratio
        selection_minimum_voice_ratio = selection_policy.get(
            "minimumVoiceRatioForAggressiveBlend"
        )
        selection_maximum_piano_ratio = selection_policy.get(
            "maximumPianoRatioForAggressiveBlend",
            maximum_piano_ratio_for_vocal_adaptation,
        )
        if selection_maximum_piano_ratio is not None:
            selection_maximum_piano_ratio = clamp(
                float(selection_maximum_piano_ratio), 0.0, 1.0
            )
        selection_source_voice_ratio = source_voice_ratio
        selection_voice_gate_applied = False
        if selection_minimum_voice_ratio is not None:
            selection_minimum_voice_ratio = clamp(
                float(selection_minimum_voice_ratio), 0.0, 1.0
            )
            if selection_source_voice_ratio < selection_minimum_voice_ratio:
                selection_ratio = 0.0
                selection_voice_gate_applied = True
        if (
            selection_maximum_piano_ratio is not None
            and source_piano_ratio > selection_maximum_piano_ratio
        ):
            selection_ratio = 0.0
            selection_voice_gate_applied = True
        low_share = clamp(
            float(selection_policy.get("lowSourceBaseShare", 1.0)), 0.0, 1.0
        )
        high_share = clamp(
            float(selection_policy.get("highSourceBaseShare", low_share)), 0.0, 1.0
        )
        effective_style_profile = copy.deepcopy(style_profile)
        effective_style_profile["selectionModel"] = interpolate_raw_selection_models(
            selection_policy.get("lowSourceSelectionModel") or {},
            selection_policy.get("highSourceSelectionModel") or {},
            selection_ratio,
        )
        selection_diagnostics = {
            "selectionPolicy": "linear-source-density-residual-share-v1",
            "selectionDensityInterpolationRatio": round_number(
                selection_density_ratio, 4
            ),
            "selectionInterpolationRatio": round_number(selection_ratio, 4),
            "lowSourceSelectionBaseShare": round_number(low_share, 6),
            "highSourceSelectionBaseShare": round_number(high_share, 6),
            "effectiveSelectionBaseShare": round_number(
                low_share + selection_ratio * (high_share - low_share), 6
            ),
            "selectionSourceVoiceRatio": round_number(
                selection_source_voice_ratio, 4
            ),
            "selectionSourcePianoRatio": round_number(source_piano_ratio, 4),
            "minimumVoiceRatioForAggressiveBlend": selection_minimum_voice_ratio,
            "maximumPianoRatioForAggressiveBlend": selection_maximum_piano_ratio,
            "selectionVoiceGateApplied": selection_voice_gate_applied,
        }

    sparse_diagnostics: dict[str, Any] = {}
    if sparse_voice_threshold is not None:
        sparse_voice_threshold = clamp(float(sparse_voice_threshold), 0.0, 1.0)
        sparse_expansion_disabled = (
            source_voice_ratio >= sparse_voice_threshold
            and (
                maximum_piano_ratio_for_vocal_adaptation is None
                or source_piano_ratio <= maximum_piano_ratio_for_vocal_adaptation
            )
        )
        if sparse_expansion_disabled:
            if effective_style_profile is style_profile:
                effective_style_profile = copy.deepcopy(style_profile)
            effective_style_profile.setdefault("decoder", {})[
                "expandSparseHarmony"
            ] = False
        sparse_diagnostics = {
            "minimumVoiceRatioForDisablingSparseExpansion": sparse_voice_threshold,
            "maximumPianoRatioForVocalAdaptation": (
                maximum_piano_ratio_for_vocal_adaptation
            ),
            "sparseExpansionDisabledForVoice": sparse_expansion_disabled,
            "effectiveExpandSparseHarmony": bool(
                (effective_style_profile.get("decoder") or {}).get(
                    "expandSparseHarmony", False
                )
            ),
        }
    effective_style_profile, physical_diagnostics = (
        adapt_physical_performance_to_source(
            effective_style_profile, factual_source_profile
        )
    )
    if not density_enabled:
        return effective_style_profile, {
            "policy": "linear-source-density-residual-share-v1",
            "sourceNotesPerSecond": round_number(source_nps, 3),
            "sourceVoiceRatio": round_number(source_voice_ratio, 4),
            "sourcePianoRatio": round_number(source_piano_ratio, 4),
            **sparse_diagnostics,
            **selection_diagnostics,
            **physical_diagnostics,
        }
    decoder = effective_style_profile.get("decoder") or {}
    policy = decoder.get("adaptiveSourceDensity") or {}
    low_nps = float(policy.get("lowSourceNotesPerSecond", 20.0))
    high_nps = float(policy.get("highSourceNotesPerSecond", 30.0))
    if high_nps <= low_nps:
        raise ValueError("Adaptive source-density high threshold must exceed its low threshold.")
    ratio = clamp((source_nps - low_nps) / (high_nps - low_nps), 0.0, 1.0)
    low_density = clamp(float(policy.get("lowDensityMultiplier", 1.0)), 0.5, 3.0)
    high_density = clamp(float(policy.get("highDensityMultiplier", low_density)), 0.5, 3.0)
    low_duration = clamp(float(policy.get("lowDurationPredictionWeight", 0.0)), 0.0, 1.0)
    high_duration = clamp(float(policy.get("highDurationPredictionWeight", low_duration)), 0.0, 1.0)
    density = low_density + ratio * (high_density - low_density)
    density_minimum_voice_ratio = policy.get(
        "minimumVoiceRatioForDensityAdaptation"
    )
    density_voice_gate_applied = False
    if density_minimum_voice_ratio is not None:
        density_minimum_voice_ratio = clamp(
            float(density_minimum_voice_ratio), 0.0, 1.0
        )
        if (
            source_voice_ratio < density_minimum_voice_ratio
            or (
                maximum_piano_ratio_for_vocal_adaptation is not None
                and source_piano_ratio > maximum_piano_ratio_for_vocal_adaptation
            )
        ):
            density = clamp(
                float(
                    policy.get(
                        "nonVocalDensityMultiplier",
                        decoder.get("preCleanupDensityMultiplier", low_density),
                    )
                ),
                0.5,
                3.0,
            )
            density_voice_gate_applied = True
    base_quota_backfill = clamp(
        float(decoder.get("quotaBackfillRatio", 1.0)), 0.0, 1.0
    )
    low_quota_backfill = policy.get("lowQuotaBackfillRatio")
    high_quota_backfill = policy.get("highQuotaBackfillRatio")
    adaptive_quota_backfill = None
    if low_quota_backfill is not None or high_quota_backfill is not None:
        if low_quota_backfill is None or high_quota_backfill is None:
            raise ValueError(
                "Adaptive quota-backfill policy requires both low and high values."
            )
        low_quota_backfill = clamp(float(low_quota_backfill), 0.0, 1.0)
        high_quota_backfill = clamp(float(high_quota_backfill), 0.0, 1.0)
        adaptive_quota_backfill = (
            low_quota_backfill
            + ratio * (high_quota_backfill - low_quota_backfill)
        )
        if density_voice_gate_applied:
            adaptive_quota_backfill = clamp(
                float(
                    policy.get(
                        "nonVocalQuotaBackfillRatio", base_quota_backfill
                    )
                ),
                0.0,
                1.0,
            )
    source_median_duration = max(
        0.0, float(factual_source_profile.get("sourceMedianDurationSeconds", 0.0))
    )
    short_duration = policy.get("shortSourceMedianDurationSeconds")
    long_duration = policy.get("longSourceMedianDurationSeconds")
    duration_ratio = ratio
    duration_signal = "source-notes-per-second"
    if short_duration is not None or long_duration is not None:
        if short_duration is None or long_duration is None:
            raise ValueError(
                "Adaptive duration policy requires both short and long median-duration thresholds."
            )
        short_duration = float(short_duration)
        long_duration = float(long_duration)
        if long_duration <= short_duration:
            raise ValueError(
                "Adaptive duration long median threshold must exceed its short threshold."
            )
        duration_ratio = clamp(
            (long_duration - source_median_duration)
            / (long_duration - short_duration),
            0.0,
            1.0,
        )
        duration_signal = "non-percussive-source-median-duration"
    duration_weight = low_duration + duration_ratio * (high_duration - low_duration)
    base_source_duration_weight = clamp(
        float(decoder.get("sourceDurationWeight", 0.85)), 0.0, 1.0
    )
    short_source_duration_weight = clamp(
        float(
            policy.get(
                "shortSourceDurationWeight", base_source_duration_weight
            )
        ),
        0.0,
        1.0,
    )
    long_source_duration_weight = clamp(
        float(
            policy.get(
                "longSourceDurationWeight", base_source_duration_weight
            )
        ),
        0.0,
        1.0,
    )
    source_duration_weight = (
        long_source_duration_weight
        + duration_ratio
        * (short_source_duration_weight - long_source_duration_weight)
    )
    minimum_voice_ratio = policy.get("minimumVoiceRatioForDurationPrediction")
    voice_gate_applied = False
    if minimum_voice_ratio is not None:
        minimum_voice_ratio = clamp(float(minimum_voice_ratio), 0.0, 1.0)
        if (
            source_voice_ratio < minimum_voice_ratio
            or (
                maximum_piano_ratio_for_vocal_adaptation is not None
                and source_piano_ratio > maximum_piano_ratio_for_vocal_adaptation
            )
        ):
            duration_weight = clamp(
                float(policy.get("nonVocalDurationPredictionWeight", 0.0)),
                0.0,
                1.0,
            )
            source_duration_weight = clamp(
                float(policy.get("nonVocalSourceDurationWeight", 1.0)),
                0.0,
                1.0,
            )
            voice_gate_applied = True

    effective = copy.deepcopy(effective_style_profile)
    effective.setdefault("decoder", {})["preCleanupDensityMultiplier"] = density
    effective["decoder"]["sourceDurationWeight"] = source_duration_weight
    if adaptive_quota_backfill is not None:
        effective["decoder"]["quotaBackfillRatio"] = adaptive_quota_backfill
    if effective.get("durationModel"):
        effective["durationModel"]["predictionWeight"] = duration_weight
    return effective, {
        "policy": "linear-source-event-density-v1",
        "sourceNotesPerSecond": round_number(source_nps, 3),
        "lowSourceNotesPerSecond": low_nps,
        "highSourceNotesPerSecond": high_nps,
        "interpolationRatio": round_number(ratio, 4),
        "durationInterpolationSignal": duration_signal,
        "sourceMedianDurationSeconds": round_number(source_median_duration, 4),
        "durationInterpolationRatio": round_number(duration_ratio, 4),
        "effectiveDensityMultiplier": round_number(density, 4),
        "effectiveQuotaBackfillRatio": (
            None
            if adaptive_quota_backfill is None
            else round_number(adaptive_quota_backfill, 4)
        ),
        "lowQuotaBackfillRatio": low_quota_backfill,
        "highQuotaBackfillRatio": high_quota_backfill,
        "quotaBackfillVoiceGateApplied": (
            adaptive_quota_backfill is not None and density_voice_gate_applied
        ),
        "minimumVoiceRatioForDensityAdaptation": density_minimum_voice_ratio,
        "densityVoiceGateApplied": density_voice_gate_applied,
        "effectiveDurationPredictionWeight": round_number(duration_weight, 4),
        "effectiveSourceDurationWeight": round_number(
            source_duration_weight, 4
        ),
        "sourceVoiceRatio": round_number(source_voice_ratio, 4),
        "sourcePianoRatio": round_number(source_piano_ratio, 4),
        "maximumPianoRatioForVocalAdaptation": (
            maximum_piano_ratio_for_vocal_adaptation
        ),
        "minimumVoiceRatioForDurationPrediction": minimum_voice_ratio,
        "voiceGateApplied": voice_gate_applied,
        **sparse_diagnostics,
        **selection_diagnostics,
        **physical_diagnostics,
    }


def _piecewise_quantile_map(
    value: float,
    source_quantiles: list[float],
    target_quantiles: list[float],
) -> float:
    """Map a duration through three learned quantile anchors.

    A global multiplier makes already-long accompaniment notes excessively
    blurry.  This monotonic map can lengthen the short/median body of a piano
    part while leaving its upper tail controlled.
    """

    if len(source_quantiles) != 3 or len(target_quantiles) != 3:
        return value
    source = [max(MIN_NOTE_SECONDS, float(item)) for item in source_quantiles]
    target = [max(MIN_NOTE_SECONDS, float(item)) for item in target_quantiles]
    if not (source[0] <= source[1] <= source[2]):
        return value
    if not (target[0] <= target[1] <= target[2]):
        return value
    if value <= source[0]:
        return value * target[0] / max(MIN_NOTE_SECONDS, source[0])
    if value <= source[1]:
        ratio = (value - source[0]) / max(1e-9, source[1] - source[0])
        return target[0] + ratio * (target[1] - target[0])
    if value <= source[2]:
        ratio = (value - source[1]) / max(1e-9, source[2] - source[1])
        return target[1] + ratio * (target[2] - target[1])
    # Do not inflate the long tail at the same rate as the median.  The final
    # physical-performance stage may add pedal resonance separately.
    return target[2] + (value - source[2]) * 0.70


def refine_left_hand_accompaniment(
    notes: list[dict[str, Any]],
    source_payload: dict[str, Any],
    style_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a learned accompaniment adapter without touching the right hand.

    The main selector is intentionally kept frozen: listeners may already like
    its sung/top line.  A second, lower-capacity profile ranks only notes below
    the configured hand split, preserves local intensity with windowed quotas,
    and learns the hold/retrigger behavior of an approved piano performance.
    """

    decoder = style_profile.get("decoder") or {}
    config = decoder.get("leftHandAccompaniment") or {}
    if not config.get("enabled"):
        return notes, {"applied": False}

    hand_split = int(round(float(config.get("handSplitMidi", 72))))
    hand_split = max(PIANO_MIN_MIDI + 1, min(PIANO_MAX_MIDI, hand_split))
    keep_ratio = clamp(float(config.get("targetKeepRatio", 1.0)), 0.1, 1.0)
    window_seconds = clamp(float(config.get("windowSeconds", 4.0)), 0.5, 16.0)
    minimum_notes_per_window = max(0, int(config.get("minimumNotesPerWindow", 1)))
    maximum_notes_per_onset = max(1, int(config.get("maximumNotesPerOnset", 2)))
    minimum_same_pitch_gap = clamp(
        float(config.get("minimumSamePitchGapSeconds", 0.18)),
        DEFAULT_DUPLICATE_ONSET_SECONDS,
        0.40,
    )
    fast_retrigger_keep_share = clamp(
        float(config.get("fastRetriggerKeepShare", 0.06)), 0.0, 1.0
    )
    remove_voice_below_split = bool(config.get("removeVoiceBelowSplit", True))
    preserve_melody_below_split = bool(
        config.get("preserveMelodyBelowSplit", True)
    )

    from piano_arranger_adapter import (
        arrangement_role as source_arrangement_role,
        instrument_family as source_instrument_family,
        normalize_source_notes,
        selection_scores,
    )

    source_mode = str(config.get("sourceMode") or "frozen-baseline").strip().lower()
    if source_mode not in {"frozen-baseline", "raw-rebuild", "hybrid"}:
        raise ValueError("Unsupported left-hand accompaniment sourceMode.")
    normalized_source = normalize_source_notes(source_payload.get("notes", []))
    model_score_by_source_index: dict[int, float] = {}
    selection_model = config.get("selectionModel") or {}
    if selection_model:
        probabilities = selection_scores(
            normalized_source, {"selectionModel": selection_model}
        )
        model_score_by_source_index = {
            int(note["sourceIndex"]): float(probability)
            for note, probability in zip(normalized_source, probabilities)
        }
    onset_score_by_source_index: dict[int, float] = model_score_by_source_index
    onset_selection_model = config.get("onsetSelectionModel") or {}
    if onset_selection_model:
        onset_probabilities = selection_scores(
            normalized_source, {"selectionModel": onset_selection_model}
        )
        onset_score_by_source_index = {
            int(note["sourceIndex"]): float(probability)
            for note, probability in zip(normalized_source, onset_probabilities)
        }
    supplemental_score_by_source_index: dict[int, float] = {}
    supplemental_selection_model = config.get("supplementalSelectionModel") or {}
    if supplemental_selection_model:
        supplemental_probabilities = selection_scores(
            normalized_source,
            {"selectionModel": supplemental_selection_model},
        )
        supplemental_score_by_source_index = {
            int(note["sourceIndex"]): float(probability)
            for note, probability in zip(normalized_source, supplemental_probabilities)
        }
    chord_completion_score_by_source_index: dict[int, float] = {}
    chord_completion_model = config.get("chordCompletionModel") or {}
    if chord_completion_model:
        completion_probabilities = selection_scores(
            normalized_source, {"selectionModel": chord_completion_model}
        )
        chord_completion_score_by_source_index = {
            int(note["sourceIndex"]): float(probability)
            for note, probability in zip(normalized_source, completion_probabilities)
        }
    supporting_tone_score_by_source_index: dict[int, float] = {}
    supporting_tone_model = config.get("supportingToneSelectionModel") or {}
    supporting_tone_families = {
        str(value).strip().lower()
        for value in config.get("supportingToneSourceFamilies", ["guitar"])
        if str(value).strip()
    }
    supporting_tone_minimum_midi = int(
        config.get("supportingToneMinimumSourceMidi", 0)
    )
    supporting_tone_maximum_midi = int(
        config.get("supportingToneMaximumSourceMidi", 127)
    )
    if supporting_tone_model:
        supporting_probabilities = selection_scores(
            normalized_source, {"selectionModel": supporting_tone_model}
        )
        supporting_tone_score_by_source_index = {
            int(note["sourceIndex"]): float(probability)
            for note, probability in zip(normalized_source, supporting_probabilities)
            if supporting_tone_minimum_midi
            <= int(note["midi"])
            <= supporting_tone_maximum_midi
            and (
                not supporting_tone_families
                or source_instrument_family(str(note.get("instrument") or ""))
                in supporting_tone_families
            )
        }
    supporting_tone_share = clamp(
        float(config.get("supportingToneAlternativeShare", 1.0)), 0.0, 1.0
    )
    supporting_tone_chord_share = clamp(
        float(config.get("supportingToneChordCompletionShare", 0.0)), 0.0, 1.0
    )

    right_notes: list[dict[str, Any]] = []
    left_candidates: list[tuple[int, dict[str, Any]]] = []
    removed_low_voice = 0
    protected_low_melody = 0
    input_left_note_count = 0
    supplemental_candidates = 0

    def append_ranked_left(sequence: int, item: dict[str, Any]) -> None:
        note = dict(item)
        try:
            source_index = int(note.get("sourceIndex"))
        except (TypeError, ValueError):
            source_index = -1
        score_override = note.pop("_leftHandScoreOverride", None)
        learned_score = (
            float(score_override)
            if score_override is not None
            else model_score_by_source_index.get(
                source_index, float(note.get("selectionProbability", 0.5))
            )
        )
        onset_learned_score = onset_score_by_source_index.get(
            source_index, learned_score
        )
        base_score = float(note.get("selectionProbability", learned_score))
        duration_salience = min(1.0, float(note.get("duration", 0.2)) / 0.80)
        bass_anchor = 1.0 if (
            note.get("arrangementRole") == "bass"
            or str(note.get("sourceInstrument") or "").lower() in BASS_INSTRUMENTS
        ) else 0.0
        ranking_score = (
            0.76 * learned_score
            + 0.12 * base_score
            + 0.07 * duration_salience
            + float(config.get("bassAnchorBonus", 0.05)) * bass_anchor
        )
        onset_ranking_score = (
            0.76 * onset_learned_score
            + 0.12 * base_score
            + 0.07 * duration_salience
            + float(config.get("bassAnchorBonus", 0.05)) * bass_anchor
        )
        note["leftHandSelectionProbability"] = round_number(learned_score, 4)
        note["leftHandRankingScore"] = round_number(ranking_score, 4)
        note["leftHandOnsetSelectionProbability"] = round_number(
            onset_learned_score, 4
        )
        note["leftHandOnsetRankingScore"] = round_number(
            onset_ranking_score, 4
        )
        # Do not materialise a fallback field for legacy profiles. Downstream
        # code deliberately falls back to the established left-hand score;
        # writing a separately rounded substitute here can change deterministic
        # tie ordering even when an experimental blend is nominally zero.
        if supporting_tone_model:
            supporting_probability = supporting_tone_score_by_source_index.get(
                source_index, learned_score
            )
            supporting_ranking_score = (
                (1.0 - supporting_tone_share) * ranking_score
                + supporting_tone_share * supporting_probability
            )
            note["leftHandSupportingToneProbability"] = round_number(
                supporting_probability, 4
            )
            note["leftHandSupportingToneRankingScore"] = round_number(
                supporting_ranking_score, 4
            )
        left_candidates.append((sequence, note))

    # In both modes the accepted upper/melody layer is copied from the frozen
    # winner. Raw rebuild replaces only accompaniment notes below the split.
    for sequence, item in enumerate(notes):
        is_low_melody = (
            int(item["midi"]) < hand_split
            and (
                item.get("arrangementRole") == "melody"
                or str(item.get("sourceInstrument") or "").lower()
                in VOICE_INSTRUMENTS
            )
        )
        protected = int(item["midi"]) >= hand_split or (
            preserve_melody_below_split and is_low_melody
        )
        if protected:
            if is_low_melody:
                protected_low_melody += 1
            right_notes.append(item)
            continue
        input_left_note_count += 1
        if source_mode == "raw-rebuild":
            continue
        if (
            remove_voice_below_split
            and str(item.get("sourceInstrument") or "").lower() in VOICE_INSTRUMENTS
        ):
            removed_low_voice += 1
            continue
        append_ranked_left(sequence, item)

    frozen_candidate_source_indices = {
        int(note.get("sourceIndex"))
        for _sequence, note in left_candidates
        if note.get("sourceIndex") is not None
    }
    quota_basis_by_window: Counter[int] = Counter(
        int(float(note["time"]) // window_seconds)
        for _sequence, note in left_candidates
    )
    onset_quota_basis_by_window: Counter[int] = Counter()
    baseline_onset_windows: set[tuple[int, int]] = set()
    for _sequence, note in left_candidates:
        window = int(float(note["time"]) // window_seconds)
        onset = int(round(float(note["time"]) * 1000))
        baseline_onset_windows.add((window, onset))
    for window, _onset in baseline_onset_windows:
        onset_quota_basis_by_window[window] += 1

    if source_mode in {"raw-rebuild", "hybrid"}:
        rebuild_minimum = int(config.get("rebuildMinimumMidi", hand_split - 26))
        rebuild_maximum = int(config.get("rebuildMaximumMidi", hand_split - 2))
        rebuild_shift = int(config.get("registerShiftSemitones", 12))
        velocity_minimum, velocity_maximum = [
            float(value)
            for value in config.get("rebuildVelocityRange", [0.38, 0.50])
        ]
        supplemental_minimum_probability = clamp(
            float(
                config.get(
                    "supplementalMinimumProbability",
                    selection_model.get("threshold", 0.5),
                )
            ),
            0.0,
            1.0,
        )
        for raw in normalized_source:
            source_instrument = str(raw.get("instrument") or "").lower()
            if source_instrument in VOICE_INSTRUMENTS:
                continue
            midi = int(raw["midi"]) + rebuild_shift
            while midi > rebuild_maximum:
                midi -= 12
            while midi < rebuild_minimum:
                midi += 12
            if not rebuild_minimum <= midi <= rebuild_maximum:
                continue
            source_index = int(raw["sourceIndex"])
            learned_score = (
                supplemental_score_by_source_index.get(source_index, 0.5)
                if source_mode == "hybrid" and supplemental_score_by_source_index
                else model_score_by_source_index.get(source_index, 0.5)
            )
            if source_mode == "hybrid" and (
                source_index in frozen_candidate_source_indices
                or learned_score < supplemental_minimum_probability
            ):
                continue
            raw_role = source_arrangement_role(raw)
            role = "bass" if raw_role == "bass" else "harmony"
            velocity = velocity_minimum + learned_score * (
                velocity_maximum - velocity_minimum
            )
            rebuilt = {
                "duration": float(raw.get("duration", 0.2)),
                "hand": "left",
                "instrument": "acoustic_piano",
                "midi": midi,
                "note": midi_to_note(midi),
                "source": source_payload.get("transcriptionProvider")
                or source_payload.get("source")
                or "polymath-transcription",
                "time": float(raw["time"]),
                "velocity": round_number(
                    clamp(velocity, velocity_minimum, velocity_maximum), 4
                ),
                "sourceVelocityBeforeArrangement": round_number(
                    clamp(float(raw.get("velocity", 0.72)), 0.05, 1.0), 4
                ),
                "sourceIndex": source_index,
                "sourceMidiBeforeArrangement": int(raw["midi"]),
                "sourceInstrument": source_instrument,
                "arrangementRole": role,
                "voice": f"{role}-{midi}",
                "articulation": "legato",
                "maximumLegatoBridgeSeconds": (
                    DEFAULT_ROLE_LEGATO_BRIDGE_SECONDS.get(role, 1.20)
                ),
                "maximumPhysicalHoldSeconds": (
                    DEFAULT_ROLE_PHYSICAL_HOLD_SECONDS.get(role, 1.50)
                ),
                "selectionProbability": round_number(learned_score, 4),
                "originalMidiBeforeRangeShift": int(raw["midi"]),
                "originalNoteBeforeRangeShift": raw.get("note")
                or midi_to_note(int(raw["midi"])),
                "globalRegisterShiftSemitones": rebuild_shift,
                "octaveShiftSemitones": midi - int(raw["midi"]),
                "wasRegisterShifted": midi != int(raw["midi"]),
                "generatedBy": "learned-left-hand-raw-rebuild-v1",
                "_leftHandScoreOverride": learned_score,
            }
            append_ranked_left(len(notes) + source_index, rebuilt)
            if source_mode == "hybrid":
                supplemental_candidates += 1
        if source_mode == "raw-rebuild":
            input_left_note_count = len(left_candidates)
            quota_basis_by_window = Counter(
                int(float(note["time"]) // window_seconds)
                for _sequence, note in left_candidates
            )

    pre_quota_onset_removed = 0
    if source_mode == "hybrid":
        pre_quota_onsets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for pair in left_candidates:
            pre_quota_onsets[int(round(float(pair[1]["time"]) * 1000))].append(pair)
        capped_candidates: list[tuple[int, dict[str, Any]]] = []
        for onset_notes in pre_quota_onsets.values():
            if len(onset_notes) <= maximum_notes_per_onset:
                capped_candidates.extend(onset_notes)
                continue
            lowest = min(
                onset_notes, key=lambda pair: (int(pair[1]["midi"]), pair[0])
            )
            remainder = sorted(
                (pair for pair in onset_notes if pair[0] != lowest[0]),
                key=lambda pair: (
                    -float(
                        pair[1].get(
                            "leftHandSupportingToneRankingScore",
                            pair[1].get("leftHandRankingScore", 0.0),
                        )
                    ),
                    -float(pair[1].get("duration", 0.0)),
                    pair[0],
                ),
            )
            selected = [lowest, *remainder[: maximum_notes_per_onset - 1]]
            capped_candidates.extend(selected)
            pre_quota_onset_removed += len(onset_notes) - len(selected)
        left_candidates = capped_candidates

    def cap_onset_notes(
        onset_notes: list[tuple[int, dict[str, Any]]],
        limit: int | None = None,
    ) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        effective_limit = max(
            1,
            min(
                maximum_notes_per_onset,
                maximum_notes_per_onset if limit is None else int(limit),
            ),
        )
        if len(onset_notes) <= effective_limit:
            return onset_notes, 0
        lowest = min(onset_notes, key=lambda pair: (int(pair[1]["midi"]), pair[0]))
        remainder = sorted(
            (pair for pair in onset_notes if pair[0] != lowest[0]),
            key=lambda pair: (
                -float(
                    pair[1].get(
                        "leftHandSupportingToneRankingScore",
                        pair[1].get("leftHandRankingScore", 0.0),
                    )
                ),
                -float(pair[1].get("duration", 0.0)),
                pair[0],
            ),
        )
        selected = [lowest, *remainder[: effective_limit - 1]]
        return selected, len(onset_notes) - len(selected)

    # Preserve the song's changing intensity instead of enforcing one global
    # density. A reference can teach us to select rhythmic onsets first; this
    # keeps complete pianist gestures together and avoids scattering isolated
    # notes across too many guitar attacks.
    onset_first_selection = bool(config.get("onsetFirstSelection", False))
    configured_size_shares = config.get("targetOnsetSizeShares") or {}
    target_onset_size_shares = (
        {
            size: max(0.0, float(configured_size_shares.get(str(size), 0.0)))
            for size in range(1, maximum_notes_per_onset + 1)
        }
        if isinstance(configured_size_shares, dict)
        else {}
    )
    use_target_size_distribution = bool(
        config.get("enforceTargetNotesPerOnset", False)
        and sum(target_onset_size_shares.values()) > 0.0
    )
    retained: list[tuple[int, dict[str, Any]]] = []
    globally_selected_onsets: list[list[tuple[int, dict[str, Any]]]] = []
    quota_removed = 0
    onset_removed = pre_quota_onset_removed
    if onset_first_selection:
        all_onsets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for pair in left_candidates:
            all_onsets[int(round(float(pair[1]["time"]) * 1000))].append(pair)
        onset_windows: dict[int, list[list[tuple[int, dict[str, Any]]]]] = defaultdict(list)
        for onset_notes in all_onsets.values():
            window = int(float(onset_notes[0][1]["time"]) // window_seconds)
            onset_windows[window].append(onset_notes)
        onset_keep_ratio = clamp(
            float(config.get("targetOnsetKeepRatio", keep_ratio)), 0.1, 1.0
        )
        target_notes_per_onset = clamp(
            float(config.get("targetNotesPerOnset", 1.5)), 1.0, 2.0
        )
        enforce_target_notes_per_onset = bool(
            config.get("enforceTargetNotesPerOnset", False)
        )
        for window_onsets in onset_windows.values():
            window_id = int(
                float(window_onsets[0][0][1]["time"]) // window_seconds
            )
            quota_basis = onset_quota_basis_by_window.get(
                window_id, len(window_onsets)
            )
            quota = max(1, int(round(quota_basis * onset_keep_ratio)))
            quota = min(len(window_onsets), quota)

            def onset_rank(onset_notes: list[tuple[int, dict[str, Any]]]) -> tuple[float, float, int]:
                scores = [
                    float(
                        note.get(
                            "leftHandOnsetRankingScore",
                            note.get("leftHandRankingScore", 0.0),
                        )
                    )
                    for _sequence, note in onset_notes
                ]
                return (
                    0.68 * max(scores) + 0.32 * (sum(scores) / len(scores)),
                    max(float(note.get("duration", 0.0)) for _sequence, note in onset_notes),
                    -min(sequence for sequence, _note in onset_notes),
                )

            ranked_onsets = sorted(window_onsets, key=onset_rank, reverse=True)
            selected_onsets = ranked_onsets[:quota]
            quota_removed += sum(len(group) for group in ranked_onsets[quota:])
            if use_target_size_distribution:
                globally_selected_onsets.extend(selected_onsets)
                continue
            extra_note_quota = (
                int(round(len(selected_onsets) * (target_notes_per_onset - 1.0)))
                if enforce_target_notes_per_onset
                else len(selected_onsets)
            )

            def second_note_rank(
                onset_notes: list[tuple[int, dict[str, Any]]]
            ) -> tuple[float, float]:
                scores = sorted(
                    (
                        float(
                            note.get(
                                "leftHandSupportingToneRankingScore",
                                note.get("leftHandRankingScore", 0.0),
                            )
                        )
                        for _sequence, note in onset_notes
                    ),
                    reverse=True,
                )
                return (
                    scores[1] if len(scores) > 1 else -1.0,
                    scores[0],
                )

            two_note_onsets = {
                id(group)
                for group in sorted(
                    (group for group in selected_onsets if len(group) > 1),
                    key=second_note_rank,
                    reverse=True,
                )[:extra_note_quota]
            }
            for onset_notes in selected_onsets:
                selected, removed = cap_onset_notes(
                    onset_notes, 2 if id(onset_notes) in two_note_onsets else 1
                )
                retained.extend(selected)
                onset_removed += removed
        if use_target_size_distribution:
            desired_counts = {
                size: int(round(len(globally_selected_onsets) * share))
                for size, share in target_onset_size_shares.items()
                if size >= 2
            }
            allocations = {id(group): 1 for group in globally_selected_onsets}
            available_groups = list(globally_selected_onsets)

            def richness_rank(
                onset_notes: list[tuple[int, dict[str, Any]]], size: int
            ) -> tuple[float, float, int]:
                scores = sorted(
                    (
                        float(
                            note.get(
                                "leftHandSupportingToneRankingScore",
                                note.get("leftHandRankingScore", 0.0),
                            )
                        )
                        for _sequence, note in onset_notes
                    ),
                    reverse=True,
                )
                return (
                    scores[size - 1] if len(scores) >= size else -1.0,
                    scores[0],
                    -min(sequence for sequence, _note in onset_notes),
                )

            for size in range(maximum_notes_per_onset, 1, -1):
                eligible = [group for group in available_groups if len(group) >= size]
                selected_for_size = sorted(
                    eligible,
                    key=lambda group: richness_rank(group, size),
                    reverse=True,
                )[: max(0, desired_counts.get(size, 0))]
                selected_ids = {id(group) for group in selected_for_size}
                for group in selected_for_size:
                    allocations[id(group)] = size
                available_groups = [
                    group for group in available_groups if id(group) not in selected_ids
                ]
            for onset_notes in globally_selected_onsets:
                selected, removed = cap_onset_notes(
                    onset_notes, allocations[id(onset_notes)]
                )
                retained.extend(selected)
                onset_removed += removed
        onset_capped = retained
    else:
        by_window: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for sequence, note in left_candidates:
            window = int(float(note["time"]) // window_seconds)
            by_window[window].append((sequence, note))
        for window_notes in by_window.values():
            window_id = int(float(window_notes[0][1]["time"]) // window_seconds)
            quota_basis = quota_basis_by_window.get(window_id, len(window_notes))
            quota = max(
                minimum_notes_per_window,
                int(round(quota_basis * keep_ratio)),
            )
            quota = min(len(window_notes), quota)
            ranked = sorted(
                window_notes,
                key=lambda pair: (
                    -float(pair[1].get("leftHandRankingScore", 0.0)),
                    -float(pair[1].get("duration", 0.0)),
                    int(pair[1]["midi"]),
                    pair[0],
                ),
            )
            selected_sequences = {sequence for sequence, _note in ranked[:quota]}
            retained.extend(
                pair for pair in window_notes if pair[0] in selected_sequences
            )
            quota_removed += len(window_notes) - quota

        # Pianist-like accompaniment generally has one bass anchor plus at most
        # one supporting chord tone at a simultaneous strike.
        onset_groups: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for pair in retained:
            onset_groups[int(round(float(pair[1]["time"]) * 1000))].append(pair)
        onset_capped: list[tuple[int, dict[str, Any]]] = []
        for onset_notes in onset_groups.values():
            selected, removed = cap_onset_notes(onset_notes)
            onset_capped.extend(selected)
            onset_removed += removed

    # The accepted onset model decides *when* the accompaniment strikes.  A
    # separately trained raw-event ranker may then correct *which chord tones*
    # occupy those same slots.  This never adds an onset or changes the frozen
    # melody/right-hand layer.
    chord_completion_changes = 0
    chord_completion_examined = 0
    chord_completion_gains: list[float] = []
    chord_completion_preserved_octave_onsets = 0
    protected_spans_by_pitch: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for protected_note in right_notes:
        start = float(protected_note["time"])
        end = start + float(
            protected_note.get("scoreDuration", protected_note.get("duration", 0.2))
        )
        protected_spans_by_pitch[int(protected_note["midi"])].append((start, end))

    def conflicts_with_protected_sustain(pitch: int, onset: float) -> bool:
        return any(
            start <= onset < end + SAME_KEY_RELEASE_GAP_SECONDS
            for start, end in protected_spans_by_pitch.get(pitch, [])
        )

    if chord_completion_score_by_source_index:
        completion_radius = clamp(
            float(config.get("chordCompletionRadiusSeconds", 0.08)), 0.03, 0.35
        )
        existing_pitch_bonus = clamp(
            float(config.get("chordCompletionExistingPitchBonus", 0.035)),
            0.0,
            0.20,
        )
        cross_family_bonus = clamp(
            float(config.get("chordCompletionCrossFamilyBonus", 0.025)),
            0.0,
            0.15,
        )
        minimum_completion_gain = clamp(
            float(config.get("chordCompletionMinimumGain", 0.0)), 0.0, 1.0
        )
        interval_preferences = {
            int(interval) % 12: clamp(float(preference), 0.0, 1.0)
            for interval, preference in (
                config.get("chordCompletionIntervalPreferences") or {}
            ).items()
        }
        interval_prior_strength = clamp(
            float(config.get("chordCompletionIntervalPriorStrength", 0.0)),
            0.0,
            0.50,
        )
        preserve_octave_doublings = bool(
            config.get("chordCompletionPreserveOctaveDoublings", True)
        )
        source_times = [float(note["time"]) for note in normalized_source]
        groups: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for pair in onset_capped:
            groups[int(round(float(pair[1]["time"]) * 1000))].append(pair)
        completed: list[tuple[int, dict[str, Any]]] = []
        for onset_notes in groups.values():
            onset_notes.sort(key=lambda pair: (int(pair[1]["midi"]), pair[0]))
            onset_time = float(onset_notes[0][1]["time"])
            original_pitch_classes = {
                int(note["midi"]) % 12 for _sequence, note in onset_notes
            }
            # Pianella's left hand frequently doubles one chord root in two
            # octaves.  Treat that as a deliberate piano voicing, not as a
            # duplicate that must be replaced by a different pitch class.
            # The structured pitch-class metric otherwise looks improved while
            # the rendered accompaniment loses its strongest acoustic pattern.
            if (
                preserve_octave_doublings
                and len(onset_notes) >= 2
                and len(original_pitch_classes) == 1
            ):
                completed.extend(onset_notes)
                chord_completion_preserved_octave_onsets += 1
                continue
            left = bisect.bisect_left(source_times, onset_time - completion_radius)
            right = bisect.bisect_right(source_times, onset_time + completion_radius)
            raw_options: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for raw in normalized_source[left:right]:
                if str(raw.get("instrument") or "").lower() in VOICE_INSTRUMENTS:
                    continue
                raw_options[int(raw["midi"]) % 12].append(raw)
            if not raw_options:
                completed.extend(onset_notes)
                continue
            existing_pitch_classes = {
                int(note["midi"]) % 12 for _sequence, note in onset_notes
            }
            option_scores: dict[int, tuple[float, dict[str, Any]]] = {}
            for pitch_class, options in raw_options.items():
                families = {
                    str(option.get("instrument") or "other") for option in options
                }
                best = max(
                    options,
                    key=lambda option: chord_completion_score_by_source_index.get(
                        int(option["sourceIndex"]), 0.0
                    ),
                )
                score = chord_completion_score_by_source_index.get(
                    int(best["sourceIndex"]), 0.0
                )
                score += cross_family_bonus * min(3, max(0, len(families) - 1))
                if pitch_class in existing_pitch_classes:
                    score += existing_pitch_bonus
                option_scores[pitch_class] = (score, best)

            used_pitch_classes: set[int] = set()
            gesture_bass_pitch_class: int | None = None
            for slot, (sequence, item) in enumerate(onset_notes):
                note = dict(item)
                original_midi = int(note["midi"])
                original_pitch_class = original_midi % 12
                available = [
                    (score, pitch_class, raw)
                    for pitch_class, (score, raw) in option_scores.items()
                    if pitch_class not in used_pitch_classes
                ]
                if not available:
                    completed.append((sequence, note))
                    continue
                def completion_score(
                    option: tuple[float, int, dict[str, Any]]
                ) -> float:
                    score, pitch_class, raw = option
                    if slot > 0 and supporting_tone_score_by_source_index:
                        supporting_score = supporting_tone_score_by_source_index.get(
                            int(raw["sourceIndex"]), score
                        )
                        score = (
                            (1.0 - supporting_tone_chord_share) * score
                            + supporting_tone_chord_share * supporting_score
                        )
                    family = str(raw.get("instrument") or "").lower()
                    bass_bonus = (
                        0.10
                        if slot == 0
                        and (
                            family in BASS_INSTRUMENTS
                            or source_arrangement_role(raw) == "bass"
                        )
                        else 0.0
                    )
                    circular_distance = min(
                        (pitch_class - original_pitch_class) % 12,
                        (original_pitch_class - pitch_class) % 12,
                    )
                    interval_bonus = 0.0
                    if gesture_bass_pitch_class is not None:
                        interval = (pitch_class - gesture_bass_pitch_class) % 12
                        interval_bonus = interval_prior_strength * interval_preferences.get(
                            interval, 0.0
                        )
                    return (
                        score
                        + bass_bonus
                        + interval_bonus
                        - 0.012 * circular_distance
                    )

                selected_option = max(
                    available,
                    key=lambda option: (
                        completion_score(option),
                        -option[1],
                    ),
                )
                selected_score, selected_pitch_class, selected_raw = selected_option
                selected_rank_score = completion_score(selected_option)
                original_option = next(
                    (
                        option
                        for option in available
                        if option[1] == original_pitch_class
                    ),
                    None,
                )
                original_rank_score = (
                    completion_score(original_option)
                    if original_option is not None
                    else 0.0
                )
                gain = selected_rank_score - original_rank_score
                if (
                    selected_pitch_class != original_pitch_class
                    and gain < minimum_completion_gain
                ):
                    selected_pitch_class = original_pitch_class
                    selected_raw = (
                        original_option[2] if original_option is not None else selected_raw
                    )
                    selected_rank_score = original_rank_score
                    gain = 0.0
                used_pitch_classes.add(selected_pitch_class)
                if gesture_bass_pitch_class is None:
                    gesture_bass_pitch_class = selected_pitch_class
                # A singleton is not automatically a bass note.  Confining
                # every first slot to the low band pulled melodic left-hand
                # tones down an octave.  Only the lower member of a multi-note
                # gesture uses the bass band; singletons retain the complete
                # accompaniment register learned from the reference.
                is_bass_slot = len(onset_notes) > 1 and slot == 0
                minimum_midi = hand_split - (26 if is_bass_slot else 18)
                maximum_midi = hand_split - (8 if is_bass_slot else 2)
                pitch_options = [
                    midi
                    for midi in range(minimum_midi, maximum_midi + 1)
                    if midi % 12 == selected_pitch_class
                    and not conflicts_with_protected_sustain(midi, onset_time)
                ]
                if not pitch_options:
                    completed.append((sequence, note))
                    continue
                selected_midi = min(
                    pitch_options,
                    key=lambda midi: (abs(midi - original_midi), midi),
                )
                chord_completion_examined += 1
                if selected_midi != original_midi:
                    chord_completion_gains.append(gain)
                    note["leftHandChordOriginalMidi"] = original_midi
                    note["leftHandChordOriginalSourceIndex"] = note.get("sourceIndex")
                    note["leftHandChordCompletionSourceIndex"] = int(
                        selected_raw["sourceIndex"]
                    )
                    note["leftHandChordCompletionScore"] = round_number(
                        chord_completion_score_by_source_index.get(
                            int(selected_raw["sourceIndex"]), 0.0
                        ),
                        4,
                    )
                    note["midi"] = selected_midi
                    note["note"] = midi_to_note(selected_midi)
                    note["sourceIndex"] = int(selected_raw["sourceIndex"])
                    note["sourceInstrument"] = str(
                        selected_raw.get("instrument") or "other"
                    )
                    note["sourceVelocityBeforeArrangement"] = round_number(
                        clamp(
                            float(selected_raw.get("velocity", 0.72)), 0.05, 1.0
                        ),
                        4,
                    )
                    role = str(note.get("arrangementRole") or "harmony")
                    note["voice"] = f"{role}-{selected_midi}"
                    chord_completion_changes += 1
                completed.append((sequence, note))
        onset_capped = completed

    source_duration_quantiles = [
        float(value) for value in config.get("sourceDurationQuantiles", [])
    ]
    target_duration_quantiles = [
        float(value) for value in config.get("targetDurationQuantiles", [])
    ]
    duration_mapped = 0
    shaped_left: list[tuple[int, dict[str, Any]]] = []
    for sequence, item in onset_capped:
        note = dict(item)
        mapped = _piecewise_quantile_map(
            float(note.get("duration", MIN_NOTE_SECONDS)),
            source_duration_quantiles,
            target_duration_quantiles,
        )
        mapped = clamp(mapped, MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)
        if abs(mapped - float(note.get("duration", mapped))) >= 0.005:
            duration_mapped += 1
        note["sourceDurationBeforeLeftHandStyle"] = round_number(
            float(note.get("duration", MIN_NOTE_SECONDS)), 6
        )
        note["duration"] = round_number(mapped)
        note["scoreDuration"] = note["duration"]
        note["visualDuration"] = note["duration"]
        note.pop("audioDuration", None)
        note.pop("releaseSeconds", None)
        shaped_left.append((sequence, note))

    # Keep only a learned small share of rapid background restrikes. For a
    # harmony note, first try to turn a redundant same-key guitar attack into a
    # different, simultaneously supported chord tone. This retains fast-song
    # energy as a pianist-like broken chord instead of deleting the rhythm or
    # producing a machine-gun repetition of one sample.
    by_pitch: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for pair in shaped_left:
        by_pitch[int(pair[1]["midi"])].append(pair)
    rapid_candidates: list[tuple[float, int]] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda pair: (float(pair[1]["time"]), pair[0]))
        for previous, current in zip(pitch_notes, pitch_notes[1:]):
            gap = float(current[1]["time"]) - float(previous[1]["time"])
            if DEFAULT_DUPLICATE_ONSET_SECONDS <= gap < minimum_same_pitch_gap:
                rapid_candidates.append(
                    (float(current[1].get("leftHandRankingScore", 0.0)), current[0])
                )
    allowed_rapid_count = int(round(len(rapid_candidates) * fast_retrigger_keep_share))
    allowed_rapid_sequences = {
        sequence
        for _score, sequence in sorted(
            rapid_candidates, key=lambda item: (-item[0], item[1])
        )[:allowed_rapid_count]
    }
    revoice_rapid_retriggers = bool(config.get("revoiceRapidRetriggers", True))
    accompaniment_floor = max(PIANO_MIN_MIDI, hand_split - 24)
    accompaniment_ceiling = hand_split - 2
    support_bucket_seconds = 0.05
    support_radius_buckets = max(
        0,
        int(
            round(
                clamp(
                    float(config.get("rapidRevoiceSupportRadiusSeconds", 0.35)),
                    0.0,
                    0.60,
                )
                / support_bucket_seconds
            )
        ),
    )
    support_by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    occupied_by_onset: dict[int, set[int]] = defaultdict(set)
    for support in notes:
        if (
            support.get("arrangementRole") != "melody"
            and str(support.get("sourceInstrument") or "").lower()
            not in VOICE_INSTRUMENTS
        ):
            support_by_bucket[
                int(round(float(support["time"]) / support_bucket_seconds))
            ].append(support)
    for _sequence, note in shaped_left:
        occupied_by_onset[int(round(float(note["time"]) * 1000))].add(
            int(note["midi"])
        )

    def supported_alternate_pitch(
        note: dict[str, Any], previous_pitch: int
    ) -> int | None:
        if not revoice_rapid_retriggers or note.get("arrangementRole") == "bass":
            return None
        onset = int(round(float(note["time"]) * 1000))
        bucket = int(round(float(note["time"]) / support_bucket_seconds))
        original_pitch = int(note["midi"])
        candidates: list[tuple[float, int]] = []
        nearby_support = [
            support
            for offset in range(-support_radius_buckets, support_radius_buckets + 1)
            for support in support_by_bucket.get(bucket + offset, [])
        ]
        for support in nearby_support:
            pitch = int(support["midi"])
            while pitch >= hand_split:
                pitch -= 12
            while pitch < accompaniment_floor:
                pitch += 12
            if not accompaniment_floor <= pitch <= accompaniment_ceiling:
                continue
            if (
                pitch in {original_pitch, previous_pitch}
                or pitch in occupied_by_onset[onset]
                or conflicts_with_protected_sustain(pitch, float(note["time"]))
            ):
                continue
            support_score = float(support.get("selectionProbability", 0.5))
            time_distance = abs(float(support["time"]) - float(note["time"]))
            # Prefer a nearby chord tone and the upper half of the left-hand
            # range; very large jumps sound like another bass player entering.
            movement_penalty = abs(pitch - original_pitch) / 24.0
            register_bonus = (pitch - accompaniment_floor) / max(
                1.0, accompaniment_ceiling - accompaniment_floor
            )
            candidates.append(
                (
                    support_score
                    - 0.16 * movement_penalty
                    - 0.22 * time_distance
                    + 0.04 * register_bonus,
                    pitch,
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]

    retriggers_merged = 0
    retriggers_revoiced = 0
    smoothed_left: list[tuple[int, dict[str, Any]]] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda pair: (float(pair[1]["time"]), pair[0]))
        merged: list[tuple[int, dict[str, Any]]] = []
        for sequence, item in pitch_notes:
            note = dict(item)
            if merged:
                previous_sequence, previous = merged[-1]
                gap = float(note["time"]) - float(previous["time"])
                if (
                    DEFAULT_DUPLICATE_ONSET_SECONDS <= gap < minimum_same_pitch_gap
                    and sequence not in allowed_rapid_sequences
                ):
                    alternate_pitch = supported_alternate_pitch(
                        note, int(previous["midi"])
                    )
                    if alternate_pitch is not None:
                        onset = int(round(float(note["time"]) * 1000))
                        occupied_by_onset[onset].discard(int(note["midi"]))
                        occupied_by_onset[onset].add(alternate_pitch)
                        note["leftHandRevoicedFromMidi"] = int(note["midi"])
                        note["leftHandRevoiceSemitones"] = (
                            alternate_pitch - int(note["midi"])
                        )
                        note["midi"] = alternate_pitch
                        note["note"] = midi_to_note(alternate_pitch)
                        note["voice"] = f"harmony-{alternate_pitch}"
                        note["hand"] = "left"
                        note["arrangementRole"] = "harmony"
                        retriggers_revoiced += 1
                    else:
                        # The mapped duration already carries natural resonance;
                        # do not extend it across several attacks and erase the
                        # short-articulation tail learned from the reference.
                        previous["leftHandMergedRetrigger"] = True
                        merged[-1] = (previous_sequence, previous)
                        retriggers_merged += 1
                        continue
            merged.append((sequence, note))
        smoothed_left.extend(merged)

    output = sorted(
        [*right_notes, *(note for _sequence, note in smoothed_left)],
        key=lambda note: (float(note["time"]), int(note["midi"])),
    )
    output_left = sum(1 for note in output if int(note["midi"]) < hand_split)
    return output, {
        "applied": True,
        "profile": str(config.get("profile") or "learned-left-hand-v1"),
        "rightHandFrozen": True,
        "sourceMode": source_mode,
        "handSplitMidi": hand_split,
        "inputRightHandNotes": len(right_notes),
        "inputLeftHandNotes": input_left_note_count,
        "scoredLeftHandNotes": len(left_candidates),
        "removedVoiceBelowSplit": removed_low_voice,
        "protectedMelodyBelowSplit": protected_low_melody,
        "targetKeepRatio": round_number(keep_ratio, 4),
        "onsetFirstSelection": onset_first_selection,
        "targetOnsetKeepRatio": (
            round_number(float(config.get("targetOnsetKeepRatio", keep_ratio)), 4)
            if onset_first_selection
            else None
        ),
        "targetNotesPerOnset": (
            round_number(float(config.get("targetNotesPerOnset", 1.5)), 4)
            if onset_first_selection
            else None
        ),
        "enforceTargetNotesPerOnset": (
            bool(config.get("enforceTargetNotesPerOnset", False))
            if onset_first_selection
            else None
        ),
        "targetOnsetSizeShares": (
            {
                str(size): round_number(share, 6)
                for size, share in target_onset_size_shares.items()
            }
            if use_target_size_distribution
            else None
        ),
        "windowSeconds": round_number(window_seconds, 3),
        "quotaRemovedNotes": quota_removed,
        "onsetCapRemovedNotes": onset_removed,
        "durationMappedNotes": duration_mapped,
        "rapidRetriggerCandidates": len(rapid_candidates),
        "rapidRetriggersPreserved": len(allowed_rapid_sequences),
        "rapidRetriggersRevoiced": retriggers_revoiced,
        "rapidRetriggersMerged": retriggers_merged,
        "outputLeftHandNotes": output_left,
        "outputRightHandNotes": len(right_notes),
        "selectionModelApplied": bool(selection_model),
        "onsetSelectionModelApplied": bool(onset_selection_model),
        "supplementalSelectionModelApplied": bool(supplemental_selection_model),
        "supplementalCandidates": supplemental_candidates,
        "chordCompletionModelApplied": bool(chord_completion_model),
        "supportingToneSelectionModelApplied": bool(supporting_tone_model),
        "supportingToneAlternativeShare": (
            round_number(supporting_tone_share, 4)
            if supporting_tone_model
            else None
        ),
        "supportingToneChordCompletionShare": (
            round_number(supporting_tone_chord_share, 4)
            if supporting_tone_model
            else None
        ),
        "supportingToneSourceFamilies": (
            sorted(supporting_tone_families) if supporting_tone_model else None
        ),
        "supportingToneMinimumSourceMidi": (
            supporting_tone_minimum_midi if supporting_tone_model else None
        ),
        "supportingToneMaximumSourceMidi": (
            supporting_tone_maximum_midi if supporting_tone_model else None
        ),
        "chordCompletionExaminedNotes": chord_completion_examined,
        "chordCompletionChangedNotes": chord_completion_changes,
        "chordCompletionMinimumGain": (
            round_number(float(config.get("chordCompletionMinimumGain", 0.0)), 4)
            if chord_completion_model
            else None
        ),
        "chordCompletionMedianAcceptedGain": (
            round_number(quantile(chord_completion_gains, 0.50), 4)
            if chord_completion_gains
            else 0.0
        ),
        "chordCompletionPreservedOctaveOnsets": (
            chord_completion_preserved_octave_onsets
        ),
        "chordCompletionIntervalPriorStrength": (
            round_number(
                float(config.get("chordCompletionIntervalPriorStrength", 0.0)),
                4,
            )
            if chord_completion_model
            else None
        ),
    }


def arrange_payload(
    payload: dict[str, Any],
    mode: str = "instrumental",
    style_profile: dict[str, Any] | None = None,
    *,
    allow_pure_piano_density_override: bool = True,
) -> dict[str, Any]:
    source_notes = [
        normalized
        for note in payload.get("notes", [])
        if (normalized := normalize_source_note(note)) is not None
    ]
    if not source_notes:
        raise ValueError("No notes inside the real 88-key piano range were available to arrange.")

    requested_style_profile = style_profile
    profile = source_profile(
        source_notes,
        payload.get("transcriptionCleanup"),
        allow_pure_piano_density_override=allow_pure_piano_density_override,
    )
    conditional_route_diagnostics: dict[str, Any] = {}
    if style_profile:
        style_profile, conditional_route_diagnostics = (
            route_conditional_learned_profile(style_profile, profile)
        )
    learned_profile_bypass_reason = None
    style_decoder = (style_profile or {}).get("decoder") or {}
    monophonic_cleanup_config = style_decoder.get("monophonicVocalCleanup") or {}
    source_notes, monophonic_vocal_cleanup = suppress_monophonic_vocal_floor_runs(
        source_notes,
        monophonic_cleanup_config,
    )
    if not source_notes:
        raise ValueError("Monophonic vocal cleanup removed every playable note.")
    performance_only_profile = bool(style_decoder.get("performanceOnly", False))
    default_pipeline_profile = bool(style_decoder.get("defaultPipeline", False))
    if (
        (performance_only_profile or default_pipeline_profile)
        and style_profile.get("schema") != "polymath-piano-arranger-profile-v1"
    ):
        raise ValueError("Unsupported default-pipeline piano arranger profile.")
    default_pipeline_piano_limit = clamp(
        float(style_decoder.get("maximumSourcePianoRatioForProfile", 0.08)),
        0.0,
        1.0,
    )
    default_pipeline_piano_preserved = bool(
        default_pipeline_profile
        and float(profile.get("pianoRatio", 0.0)) > default_pipeline_piano_limit
    )
    if default_pipeline_piano_preserved:
        # A mixed score with a meaningful detected-piano layer already carries
        # instrument-specific voicing and touch.  The cross-song full-mix
        # profile was trained mainly on non-piano stems and regressed on the
        # locked 22 holdout.  Route such inputs through the frozen production
        # decoder instead of forcing its guitar/melody/touch priors.
        learned_profile_bypass_reason = "meaningful-piano-source-preserved"
        style_profile = None
        style_decoder = {}
        performance_only_profile = False
        default_pipeline_profile = False
    if performance_only_profile:
        learned_profile_bypass_reason = (
            "genuine-solo-piano-preserved"
            if profile["detectedAcousticPianoPerformance"]
            else "selection-frozen-performance-only-profile"
        )
    elif default_pipeline_profile:
        learned_profile_bypass_reason = "default-pipeline-profile-applied"
    elif style_profile and profile["detectedAcousticPianoPerformance"]:
        learned_profile_bypass_reason = "genuine-solo-piano-preserved"
    if (
        style_profile
        and not performance_only_profile
        and not default_pipeline_profile
        and not profile["detectedAcousticPianoPerformance"]
    ):
        # MuScriptor has already detected the instruments. The explicit Piano
        # route now hands that factual score to a separate supervised arranger;
        # Band never calls this stage. A genuine solo-piano performance remains
        # on the preservation path below so we do not rewrite a pianist who is
        # already playing the requested instrument.
        from piano_arranger_adapter import arrange_with_profile

        effective_style_profile, density_adaptation = adapt_profile_to_source_density(
            style_profile, profile
        )
        learned = arrange_with_profile(payload, mode, effective_style_profile)
        learned["pianoArrangement"]["sourceProfile"] = profile
        if density_adaptation is not None:
            learned["pianoArrangement"]["sourceDensityAdaptation"] = density_adaptation
        learned["notes"], focused_melody_collisions = (
            collapse_focused_melody_collisions(learned["notes"])
        )
        effective_decoder = effective_style_profile.get("decoder") or {}
        authored_two_hand_style = effective_decoder.get("authoredTwoHandStyle") or {}
        if authored_two_hand_style.get("enabled"):
            learned["notes"], range_compaction = compact_authored_two_hand_register(
                learned["notes"], authored_two_hand_style
            )
        else:
            preferred_shift = int(
                round(
                    float(
                        effective_decoder.get(
                            "preferredGlobalRegisterShiftSemitones",
                            PIANELLA_PREFERRED_GLOBAL_SHIFT,
                        )
                    )
                )
            )
            learned["notes"], range_compaction = compact_pianella_register(
                learned["notes"], preferred_shift
            )
        learned["notes"], range_fold_collisions = merge_phrase_retriggers(learned["notes"])
        learned["notes"], authored_duration_style = apply_authored_duration_style(
            learned["notes"], authored_two_hand_style
        )
        physical_limits = learned_physical_performance_limits(
            effective_style_profile
        )
        for note in learned["notes"]:
            role = str(note.get("arrangementRole") or "harmony")
            note["scoreDuration"] = note["duration"]
            note["visualDuration"] = note["duration"]
            note["voice"] = (
                "melody"
                if role == "melody"
                else f"{role}-{int(note['midi'])}"
                if authored_two_hand_style.get("enabled")
                else role
                if role == "bass"
                else f"harmony-{int(note['midi'])}"
            )
            note.setdefault("articulation", "legato")
            note["maximumLegatoBridgeSeconds"] = physical_limits[
                "maximumLegatoBridgeSeconds"
            ].get(role, 1.20)
            note["maximumPhysicalHoldSeconds"] = physical_limits[
                "maximumPhysicalHoldSeconds"
            ].get(role, 1.50)
            note.pop("audioDuration", None)
            note.pop("releaseSeconds", None)
        learned["notes"], expression = shape_melody_forward_expression(
            learned["notes"], effective_style_profile
        )
        learned["notes"], left_hand_accompaniment = refine_left_hand_accompaniment(
            learned["notes"], payload, effective_style_profile
        )
        (
            learned["notes"],
            simultaneous_left_hand_duplicates,
        ) = collapse_exact_left_hand_duplicates(learned["notes"])
        retrigger_diagnostics = annotate_musical_retriggers(learned["notes"])
        learned["notes"], gesture_expression = shape_gesture_coherent_expression(
            learned["notes"], effective_style_profile
        )
        learned = shape_piano_performance(learned, infer_pedal=True)
        learned = synchronize_generated_visual_holds(learned)
        adaptive_mix_config = (
            (effective_style_profile.get("decoder") or {})
            .get("gestureDynamics", {})
            .get("adaptiveMelodyMixBalance", {})
        )
        learned["notes"], adaptive_mix = adaptive_melody_mix_balance(
            learned["notes"], adaptive_mix_config
        )
        learned["performance"]["profile"] = "polymath-learned-piano-arranger-v2"
        learned["performance"]["defaultAutoplayReleaseSeconds"] = 0.62
        learned["performance"]["melodyForwardDynamics"] = True
        learned["performance"]["authoredTwoHandStyle"] = bool(
            authored_two_hand_style.get("enabled")
        )
        learned["pianoArrangement"]["version"] = 6
        learned["pianoArrangement"]["expression"] = expression
        learned["pianoArrangement"]["gestureExpression"] = gesture_expression
        learned["pianoArrangement"]["adaptiveMelodyMixBalance"] = adaptive_mix
        learned["pianoArrangement"]["compactRange"] = range_compaction
        learned["pianoArrangement"]["leftHandAccompaniment"] = (
            left_hand_accompaniment
        )
        learned["pianoArrangement"][
            "collapsedSimultaneousLeftHandStrikes"
        ] = simultaneous_left_hand_duplicates
        learned["pianoArrangement"]["authoredDurationStyle"] = (
            authored_duration_style
        )
        learned["pianoArrangement"]["rangeFoldCollisionsRemoved"] = range_fold_collisions
        learned["pianoArrangement"]["retriggerPolicy"] = {
            "profile": "source-aware-natural-resonance-v1",
            "duplicateOnsetSeconds": DEFAULT_DUPLICATE_ONSET_SECONDS,
            "artificialCollisionWindowSeconds": DEFAULT_COLLISION_WINDOW_SECONDS,
            **retrigger_diagnostics,
        }
        learned["pianoArrangement"]["focusedMelodyOnsetCollisionsRemoved"] = (
            focused_melody_collisions
        )
        final_role_counts = Counter(
            note.get("arrangementRole", "harmony") for note in learned["notes"]
        )
        final_vocal_melody_notes = sum(
            1
            for note in learned["notes"]
            if note.get("sourceInstrument") in VOICE_INSTRUMENTS
            and note.get("arrangementRole") == "melody"
        )
        learned["pianoArrangement"]["outputNoteCount"] = len(learned["notes"])
        final_duration = max(
            (
                float(note["time"]) + float(note.get("duration", 0.0))
                for note in learned["notes"]
            ),
            default=0.0,
        )
        learned["pianoArrangement"]["outputNotesPerSecond"] = round_number(
            len(learned["notes"]) / max(1.0, final_duration), 3
        )
        learned["pianoArrangement"]["roleCounts"] = dict(
            sorted(final_role_counts.items())
        )
        learned["pianoArrangement"]["vocalMelodyNotes"] = final_vocal_melody_notes
        learned["vocalMelodyIncluded"] = final_vocal_melody_notes > 0
        learned["pianoArrangement"]["physicalPerformanceRoleLimits"] = (
            physical_limits
        )
        learned["pianoRangeNormalization"] = range_compaction
        learned["pianoArrangement"]["physicalPerformance"] = {
            **(learned.get("pianoPerformance") or {}),
            "profile": "written-key-hold-damper-v1",
        }
        if conditional_route_diagnostics:
            learned["pianoArrangement"]["conditionalLearnedRoute"] = (
                conditional_route_diagnostics
            )
        if monophonic_cleanup_config:
            learned["pianoArrangement"]["monophonicVocalCleanup"] = (
                monophonic_vocal_cleanup
            )
        return learned
    percussion_removed = sum(
        1 for note in source_notes if note["instrument"] in PERCUSSION_INSTRUMENTS
    )
    working = [
        note for note in source_notes if note["instrument"] not in PERCUSSION_INSTRUMENTS
    ]
    role_notes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if profile["detectedAcousticPianoPerformance"]:
        for source in working:
            role_notes["source_piano"].append(
                arranged_note(
                    source,
                    midi=source["midi"],
                    role="source_piano",
                    velocity=clamp(source["velocity"], 0.25, 0.95),
                    duration_minimum=MIN_NOTE_SECONDS,
                    duration_maximum=MAX_NOTE_SECONDS,
                )
            )
        arranger_profile = "acoustic-piano-preserve"
        density_limit = int(
            profile.get(
                "directPianoDensityLimitNotesPerSecond",
                MAX_DIRECT_PIANO_NOTES_PER_SECOND,
            )
        )
    else:
        voice_notes = [
            note for note in working if note["instrument"] in VOICE_INSTRUMENTS
        ]
        if mode == "full" and voice_notes:
            role_notes["melody"] = select_lead(
                # Keep sung syllable changes while still collapsing simultaneous
                # pitch alternatives emitted by the separator.
                voice_notes, window_seconds=0.045
            )
        else:
            explicit_leads = [
                note for note in working if note["instrument"] in LEAD_INSTRUMENTS
            ]
            fallback_leads = [
                note
                for note in working
                if note["instrument"] not in BASS_INSTRUMENTS
                and note["instrument"] not in VOICE_INSTRUMENTS
                and note["midi"] >= 60
            ]
            source_duration = max(
                (note["time"] + note["duration"] for note in working),
                default=0,
            )
            minimum_explicit_leads = max(16, math.ceil(source_duration * 0.20))
            role_notes["melody"] = select_lead(
                explicit_leads
                if len(explicit_leads) >= minimum_explicit_leads
                else fallback_leads,
                window_seconds=0.16,
            )

        explicit_bass = [
            note for note in working if note["instrument"] in BASS_INSTRUMENTS
        ]
        fallback_bass = [
            note
            for note in working
            if note["instrument"] not in VOICE_INSTRUMENTS and note["midi"] <= 52
        ]
        role_notes["bass"] = select_bass(explicit_bass or fallback_bass)

        harmony_sources = [
            note
            for note in working
            if note["instrument"] not in VOICE_INSTRUMENTS
            and note["instrument"] not in BASS_INSTRUMENTS
        ]
        default_full_mix = (
            style_decoder.get("defaultFullMix")
            if default_pipeline_profile
            else {}
        ) or {}
        harmony_window_seconds = clamp(
            float(default_full_mix.get("harmonyWindowSeconds", 0.30)),
            0.06,
            0.40,
        )
        maximum_harmony_pitch_classes = int(
            clamp(
                int(
                    default_full_mix.get(
                        "maximumHarmonyPitchClasses",
                        MAX_HARMONY_PITCH_CLASSES,
                    )
                ),
                1,
                4,
            )
        )
        cyclic_harmony_config = default_full_mix.get("sourceSupportedCyclicHarmony") or {}
        cyclic_harmony: dict[str, Any] = {
            "applied": False,
            "profile": "source-supported-cyclic-harmony-v1",
            "reason": "disabled",
        }
        if cyclic_harmony_config.get("enabled"):
            effective_cyclic_harmony_config = dict(cyclic_harmony_config)
            effective_cyclic_harmony_config["_sourcePianoRatio"] = float(
                profile.get("pianoRatio", 0.0)
            )
            effective_cyclic_harmony_config["_bassContextNotes"] = list(
                explicit_bass or fallback_bass
            )
            effective_cyclic_harmony_config["_preferredOnsets"] = sorted(
                {
                    float(note["time"])
                    for role in ("melody", "bass")
                    for note in role_notes.get(role, [])
                }
            )
            effective_cyclic_harmony_config["_melodyOnsets"] = sorted(
                {
                    float(note["time"])
                    for note in role_notes.get("melody", [])
                }
            )
            role_notes["harmony"], cyclic_harmony = source_supported_cyclic_harmony(
                harmony_sources,
                effective_cyclic_harmony_config,
            )
        else:
            role_notes["harmony"] = compact_harmony(
                harmony_sources,
                window_seconds=harmony_window_seconds,
                maximum_pitch_classes=maximum_harmony_pitch_classes,
            )
        arranger_profile = "full-mix-piano-reduction"
        density_limit = MAX_ARRANGED_NOTES_PER_SECOND

    clustered, cluster_removed = limit_onset_clusters(
        [note for notes in role_notes.values() for note in notes]
    )
    clustered_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in clustered:
        clustered_by_role[note["arrangementRole"]].append(note)
    density_limited, density_removed = limit_density(
        clustered_by_role, density_limit
    )
    if arranger_profile == 'full-mix-piano-reduction':
        legato_notes, legato_extended = shape_legato(density_limited)
        arranged_notes, retriggers_removed = merge_phrase_retriggers(legato_notes)
    else:
        legato_extended = 0
        arranged_notes, retriggers_removed = suppress_rapid_retriggers(
            density_limited
        )

    authored_two_hand_style = style_decoder.get("authoredTwoHandStyle") or {}
    if (
        arranger_profile == "full-mix-piano-reduction"
        and authored_two_hand_style.get("enabled")
    ):
        arranged_notes, range_compaction = compact_authored_two_hand_register(
            arranged_notes, authored_two_hand_style
        )
    else:
        arranged_notes, range_compaction = compact_pianella_register(arranged_notes)
    melody_register_policy = (
        default_full_mix.get("adaptiveMelodyRegisterSeparation") or {}
        if arranger_profile == "full-mix-piano-reduction" and default_pipeline_profile
        else {}
    )
    arranged_notes, melody_register_separation = adapt_melody_register_separation(
        arranged_notes,
        melody_register_policy,
    )
    if arranger_profile == "full-mix-piano-reduction":
        arranged_notes, range_fold_collisions = merge_phrase_retriggers(arranged_notes)
    else:
        arranged_notes, range_fold_collisions = suppress_rapid_retriggers(arranged_notes)
    retriggers_removed += range_fold_collisions
    arranged_notes, expression = shape_melody_forward_expression(arranged_notes)
    gesture_expression: dict[str, Any] = {
        "applied": False,
        "profile": "gesture-coherent-piano-dynamics-v1",
        "reason": "performance-only-profile-disabled",
    }
    if (
        (performance_only_profile or default_pipeline_profile)
        and arranger_profile == "full-mix-piano-reduction"
    ):
        arranged_notes, gesture_expression = shape_gesture_coherent_expression(
            arranged_notes, style_profile
        )
    if arranger_profile == "acoustic-piano-preserve":
        arranged_notes, direct_piano_hold_stabilization = (
            stabilize_direct_piano_score_holds(arranged_notes)
        )
    else:
        direct_piano_hold_stabilization = {
            "applied": False,
            "profile": "direct-piano-short-hold-stabilizer-v2",
            "reason": "not-source-piano-preservation-route",
        }
    retrigger_diagnostics = annotate_musical_retriggers(arranged_notes)

    vocal_melody_notes = sum(
        1
        for note in arranged_notes
        if note.get("sourceInstrument") in VOICE_INSTRUMENTS
        and note.get("arrangementRole") == "melody"
    )
    role_counts = Counter(note["arrangementRole"] for note in arranged_notes)
    # Source events can already carry physical-performance fields from an
    # earlier processing pass.  The arranger has now changed their musical
    # duration, so stale holds must not overwrite the new score during the
    # final pianist pass.
    physical_limits = learned_physical_performance_limits(
        style_profile
        if performance_only_profile or default_pipeline_profile
        else None
    )
    for note in arranged_notes:
        note["scoreDuration"] = note["duration"]
        note["visualDuration"] = note["duration"]
        if arranger_profile == "full-mix-piano-reduction":
            role = str(note.get("arrangementRole") or "harmony")
            note["voice"] = (
                role
                if role in {"melody", "bass"}
                else f"harmony-{int(note['midi'])}"
            )
            note.setdefault("articulation", "legato")
            note["maximumLegatoBridgeSeconds"] = physical_limits[
                "maximumLegatoBridgeSeconds"
            ].get(role, 1.20)
            note["maximumPhysicalHoldSeconds"] = physical_limits[
                "maximumPhysicalHoldSeconds"
            ].get(role, 1.50)
        note.pop("audioDuration", None)
        note.pop("releaseSeconds", None)
    output = dict(payload)
    output["instrument"] = "piano"
    output["notes"] = arranged_notes
    output["pianoRangeNormalization"] = range_compaction
    output["instrumentGroups"] = ["acoustic_piano"]
    output["vocalMelodyIncluded"] = vocal_melody_notes > 0
    output["arrangementProfile"] = "piano-reduction-with-physical-performance-v5"
    output["performance"] = {
        **(payload.get("performance") or {}),
        "profile": "polymath-piano-arranger-v1",
        "arrangerProfile": arranger_profile,
        "preserveScoreDurations": True,
        "sameKeyRetriggerGapSeconds": SAME_KEY_RELEASE_GAP_SECONDS,
        "defaultAutoplayReleaseSeconds": 0.42,
    }
    output["pianoArrangement"] = {
        "version": 1,
        "profile": arranger_profile,
        "mode": mode,
        "sourceNoteCount": len(payload.get("notes", [])),
        "normalizedSourceNoteCount": len(source_notes),
        "outputNoteCount": len(arranged_notes),
        "outputNotesPerSecond": round_number(notes_per_second(arranged_notes), 3),
        "outputMaximumOnsetCluster": maximum_onset_cluster(arranged_notes),
        "pianoRange": {
            "minimumMidi": COMPACT_PIANO_MIN_MIDI,
            "maximumMidi": COMPACT_PIANO_MAX_MIDI,
            "minimumNote": "A1",
            "maximumNote": "C8",
        },
        "compactRange": range_compaction,
        "adaptiveMelodyRegisterSeparation": melody_register_separation,
        "rangeFoldCollisionsRemoved": range_fold_collisions,
        "densityLimitNotesPerSecond": density_limit,
        "maximumOnsetCluster": MAX_ONSET_CLUSTER,
        "minimumSameKeyRetriggerMs": round(DEFAULT_DUPLICATE_ONSET_SECONDS * 1000),
        "removedPercussionNotes": percussion_removed,
        "removedForOnsetClusterLimit": cluster_removed,
        "removedForDensityLimit": density_removed,
        "removedRapidRetriggers": retriggers_removed,
        "retriggerPolicy": {
            "profile": "source-aware-natural-resonance-v1",
            "duplicateOnsetSeconds": DEFAULT_DUPLICATE_ONSET_SECONDS,
            "artificialCollisionWindowSeconds": DEFAULT_COLLISION_WINDOW_SECONDS,
            **retrigger_diagnostics,
        },
        "vocalMelodyNotes": vocal_melody_notes,
        "roleCounts": dict(sorted(role_counts.items())),
        "gestureExpression": gesture_expression,
        "sourceSupportedCyclicHarmony": (
            cyclic_harmony
            if arranger_profile == "full-mix-piano-reduction"
            else {
                "applied": False,
                "profile": "source-supported-cyclic-harmony-v1",
                "reason": "source-piano-preservation-route",
            }
        ),
        "directPianoHoldStabilization": direct_piano_hold_stabilization,
        **profile,
    }
    output["transcriptionCleanup"] = {
        **(payload.get("transcriptionCleanup") or {}),
        "outputNotes": len(arranged_notes),
        "vocalMelodyNotes": vocal_melody_notes,
        "arrangerProfile": arranger_profile,
        "arrangedNotesPerSecond": round_number(notes_per_second(arranged_notes), 3),
    }
    # Turn symbolic durations into a physically plausible performance layer.
    # `duration` remains the written/visual value; `audioDuration` represents
    # how long the key is actually held, while the release tail and explicitly
    # labelled pedal events let the string continue naturally.  Keeping these
    # concepts separate prevents both typewriter cut-offs and stuck overlaps.
    output = shape_piano_performance(output, infer_pedal=True)
    if arranger_profile == 'full-mix-piano-reduction':
        output = synchronize_generated_visual_holds(output)
    adaptive_mix_config = (
        style_decoder.get("gestureDynamics", {}).get(
            "adaptiveMelodyMixBalance", {}
        )
        if performance_only_profile or default_pipeline_profile
        else {}
    )
    output['notes'], adaptive_mix = adaptive_melody_mix_balance(
        output['notes'], adaptive_mix_config
    )
    output['pianoArrangement']['adaptiveMelodyMixBalance'] = adaptive_mix
    output['performance']['profile'] = 'polymath-piano-arranger-v5'
    output['performance']['defaultAutoplayReleaseSeconds'] = 0.62
    output['performance']['melodyForwardDynamics'] = True
    output['pianoArrangement']['version'] = 5
    output['pianoArrangement']['maximumHarmonyPitchClasses'] = (
        maximum_harmony_pitch_classes
        if arranger_profile == 'full-mix-piano-reduction'
        else MAX_HARMONY_PITCH_CLASSES
    )
    output['pianoArrangement']['legatoExtendedNotes'] = legato_extended
    if requested_style_profile:
        output['pianoArrangement']['requestedLearnedProfileId'] = (
            requested_style_profile.get('id')
        )
        output['pianoArrangement']['learnedProfileBypassReason'] = (
            learned_profile_bypass_reason or 'profile-not-applied'
        )
        output['pianoArrangement']['performanceOnlyProfileApplied'] = bool(
            performance_only_profile and gesture_expression.get('applied')
        )
        output['pianoArrangement']['defaultPipelineProfileApplied'] = bool(
            default_pipeline_profile
        )
        output['pianoArrangement']['defaultPipelinePianoPreservationGate'] = {
            'applied': default_pipeline_piano_preserved,
            'sourcePianoRatio': round_number(
                float(profile.get('pianoRatio', 0.0)), 4
            ),
            'maximumSourcePianoRatioForProfile': round_number(
                default_pipeline_piano_limit, 4
            ),
        }
        if default_pipeline_profile and arranger_profile == 'full-mix-piano-reduction':
            output['pianoArrangement']['defaultFullMixTuning'] = {
                'harmonyWindowSeconds': round_number(harmony_window_seconds, 4),
                'maximumHarmonyPitchClasses': maximum_harmony_pitch_classes,
            }
            if cyclic_harmony_config:
                output['pianoArrangement']['defaultFullMixTuning'][
                    'sourceSupportedCyclicHarmonyEnabled'
                ] = bool(cyclic_harmony_config.get('enabled'))
    if conditional_route_diagnostics:
        output['pianoArrangement']['conditionalLearnedRoute'] = (
            conditional_route_diagnostics
        )
    if monophonic_cleanup_config:
        output['pianoArrangement']['monophonicVocalCleanup'] = (
            monophonic_vocal_cleanup
        )
    output['pianoArrangement']['expression'] = expression
    output['pianoArrangement']['physicalPerformance'] = {
        **(output.get('pianoPerformance') or {}),
        'profile': 'written-key-hold-damper-v1',
    }
    output['pianoArrangement']['physicalPerformanceRoleLimits'] = physical_limits
    return output


def merge_phrase_retriggers(
    notes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    '''Merge continuing tones while retaining clearly articulated repetitions.'''
    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_pitch[note['midi']].append(note)
    output: list[dict[str, Any]] = []
    removed = 0
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(
            key=lambda note: (
                note['time'],
                ROLE_PRIORITY.get(note['arrangementRole'], 9),
            )
        )
        merged: list[dict[str, Any]] = []
        for note in pitch_notes:
            if not merged:
                merged.append(note)
                continue
            previous = merged[-1]
            previous_end = previous['time'] + previous['duration']
            note_end = note['time'] + note['duration']
            decision = classify_same_key_retrigger(
                previous,
                note,
                duplicate_seconds=DEFAULT_DUPLICATE_ONSET_SECONDS,
                collision_seconds=DEFAULT_COLLISION_WINDOW_SECONDS,
            )
            if decision == "musical-repeat":
                merged.append(note)
                continue
            preferred = min(
                (previous, note),
                key=lambda item: (
                    ROLE_PRIORITY.get(item['arrangementRole'], 9),
                    -item['velocity'],
                ),
            )
            preferred = dict(preferred)
            preferred['time'] = min(previous['time'], note['time'])
            preferred['duration'] = round_number(
                clamp(
                    max(previous_end, note_end) - preferred['time'],
                    MIN_NOTE_SECONDS,
                    MAX_NOTE_SECONDS,
                )
            )
            preferred['velocity'] = max(previous['velocity'], note['velocity'])
            merged[-1] = preferred
            removed += 1

        for index, note in enumerate(merged[:-1]):
            following = merged[index + 1]
            latest_end = following['time'] - SAME_KEY_RELEASE_GAP_SECONDS
            if note['time'] + note['duration'] > latest_end:
                note['duration'] = round_number(
                    max(MIN_NOTE_SECONDS, latest_end - note['time'])
                )
        output.extend(merged)
    return sorted(output, key=lambda note: (note['time'], note['midi'])), removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "instrumental"), default="instrumental")
    parser.add_argument(
        "--profile",
        help=(
            "Optional learned piano-arranger JSON profile. Full arranger profiles "
            "replace selection only for a detected full mix; profiles with "
            "decoder.performanceOnly keep the default notes and adjust performance only."
        ),
    )
    parser.add_argument(
        "--density-multiplier",
        type=float,
        help="Research-only decoder override used for reproducible validation sweeps.",
    )
    parser.add_argument(
        "--sparse-expansion",
        choices=("enabled", "disabled"),
        help="Research-only override for generated sparse-harmony expansion.",
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        help="Research-only override for the learned note-selection threshold.",
    )
    parser.add_argument(
        "--duration-prediction-weight",
        type=float,
        help="Research-only override for blending learned and source note durations.",
    )
    parser.add_argument(
        "--quota-backfill-ratio",
        type=float,
        help="Research-only minimum fraction of each density quota filled below the selection threshold.",
    )
    parser.add_argument(
        "--adaptive-density-multiplier",
        type=float,
        help="Research-only override for both endpoints of the adaptive source-density policy.",
    )
    parser.add_argument(
        "--long-source-duration-weight",
        type=float,
        help="Research-only source-duration blend used when the detected source has long events.",
    )
    parser.add_argument(
        "--preferred-global-register-shift-semitones",
        type=int,
        choices=(-24, -12, 0, 12, 24),
        help="Research-only whole-score octave preference before safe edge folding.",
    )
    parser.add_argument(
        "--chord-completion-radius-seconds",
        type=float,
        help="Research-only local raw-evidence radius for left-hand chord completion.",
    )
    parser.add_argument(
        "--chord-completion-existing-pitch-bonus",
        type=float,
        help="Research-only stability bonus that prevents unnecessary chord-tone replacement.",
    )
    parser.add_argument(
        "--chord-completion-cross-family-bonus",
        type=float,
        help="Research-only reward for a pitch class supported by multiple source stems.",
    )
    parser.add_argument(
        "--chord-completion-minimum-gain",
        type=float,
        help="Research-only evidence margin required before replacing a 3B chord tone.",
    )
    parser.add_argument(
        "--chord-completion-interval-prior-strength",
        type=float,
        help="Research-only strength of the learned pianist chord-interval preference.",
    )
    parser.add_argument(
        "--left-hand-onset-keep-ratio",
        type=float,
        help="Research-only share of frozen accompaniment onsets retained per window.",
    )
    parser.add_argument(
        "--left-hand-enforce-notes-per-onset",
        choices=("enabled", "disabled"),
        help="Research-only enforcement of the ideal sheet's one-vs-two-note onset mix.",
    )
    parser.add_argument(
        "--left-hand-target-notes-per-onset",
        type=float,
        help="Research-only pre-cleanup one-vs-two-note accompaniment target.",
    )
    parser.add_argument(
        "--left-hand-fast-retrigger-keep-share",
        type=float,
        help="Research-only share of fast left-hand repeats preserved as intentional attacks.",
    )
    parser.add_argument(
        "--melody-gain",
        type=float,
        help="Research-only vocal-melody performance gain for rendered balance calibration.",
    )
    parser.add_argument(
        "--melody-octave-shift",
        type=int,
        help=(
            "Research-only octave shift applied to the learned melody role "
            "before whole-score register planning."
        ),
    )
    parser.add_argument(
        "--left-gain-during-melody",
        type=float,
        help="Research-only left accompaniment gain while the melody is sounding.",
    )
    parser.add_argument(
        "--right-gain-during-melody",
        type=float,
        help="Research-only upper-harmony gain while the melody is sounding.",
    )
    parser.add_argument(
        "--adaptive-melody-onset-delay-seconds",
        type=float,
        help=(
            "Research-only whole-melody onset delay used only when the adaptive "
            "register-separation gate activates."
        ),
    )
    parser.add_argument(
        "--melody-performance-gain",
        type=float,
        help="Research-only final voice gain for the arranged melody role.",
    )
    parser.add_argument(
        "--accompaniment-performance-gain",
        type=float,
        help=(
            "Research-only final accompaniment gain while melody is active; "
            "does not alter hammer velocity."
        ),
    )
    parser.add_argument(
        "--duck-overlapping-accompaniment",
        choices=("enabled", "disabled"),
        help=(
            "Research-only overlap-aware accompaniment ducking around melody "
            "spans instead of same-onset ducking only."
        ),
    )
    parser.add_argument(
        "--performance-gain-look-behind-seconds",
        type=float,
        help="Research-only lead-in window for overlap-aware accompaniment gain.",
    )
    parser.add_argument(
        "--performance-gain-look-ahead-seconds",
        type=float,
        help="Research-only release window for overlap-aware accompaniment gain.",
    )
    parser.add_argument(
        "--adaptive-melody-mix-balance",
        choices=("enabled", "disabled"),
        help=(
            "Research-only inference-safe sampled-piano energy balancing; "
            "changes performance gain only."
        ),
    )
    parser.add_argument(
        "--target-estimated-melody-share",
        type=float,
        help="Research-only sampled-piano melody share target for adaptive balance.",
    )
    parser.add_argument(
        "--pure-piano-density-override",
        choices=("enabled", "disabled"),
        default="enabled",
        help=(
            "Enable the inference-safe high-density pure-acoustic-piano route. "
            "The disabled value exists only to reproduce the frozen production "
            "baseline during blind evaluation."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    style_profile = None
    if args.profile:
        style_profile = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
        decoder = style_profile.setdefault("decoder", {})
        if args.melody_octave_shift is not None:
            style_profile.setdefault("roles", {}).setdefault("melody", {})[
                "octaveShift"
            ] = int(
                clamp(
                    round(float(args.melody_octave_shift) / 12.0) * 12,
                    -48,
                    48,
                )
            )
        if args.density_multiplier is not None:
            decoder["preCleanupDensityMultiplier"] = clamp(
                float(args.density_multiplier), 0.5, 3.0
            )
        if args.sparse_expansion is not None:
            decoder["expandSparseHarmony"] = args.sparse_expansion == "enabled"
        if args.selection_threshold is not None:
            style_profile.setdefault("selectionModel", {})["threshold"] = clamp(
                float(args.selection_threshold), 0.01, 0.99
            )
        if args.duration_prediction_weight is not None:
            style_profile.setdefault("durationModel", {})["predictionWeight"] = clamp(
                float(args.duration_prediction_weight), 0.0, 1.0
            )
        if args.quota_backfill_ratio is not None:
            decoder["quotaBackfillRatio"] = clamp(
                float(args.quota_backfill_ratio), 0.0, 1.0
            )
        adaptive_density = decoder.setdefault("adaptiveSourceDensity", {})
        if args.adaptive_density_multiplier is not None:
            density = clamp(float(args.adaptive_density_multiplier), 0.5, 3.0)
            adaptive_density["lowDensityMultiplier"] = density
            adaptive_density["highDensityMultiplier"] = density
        if args.long_source_duration_weight is not None:
            adaptive_density["longSourceDurationWeight"] = clamp(
                float(args.long_source_duration_weight), 0.0, 1.0
            )
        if args.adaptive_melody_onset_delay_seconds is not None:
            decoder.setdefault("defaultFullMix", {}).setdefault(
                "adaptiveMelodyRegisterSeparation", {}
            )["onsetDelaySeconds"] = clamp(
                float(args.adaptive_melody_onset_delay_seconds), -0.08, 0.08
            )
        gesture_dynamics = decoder.setdefault("gestureDynamics", {})
        if args.melody_performance_gain is not None:
            gesture_dynamics["melodyPerformanceGain"] = clamp(
                float(args.melody_performance_gain), 0.5, 1.5
            )
        if args.accompaniment_performance_gain is not None:
            gesture_dynamics["accompanimentGainDuringMelody"] = clamp(
                float(args.accompaniment_performance_gain), 0.25, 1.25
            )
        if args.duck_overlapping_accompaniment is not None:
            gesture_dynamics["duckOverlappingAccompaniment"] = (
                args.duck_overlapping_accompaniment == "enabled"
            )
        if args.performance_gain_look_behind_seconds is not None:
            gesture_dynamics["performanceGainLookBehindSeconds"] = clamp(
                float(args.performance_gain_look_behind_seconds), 0.0, 0.30
            )
        if args.performance_gain_look_ahead_seconds is not None:
            gesture_dynamics["performanceGainLookAheadSeconds"] = clamp(
                float(args.performance_gain_look_ahead_seconds), 0.0, 0.30
            )
        adaptive_mix = gesture_dynamics.setdefault(
            "adaptiveMelodyMixBalance", {}
        )
        if args.adaptive_melody_mix_balance is not None:
            adaptive_mix["enabled"] = (
                args.adaptive_melody_mix_balance == "enabled"
            )
        if args.target_estimated_melody_share is not None:
            adaptive_mix["targetEstimatedMelodyShare"] = clamp(
                float(args.target_estimated_melody_share), 0.45, 0.75
            )
        if args.preferred_global_register_shift_semitones is not None:
            decoder["preferredGlobalRegisterShiftSemitones"] = int(
                args.preferred_global_register_shift_semitones
            )
        left_hand = decoder.setdefault("leftHandAccompaniment", {})
        if args.chord_completion_radius_seconds is not None:
            left_hand["chordCompletionRadiusSeconds"] = clamp(
                float(args.chord_completion_radius_seconds), 0.03, 0.35
            )
        if args.chord_completion_existing_pitch_bonus is not None:
            left_hand["chordCompletionExistingPitchBonus"] = clamp(
                float(args.chord_completion_existing_pitch_bonus), 0.0, 0.20
            )
        if args.chord_completion_cross_family_bonus is not None:
            left_hand["chordCompletionCrossFamilyBonus"] = clamp(
                float(args.chord_completion_cross_family_bonus), 0.0, 0.15
            )
        if args.chord_completion_minimum_gain is not None:
            left_hand["chordCompletionMinimumGain"] = clamp(
                float(args.chord_completion_minimum_gain), 0.0, 1.0
            )
        if args.chord_completion_interval_prior_strength is not None:
            left_hand["chordCompletionIntervalPriorStrength"] = clamp(
                float(args.chord_completion_interval_prior_strength), 0.0, 0.50
            )
        if args.left_hand_onset_keep_ratio is not None:
            left_hand["targetOnsetKeepRatio"] = clamp(
                float(args.left_hand_onset_keep_ratio), 0.1, 1.0
            )
        if args.left_hand_enforce_notes_per_onset is not None:
            left_hand["enforceTargetNotesPerOnset"] = (
                args.left_hand_enforce_notes_per_onset == "enabled"
            )
        if args.left_hand_target_notes_per_onset is not None:
            left_hand["targetNotesPerOnset"] = clamp(
                float(args.left_hand_target_notes_per_onset), 1.0, 2.0
            )
        if args.left_hand_fast_retrigger_keep_share is not None:
            left_hand["fastRetriggerKeepShare"] = clamp(
                float(args.left_hand_fast_retrigger_keep_share), 0.0, 1.0
            )
        melody_balance = decoder.setdefault("melodyForwardBalance", {})
        if any(
            value is not None
            for value in (
                args.melody_gain,
                args.left_gain_during_melody,
                args.right_gain_during_melody,
            )
        ):
            melody_balance["enabled"] = True
        if args.melody_gain is not None:
            melody_balance["melodyGain"] = clamp(
                float(args.melody_gain), 0.8, 1.2
            )
        if args.left_gain_during_melody is not None:
            melody_balance["leftGainDuringMelody"] = clamp(
                float(args.left_gain_during_melody), 0.25, 1.0
            )
        if args.right_gain_during_melody is not None:
            melody_balance["rightGainDuringMelody"] = clamp(
                float(args.right_gain_during_melody), 0.25, 1.0
            )
    arranged = arrange_payload(
        payload,
        args.mode,
        style_profile=style_profile,
        allow_pure_piano_density_override=(
            args.pure_piano_density_override == "enabled"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(arranged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "profile": arranged["pianoArrangement"]["profile"],
                "sourceNotes": arranged["pianoArrangement"]["sourceNoteCount"],
                "outputNotes": arranged["pianoArrangement"]["outputNoteCount"],
                "notesPerSecond": arranged["pianoArrangement"]["outputNotesPerSecond"],
                "vocalMelodyNotes": arranged["pianoArrangement"]["vocalMelodyNotes"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
