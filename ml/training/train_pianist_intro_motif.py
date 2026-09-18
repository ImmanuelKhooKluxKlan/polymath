"""Mine and apply a repeated pianist intro motif without copying target notes.

This is a deliberately narrow research adapter.  It discovers a repeated
gesture cycle in an approved piano reference, stores the cycle as intervals
relative to its bass anchor, and rebuilds only a source's pre-vocal intro.  The
application side sees the source transcription and the learned relative
template; it never reads target notes.

The profile is same-song style conditioning, not evidence of cross-song
generalisation.  It therefore remains research-only until a larger licensed
corpus and complete-song holdouts support promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from .analyze_pianist_gesture_patterns import (
        group_onsets,
        normalize_notes,
        pitch_set,
        set_f1,
    )
    from .analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import (
        group_onsets,
        normalize_notes,
        pitch_set,
        set_f1,
    )
    from analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import load_song_sequences, rounded


VOICE_INSTRUMENTS = {"voice"}
PERCUSSION_MARKERS = ("drum", "percussion", "timpani")
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def source_instrument(note: dict[str, Any]) -> str:
    return str(note.get("instrument") or note.get("sourceInstrument") or "other").lower()


def is_voice(note: dict[str, Any]) -> bool:
    return source_instrument(note) in VOICE_INSTRUMENTS


def is_percussion(note: dict[str, Any]) -> bool:
    value = source_instrument(note)
    return any(marker in value for marker in PERCUSSION_MARKERS)


def finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def repeating_prefix(
    groups: list[list[dict[str, Any]]],
    *,
    minimum_period: int = 4,
    maximum_period: int = 24,
    minimum_cycles: int = 3,
    minimum_cycle_agreement: float = 0.95,
) -> dict[str, Any]:
    """Return the strongest non-trivial repeated hand-occupancy prefix."""

    states = [occupancy(group) for group in groups]
    candidates: list[dict[str, Any]] = []
    maximum_period = min(maximum_period, len(states) // minimum_cycles)
    for period in range(minimum_period, maximum_period + 1):
        pattern = states[:period]
        if len(set(pattern)) < 2:
            continue
        agreements: list[float] = []
        cycles = 1
        while (cycles + 1) * period <= len(states):
            observed = states[cycles * period : (cycles + 1) * period]
            agreement = sum(a == b for a, b in zip(pattern, observed)) / period
            if agreement < minimum_cycle_agreement:
                break
            agreements.append(agreement)
            cycles += 1
        if cycles < minimum_cycles:
            continue
        confidence = sum(agreements) / max(1, len(agreements))
        candidates.append(
            {
                "periodGestures": period,
                "cycles": cycles,
                "coveredGestures": period * cycles,
                "agreement": confidence,
                "pattern": pattern,
            }
        )
    if not candidates:
        raise ValueError("No repeated multi-hand prefix met the motif confidence gate")
    return max(
        candidates,
        key=lambda item: (
            int(item["coveredGestures"]),
            float(item["agreement"]),
            -int(item["periodGestures"]),
        ),
    )


def infer_grid(
    groups: list[list[dict[str, Any]]],
    period: int,
    cycles: int,
    subdivisions: Iterable[int] = (8, 12, 16, 20, 24, 32),
) -> dict[str, Any]:
    """Find the simplest clock whose slot pattern repeats across cycles."""

    # Include the start of the following (possibly partial) cycle when it is
    # present.  That boundary gives us one extra complete duration estimate
    # without learning any additional target gestures.
    boundary_count = cycles + 1 if len(groups) > cycles * period else cycles
    starts = [
        float(groups[index * period][0]["time"]) for index in range(boundary_count)
    ]
    durations = [right - left for left, right in zip(starts, starts[1:])]
    if not durations:
        raise ValueError("At least two motif cycles are required")
    trials: list[dict[str, Any]] = []
    for division in subdivisions:
        patterns: list[tuple[int, ...]] = []
        residuals: list[float] = []
        valid = True
        for cycle in range(min(cycles, len(starts) - 1)):
            start = starts[cycle]
            duration = starts[cycle + 1] - start
            slots = []
            for offset in range(period):
                position = (
                    (float(groups[cycle * period + offset][0]["time"]) - start)
                    / max(1e-9, duration)
                    * division
                )
                slot = int(round(position))
                slots.append(slot)
                residuals.append(abs(position - slot))
            if len(set(slots)) != period or min(slots) < 0 or max(slots) >= division:
                valid = False
                break
            patterns.append(tuple(slots))
        if not valid or not patterns:
            continue
        counts = Counter(patterns)
        modal, modal_count = counts.most_common(1)[0]
        trials.append(
            {
                "subdivisions": division,
                "slots": list(modal),
                "cyclePatternAgreement": modal_count / len(patterns),
                "meanSlotResidual": sum(residuals) / len(residuals),
            }
        )
    if not trials:
        raise ValueError("No collision-free motif clock was found")
    best = max(
        trials,
        key=lambda item: (
            float(item["cyclePatternAgreement"]),
            -float(item["meanSlotResidual"]),
            -int(item["subdivisions"]),
        ),
    )
    best["cycleDurationSeconds"] = median(durations)
    best["pulseSeconds"] = best["cycleDurationSeconds"] / int(best["subdivisions"])
    best["trials"] = trials
    return best


def note_duration(note: dict[str, Any]) -> float:
    return max(0.03, finite(note.get("scoreDuration", note.get("duration")), 0.2))


def motif_note_templates(
    groups: list[list[dict[str, Any]]],
    *,
    period: int,
    cycles: int,
    pulse_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode stable pitches and the one moving right-hand voice by interval."""

    roots = [
        min(int(note["midi"]) for note in groups[cycle * period])
        for cycle in range(cycles)
    ]
    # The changing right-hand voice is the only non-consensus right interval
    # at the first gesture.  The highest interval is the stable top pedal.
    first_rows = []
    for cycle, root in enumerate(roots):
        first_rows.append(
            [
                int(note["midi"]) - root
                for note in groups[cycle * period]
                if explicit_hand(note) == "right"
            ]
        )
    right_counts = Counter(value for row in first_rows for value in row)
    stable_right = {
        value for value, count in right_counts.items() if count >= math.ceil(0.75 * cycles)
    }
    moving_intervals: list[int] = []
    for row in first_rows:
        variable = [value for value in row if value not in stable_right]
        if len(variable) != 1:
            raise ValueError("The repeated prefix does not expose one moving right voice")
        moving_intervals.append(variable[0])

    templates: list[dict[str, Any]] = []
    stable_threshold = math.ceil(0.75 * cycles)
    for slot in range(period):
        stable_counts: Counter[tuple[int, str]] = Counter()
        stable_durations: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
        moving_durations: list[float] = []
        moving_present = 0
        velocities: list[float] = []
        for cycle, root in enumerate(roots):
            group = groups[cycle * period + slot]
            velocities.extend(finite(note.get("velocity"), 0.7) for note in group)
            moving_interval = moving_intervals[cycle]
            for note in group:
                key = (int(note["midi"]) - root, explicit_hand(note))
                if key == (moving_interval, "right"):
                    moving_present += 1
                    moving_durations.append(note_duration(note) / pulse_seconds)
                else:
                    stable_counts[key] += 1
                    stable_durations[key].append(note_duration(note) / pulse_seconds)
        stable = []
        for (interval, hand), count in sorted(stable_counts.items()):
            if count < stable_threshold:
                continue
            stable.append(
                {
                    "intervalFromBass": interval,
                    "hand": hand,
                    "durationPulses": rounded(median(stable_durations[(interval, hand)])),
                    "cycleSupport": rounded(count / cycles),
                }
            )
        stable_keys = {
            (int(note["intervalFromBass"]), str(note["hand"])) for note in stable
        }
        first_cycle_notes: list[dict[str, Any]] = []
        first_group = groups[slot]
        first_root = roots[0]
        first_moving_interval = moving_intervals[0]
        for note in first_group:
            interval = int(note["midi"]) - first_root
            hand = explicit_hand(note)
            key = (interval, hand)
            if key in stable_keys or key == (first_moving_interval, "right"):
                continue
            # Keep a one-cycle opening ornament only when its interval is a
            # stable pedal elsewhere in the repeated phrase.  This recovers a
            # deliberate pickup without memorising arbitrary one-off noise.
            if hand == "right" and interval in stable_right:
                first_cycle_notes.append(
                    {
                        "intervalFromBass": interval,
                        "hand": hand,
                        "durationPulses": rounded(note_duration(note) / pulse_seconds),
                        "evidence": "globally-stable-pedal-first-cycle-only",
                    }
                )
        template: dict[str, Any] = {
            "gestureOffset": slot,
            "stableNotes": stable,
            "velocityMedian": rounded(median(velocities)),
        }
        if moving_present >= stable_threshold:
            template["movingRightVoice"] = {
                "enabled": True,
                "durationPulses": rounded(median(moving_durations)),
                "cycleSupport": rounded(moving_present / cycles),
            }
        if first_cycle_notes:
            template["firstCycleNotes"] = first_cycle_notes
        templates.append(template)
    return templates, {
        "bassMidiByCycle": roots,
        "movingIntervalByCycle": moving_intervals,
        "stableRightIntervals": sorted(stable_right),
    }


def source_groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    notes = []
    for source_index, item in enumerate(payload.get("notes") or []):
        if not isinstance(item, dict) or is_percussion(item):
            continue
        time = finite(item.get("time", item.get("startTime", item.get("start"))), -1)
        midi = int(round(finite(item.get("midi", item.get("pitch")), -1)))
        if time < 0 or not 21 <= midi <= 108:
            continue
        note = dict(item)
        note.update({"time": time, "midi": midi, "sourceIndex": source_index})
        notes.append(note)
    return group_onsets(sorted(notes, key=lambda note: (note["time"], note["midi"])), 0.035)


def estimate_source_pulse(groups: list[list[dict[str, Any]]], end_time: float) -> float:
    times = [float(group[0]["time"]) for group in groups if float(group[0]["time"]) < end_time]
    gaps = [right - left for left, right in zip(times, times[1:]) if 0.075 <= right - left <= 0.23]
    if not gaps:
        raise ValueError("The source intro has no usable pulse evidence")
    # Millisecond binning makes a repeated 150 ms grid dominate one noisy gap.
    bins = Counter(int(round(gap * 1000 / 10)) * 10 for gap in gaps)
    mode_ms = bins.most_common(1)[0][0]
    neighbors = [gap for gap in gaps if abs(gap * 1000 - mode_ms) <= 15]
    return median(neighbors or gaps)


def nearest_group(
    groups: list[list[dict[str, Any]]], time: float, maximum_distance: float
) -> list[dict[str, Any]]:
    eligible = [group for group in groups if not all(is_voice(note) for note in group)]
    if not eligible:
        return []
    group = min(eligible, key=lambda item: abs(float(item[0]["time"]) - time))
    return group if abs(float(group[0]["time"]) - time) <= maximum_distance else []


def source_moving_pitch(
    group: list[dict[str, Any]],
    *,
    bass_midi: int,
    top_midi: int,
    stable_pitch_classes: set[int],
) -> int:
    midis = sorted({int(note["midi"]) for note in group if not is_voice(note)})
    candidates = [
        midi
        for midi in midis
        if bass_midi + 7 <= midi < top_midi and midi % 12 not in stable_pitch_classes
    ]
    if candidates:
        return max(candidates)
    root_high = bass_midi + 12
    return min(midis, key=lambda midi: (abs(midi - root_high), -midi)) if midis else root_high


def nearest_octave_shift(
    source_midi: float,
    target_midi: float,
    *,
    minimum_shift: int = -36,
    maximum_shift: int = 36,
) -> int:
    """Return the least octave displacement that approaches a target register."""

    shifts = range(
        int(math.ceil(minimum_shift / 12.0)) * 12,
        int(math.floor(maximum_shift / 12.0)) * 12 + 1,
        12,
    )
    return min(
        shifts,
        key=lambda shift: (
            abs(float(source_midi) + shift - float(target_midi)),
            abs(shift),
            shift,
        ),
    )


def infer_intro_register_application(
    source: dict[str, Any],
    templates: list[dict[str, Any]],
    pitch_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Learn source-relative octave placement without storing time coordinates.

    The original adapter used a fixed ``+12`` moving-voice shift and treated the
    source guitar root as the output bass.  For Kiss Me that moved Pianella's
    D#2/D#4 opening to D#3/D#5.  The profile already knows the pianist's
    relative register, so fit octave-only shifts once during training and let
    inference apply those shifts to source-detected pitches.
    """

    groups = source_groups(source)
    non_voice = [group for group in groups if not all(is_voice(note) for note in group)]
    if not non_voice:
        raise ValueError("The source has no harmonic group for register calibration")
    initial = non_voice[0]
    source_midis = sorted({int(note["midi"]) for note in initial if not is_voice(note)})
    if len(source_midis) < 3:
        raise ValueError("The source register anchor needs at least three pitches")
    reference_basses = [int(value) for value in pitch_analysis.get("bassMidiByCycle") or []]
    moving_intervals = [
        int(value) for value in pitch_analysis.get("movingIntervalByCycle") or []
    ]
    if not reference_basses or not moving_intervals:
        raise ValueError("The learned motif has no bass or moving-voice register evidence")

    source_bass = min(source_midis)
    source_top = max(source_midis)
    reference_bass = float(median(reference_basses))
    bass_shift = nearest_octave_shift(source_bass, reference_bass)
    stable_intervals = {
        int(note["intervalFromBass"]) % 12
        for template in templates
        for note in template.get("stableNotes") or []
    }
    stable_source_pitch_classes = {
        (source_bass + interval) % 12 for interval in stable_intervals
    }
    stable_source_pitch_classes.add(source_top % 12)
    source_moving = source_moving_pitch(
        initial,
        bass_midi=source_bass,
        top_midi=source_top,
        stable_pitch_classes=stable_source_pitch_classes,
    )
    reference_moving = reference_bass + float(median(moving_intervals))
    moving_shift = nearest_octave_shift(source_moving, reference_moving)
    return {
        "method": "learned-reference-register-minus-source-register-octave-only-v1",
        "sourceBassMidi": source_bass,
        "referenceBassMedianMidi": rounded(reference_bass),
        "bassSourceOctaveShift": bass_shift,
        "sourceMovingMidi": source_moving,
        "referenceMovingMedianMidi": rounded(reference_moving),
        "movingSourceOctaveShift": moving_shift,
        "targetReadAtInference": False,
    }


def infer_pre_voice_cadence(
    reference: list[list[dict[str, Any]]],
    source: dict[str, Any],
    motif: dict[str, Any],
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Learn a relative closing gesture after a repeated intro prefix.

    The Kiss Me pianist repeats seven complete arpeggio cycles, then replaces
    the final cycle's last two attacks with one held cadence immediately before
    the vocal entry.  Repeating a complete eighth cycle creates the audible
    stutter.  Store only relative intervals, hand labels and pulse-relative
    timing; inference never reads the target score.
    """

    groups = source_groups(source)
    voice_times = [
        float(note["time"])
        for group in groups
        for note in group
        if is_voice(note)
    ]
    if not voice_times:
        return {"enabled": False, "reason": "source-has-no-voice-boundary"}
    first_voice = min(voice_times)
    intro_reference = [
        group for group in reference if float(group[0]["time"]) < first_voice
    ]
    prefix_gestures = int(motif["periodGestures"]) * int(motif["cycles"])
    tail = intro_reference[prefix_gestures:]
    attacked_slots = [int(value) for value in grid["slots"]]
    if len(tail) < 1 or len(tail) > len(attacked_slots):
        return {
            "enabled": False,
            "reason": "reference-has-no-single-partial-cycle-cadence",
            "tailGestures": len(tail),
        }

    cadence = tail[-1]
    source_non_voice = [
        group for group in groups if not all(is_voice(note) for note in group)
    ]
    source_pulse = estimate_source_pulse(source_non_voice, first_voice)
    cadence_time = float(cadence[0]["time"])
    evidence = nearest_group(
        source_non_voice,
        cadence_time,
        max(0.25, source_pulse * 2.1),
    )
    evidence_midis = sorted(
        {int(note["midi"]) for note in evidence if not is_voice(note)}
    )
    if not evidence_midis:
        return {"enabled": False, "reason": "cadence-has-no-source-harmony"}

    reference_root = min(int(note["midi"]) for note in cadence)
    source_root = min(evidence_midis)
    root_shift = nearest_octave_shift(source_root, reference_root)
    notes = [
        {
            "intervalFromRoot": int(note["midi"]) - reference_root,
            "hand": explicit_hand(note),
            "durationPulses": rounded(note_duration(note) / source_pulse),
        }
        for note in sorted(cadence, key=lambda item: int(item["midi"]))
    ]
    return {
        "enabled": True,
        "method": "partial-final-cycle-relative-cadence-v1",
        "replaceFromGridSlot": attacked_slots[len(tail) - 1],
        "voiceLeadPulses": rounded((first_voice - cadence_time) / source_pulse),
        "velocityMedian": rounded(
            median(finite(note.get("velocity"), 0.7) for note in cadence)
        ),
        "sourceRootMidiAtTraining": source_root,
        "referenceRootMidiAtTraining": reference_root,
        "rootSourceOctaveShift": root_shift,
        "notes": notes,
        "tailGestures": len(tail),
        "targetReadAtInference": False,
    }


def source_cycle_transition_anchors(
    groups: list[list[dict[str, Any]]],
    *,
    start: float,
    cutoff: float,
    cycle_seconds: float,
    cycles: int,
    source_bass_midi: int,
    top_source_midi: int,
    stable_pitch_classes: set[int],
) -> list[float]:
    """Locate source harmonic transitions nearest each expected motif cycle.

    Dense full-mix transcriptions often expose the repeating inner voice more
    reliably than they expose every individual piano attack.  Transition
    anchors let the learned motif follow that source rubato while retaining a
    fixed-grid fallback whenever no credible change is observed.
    """

    transitions: list[float] = []
    previous_moving: int | None = None
    for group in groups:
        time = float(group[0]["time"])
        if time < start - 0.05 or time > cutoff + cycle_seconds:
            continue
        midis = {int(note["midi"]) for note in group if not is_voice(note)}
        if len(midis) < 3:
            continue
        moving = source_moving_pitch(
            group,
            bass_midi=source_bass_midi,
            top_midi=top_source_midi,
            stable_pitch_classes=stable_pitch_classes,
        )
        if previous_moving is None or moving != previous_moving:
            transitions.append(time)
            previous_moving = moving

    anchors = [start]
    search_radius = max(0.35, cycle_seconds * 0.48)
    for cycle in range(1, cycles + 1):
        expected = start + cycle * cycle_seconds
        eligible = [
            time
            for time in transitions
            if time > anchors[-1] + cycle_seconds * 0.45
            and abs(time - expected) <= search_radius
        ]
        anchors.append(
            min(eligible, key=lambda time: (abs(time - expected), time))
            if eligible
            else expected
        )
    return anchors


def generated_note(
    *,
    midi: int,
    time: float,
    duration: float,
    velocity: float,
    hand: str,
    role: str,
    cycle: int,
    grid_slot: int,
) -> dict[str, Any]:
    return {
        "duration": rounded(max(0.05, duration)),
        "scoreDuration": rounded(max(0.05, duration)),
        "visualDuration": rounded(max(0.05, duration)),
        "audioDuration": rounded(max(0.05, duration)),
        "releaseSeconds": 0.62,
        "hand": hand,
        "instrument": "acoustic_piano",
        "midi": midi,
        "note": midi_to_note(midi),
        "source": "polymath-pianist-motif-adapter",
        "time": rounded(time),
        "velocity": rounded(max(0.05, min(1.0, velocity))),
        "sourceInstrument": "learned_pianist_motif",
        "arrangementRole": role,
        "voice": "melody" if role == "melody" else f"{role}-{midi}",
        "articulation": "legato",
        "generatedBy": "pianist-intro-motif-v1",
        "motifCycle": cycle,
        "motifGridSlot": grid_slot,
    }


def apply_intro_motif(
    source: dict[str, Any], candidate: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = source_groups(source)
    voices = [
        float(note["time"])
        for group in groups
        for note in group
        if is_voice(note)
    ]
    if not voices:
        raise ValueError("A pre-vocal boundary cannot be inferred without voice events")
    first_voice = min(voices)
    non_voice_groups = [group for group in groups if not all(is_voice(note) for note in group)]
    intro_groups = [group for group in non_voice_groups if float(group[0]["time"]) < first_voice]
    if len(intro_groups) < 8:
        raise ValueError("The source has too little instrumental intro evidence")
    pulse = estimate_source_pulse(non_voice_groups, first_voice)
    subdivisions = int(profile["clock"]["subdivisions"])
    cycle_seconds = pulse * subdivisions
    start = float(intro_groups[0][0]["time"])
    cutoff = first_voice - 2.0 * pulse
    initial = nearest_group(non_voice_groups, start, pulse)
    initial_midis = sorted({int(note["midi"]) for note in initial if not is_voice(note)})
    if len(initial_midis) < 3:
        raise ValueError("The intro anchor needs at least three harmonic pitches")
    source_bass_midi = min(initial_midis)
    bass_shift = int(profile.get("application", {}).get("bassSourceOctaveShift", 0))
    bass_midi = source_bass_midi + bass_shift
    top_source_midi = max(initial_midis)
    stable_intervals = {
        int(note["intervalFromBass"]) % 12
        for template in profile["templates"]
        for note in template.get("stableNotes") or []
    }
    stable_source_pitch_classes = {
        (source_bass_midi + interval) % 12
        for interval in stable_intervals
    }
    # The high pedal in the learned score is two octaves plus its relative
    # chord interval.  Its source equivalent is the highest opening pitch.
    stable_source_pitch_classes.add(top_source_midi % 12)
    moving_shift = int(profile.get("application", {}).get("movingSourceOctaveShift", 12))

    application = profile.get("application", {})
    source_anchor_blend = max(
        0.0, min(1.0, float(application.get("sourceCycleAnchorBlend", 0.0)))
    )
    source_tempo_blend = max(
        0.0, min(1.0, float(application.get("sourceCycleTempoBlend", 0.0)))
    )
    minimum_cycle_separation_ratio = max(
        0.0,
        min(
            1.0,
            float(application.get("minimumCycleSeparationRatio", 0.95)),
        ),
    )
    minimum_cycle_separation = cycle_seconds * minimum_cycle_separation_ratio
    planned_cycles = max(1, math.ceil((cutoff - start) / cycle_seconds))
    source_anchors = source_cycle_transition_anchors(
        non_voice_groups,
        start=start,
        cutoff=cutoff,
        cycle_seconds=cycle_seconds,
        cycles=planned_cycles,
        source_bass_midi=source_bass_midi,
        top_source_midi=top_source_midi,
        stable_pitch_classes=stable_source_pitch_classes,
    )

    output_notes: list[dict[str, Any]] = []
    cycles = 0
    generated_gestures = 0
    adjusted_cycle_boundaries = 0
    previous_cycle_start: float | None = None
    cadence = profile.get("preVoiceCadence") or {}
    cadence_enabled = bool(cadence.get("enabled"))
    cadence_cycle = planned_cycles - 1
    cadence_replace_from_slot = int(cadence.get("replaceFromGridSlot", subdivisions))
    cadence_time = (
        first_voice - float(cadence.get("voiceLeadPulses", 2.0)) * pulse
        if cadence_enabled
        else None
    )
    while start + cycles * cycle_seconds < cutoff:
        fixed_cycle_start = start + cycles * cycle_seconds
        source_cycle_start = source_anchors[cycles]
        cycle_start = (
            fixed_cycle_start * (1.0 - source_anchor_blend)
            + source_cycle_start * source_anchor_blend
        )
        if previous_cycle_start is not None:
            earliest_cycle_start = previous_cycle_start + minimum_cycle_separation
            if cycle_start < earliest_cycle_start:
                cycle_start = earliest_cycle_start
                adjusted_cycle_boundaries += 1
        source_cycle_duration = (
            source_anchors[cycles + 1] - source_anchors[cycles]
            if cycles + 1 < len(source_anchors)
            else cycle_seconds
        )
        source_cycle_pulse = source_cycle_duration / subdivisions
        local_pulse = (
            pulse * (1.0 - source_tempo_blend)
            + source_cycle_pulse * source_tempo_blend
        )
        evidence = nearest_group(non_voice_groups, cycle_start, max(0.20, pulse * 1.6))
        moving_source = source_moving_pitch(
            evidence,
            bass_midi=source_bass_midi,
            top_midi=top_source_midi,
            stable_pitch_classes=stable_source_pitch_classes,
        )
        moving_output = moving_source + moving_shift
        for grid_slot, template in zip(profile["clock"]["attackedSlots"], profile["templates"]):
            if (
                cadence_enabled
                and cycles == cadence_cycle
                and int(grid_slot) >= cadence_replace_from_slot
            ):
                continue
            time = cycle_start + int(grid_slot) * local_pulse
            if time >= cutoff:
                continue
            # Source transcription velocities describe the mixed recording,
            # not the pianist's authored phrase contour.  Kiss Me's opening
            # source attacks are all tied at 0.38, while the approved pianist
            # intentionally accents the final three gestures.  Replacing the
            # learned value with source velocity erased that phrasing.  Keep
            # the learned, tempo-independent touch here; later adapters may
            # add a separately calibrated section-energy envelope.
            gesture_velocity = float(template["velocityMedian"])
            wrote = False
            for stable in template.get("stableNotes") or []:
                midi = bass_midi + int(stable["intervalFromBass"])
                hand = str(stable["hand"])
                output_notes.append(
                    generated_note(
                        midi=midi,
                        time=time,
                        duration=float(stable["durationPulses"]) * local_pulse,
                        velocity=gesture_velocity,
                        hand=hand,
                        role="bass" if hand == "left" and midi == bass_midi else "harmony",
                        cycle=cycles,
                        grid_slot=int(grid_slot),
                    )
                )
                wrote = True
            if cycles == 0:
                for opening in template.get("firstCycleNotes") or []:
                    midi = bass_midi + int(opening["intervalFromBass"])
                    hand = str(opening["hand"])
                    output_notes.append(
                        generated_note(
                            midi=midi,
                            time=time,
                            duration=float(opening["durationPulses"]) * local_pulse,
                            velocity=gesture_velocity,
                            hand=hand,
                            role="harmony",
                            cycle=cycles,
                            grid_slot=int(grid_slot),
                        )
                    )
                    wrote = True
            moving = template.get("movingRightVoice") or {}
            if moving.get("enabled"):
                output_notes.append(
                    generated_note(
                        midi=moving_output,
                        time=time,
                        duration=float(moving["durationPulses"]) * local_pulse,
                        velocity=gesture_velocity,
                        hand="right",
                        role="melody",
                        cycle=cycles,
                        grid_slot=int(grid_slot),
                    )
                )
                wrote = True
            generated_gestures += int(wrote)
        previous_cycle_start = cycle_start
        cycles += 1

    generated_cadence_notes = 0
    if cadence_enabled and cadence_time is not None and cadence_time < first_voice:
        evidence = nearest_group(
            non_voice_groups,
            cadence_time,
            max(0.25, pulse * 2.1),
        )
        evidence_midis = sorted(
            {int(note["midi"]) for note in evidence if not is_voice(note)}
        )
        if evidence_midis:
            cadence_root = min(evidence_midis) + int(
                cadence.get("rootSourceOctaveShift", 0)
            )
            cadence_velocity = float(cadence.get("velocityMedian", 0.7))
            for cadence_note in cadence.get("notes") or []:
                midi = cadence_root + int(cadence_note["intervalFromRoot"])
                hand = str(cadence_note["hand"])
                output_notes.append(
                    generated_note(
                        midi=midi,
                        time=cadence_time,
                        duration=float(cadence_note["durationPulses"]) * pulse,
                        velocity=cadence_velocity,
                        hand=hand,
                        role=(
                            "bass"
                            if hand == "left" and int(cadence_note["intervalFromRoot"]) == 0
                            else "harmony"
                        ),
                        cycle=cadence_cycle,
                        grid_slot=cadence_replace_from_slot,
                    )
                )
                generated_cadence_notes += 1
            generated_gestures += int(generated_cadence_notes > 0)

    retained = [
        dict(note)
        for note in candidate.get("notes") or []
        if finite(note.get("time"), -1) >= cutoff
    ]
    output = copy.deepcopy(candidate)
    output["notes"] = sorted(
        [*output_notes, *retained],
        key=lambda note: (finite(note.get("time"), 0), int(note.get("midi", 0))),
    )
    arrangement = output.setdefault("pianoArrangement", {})
    diagnostics = {
        "applied": True,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "scope": "pre-vocal-intro-only",
        "firstSourceOnsetSeconds": rounded(start),
        "firstVoiceOnsetSeconds": rounded(first_voice),
        "replacementCutoffSeconds": rounded(cutoff),
        "sourcePulseSeconds": rounded(pulse),
        "sourceBassMidi": source_bass_midi,
        "bassSourceOctaveShift": bass_shift,
        "outputBassMidi": bass_midi,
        "movingSourceOctaveShift": moving_shift,
        "sourceCycleAnchorBlend": rounded(source_anchor_blend),
        "sourceCycleTempoBlend": rounded(source_tempo_blend),
        "minimumCycleSeparationRatio": rounded(minimum_cycle_separation_ratio),
        "minimumCycleSeparationSeconds": rounded(minimum_cycle_separation),
        "adjustedCycleBoundaries": adjusted_cycle_boundaries,
        "preVoiceCadenceApplied": generated_cadence_notes > 0,
        "preVoiceCadenceTimeSeconds": rounded(cadence_time) if cadence_time is not None else None,
        "preVoiceCadenceNotes": generated_cadence_notes,
        "sourceCycleTransitionAnchors": [rounded(value) for value in source_anchors],
        "cycleSeconds": rounded(cycle_seconds),
        "cycles": cycles,
        "generatedGestures": generated_gestures,
        "generatedNotes": len(output_notes),
        "retainedCandidateNotes": len(retained),
        "inferenceReadTarget": False,
    }
    arrangement["pianistMotifGrammar"] = diagnostics
    arrangement["outputNoteCount"] = len(output["notes"])
    output["arrangementProfile"] = f"{candidate.get('arrangementProfile', 'piano')}-motif-v1"
    return output, diagnostics


def evaluation(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    matches, missing, extra, score = align_sequences(
        reference, candidate, maximum_time_distance=0.55, gap_cost=0.85
    )
    pc_f1 = []
    exact_key_f1 = []
    exact = 0
    exact_key_sets = 0
    occupancy_hits = 0
    onset_errors: list[float] = []
    velocity_errors: list[float] = []
    duration_errors: list[float] = []
    for reference_index, candidate_index in matches:
        target = reference[reference_index]
        observed = candidate[candidate_index]
        pc_f1.append(set_f1(pitch_set(target, True), pitch_set(observed, True)))
        exact_key_f1.append(set_f1(pitch_set(target), pitch_set(observed)))
        exact += int(pitch_set(target, True) == pitch_set(observed, True))
        exact_key_sets += int(pitch_set(target) == pitch_set(observed))
        occupancy_hits += int(occupancy(target) == occupancy(observed))
        onset_errors.append(
            abs(float(target[0]["time"]) - float(observed[0]["time"]))
        )
        target_velocity = median(
            finite(note.get("velocity"), 0.7) for note in target
        )
        observed_velocity = median(
            finite(note.get("velocity"), 0.7) for note in observed
        )
        velocity_errors.append(abs(target_velocity - observed_velocity))

        # Duration is scored only on exact-key correspondences.  A duration
        # error on the wrong pitch is not an articulation measurement.
        target_by_midi = {int(note["midi"]): note for note in target}
        observed_by_midi = {int(note["midi"]): note for note in observed}
        for midi in target_by_midi.keys() & observed_by_midi.keys():
            duration_errors.append(
                abs(
                    note_duration(target_by_midi[midi])
                    - note_duration(observed_by_midi[midi])
                )
            )
    reference_recall = len(matches) / max(1, len(reference))
    candidate_precision = len(matches) / max(1, len(candidate))
    mean_pitch_class_f1 = sum(pc_f1) / max(1, len(pc_f1))
    mean_exact_key_f1 = sum(exact_key_f1) / max(1, len(exact_key_f1))
    exact_pitch_class_rate = exact / max(1, len(matches))
    exact_key_set_rate = exact_key_sets / max(1, len(matches))
    occupancy_accuracy = occupancy_hits / max(1, len(matches))
    return {
        "referenceGestures": len(reference),
        "candidateGestures": len(candidate),
        "alignedGestures": len(matches),
        "missingReferenceGestures": len(missing),
        "extraCandidateGestures": len(extra),
        "referenceGestureRecall": rounded(reference_recall),
        "candidateGesturePrecision": rounded(candidate_precision),
        "meanPitchClassF1": rounded(mean_pitch_class_f1),
        "exactPitchClassSetRate": rounded(exact_pitch_class_rate),
        "meanExactKeyF1": rounded(mean_exact_key_f1),
        "exactKeySetRate": rounded(exact_key_set_rate),
        "occupancyAccuracy": rounded(occupancy_accuracy),
        # Matched-only averages can look excellent while missing entire
        # gestures.  These coverage-adjusted variants count every missing
        # reference gesture as zero and are the correct promotion metrics.
        "coverageAdjustedPitchClassF1": rounded(
            mean_pitch_class_f1 * reference_recall
        ),
        "coverageAdjustedExactPitchClassRate": rounded(
            exact_pitch_class_rate * reference_recall
        ),
        "coverageAdjustedExactKeyF1": rounded(
            mean_exact_key_f1 * reference_recall
        ),
        "coverageAdjustedExactKeySetRate": rounded(
            exact_key_set_rate * reference_recall
        ),
        "coverageAdjustedOccupancyAccuracy": rounded(
            occupancy_accuracy * reference_recall
        ),
        "onsetMaeSeconds": rounded(sum(onset_errors) / max(1, len(onset_errors))),
        "gestureVelocityMae": rounded(
            sum(velocity_errors) / max(1, len(velocity_errors))
        ),
        "exactKeyDurationMaeSeconds": rounded(
            sum(duration_errors) / max(1, len(duration_errors))
        ),
        "exactKeyDurationComparisons": len(duration_errors),
        "sequenceScore": rounded(score),
    }


def slice_groups(
    groups: list[list[dict[str, Any]]], start: float, end: float
) -> list[list[dict[str, Any]]]:
    return [group for group in groups if start <= float(group[0]["time"]) < end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--source-cycle-anchor-blend", type=float, default=0.0)
    parser.add_argument("--source-cycle-tempo-blend", type=float, default=0.0)
    parser.add_argument("--minimum-cycle-separation-ratio", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 <= args.source_cycle_anchor_blend <= 1.0:
        parser.error("--source-cycle-anchor-blend must be between 0 and 1")
    if not 0.0 <= args.source_cycle_tempo_blend <= 1.0:
        parser.error("--source-cycle-tempo-blend must be between 0 and 1")
    if not 0.0 <= args.minimum_cycle_separation_ratio <= 1.0:
        parser.error("--minimum-cycle-separation-ratio must be between 0 and 1")

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    row = next(
        (item for item in manifest.get("songs") or [] if item.get("id") == args.song_id),
        None,
    )
    if row is None:
        raise ValueError(f"Song id not found in manifest: {args.song_id}")
    reference, baseline, description = load_song_sequences(row, 0.035)
    motif = repeating_prefix(reference)
    grid = infer_grid(
        reference,
        int(motif["periodGestures"]),
        int(motif["cycles"]),
    )
    templates, pitch_analysis = motif_note_templates(
        reference,
        period=int(motif["periodGestures"]),
        cycles=int(motif["cycles"]),
        pulse_seconds=float(grid["pulseSeconds"]),
    )
    source = json.loads(Path(row["source"]).read_text(encoding="utf-8-sig"))
    register_application = infer_intro_register_application(
        source,
        templates,
        pitch_analysis,
    )
    pre_voice_cadence = infer_pre_voice_cadence(
        reference,
        source,
        motif,
        grid,
    )
    profile = {
        "schema": "polymath-pianist-intro-motif-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
        "training": {
            "songId": args.song_id,
            "reference": description["reference"],
            "method": "repeated relative-interval gesture motif mining",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
        },
        "motif": motif,
        "clock": {
            "subdivisions": int(grid["subdivisions"]),
            "attackedSlots": list(grid["slots"]),
            "cycleDurationSeconds": rounded(grid["cycleDurationSeconds"]),
            "referencePulseSeconds": rounded(grid["pulseSeconds"]),
            "cyclePatternAgreement": rounded(grid["cyclePatternAgreement"]),
            "meanSlotResidual": rounded(grid["meanSlotResidual"]),
        },
        "templates": templates,
        "pitchAnalysis": pitch_analysis,
        "preVoiceCadence": pre_voice_cadence,
        "application": {
            "scope": "pre-vocal-intro-only",
            **register_application,
            "sourceCycleAnchorBlend": args.source_cycle_anchor_blend,
            "sourceCycleTempoBlend": args.source_cycle_tempo_blend,
            "minimumCycleSeparationRatio": args.minimum_cycle_separation_ratio,
            "targetReadAtInference": False,
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    candidate_payload = json.loads(Path(row["candidate"]).read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_intro_motif(source, candidate_payload, profile)
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    # Reload through the exact manifest/trust-window path used for the sealed
    # baseline.  Scoring the full generated payload here would compare the
    # candidate's untrusted tail against a reference intentionally capped at
    # 150 s and falsely report hundreds of extra gestures.
    generated_row = {**row, "candidate": str(output_path)}
    _, generated_groups, _ = load_song_sequences(generated_row, 0.035)
    start = float(diagnostics["firstSourceOnsetSeconds"])
    cutoff = float(diagnostics["replacementCutoffSeconds"])
    report = {
        "schema": "polymath-pianist-intro-motif-training-report-v1",
        "evidenceBoundary": (
            "Same-song, pre-vocal, trusted-reference style conditioning. The target is "
            "used to learn a relative motif but is not read by the application pass. "
            "This is not a cross-song promotion result."
        ),
        "profile": str(profile_path),
        "candidate": str(output_path),
        "motif": motif,
        "clock": profile["clock"],
        "pitchAnalysis": pitch_analysis,
        "application": diagnostics,
        "introEvaluation": {
            "baseline": evaluation(
                slice_groups(reference, start, cutoff),
                slice_groups(baseline, start, cutoff),
            ),
            "motifCandidate": evaluation(
                slice_groups(reference, start, cutoff),
                slice_groups(generated_groups, start, cutoff),
            ),
        },
        "trustedWindowEvaluation": {
            "baseline": evaluation(reference, baseline),
            "motifCandidate": evaluation(reference, generated_groups),
        },
        "decision": "LISTENING_CANDIDATE_RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
