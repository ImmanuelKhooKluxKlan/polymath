"""Fail-closed routing for narrow, same-song pianist intro profiles.

The generic Polymath arranger remains the default for every transcription.  A
profile in this module may replace only the pre-vocal intro, and only when the
source satisfies every independent identity check stored with that profile.

The current profile was fitted to a single, locally held reference and is
marked research-only.  Production callers must opt in explicitly; otherwise a
positive match is reported as policy-blocked and the score is returned byte-for-
byte unchanged at this stage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from statistics import median
from typing import Any


VOICE_INSTRUMENTS = {"voice", "vocals", "vocal"}
PERCUSSION_MARKERS = ("drum", "percussion", "timpani")
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _source_instrument(note: dict[str, Any]) -> str:
    return str(note.get("instrument") or note.get("sourceInstrument") or "other").lower()


def _is_voice(note: dict[str, Any]) -> bool:
    instrument = _source_instrument(note)
    return instrument in VOICE_INSTRUMENTS or instrument.startswith("vocal")


def _is_percussion(note: dict[str, Any]) -> bool:
    instrument = _source_instrument(note)
    return any(marker in instrument for marker in PERCUSSION_MARKERS)


def _group_onsets(
    notes: list[dict[str, Any]], window: float = 0.035
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in sorted(notes, key=lambda item: (float(item["time"]), int(item["midi"]))):
        if not groups or float(note["time"]) - float(groups[-1][0]["time"]) > window:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _source_groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    notes: list[dict[str, Any]] = []
    for source_index, item in enumerate(payload.get("notes") or []):
        if not isinstance(item, dict) or _is_percussion(item):
            continue
        time = _finite(item.get("time", item.get("startTime", item.get("start"))), -1)
        midi = int(round(_finite(item.get("midi", item.get("pitch")), -1)))
        if time < 0 or not 21 <= midi <= 108:
            continue
        note = dict(item)
        note.update({"time": time, "midi": midi, "sourceIndex": source_index})
        notes.append(note)
    return _group_onsets(notes)


def _estimate_source_pulse(
    groups: list[list[dict[str, Any]]], end_time: float
) -> float:
    times = [float(group[0]["time"]) for group in groups if float(group[0]["time"]) < end_time]
    gaps = [right - left for left, right in zip(times, times[1:]) if 0.075 <= right - left <= 0.23]
    if not gaps:
        raise ValueError("source intro has no usable pulse evidence")
    bins = Counter(int(round(gap * 1000 / 10)) * 10 for gap in gaps)
    mode_ms = bins.most_common(1)[0][0]
    neighbors = [gap for gap in gaps if abs(gap * 1000 - mode_ms) <= 15]
    return median(neighbors or gaps)


def _nearest_group(
    groups: list[list[dict[str, Any]]], time: float, maximum_distance: float
) -> list[dict[str, Any]]:
    eligible = [group for group in groups if not all(_is_voice(note) for note in group)]
    if not eligible:
        return []
    group = min(eligible, key=lambda item: abs(float(item[0]["time"]) - time))
    return group if abs(float(group[0]["time"]) - time) <= maximum_distance else []


def _source_moving_pitch(
    group: list[dict[str, Any]],
    *,
    bass_midi: int,
    top_midi: int,
    stable_pitch_classes: set[int],
) -> int:
    midis = sorted({int(note["midi"]) for note in group if not _is_voice(note)})
    candidates = [
        midi
        for midi in midis
        if bass_midi + 7 <= midi < top_midi and midi % 12 not in stable_pitch_classes
    ]
    if candidates:
        return max(candidates)
    root_high = bass_midi + 12
    return min(midis, key=lambda midi: (abs(midi - root_high), -midi)) if midis else root_high


def _source_cycle_transition_anchors(
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
    transitions: list[float] = []
    previous_moving: int | None = None
    for group in groups:
        time = float(group[0]["time"])
        if time < start - 0.05 or time > cutoff + cycle_seconds:
            continue
        midis = {int(note["midi"]) for note in group if not _is_voice(note)}
        if len(midis) < 3:
            continue
        moving = _source_moving_pitch(
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


def _cycle_anchor_blend(
    application: dict[str, Any],
    *,
    cycle: int,
    planned_cycles: int,
    fallback: float,
) -> float:
    envelope = application.get("sourceCycleAnchorEnvelope") or {}
    if not bool(envelope.get("enabled")) or planned_cycles <= 1:
        return max(0.0, min(1.0, fallback))
    edge = max(0.0, min(1.0, float(envelope.get("edgeBlend", fallback))))
    peak = max(0.0, min(1.0, float(envelope.get("peakBlend", fallback))))
    power = max(0.05, float(envelope.get("power", 1.0)))
    position = cycle / max(1, planned_cycles - 1)
    shape = max(0.0, math.sin(math.pi * position)) ** power
    return max(0.0, min(1.0, edge + (peak - edge) * shape))


def _generated_note(
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
        "duration": _rounded(max(0.05, duration)),
        "scoreDuration": _rounded(max(0.05, duration)),
        "visualDuration": _rounded(max(0.05, duration)),
        "audioDuration": _rounded(max(0.05, duration)),
        "releaseSeconds": 0.62,
        "hand": hand,
        "instrument": "acoustic_piano",
        "midi": midi,
        "note": _midi_to_note(midi),
        "source": "polymath-pianist-motif-adapter",
        "time": _rounded(time),
        "velocity": _rounded(max(0.05, min(1.0, velocity))),
        "sourceInstrument": "learned_pianist_motif",
        "arrangementRole": role,
        "voice": "melody" if role == "melody" else f"{role}-{midi}",
        "articulation": "legato",
        "generatedBy": "pianist-intro-motif-v1",
        "motifCycle": cycle,
        "motifGridSlot": grid_slot,
    }


def _profile_sha256(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalised_title_tokens(value: Any) -> set[str]:
    normalised = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return set(re.findall(r"[a-z0-9]+", normalised.lower()))


def _note_span_seconds(source: dict[str, Any]) -> float | None:
    ends: list[float] = []
    for note in source.get("notes") or []:
        if not isinstance(note, dict):
            continue
        time = _finite(note.get("time", note.get("startTime", note.get("start"))), -1)
        if time < 0:
            continue
        duration = _finite(
            note.get("audioDuration", note.get("scoreDuration", note.get("duration"))),
            0.0,
        )
        ends.append(time + max(0.0, duration))
    return max(ends) if ends else None


def _set_f1(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return 2.0 * len(left & right) / max(1, len(left) + len(right))


def _range_pass(value: float | None, bounds: list[Any]) -> bool:
    return value is not None and len(bounds) == 2 and float(bounds[0]) <= value <= float(bounds[1])


def evaluate_exact_song_match(
    source: dict[str, Any], mode: str, profile: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every stored identity signal; any uncertainty is a rejection."""

    routing = profile.get("routing") or {}
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, observed: Any, expected: Any, passed: bool) -> None:
        checks[name] = {"passed": bool(passed), "observed": observed, "expected": expected}

    expected_hash = str(profile.get("profileSha256") or "")
    actual_hash = _profile_sha256(profile)
    add("profileIntegrity", actual_hash, expected_hash, bool(expected_hash) and actual_hash == expected_hash)
    add("profileEnabled", bool(profile.get("enabled")), True, bool(profile.get("enabled")))

    allowed_modes = [str(value) for value in routing.get("allowedModes") or []]
    add("mode", mode, allowed_modes, bool(allowed_modes) and mode in allowed_modes)

    title_tokens = _normalised_title_tokens(source.get("title"))
    required_tokens = {str(value).lower() for value in routing.get("titleRequiredTokens") or []}
    forbidden_tokens = {str(value).lower() for value in routing.get("titleForbiddenTokens") or []}
    add(
        "titleTokens",
        sorted(title_tokens),
        {"required": sorted(required_tokens), "forbidden": sorted(forbidden_tokens)},
        bool(required_tokens) and required_tokens <= title_tokens and not (forbidden_tokens & title_tokens),
    )

    span = _note_span_seconds(source)
    span_range = list(routing.get("noteSpanSecondsRange") or [])
    add("noteSpanSeconds", _rounded(span) if span is not None else None, span_range, _range_pass(span, span_range))

    try:
        groups = _source_groups(source)
        voices = [float(note["time"]) for group in groups for note in group if _is_voice(note)]
        non_voice = [group for group in groups if not all(_is_voice(note) for note in group)]
        first_source = float(non_voice[0][0]["time"]) if non_voice else None
        first_voice = min(voices) if voices else None
        source_range = list(routing.get("firstSourceOnsetSecondsRange") or [])
        voice_range = list(routing.get("firstVoiceOnsetSecondsRange") or [])
        add("firstSourceOnsetSeconds", _rounded(first_source) if first_source is not None else None, source_range, _range_pass(first_source, source_range))
        add("firstVoiceOnsetSeconds", _rounded(first_voice) if first_voice is not None else None, voice_range, _range_pass(first_voice, voice_range))

        if first_source is None or first_voice is None:
            raise ValueError("source or vocal onset is missing")
        pulse = _estimate_source_pulse(non_voice, first_voice)
        pulse_range = list(routing.get("sourcePulseSecondsRange") or [])
        add("sourcePulseSeconds", _rounded(pulse), pulse_range, _range_pass(pulse, pulse_range))

        initial = _nearest_group(non_voice, first_source, max(0.20, pulse * 1.6))
        opening = sorted({int(note["midi"]) for note in initial if not _is_voice(note)})
        expected_opening = {int(value) for value in routing.get("openingMidiSet") or []}
        opening_f1 = _set_f1(set(opening), expected_opening)
        minimum_opening_f1 = float(routing.get("minimumOpeningMidiF1", 1.0))
        add(
            "openingMidiSet",
            {"midis": opening, "f1": _rounded(opening_f1)},
            {"midis": sorted(expected_opening), "minimumF1": minimum_opening_f1},
            bool(expected_opening) and opening_f1 >= minimum_opening_f1,
        )

        bass_range = list(routing.get("openingBassMidiRange") or [])
        top_range = list(routing.get("openingTopMidiRange") or [])
        source_bass = min(opening) if opening else None
        source_top = max(opening) if opening else None
        add("openingBassMidi", source_bass, bass_range, _range_pass(float(source_bass) if source_bass is not None else None, bass_range))
        add("openingTopMidi", source_top, top_range, _range_pass(float(source_top) if source_top is not None else None, top_range))

        expected_moving = [int(value) % 12 for value in routing.get("movingPitchClassSequence") or []]
        if source_bass is None or source_top is None or not expected_moving:
            raise ValueError("harmonic fingerprint configuration is incomplete")
        stable_intervals = {
            int(note["intervalFromBass"]) % 12
            for template in profile.get("templates") or []
            for note in template.get("stableNotes") or []
        }
        stable_pitch_classes = {(source_bass + interval) % 12 for interval in stable_intervals}
        stable_pitch_classes.add(source_top % 12)
        cycle_seconds = pulse * int(profile.get("clock", {}).get("subdivisions", 16))
        cutoff = first_voice - 2.0 * pulse
        anchors = _source_cycle_transition_anchors(
            non_voice,
            start=first_source,
            cutoff=cutoff,
            cycle_seconds=cycle_seconds,
            cycles=len(expected_moving),
            source_bass_midi=source_bass,
            top_source_midi=source_top,
            stable_pitch_classes=stable_pitch_classes,
        )
        observed_moving: list[int] = []
        for anchor in anchors[: len(expected_moving)]:
            evidence = _nearest_group(non_voice, anchor, max(0.20, pulse * 1.6))
            observed_moving.append(
                _source_moving_pitch(
                    evidence,
                    bass_midi=source_bass,
                    top_midi=source_top,
                    stable_pitch_classes=stable_pitch_classes,
                )
                % 12
            )
        agreement = sum(a == b for a, b in zip(observed_moving, expected_moving)) / max(1, len(expected_moving))
        minimum_agreement = float(routing.get("minimumMovingSequenceAgreement", 1.0))
        add(
            "movingPitchClassSequence",
            {"sequence": observed_moving, "agreement": _rounded(agreement)},
            {"sequence": expected_moving, "minimumAgreement": minimum_agreement},
            len(observed_moving) == len(expected_moving) and agreement >= minimum_agreement,
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        checks["sourceEvidenceReadable"] = {
            "passed": False,
            "observed": type(error).__name__,
            "expected": "complete and finite intro evidence",
        }

    failed = [name for name, result in checks.items() if not result["passed"]]
    return {
        "profileId": profile.get("id"),
        "matched": not failed,
        "failedChecks": failed,
        "checks": checks,
    }


def apply_intro_motif(
    source: dict[str, Any], candidate: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the trained relative motif without reading any target score."""

    groups = _source_groups(source)
    voices = [float(note["time"]) for group in groups for note in group if _is_voice(note)]
    if not voices:
        raise ValueError("a pre-vocal boundary cannot be inferred without voice events")
    first_voice = min(voices)
    non_voice_groups = [group for group in groups if not all(_is_voice(note) for note in group)]
    intro_groups = [group for group in non_voice_groups if float(group[0]["time"]) < first_voice]
    if len(intro_groups) < 8:
        raise ValueError("source has too little instrumental intro evidence")
    pulse = _estimate_source_pulse(non_voice_groups, first_voice)
    subdivisions = int(profile["clock"]["subdivisions"])
    cycle_seconds = pulse * subdivisions
    start = float(intro_groups[0][0]["time"])
    cutoff = first_voice - 2.0 * pulse
    initial = _nearest_group(non_voice_groups, start, pulse)
    initial_midis = sorted({int(note["midi"]) for note in initial if not _is_voice(note)})
    if len(initial_midis) < 3:
        raise ValueError("intro anchor needs at least three harmonic pitches")
    source_bass_midi = min(initial_midis)
    application = profile.get("application", {})
    bass_shift = int(application.get("bassSourceOctaveShift", 0))
    bass_midi = source_bass_midi + bass_shift
    top_source_midi = max(initial_midis)
    stable_intervals = {
        int(note["intervalFromBass"]) % 12
        for template in profile["templates"]
        for note in template.get("stableNotes") or []
    }
    stable_source_pitch_classes = {(source_bass_midi + interval) % 12 for interval in stable_intervals}
    stable_source_pitch_classes.add(top_source_midi % 12)
    moving_shift = int(application.get("movingSourceOctaveShift", 12))

    source_anchor_blend = max(0.0, min(1.0, float(application.get("sourceCycleAnchorBlend", 0.0))))
    source_tempo_blend = max(0.0, min(1.0, float(application.get("sourceCycleTempoBlend", 0.0))))
    minimum_cycle_separation_ratio = max(0.0, min(1.0, float(application.get("minimumCycleSeparationRatio", 0.95))))
    minimum_cycle_separation = cycle_seconds * minimum_cycle_separation_ratio
    planned_cycles = max(1, math.ceil((cutoff - start) / cycle_seconds))
    source_anchors = _source_cycle_transition_anchors(
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
    cadence_time = first_voice - float(cadence.get("voiceLeadPulses", 2.0)) * pulse if cadence_enabled else None
    attacked_slots = [int(value) for value in profile["clock"]["attackedSlots"]]
    timing_offsets = [float(value) for value in profile["clock"].get("attackedSlotOffsetsPulses", [])]
    if timing_offsets and len(timing_offsets) != len(attacked_slots):
        raise ValueError("attackedSlotOffsetsPulses must match attackedSlots length")
    timing_offset_by_slot = {slot: timing_offsets[index] if timing_offsets else 0.0 for index, slot in enumerate(attacked_slots)}
    applied_anchor_blends: list[float] = []

    while start + cycles * cycle_seconds < cutoff:
        fixed_cycle_start = start + cycles * cycle_seconds
        source_cycle_start = source_anchors[cycles]
        local_anchor_blend = _cycle_anchor_blend(
            application,
            cycle=cycles,
            planned_cycles=planned_cycles,
            fallback=source_anchor_blend,
        )
        applied_anchor_blends.append(local_anchor_blend)
        cycle_start = fixed_cycle_start * (1.0 - local_anchor_blend) + source_cycle_start * local_anchor_blend
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
        local_pulse = pulse * (1.0 - source_tempo_blend) + source_cycle_pulse * source_tempo_blend
        evidence = _nearest_group(non_voice_groups, cycle_start, max(0.20, pulse * 1.6))
        moving_output = _source_moving_pitch(
            evidence,
            bass_midi=source_bass_midi,
            top_midi=top_source_midi,
            stable_pitch_classes=stable_source_pitch_classes,
        ) + moving_shift

        for grid_slot, template in zip(attacked_slots, profile["templates"]):
            if cadence_enabled and cycles == cadence_cycle and grid_slot >= cadence_replace_from_slot:
                continue
            time = cycle_start + (grid_slot + timing_offset_by_slot[grid_slot]) * local_pulse
            if time >= cutoff:
                continue
            gesture_velocity = float(template["velocityMedian"])
            wrote = False
            for stable in template.get("stableNotes") or []:
                midi = bass_midi + int(stable["intervalFromBass"])
                hand = str(stable["hand"])
                output_notes.append(
                    _generated_note(
                        midi=midi,
                        time=time,
                        duration=float(stable["durationPulses"]) * local_pulse,
                        velocity=gesture_velocity,
                        hand=hand,
                        role="bass" if hand == "left" and midi == bass_midi else "harmony",
                        cycle=cycles,
                        grid_slot=grid_slot,
                    )
                )
                wrote = True
            if cycles == 0:
                for opening in template.get("firstCycleNotes") or []:
                    midi = bass_midi + int(opening["intervalFromBass"])
                    hand = str(opening["hand"])
                    output_notes.append(
                        _generated_note(
                            midi=midi,
                            time=time,
                            duration=float(opening["durationPulses"]) * local_pulse,
                            velocity=gesture_velocity,
                            hand=hand,
                            role="harmony",
                            cycle=cycles,
                            grid_slot=grid_slot,
                        )
                    )
                    wrote = True
            moving = template.get("movingRightVoice") or {}
            if moving.get("enabled"):
                output_notes.append(
                    _generated_note(
                        midi=moving_output,
                        time=time,
                        duration=float(moving["durationPulses"]) * local_pulse,
                        velocity=gesture_velocity,
                        hand="right",
                        role="melody",
                        cycle=cycles,
                        grid_slot=grid_slot,
                    )
                )
                wrote = True
            generated_gestures += int(wrote)
        previous_cycle_start = cycle_start
        cycles += 1

    generated_cadence_notes = 0
    if cadence_enabled and cadence_time is not None and cadence_time < first_voice:
        evidence = _nearest_group(non_voice_groups, cadence_time, max(0.25, pulse * 2.1))
        evidence_midis = sorted({int(note["midi"]) for note in evidence if not _is_voice(note)})
        if evidence_midis:
            cadence_root = min(evidence_midis) + int(cadence.get("rootSourceOctaveShift", 0))
            cadence_velocity = float(cadence.get("velocityMedian", 0.7))
            for cadence_note in cadence.get("notes") or []:
                midi = cadence_root + int(cadence_note["intervalFromRoot"])
                hand = str(cadence_note["hand"])
                output_notes.append(
                    _generated_note(
                        midi=midi,
                        time=cadence_time,
                        duration=float(cadence_note["durationPulses"]) * pulse,
                        velocity=cadence_velocity,
                        hand=hand,
                        role="bass" if hand == "left" and int(cadence_note["intervalFromRoot"]) == 0 else "harmony",
                        cycle=cadence_cycle,
                        grid_slot=cadence_replace_from_slot,
                    )
                )
                generated_cadence_notes += 1
            generated_gestures += int(generated_cadence_notes > 0)

    retained = [
        dict(note)
        for note in candidate.get("notes") or []
        if _finite(note.get("time"), -1) >= cutoff
    ]
    output = copy.deepcopy(candidate)
    output["notes"] = sorted(
        [*output_notes, *retained],
        key=lambda note: (_finite(note.get("time"), 0), int(note.get("midi", 0))),
    )
    diagnostics = {
        "applied": True,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "scope": "pre-vocal-intro-only",
        "firstSourceOnsetSeconds": _rounded(start),
        "firstVoiceOnsetSeconds": _rounded(first_voice),
        "replacementCutoffSeconds": _rounded(cutoff),
        "sourcePulseSeconds": _rounded(pulse),
        "sourceBassMidi": source_bass_midi,
        "bassSourceOctaveShift": bass_shift,
        "outputBassMidi": bass_midi,
        "movingSourceOctaveShift": moving_shift,
        "sourceCycleAnchorBlend": _rounded(source_anchor_blend),
        "sourceCycleAnchorEnvelope": application.get("sourceCycleAnchorEnvelope"),
        "appliedSourceCycleAnchorBlends": [_rounded(value) for value in applied_anchor_blends],
        "sourceCycleTempoBlend": _rounded(source_tempo_blend),
        "attackedSlotOffsetsPulses": [_rounded(timing_offset_by_slot[slot]) for slot in attacked_slots],
        "minimumCycleSeparationRatio": _rounded(minimum_cycle_separation_ratio),
        "minimumCycleSeparationSeconds": _rounded(minimum_cycle_separation),
        "adjustedCycleBoundaries": adjusted_cycle_boundaries,
        "preVoiceCadenceApplied": generated_cadence_notes > 0,
        "preVoiceCadenceTimeSeconds": _rounded(cadence_time) if cadence_time is not None else None,
        "preVoiceCadenceNotes": generated_cadence_notes,
        "sourceCycleTransitionAnchors": [_rounded(value) for value in source_anchors],
        "cycleSeconds": _rounded(cycle_seconds),
        "cycles": cycles,
        "generatedGestures": generated_gestures,
        "generatedNotes": len(output_notes),
        "retainedCandidateNotes": len(retained),
        "inferenceReadTarget": False,
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistMotifGrammar"] = diagnostics
    arrangement["outputNoteCount"] = len(output["notes"])
    output["arrangementProfile"] = f"{candidate.get('arrangementProfile', 'piano')}-motif-v1"
    return output, diagnostics


def apply_matching_exact_song_intro(
    source: dict[str, Any],
    candidate: dict[str, Any],
    mode: str,
    profiles: list[dict[str, Any]] | None,
    *,
    allow_research_profiles: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the first fully matched profile, or return the candidate unchanged."""

    considered: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": "polymath-exact-song-intro-router-v1",
        "applied": False,
        "failClosed": True,
        "researchProfilesAllowed": bool(allow_research_profiles),
        "reason": "no-profile-matched",
        "consideredProfiles": considered,
    }
    for profile in profiles or []:
        match = evaluate_exact_song_match(source, mode, profile)
        considered.append(match)
        if not match["matched"]:
            continue
        training = profile.get("training") or {}
        research_only = (
            str(training.get("decision") or "").upper() == "RESEARCH_ONLY"
            or training.get("commercialUseAllowed") is False
        )
        if research_only and not allow_research_profiles:
            match["policyBlocked"] = True
            summary.update(
                {
                    "reason": "matched-research-profile-disabled",
                    "matchedProfileId": profile.get("id"),
                }
            )
            return candidate, summary
        try:
            output, motif = apply_intro_motif(source, candidate, profile)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            match["applicationError"] = type(error).__name__
            summary.update(
                {
                    "reason": "matched-profile-application-failed",
                    "matchedProfileId": profile.get("id"),
                }
            )
            return candidate, summary
        summary.update(
            {
                "applied": True,
                "reason": "strict-profile-match",
                "matchedProfileId": profile.get("id"),
                "motif": motif,
            }
        )
        output.setdefault("pianoArrangement", {})["exactSongIntroRouting"] = summary
        return output, summary
    return candidate, summary
