"""Correct one repeated gesture while preserving destination performance.

Full-phrase copying is unsafe when two otherwise equivalent phrases breathe at
different times.  This adapter uses a surrounding input recurrence only as
evidence that one learned pianist gesture applies again.  It replaces the keys
at the matched destination onset, while retaining that destination's onset,
shared velocity, and articulation.  The destination reference is never read by
``apply_profile``.

Profiles are deliberately labelled same-song research.  They are building
blocks for a future context decoder, not proof of unseen-song generalisation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .analyze_pianist_gesture_patterns import group_onsets, normalize_notes, pitch_set, set_f1
    from .analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import evaluation, finite, midi_to_note, note_duration
    from .train_pianist_repetition_adapter import (
        detect_repeat,
        infer_transposition,
        slice_groups,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import group_onsets, normalize_notes, pitch_set, set_f1
    from analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import evaluation, finite, midi_to_note, note_duration
    from train_pianist_repetition_adapter import detect_repeat, infer_transposition, slice_groups


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def gesture_velocity(group: list[dict[str, Any]]) -> float:
    return median(finite(note.get("velocity"), 0.7) for note in group)


def nearest_group_index(
    groups: list[list[dict[str, Any]]], value: float, maximum_distance: float
) -> int | None:
    if not groups:
        return None
    index = min(
        range(len(groups)),
        key=lambda item: abs(float(groups[item][0]["time"]) - value),
    )
    if abs(float(groups[index][0]["time"]) - value) > maximum_distance:
        return None
    return index


def clean_learned_note(note: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "_payloadIndex",
        "sourceIndex",
        "originalTime",
        "originalDuration",
        "referenceTime",
        "referenceDuration",
    }
    return {key: value for key, value in note.items() if key not in excluded}


def context_record(
    group: list[dict[str, Any]], template_start: float
) -> dict[str, Any]:
    return {
        "relativeTime": rounded(float(group[0]["time"]) - template_start),
        "midis": sorted({int(note["midi"]) for note in group}),
        "occupancy": occupancy(group),
    }


def recorded_context_group(
    record: dict[str, Any], template_start: float
) -> list[dict[str, Any]]:
    return [
        {
            "time": template_start + float(record["relativeTime"]),
            "midi": int(midi),
            "duration": 0.2,
            "velocity": 0.7,
            "hand": "left" if int(midi) < 60 else "right",
        }
        for midi in record["midis"]
    ]


def fit_profile(
    reference_groups: list[list[dict[str, Any]]],
    *,
    profile_id: str,
    song_id: str,
    template_start: float,
    template_end: float,
    target_time: float,
    repeat_search_minimum: float,
    repeat_search_maximum: float,
    minimum_input_pitch_f1: float = 0.80,
    maximum_input_normalized_score: float = 0.25,
    learned_minimum_midi: int = 21,
    learned_maximum_midi: int = 108,
    learned_midis: set[int] | None = None,
) -> dict[str, Any]:
    context = slice_groups(reference_groups, template_start, template_end)
    target_index = nearest_group_index(context, target_time, 0.30)
    if target_index is None:
        raise ValueError("No approved gesture was found near target_time")
    target = context[target_index]
    learned_target = [
        note
        for note in target
        if learned_minimum_midi <= int(note["midi"]) <= learned_maximum_midi
        and (learned_midis is None or int(note["midi"]) in learned_midis)
    ]
    if not learned_target:
        raise ValueError("The learned MIDI register filter removed every target note")
    profile = {
        "schema": "polymath-pianist-contextual-gesture-profile-v1",
        "id": profile_id,
        "enabled": True,
        "training": {
            "songId": song_id,
            "method": "input-recurrence-conditioned-single-gesture-correction",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
            "learnedMidiRange": [learned_minimum_midi, learned_maximum_midi],
            "learnedMidiAllowlist": (
                sorted(learned_midis) if learned_midis is not None else None
            ),
        },
        "context": {
            "templateStartSeconds": template_start,
            "templateEndSeconds": template_end,
            "targetTimeSeconds": float(target[0]["time"]),
            "targetRelativeTimeSeconds": rounded(
                float(target[0]["time"]) - template_start
            ),
            "targetPitchClasses": sorted(pitch_set(target, True)),
            "targetOccupancy": occupancy(target),
            "targetOrdinal": target_index,
            "gestures": [
                context_record(group, template_start) for group in context
            ],
        },
        "learnedGesture": {
            "notes": [clean_learned_note(note) for note in learned_target],
            "noteCount": len(learned_target),
            "pitchClasses": sorted(pitch_set(learned_target, True)),
            "occupancy": occupancy(learned_target),
        },
        "detection": {
            "minimumOffsetSeconds": repeat_search_minimum,
            "maximumOffsetSeconds": repeat_search_maximum,
            "coarseStepSeconds": 0.10,
            "fineStepSeconds": 0.01,
            "maximumTimeDistanceSeconds": 0.35,
            "gapCost": 0.85,
            "minimumMeanPitchClassF1": minimum_input_pitch_f1,
            "maximumNormalizedScore": maximum_input_normalized_score,
            "maximumTargetTimeDistanceSeconds": 0.40,
            "minimumTemplateTargetPitchClassF1": 0.50,
            "minimumTranspositionGainOverUnison": 0.08,
        },
        "application": {
            "targetReadAtInference": False,
            "preserveDestinationOnset": True,
            "preserveDestinationVelocity": True,
            "preserveDestinationArticulation": True,
            "replaceCompleteOnsetGesture": True,
            "structureMode": "replace-learned-gesture",
            "maximumCoalescingDistanceSeconds": 0.22,
            "longMelodyHold": fit_long_melody_hold(
                reference_groups,
                evidence_end=template_start,
            ),
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    return profile


def destination_duration(
    midi: int,
    destination: list[dict[str, Any]],
    learned_note: dict[str, Any],
    learned_pitch_classes: set[int] | None = None,
    learned_midis: set[int] | None = None,
) -> float:
    exact = [note for note in destination if int(note["midi"]) == midi]
    if exact:
        return note_duration(exact[0])
    supporting = [
        note
        for note in destination
        if (
            learned_midis is not None
            and int(note["midi"]) in learned_midis
        )
        or (
            learned_midis is None
            and learned_pitch_classes is not None
            and int(note["midi"]) % 12 in learned_pitch_classes
        )
    ]
    if supporting:
        # When other correct tones survive at this destination, they reveal
        # the local articulation.  A missing tone joins that physical chord
        # rather than importing a longer/shorter hold from another phrase.
        return median(note_duration(note) for note in supporting)
    # A destination note with another pitch often belongs to the very error
    # being corrected.  Its release is not evidence for a newly introduced
    # key.  When no same pitch class exists, retain the learned motif's
    # articulation instead (the repeated gesture still keeps its own onset,
    # velocity, and every duration that has direct pitch support).
    return note_duration(learned_note)


def destination_exact_or_learned_duration(
    midi: int,
    destination: list[dict[str, Any]],
    learned_note: dict[str, Any],
) -> float:
    """Keep a detected key's hold, but restore a missing key's own hold.

    Pianist voicings often release two notes in one onset at different times.
    Sharing the shortest or median destination duration is therefore unsafe
    when the missing tone is the sustained voice.  Exact-key evidence remains
    authoritative; only a truly absent key inherits the supervised gesture's
    articulation.
    """

    exact = [note for note in destination if int(note["midi"]) == midi]
    return note_duration(exact[0]) if exact else note_duration(learned_note)


def preserve_allowed_destination_pitch_classes(
    destination: list[dict[str, Any]],
    allowed_pitch_classes: set[int],
) -> list[dict[str, Any]]:
    """Prune harmonic outliers without changing valid destination keys."""

    output = []
    for destination_note in destination:
        if int(destination_note["midi"]) % 12 not in allowed_pitch_classes:
            continue
        note = clean_learned_note(destination_note)
        note["source"] = "polymath-pianist-contextual-harmonic-mask"
        note["sourceInstrument"] = "destination_piano_harmonic_mask"
        note["generatedBy"] = "pianist-contextual-gesture-adapter-v1"
        output.append(note)
    return output


def preserve_allowed_destination_exact_keys(
    destination: list[dict[str, Any]],
    allowed_midis: set[int],
) -> list[dict[str, Any]]:
    """Prune wrong-register doublings while preserving detected performance.

    Pitch-class masks deliberately treat octave equivalents as the same note.
    That is useful for harmony, but it cannot distinguish a Pianella voicing
    such as A#5 from a spurious A#4 doubling.  This stricter mask is only used
    after contextual recurrence detection has supplied an exact MIDI allowlist.
    The caller's empty-generation guard leaves the destination untouched when
    none of its notes are supported, so uncertain octave evidence cannot erase
    a gesture.
    """

    output = []
    for destination_note in destination:
        if int(destination_note["midi"]) not in allowed_midis:
            continue
        note = clean_learned_note(destination_note)
        note["source"] = "polymath-pianist-contextual-exact-voicing-mask"
        note["sourceInstrument"] = "destination_piano_exact_voicing_mask"
        note["generatedBy"] = "pianist-contextual-gesture-adapter-v1"
        output.append(note)
    return output


def correct_destination_registers(
    destination: list[dict[str, Any]],
    learned_notes: list[dict[str, Any]],
    *,
    transposition: int = 0,
) -> list[dict[str, Any]]:
    """Correct octave placement for learned voices without replacing the chord.

    Audio transcription commonly detects the right pitch class in the wrong
    octave.  For each supervised voice this keeps an exact destination key when
    present; otherwise it replaces the closest same-pitch-class key and carries
    over that detected key's duration.  Notes belonging to other pitch classes
    are untouched, which lets a learned accompaniment correction coexist with a
    changing melody voice.
    """

    shifted_learned = [
        {**note, "midi": int(note["midi"]) + transposition}
        for note in learned_notes
        if 21 <= int(note["midi"]) + transposition <= 108
    ]
    learned_midis = {int(note["midi"]) for note in shifted_learned}
    learned_pitch_classes = {midi % 12 for midi in learned_midis}
    destination_time = float(destination[0]["time"])
    velocity = gesture_velocity(destination)
    output = [
        clean_learned_note(note)
        for note in destination
        if int(note["midi"]) % 12 not in learned_pitch_classes
        or int(note["midi"]) in learned_midis
    ]
    present = {int(note["midi"]) for note in output}
    for learned_note in shifted_learned:
        midi = int(learned_note["midi"])
        if midi in present:
            continue
        same_pitch_class = [
            note
            for note in destination
            if int(note["midi"]) % 12 == midi % 12
        ]
        evidence = min(
            same_pitch_class,
            key=lambda note: abs(int(note["midi"]) - midi),
        ) if same_pitch_class else None
        learned_duration = note_duration(learned_note)
        peer_holds = []
        for peer in shifted_learned:
            peer_midi = int(peer["midi"])
            if peer_midi == midi or peer_midi not in present:
                continue
            peer_duration = note_duration(peer)
            ratio = learned_duration / max(0.03, peer_duration)
            if not 0.75 <= ratio <= 1.33:
                continue
            exact_destination_peer = next(
                (
                    note
                    for note in destination
                    if int(note["midi"]) == peer_midi
                ),
                None,
            )
            if exact_destination_peer is not None:
                peer_holds.append(note_duration(exact_destination_peer))
        duration = (
            median(peer_holds)
            if peer_holds
            else note_duration(evidence)
            if evidence is not None
            else learned_duration
        )
        note = clean_learned_note(learned_note)
        note.update(
            {
                "midi": midi,
                "note": midi_to_note(midi),
                "time": rounded(destination_time),
                "duration": rounded(duration),
                "scoreDuration": rounded(duration),
                "visualDuration": rounded(duration),
                "audioDuration": rounded(duration),
                "velocity": rounded(velocity),
                "hand": explicit_hand({**learned_note, "midi": midi}),
                "source": "polymath-pianist-contextual-register-correction",
                "sourceInstrument": "learned_pianist_register_context",
                "generatedBy": "pianist-contextual-gesture-adapter-v1",
            }
        )
        output.append(note)
        present.add(midi)
    return sorted(output, key=lambda note: int(note["midi"]))


def complete_destination_pitch_classes(
    destination: list[dict[str, Any]],
    learned_notes: list[dict[str, Any]],
    *,
    transposition: int = 0,
    groups: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Keep valid destination voicing and fill one musically placed missing tone."""

    shifted_learned = [
        {**note, "midi": int(note["midi"]) + transposition}
        for note in learned_notes
        if 21 <= int(note["midi"]) + transposition <= 108
    ]
    allowed = {int(note["midi"]) % 12 for note in shifted_learned}
    output = preserve_allowed_destination_pitch_classes(destination, allowed)
    if not output:
        # With no valid destination anchor, octave placement becomes a guess.
        # A conservative adapter leaves that case to a richer model.
        return []
    present = {int(note["midi"]) % 12 for note in output}
    missing = sorted(allowed - present)
    learned_occupancy = occupancy(shifted_learned)
    destination_occupancy = occupancy(output)
    destination_time = float(destination[0]["time"])
    destination_velocity = gesture_velocity(destination)
    destination_hold = median(note_duration(note) for note in output)
    next_onset_gap = next(
        (
            float(group[0]["time"]) - destination_time
            for group in (groups or [])
            if float(group[0]["time"]) > destination_time + 0.03
        ),
        None,
    )
    anchor = max(int(note["midi"]) for note in output)

    for pitch_class in missing:
        candidates = [midi for midi in range(21, 109) if midi % 12 == pitch_class]
        needs_left = (
            learned_occupancy in {"left-only", "both"}
            and destination_occupancy == "right-only"
        )
        needs_right = (
            learned_occupancy in {"right-only", "both"}
            and destination_occupancy == "left-only"
        )
        if needs_left:
            register = [midi for midi in candidates if midi < 60]
            midi = max(register) if register else min(candidates, key=lambda value: abs(value - anchor))
            hand = "left"
        elif needs_right:
            register = [midi for midi in candidates if midi >= 60]
            midi = min(register, key=lambda value: abs(value - anchor)) if register else min(
                candidates, key=lambda value: abs(value - anchor)
            )
            hand = "right"
        else:
            close_above = [midi for midi in candidates if 0 < midi - anchor <= 7]
            midi = min(close_above) if close_above else min(
                candidates,
                key=lambda value: (abs(value - anchor), value),
            )
            hand = "left" if midi < 60 else "right"
        templates = [
            note
            for note in shifted_learned
            if int(note["midi"]) % 12 == pitch_class
        ]
        same_hand = [note for note in templates if explicit_hand(note) == hand]
        template = min(
            same_hand or templates,
            key=lambda note: abs(int(note["midi"]) - midi),
        )
        note = clean_learned_note(template)
        generated_hold = (
            min(destination_hold, next_onset_gap)
            if needs_left and next_onset_gap is not None
            else destination_hold
        )
        note.update(
            {
                "midi": midi,
                "note": midi_to_note(midi),
                "time": rounded(destination_time),
                "duration": rounded(generated_hold),
                "scoreDuration": rounded(generated_hold),
                "visualDuration": rounded(generated_hold),
                "audioDuration": rounded(generated_hold),
                "velocity": rounded(destination_velocity),
                "hand": hand,
                "source": "polymath-pianist-contextual-pitch-completion",
                "sourceInstrument": "learned_pianist_harmonic_context",
                "generatedBy": "pianist-contextual-gesture-adapter-v1",
            }
        )
        output.append(note)
        destination_occupancy = occupancy(output)
        anchor = max(anchor, midi)
    return sorted(output, key=lambda note: int(note["midi"]))


def add_learned_keys_to_destination(
    destination: list[dict[str, Any]],
    learned_notes: list[dict[str, Any]],
    *,
    transposition: int = 0,
) -> list[dict[str, Any]]:
    """Layer absent learned keys under/over a destination without deleting it."""

    output = [clean_learned_note(note) for note in destination]
    present = {int(note["midi"]) for note in output}
    destination_time = float(destination[0]["time"])
    velocity = gesture_velocity(destination)
    for learned_note in learned_notes:
        midi = int(learned_note["midi"]) + transposition
        if midi in present or not 21 <= midi <= 108:
            continue
        duration = note_duration(learned_note)
        note = clean_learned_note(learned_note)
        note.update(
            {
                "midi": midi,
                "note": midi_to_note(midi),
                "time": rounded(destination_time),
                "duration": rounded(duration),
                "scoreDuration": rounded(duration),
                "visualDuration": rounded(duration),
                "audioDuration": rounded(duration),
                "velocity": rounded(velocity),
                "hand": explicit_hand({**learned_note, "midi": midi}),
                "source": "polymath-pianist-contextual-layering",
                "sourceInstrument": "learned_pianist_layer",
                "generatedBy": "pianist-contextual-gesture-adapter-v1",
            }
        )
        output.append(note)
        present.add(midi)
    return sorted(output, key=lambda note: int(note["midi"]))


def fit_long_melody_hold(
    reference_groups: list[list[dict[str, Any]]],
    *,
    evidence_end: float,
) -> dict[str, Any]:
    """Learn phrase-spanning melody sustain from earlier approved material."""

    ratios: list[float] = []
    for index, group in enumerate(reference_groups):
        onset = float(group[0]["time"])
        midis = sorted({int(note["midi"]) for note in group})
        if onset >= evidence_end or len(midis) != 1 or midis[0] < 79:
            continue
        if occupancy(group) != "right-only":
            continue
        cue_time = next(
            (
                float(following[0]["time"])
                for following in reference_groups[index + 1 :]
                if float(following[0]["time"]) < evidence_end
                and any(
                    explicit_hand(note) == "right"
                    and int(note["midi"]) >= midis[0] - 7
                    for note in following
                )
            ),
            None,
        )
        if cue_time is None:
            continue
        gap = cue_time - onset
        duration = max(note_duration(note) for note in group)
        ratio = duration / max(0.03, gap)
        if 1.20 <= gap <= 2.10 and 0.40 <= ratio <= 1.0:
            ratios.append(ratio)
    return {
        "holdToNextHighCueRatio": rounded(median(ratios) if ratios else 0.76),
        "trainingExamples": len(ratios),
        "evidenceEndSeconds": rounded(evidence_end),
        "minimumCueGapSeconds": 1.20,
        "maximumCueGapSeconds": 2.10,
        "highCueFloorSemitonesBelowTarget": 7,
        "releaseGuardSeconds": 0.03,
    }


def trained_long_melody_duration(
    groups: list[list[dict[str, Any]]],
    *,
    destination_time: float,
    midi: int,
    learned_note: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[float, float | None]:
    """Extend one melody note toward the next high-register input cue."""

    floor = midi - int(settings.get("highCueFloorSemitonesBelowTarget", 7))
    cue_time = next(
        (
            float(group[0]["time"])
            for group in groups
            if float(group[0]["time"]) > destination_time + 0.03
            and any(
                explicit_hand(note) == "right" and int(note["midi"]) >= floor
                for note in group
            )
        ),
        None,
    )
    if cue_time is None:
        return note_duration(learned_note), None
    gap = cue_time - destination_time
    minimum_gap = float(settings.get("minimumCueGapSeconds", 1.20))
    maximum_gap = float(settings.get("maximumCueGapSeconds", 2.10))
    if not minimum_gap <= gap <= maximum_gap:
        return note_duration(learned_note), cue_time
    ratio = float(settings.get("holdToNextHighCueRatio", 0.76))
    release_guard = float(settings.get("releaseGuardSeconds", 0.03))
    duration = min(gap - release_guard, ratio * gap)
    return max(note_duration(learned_note), duration), cue_time


def nearby_isolated_pitch_evidence(
    groups: list[list[dict[str, Any]]],
    *,
    destination_time: float,
    midi: int,
    maximum_distance: float,
    excluded_indices: set[int] | None = None,
) -> dict[str, Any] | None:
    """Find one displaced note that can safely be folded into a chord.

    Source separation and onset detection sometimes place a melody tone a few
    milliseconds after its accompaniment. When the learned gesture says that
    pitch belongs in the chord, an isolated nearby strike is stronger duration
    evidence than the accompaniment notes. Multi-note neighboring gestures
    are excluded because moving one tone could damage a legitimate next chord.
    """

    excluded = excluded_indices or set()
    options: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for group in groups:
        group_time = float(group[0]["time"])
        distance = abs(group_time - destination_time)
        # This rule repairs a delayed melody onset. An earlier strike is more
        # likely to be a legitimate preceding note and is never pulled forward.
        if group_time <= destination_time or distance > maximum_distance:
            continue
        if len(pitch_set(group)) != 1:
            continue
        for note in group:
            payload_index = int(note.get("_payloadIndex", -1))
            if payload_index in excluded:
                continue
            candidate_midi = int(note["midi"])
            if candidate_midi % 12 != midi % 12:
                continue
            options.append(
                (
                    (
                        0.0 if candidate_midi == midi else 1.0,
                        distance,
                        group_time,
                    ),
                    note,
                )
            )
    return min(options, key=lambda item: item[0])[1] if options else None


def reanchor_delayed_isolated_gesture(
    groups: list[list[dict[str, Any]]],
    *,
    destination: list[dict[str, Any]],
    learned_midis: list[int],
    maximum_distance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Anchor a late solo tone to the preceding partial accompaniment chord."""

    if not destination or len(pitch_set(destination, True)) != 1:
        return destination, None
    destination_time = float(destination[0]["time"])
    learned_pitch_classes = {midi % 12 for midi in learned_midis}
    current_pitch_classes = pitch_set(destination, True)
    current_f1 = set_f1(learned_pitch_classes, current_pitch_classes)
    options: list[tuple[tuple[float, ...], list[dict[str, Any]], float]] = []
    for group in groups:
        group_time = float(group[0]["time"])
        distance = destination_time - group_time
        if distance <= 1e-9 or distance > maximum_distance:
            continue
        prior_pitch_classes = pitch_set(group, True)
        if not (prior_pitch_classes & learned_pitch_classes):
            continue
        combined_f1 = set_f1(
            learned_pitch_classes,
            prior_pitch_classes | current_pitch_classes,
        )
        prior_f1 = set_f1(learned_pitch_classes, prior_pitch_classes)
        gain = combined_f1 - max(current_f1, prior_f1)
        if gain < 0.10:
            continue
        options.append(((-gain, -combined_f1, distance), group, combined_f1))
    if not options:
        return destination, None
    _rank, selected, combined_f1 = min(options, key=lambda item: item[0])
    return selected, {
        "fromTimeSeconds": rounded(destination_time),
        "toTimeSeconds": rounded(float(selected[0]["time"])),
        "combinedPitchClassF1": rounded(combined_f1),
        "reason": "late-isolated-tone-plus-preceding-partial-chord",
    }


def coalesced_note_duration(
    *,
    midi: int,
    learned_note: dict[str, Any],
    learned_notes: list[dict[str, Any]],
    destination: list[dict[str, Any]],
    learned_pitch_classes: set[int],
    learned_midis: set[int],
    evidence: dict[str, Any],
    destination_time: float,
) -> tuple[float, str]:
    """Choose chord-shared or release-preserving articulation from training."""

    learned_duration = note_duration(learned_note)
    peer_durations = [
        note_duration(note) for note in learned_notes if note is not learned_note
    ]
    peer_duration = median(peer_durations) if peer_durations else learned_duration
    ratio = learned_duration / max(0.03, peer_duration)
    if 0.75 <= ratio <= 1.33:
        return (
            destination_duration(
                midi,
                destination,
                learned_note,
                learned_pitch_classes,
                learned_midis,
            ),
            "destination-chord-articulation",
        )
    original_duration = note_duration(evidence)
    return (
        max(
            0.03,
            original_duration + float(evidence["time"]) - destination_time,
        ),
        "detected-release-preserved",
    )


def regularized_transposition(
    inferred: int,
    trials: list[dict[str, Any]],
    minimum_gain_over_unison: float,
) -> tuple[int, float]:
    """Prefer an untransposed recurrence when noisy notes create a near tie."""

    scores = {
        int(row["semitones"]): float(row["meanPitchClassF1"]) for row in trials
    }
    gain = scores.get(inferred, 0.0) - scores.get(0, 0.0)
    if inferred and gain < minimum_gain_over_unison:
        return 0, gain
    return inferred, gain


def local_time_scale(
    template: list[list[dict[str, Any]]],
    repeated: list[list[dict[str, Any]]],
    matches: list[tuple[int, int]],
    source_index: int,
) -> float:
    """Estimate local tempo change from intervals surrounding one gesture."""

    by_source = {left: right for left, right in matches}
    ratios: list[float] = []
    for neighbor in (source_index - 1, source_index + 1):
        if neighbor not in by_source or source_index not in by_source:
            continue
        source_gap = abs(
            float(template[neighbor][0]["time"])
            - float(template[source_index][0]["time"])
        )
        destination_gap = abs(
            float(repeated[by_source[neighbor]][0]["time"])
            - float(repeated[by_source[source_index]][0]["time"])
        )
        if source_gap > 0.03 and destination_gap > 0.03:
            ratios.append(destination_gap / source_gap)
    return max(0.50, min(2.0, median(ratios) if ratios else 1.0))


def projected_destination_time(
    template: list[list[dict[str, Any]]],
    repeated: list[list[dict[str, Any]]],
    matches: list[tuple[int, int]],
    source_index: int,
    fallback_offset: float,
) -> float:
    """Project one masked target with a robust local time transform.

    The target match is deliberately excluded: its notes are exactly what the
    adapter is trying to repair, so using their pitch similarity can pull the
    destination onto an adjacent accompaniment strike.  A Theil-Sen-style
    median slope over the remaining context tolerates one wrongly paired
    neighbor while retaining the phrase's local tempo and position.
    """

    usable = [(left, right) for left, right in matches if left != source_index]
    source_time = float(template[source_index][0]["time"])
    if len(usable) >= 2:
        points = [
            (
                float(template[left][0]["time"]),
                float(repeated[right][0]["time"]),
            )
            for left, right in usable
        ]
        slopes = [
            (right_destination - left_destination)
            / (right_source - left_source)
            for index, (left_source, left_destination) in enumerate(points)
            for right_source, right_destination in points[index + 1 :]
            if right_source - left_source > 0.03
            and right_destination - left_destination > 0.0
        ]
        if slopes:
            # Local phrase tempo can stretch, but a wildly large pairwise
            # slope usually represents one incorrect sequence correspondence.
            # Taking the median first makes that pair an outlier; the clamp is
            # only a final physical guardrail.
            slope = max(0.50, min(2.0, median(slopes)))
            intercept = median(
                destination_time - slope * template_time
                for template_time, destination_time in points
            )
            return slope * source_time + intercept
    if usable:
        nearest = sorted(
            usable,
            key=lambda pair: abs(pair[0] - source_index),
        )[:4]
        local_offset = median(
            float(repeated[right][0]["time"])
            - float(template[left][0]["time"])
            for left, right in nearest
        )
        return source_time + local_offset
    return source_time + fallback_offset


def select_projected_destination(
    groups: list[list[dict[str, Any]]],
    *,
    mapped_destination: list[dict[str, Any]],
    projected_time: float,
    maximum_distance: float,
    expected_pitch_classes: set[int] | None = None,
    direct_offset_time: float | None = None,
    minimum_improvement: float = 0.025,
    maximum_pitch_class_f1_loss: float = 0.05,
) -> tuple[list[dict[str, Any]], str]:
    """Prefer a better-timed strike without jumping to a worse harmony."""

    if not groups:
        return mapped_destination, "sequence-alignment"
    if direct_offset_time is not None and mapped_destination:
        direct_nearest = min(
            groups,
            key=lambda group: abs(float(group[0]["time"]) - direct_offset_time),
        )
        direct_distance = abs(float(direct_nearest[0]["time"]) - direct_offset_time)
        mapped_direct_distance = abs(
            float(mapped_destination[0]["time"]) - direct_offset_time
        )
        if expected_pitch_classes:
            direct_f1 = set_f1(
                expected_pitch_classes,
                pitch_set(direct_nearest, True),
            )
            mapped_direct_f1 = set_f1(
                expected_pitch_classes,
                pitch_set(mapped_destination, True),
            )
        else:
            direct_f1 = mapped_direct_f1 = 1.0
        if (
            direct_nearest is not mapped_destination
            and direct_distance <= maximum_distance
            and direct_distance + minimum_improvement < mapped_direct_distance
            and direct_f1 > mapped_direct_f1 + maximum_pitch_class_f1_loss
        ):
            return direct_nearest, "direct-recurrence-clock"
    nearest = min(groups, key=lambda group: abs(float(group[0]["time"]) - projected_time))
    nearest_distance = abs(float(nearest[0]["time"]) - projected_time)
    mapped_distance = (
        abs(float(mapped_destination[0]["time"]) - projected_time)
        if mapped_destination
        else float("inf")
    )
    pitch_safe = True
    if expected_pitch_classes and mapped_destination:
        nearest_f1 = set_f1(
            expected_pitch_classes,
            pitch_set(nearest, True),
        )
        mapped_f1 = set_f1(
            expected_pitch_classes,
            pitch_set(mapped_destination, True),
        )
        pitch_safe = nearest_f1 + maximum_pitch_class_f1_loss >= mapped_f1
    direct_time_safe = True
    if direct_offset_time is not None and mapped_destination:
        nearest_direct_distance = abs(
            float(nearest[0]["time"]) - direct_offset_time
        )
        mapped_direct_distance = abs(
            float(mapped_destination[0]["time"]) - direct_offset_time
        )
        direct_time_safe = not (
            mapped_direct_distance + minimum_improvement
            < nearest_direct_distance
        )
    if (
        nearest_distance <= maximum_distance
        and nearest_distance + minimum_improvement < mapped_distance
        and pitch_safe
        and direct_time_safe
    ):
        return nearest, "neighbor-time-projection"
    return mapped_destination, "sequence-alignment"


def apply_profile(
    candidate: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = group_onsets(normalize_notes(candidate), 0.035)
    context = profile["context"]
    detection = profile["detection"]
    template_start = float(context["templateStartSeconds"])
    template_end = float(context["templateEndSeconds"])
    best, ranked = detect_repeat(
        groups,
        template_start=template_start,
        template_end=template_end,
        minimum_offset=float(detection["minimumOffsetSeconds"]),
        maximum_offset=float(detection["maximumOffsetSeconds"]),
        coarse_step=float(detection["coarseStepSeconds"]),
        fine_step=float(detection["fineStepSeconds"]),
        maximum_time_distance=float(detection["maximumTimeDistanceSeconds"]),
        gap_cost=float(detection["gapCost"]),
    )
    context_matches: list[tuple[int, int]] = []
    recorded_context = context.get("gestures") or []
    if recorded_context and context.get("targetOrdinal") is not None:
        trained_context = [
            recorded_context_group(record, template_start)
            for record in recorded_context
        ]
        context_matches, _, _, _ = align_sequences(
            trained_context,
            best["template"],
            maximum_time_distance=0.55,
            gap_cost=0.85,
        )
        source_index = dict(context_matches).get(int(context["targetOrdinal"]))
    else:
        # Backward-compatible fallback for early experimental profiles.
        source_index = nearest_group_index(
            best["template"],
            float(context["targetTimeSeconds"]),
            float(detection["maximumTargetTimeDistanceSeconds"]),
        )
    match_by_source = {left: right for left, right in best["matches"]}
    destination_index = (
        match_by_source.get(source_index) if source_index is not None else None
    )
    source_target_f1 = (
        set_f1(
            set(int(value) for value in context["targetPitchClasses"]),
            pitch_set(best["template"][source_index], True),
        )
        if source_index is not None
        else 0.0
    )
    context_confidence_passed = (
        float(best["meanPitchClassF1"])
        >= float(detection["minimumMeanPitchClassF1"])
        and float(best["normalizedScore"])
        <= float(detection["maximumNormalizedScore"])
        and source_target_f1
        >= float(detection["minimumTemplateTargetPitchClassF1"])
    )

    raw_transposition, transposition_trials = infer_transposition(
        best["template"], best["repeated"], best["matches"]
    )
    transposition, transposition_gain = regularized_transposition(
        raw_transposition,
        transposition_trials,
        float(detection.get("minimumTranspositionGainOverUnison", 0.08)),
    )
    mapped_destination = (
        best["repeated"][destination_index]
        if destination_index is not None
        else []
    )
    projected_time = (
        projected_destination_time(
            best["template"],
            best["repeated"],
            best["matches"],
            source_index,
            float(best["offsetSeconds"]),
        )
        if source_index is not None
        else None
    )
    direct_offset_time = (
        float(best["template"][source_index][0]["time"])
        + float(best["offsetSeconds"])
        if source_index is not None
        else None
    )
    destination, destination_selection = (
        select_projected_destination(
            groups,
            mapped_destination=mapped_destination,
            projected_time=projected_time,
            maximum_distance=float(detection["maximumTargetTimeDistanceSeconds"]),
            expected_pitch_classes={
                (int(value) + transposition) % 12
                for value in (
                    profile["learnedGesture"].get("pitchClasses")
                    or [
                        int(note["midi"]) % 12
                        for note in profile["learnedGesture"]["notes"]
                    ]
                )
            },
            direct_offset_time=direct_offset_time,
        )
        if projected_time is not None
        else (mapped_destination, "sequence-alignment")
    )
    confidence_passed = context_confidence_passed and bool(destination)
    articulation_source = str(
        profile.get("application", {}).get(
            "articulationSource", "destination-supported"
        )
    )
    structure_mode = str(
        profile.get("application", {}).get(
            "structureMode", "replace-learned-gesture"
        )
    )
    maximum_coalescing_distance = float(
        profile.get("application", {}).get(
            "maximumCoalescingDistanceSeconds", 0.22
        )
    )
    destination_reanchor = None
    if (
        confidence_passed
        and articulation_source == "nearby-isolated-pitch-coalesced"
    ):
        destination, destination_reanchor = reanchor_delayed_isolated_gesture(
            groups,
            destination=destination,
            learned_midis=[
                int(note["midi"]) + transposition
                for note in profile["learnedGesture"]["notes"]
            ],
            maximum_distance=maximum_coalescing_distance,
        )
    articulation_scale = (
        local_time_scale(
            best["template"],
            best["repeated"],
            best["matches"],
            source_index,
        )
        if source_index is not None
        and articulation_source == "learned-local-ioi-scaled"
        else 1.0
    )
    removed_indices = {
        int(note["_payloadIndex"]) for note in destination
    } if confidence_passed else set()
    destination_indices = set(removed_indices)
    coalesced_indices: set[int] = set()
    coalesced_evidence: list[dict[str, Any]] = []
    long_hold_evidence: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    if confidence_passed and destination:
        destination_time = float(destination[0]["time"])
        velocity = gesture_velocity(destination)
        learned_pitch_classes = {
            (int(note["midi"]) + transposition) % 12
            for note in profile["learnedGesture"]["notes"]
        }
        learned_midis = {
            int(note["midi"]) + transposition
            for note in profile["learnedGesture"]["notes"]
        }
        if structure_mode == "preserve-destination-allowed-pitch-classes":
            generated.extend(
                preserve_allowed_destination_pitch_classes(
                    destination,
                    learned_pitch_classes,
                )
            )
        elif structure_mode == "preserve-destination-allowed-exact-keys":
            generated.extend(
                preserve_allowed_destination_exact_keys(
                    destination,
                    learned_midis,
                )
            )
        elif structure_mode == "correct-destination-registers":
            generated.extend(
                correct_destination_registers(
                    destination,
                    profile["learnedGesture"]["notes"],
                    transposition=transposition,
                )
            )
        elif structure_mode == "complete-destination-pitch-classes":
            generated.extend(
                complete_destination_pitch_classes(
                    destination,
                    profile["learnedGesture"]["notes"],
                    transposition=transposition,
                    groups=groups,
                )
            )
        elif structure_mode == "add-learned-keys-to-destination":
            generated.extend(
                add_learned_keys_to_destination(
                    destination,
                    profile["learnedGesture"]["notes"],
                    transposition=transposition,
                )
            )
        learned_notes_to_generate = (
            []
            if structure_mode
            in {
                "preserve-destination-allowed-pitch-classes",
                "preserve-destination-allowed-exact-keys",
                "correct-destination-registers",
                "complete-destination-pitch-classes",
                "add-learned-keys-to-destination",
            }
            else profile["learnedGesture"]["notes"]
        )
        for learned_note in learned_notes_to_generate:
            midi = int(learned_note["midi"]) + transposition
            if not 21 <= midi <= 108:
                continue
            destination_has_pitch = any(
                int(note["midi"]) == midi
                or int(note["midi"]) % 12 == midi % 12
                for note in destination
            )
            nearby_evidence = None
            if (
                articulation_source == "nearby-isolated-pitch-coalesced"
                and not destination_has_pitch
            ):
                nearby_evidence = nearby_isolated_pitch_evidence(
                    groups,
                    destination_time=destination_time,
                    midi=midi,
                    maximum_distance=maximum_coalescing_distance,
                    excluded_indices=destination_indices | coalesced_indices,
                )
            if nearby_evidence is not None:
                original_duration = note_duration(nearby_evidence)
                duration, duration_strategy = coalesced_note_duration(
                    midi=midi,
                    learned_note=learned_note,
                    learned_notes=profile["learnedGesture"]["notes"],
                    destination=destination,
                    learned_pitch_classes=learned_pitch_classes,
                    learned_midis=learned_midis,
                    evidence=nearby_evidence,
                    destination_time=destination_time,
                )
                evidence_index = int(nearby_evidence["_payloadIndex"])
                coalesced_indices.add(evidence_index)
                coalesced_evidence.append(
                    {
                        "sourceTimeSeconds": rounded(float(nearby_evidence["time"])),
                        "sourceMidi": int(nearby_evidence["midi"]),
                        "generatedMidi": midi,
                        "timeOffsetSeconds": rounded(
                            float(nearby_evidence["time"]) - destination_time
                        ),
                        "originalDurationSeconds": rounded(original_duration),
                        "coalescedDurationSeconds": rounded(duration),
                        "durationStrategy": duration_strategy,
                        "releaseTimePreserved": (
                            duration_strategy == "detected-release-preserved"
                        ),
                    }
                )
            elif articulation_source == "learned-local-ioi-scaled":
                duration = note_duration(learned_note) * articulation_scale
            elif articulation_source == "learned-source":
                # A strongly matched recurrence can already contain the right
                # articulation.  Preserve it verbatim when local IOI matches
                # are less trustworthy than the supervised source release.
                duration = note_duration(learned_note)
            elif articulation_source == "destination-shared-shortest":
                duration = min(
                    (note_duration(note) for note in destination),
                    default=note_duration(learned_note),
                )
            elif articulation_source == "destination-exact-else-learned":
                duration = destination_exact_or_learned_duration(
                    midi,
                    destination,
                    learned_note,
                )
            elif articulation_source == "destination-harmonic-median":
                duration = destination_duration(
                    midi,
                    destination,
                    learned_note,
                    learned_pitch_classes,
                    None,
                )
            elif articulation_source == "trained-long-melody-hold":
                duration, cue_time = trained_long_melody_duration(
                    groups,
                    destination_time=destination_time,
                    midi=midi,
                    learned_note=learned_note,
                    settings=profile.get("application", {}).get(
                        "longMelodyHold", {}
                    ),
                )
                long_hold_evidence.append(
                    {
                        "midi": midi,
                        "nextHighCueTimeSeconds": (
                            rounded(cue_time) if cue_time is not None else None
                        ),
                        "durationSeconds": rounded(duration),
                    }
                )
            else:
                duration = destination_duration(
                    midi,
                    destination,
                    learned_note,
                    learned_pitch_classes,
                    learned_midis,
                )
            note = {
                key: value
                for key, value in learned_note.items()
                if key
                not in {
                    "time",
                    "startTime",
                    "duration",
                    "scoreDuration",
                    "visualDuration",
                    "audioDuration",
                    "velocity",
                }
            }
            note.update(
                {
                    "midi": midi,
                    "note": midi_to_note(midi),
                    "time": rounded(destination_time),
                    "duration": rounded(duration),
                    "scoreDuration": rounded(duration),
                    "visualDuration": rounded(duration),
                    "audioDuration": rounded(duration),
                    "velocity": rounded(velocity),
                    "hand": explicit_hand({**learned_note, "midi": midi}),
                    "source": "polymath-pianist-contextual-gesture-adapter",
                    "sourceInstrument": "learned_pianist_context",
                    "generatedBy": "pianist-contextual-gesture-adapter-v1",
                }
            )
            generated.append(note)

    removed_indices.update(coalesced_indices)

    output = copy.deepcopy(candidate)
    applied = confidence_passed and bool(generated)
    if applied:
        retained = [
            note
            for index, note in enumerate(output.get("notes") or [])
            if index not in removed_indices
        ]
        output["notes"] = sorted(
            retained + generated,
            key=lambda note: (
                finite(note.get("time", note.get("startTime")), 0.0),
                int(note.get("midi", 0)),
            ),
        )
    diagnostics = {
        "applied": applied,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "detectedOffsetSeconds": rounded(best["offsetSeconds"]),
        "inputTemplateGestures": int(best["templateGestures"]),
        "inputRepeatGestures": int(best["repeatGestures"]),
        "inputMatchedGestures": int(best["matchedGestures"]),
        "inputMeanPitchClassF1": rounded(best["meanPitchClassF1"]),
        "inputNormalizedSequenceScore": rounded(best["normalizedScore"]),
        "templateTargetPitchClassF1": rounded(source_target_f1),
        "trainedContextMatches": len(context_matches),
        "confidencePassed": confidence_passed,
        "targetMatchedBySequence": destination_index is not None,
        "inferredTranspositionSemitones": transposition,
        "rawInferredTranspositionSemitones": raw_transposition,
        "transpositionGainOverUnison": rounded(transposition_gain),
        "destinationTimeSeconds": (
            rounded(float(destination[0]["time"])) if destination else None
        ),
        "mappedDestinationTimeSeconds": (
            rounded(float(mapped_destination[0]["time"]))
            if mapped_destination
            else None
        ),
        "projectedDestinationTimeSeconds": (
            rounded(projected_time) if projected_time is not None else None
        ),
        "directOffsetTimeSeconds": (
            rounded(direct_offset_time)
            if direct_offset_time is not None
            else None
        ),
        "destinationSelection": destination_selection,
        "removedDestinationNotes": len(destination_indices),
        "removedCandidateNotes": len(removed_indices) if applied else 0,
        "coalescedNearbyNotes": len(coalesced_indices),
        "coalescedEvidence": coalesced_evidence,
        "longMelodyHoldEvidence": long_hold_evidence,
        "destinationReanchor": destination_reanchor,
        "generatedNotes": len(generated),
        "preservedVelocity": rounded(gesture_velocity(destination)) if destination else None,
        "articulationSource": articulation_source,
        "structureMode": structure_mode,
        "articulationTimeScale": rounded(articulation_scale),
        "targetReadAtInference": False,
        "runnerUpOffsets": [
            {
                "offsetSeconds": rounded(row["offsetSeconds"]),
                "normalizedScore": rounded(row["normalizedScore"]),
            }
            for row in ranked[:5]
        ],
        "transpositionTrials": [
            {
                "semitones": int(row["semitones"]),
                "meanPitchClassF1": rounded(row["meanPitchClassF1"]),
            }
            for row in transposition_trials
        ],
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistContextualGesture"] = diagnostics
    arrangement["outputNoteCount"] = len(output.get("notes") or [])
    return output, diagnostics


def contextual_gate(
    baseline_local: dict[str, Any],
    candidate_local: dict[str, Any],
    baseline_full: dict[str, Any],
    candidate_full: dict[str, Any],
) -> tuple[bool, str]:
    def metric(values: dict[str, Any], name: str, fallback: str) -> float:
        return float(values.get(name, values[fallback]))

    onset_safe = (
        float(candidate_local["referenceGestureRecall"])
        >= float(baseline_local["referenceGestureRecall"])
        and float(candidate_local["candidateGesturePrecision"])
        >= float(baseline_local["candidateGesturePrecision"])
        and float(candidate_local["onsetMaeSeconds"])
        <= float(baseline_local["onsetMaeSeconds"]) + 1e-6
    )
    local_structure_gain = (
        float(candidate_local["coverageAdjustedPitchClassF1"])
        > float(baseline_local["coverageAdjustedPitchClassF1"])
        or float(candidate_local["coverageAdjustedExactPitchClassRate"])
        > float(baseline_local["coverageAdjustedExactPitchClassRate"])
        or metric(
            candidate_local,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        > metric(
            baseline_local,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        or float(candidate_local["coverageAdjustedOccupancyAccuracy"])
        > float(baseline_local["coverageAdjustedOccupancyAccuracy"])
    )
    local_safe = (
        float(candidate_local["coverageAdjustedPitchClassF1"])
        >= float(baseline_local["coverageAdjustedPitchClassF1"])
        and float(candidate_local["coverageAdjustedExactPitchClassRate"])
        >= float(baseline_local["coverageAdjustedExactPitchClassRate"])
        and metric(
            candidate_local,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        >= metric(
            baseline_local,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        and metric(
            candidate_local,
            "coverageAdjustedExactKeySetRate",
            "coverageAdjustedExactPitchClassRate",
        )
        >= metric(
            baseline_local,
            "coverageAdjustedExactKeySetRate",
            "coverageAdjustedExactPitchClassRate",
        )
        and float(candidate_local["coverageAdjustedOccupancyAccuracy"])
        >= float(baseline_local["coverageAdjustedOccupancyAccuracy"])
        and float(candidate_local["gestureVelocityMae"])
        <= float(baseline_local["gestureVelocityMae"]) + 1e-6
        and float(candidate_local["exactKeyDurationMaeSeconds"])
        <= float(baseline_local["exactKeyDurationMaeSeconds"]) + 0.01
        and float(candidate_local["sequenceScore"])
        < float(baseline_local["sequenceScore"])
    )
    full_safe = (
        float(candidate_full["coverageAdjustedPitchClassF1"])
        >= float(baseline_full["coverageAdjustedPitchClassF1"])
        and float(candidate_full["coverageAdjustedExactPitchClassRate"])
        >= float(baseline_full["coverageAdjustedExactPitchClassRate"])
        and metric(
            candidate_full,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        >= metric(
            baseline_full,
            "coverageAdjustedExactKeyF1",
            "coverageAdjustedPitchClassF1",
        )
        and metric(
            candidate_full,
            "coverageAdjustedExactKeySetRate",
            "coverageAdjustedExactPitchClassRate",
        )
        >= metric(
            baseline_full,
            "coverageAdjustedExactKeySetRate",
            "coverageAdjustedExactPitchClassRate",
        )
        and float(candidate_full["coverageAdjustedOccupancyAccuracy"])
        >= float(baseline_full["coverageAdjustedOccupancyAccuracy"])
        and float(candidate_full["onsetMaeSeconds"])
        <= float(baseline_full["onsetMaeSeconds"]) + 1e-6
        and float(candidate_full["gestureVelocityMae"])
        <= float(baseline_full["gestureVelocityMae"]) + 1e-6
        and float(candidate_full["exactKeyDurationMaeSeconds"])
        <= float(baseline_full["exactKeyDurationMaeSeconds"]) + 0.001
        and float(candidate_full["sequenceScore"])
        < float(baseline_full["sequenceScore"])
    )
    if onset_safe and local_structure_gain and local_safe and full_safe:
        return True, "safe-contextual-gesture-refinement"
    return False, "gate-not-met"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--template-start", type=float, required=True)
    parser.add_argument("--template-end", type=float, required=True)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--search-min-offset", type=float, required=True)
    parser.add_argument("--search-max-offset", type=float, required=True)
    parser.add_argument("--minimum-input-pitch-f1", type=float, default=0.80)
    parser.add_argument("--maximum-input-normalized-score", type=float, default=0.25)
    parser.add_argument("--learned-minimum-midi", type=int, default=21)
    parser.add_argument("--learned-maximum-midi", type=int, default=108)
    parser.add_argument(
        "--learned-midis",
        help="Optional comma-separated exact MIDI notes to learn from the source gesture",
    )
    parser.add_argument(
        "--articulation-source",
        choices=(
            "destination-supported",
            "destination-shared-shortest",
            "destination-exact-else-learned",
            "destination-harmonic-median",
            "trained-long-melody-hold",
            "learned-source",
            "learned-local-ioi-scaled",
            "nearby-isolated-pitch-coalesced",
        ),
        default="destination-supported",
    )
    parser.add_argument(
        "--structure-mode",
        choices=(
            "replace-learned-gesture",
            "preserve-destination-allowed-pitch-classes",
            "preserve-destination-allowed-exact-keys",
            "correct-destination-registers",
            "complete-destination-pitch-classes",
            "add-learned-keys-to-destination",
        ),
        default="replace-learned-gesture",
    )
    args = parser.parse_args()
    learned_midis = (
        {
            int(value.strip())
            for value in args.learned_midis.split(",")
            if value.strip()
        }
        if args.learned_midis
        else None
    )

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    row = next(
        (item for item in manifest.get("songs") or [] if item.get("id") == args.song_id),
        None,
    )
    if row is None:
        raise ValueError(f"Song {args.song_id!r} was not found in the manifest")
    reference, _, _ = load_song_sequences(row, 0.035)
    profile = fit_profile(
        reference,
        profile_id=args.profile_id,
        song_id=args.song_id,
        template_start=args.template_start,
        template_end=args.template_end,
        target_time=args.target_time,
        repeat_search_minimum=args.search_min_offset,
        repeat_search_maximum=args.search_max_offset,
        minimum_input_pitch_f1=args.minimum_input_pitch_f1,
        maximum_input_normalized_score=args.maximum_input_normalized_score,
        learned_minimum_midi=args.learned_minimum_midi,
        learned_maximum_midi=args.learned_maximum_midi,
        learned_midis=learned_midis,
    )
    profile["application"]["articulationSource"] = args.articulation_source
    profile["application"]["structureMode"] = args.structure_mode
    profile["profileSha256"] = sha256_json(profile)
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    input_path = Path(args.input_candidate).resolve()
    input_payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_profile(input_payload, profile)
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    input_row = {**row, "candidate": str(input_path)}
    output_row = {**row, "candidate": str(output_path)}
    _, baseline_groups, _ = load_song_sequences(input_row, 0.035)
    _, output_groups, _ = load_song_sequences(output_row, 0.035)
    destination_time = finite(diagnostics.get("destinationTimeSeconds"), -99.0)
    local_start, local_end = destination_time - 0.65, destination_time + 0.65
    baseline_local_eval = evaluation(
        slice_groups(reference, local_start, local_end),
        slice_groups(baseline_groups, local_start, local_end),
    )
    candidate_local_eval = evaluation(
        slice_groups(reference, local_start, local_end),
        slice_groups(output_groups, local_start, local_end),
    )
    baseline_full_eval = evaluation(reference, baseline_groups)
    candidate_full_eval = evaluation(reference, output_groups)
    passed, reason = contextual_gate(
        baseline_local_eval,
        candidate_local_eval,
        baseline_full_eval,
        candidate_full_eval,
    )
    report = {
        "schema": "polymath-pianist-contextual-gesture-training-report-v1",
        "evidenceBoundary": (
            "The source gesture is supervised; the recurrence is detected from input only. "
            "This remains same-song research and is not unseen-song accuracy."
        ),
        "profile": str(profile_path),
        "inputCandidate": str(input_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutGestureEvaluation": {
            "baseline": baseline_local_eval,
            "candidate": candidate_local_eval,
        },
        "trustedWindowEvaluation": {
            "baseline": baseline_full_eval,
            "candidate": candidate_full_eval,
        },
        "evaluationGate": reason,
        "decision": "LISTENING_CANDIDATE_RESEARCH_ONLY" if passed else "REJECT",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
