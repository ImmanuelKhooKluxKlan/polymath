"""Reduce MuScriptor events into an idiomatic, playable 88-key piano arrangement.

The transcription model reports what it hears. A piano cover needs a second
stage that decides what a pianist should actually play. This module preserves
genuine acoustic-piano sources, but reduces full mixes to melody, bass, and
compact harmony while removing percussion and excessive event density.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from omr.polymath_omr.performance import shape_piano_performance


PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108
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
MAX_HARMONY_PITCH_CLASSES = 4
MAX_DIRECT_CLEANUP_PRESSURE = 0.18

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
) -> dict[str, Any]:
    counts = Counter(note["instrument"] for note in notes)
    total = max(1, len(notes))
    acoustic_ratio = counts["acoustic_piano"] / total
    piano_ratio = sum(counts[name] for name in PIANO_INSTRUMENTS) / total
    density = notes_per_second(notes)
    onset_cluster = maximum_onset_cluster(notes)
    cleanup = cleanup or {}
    cleanup_input = max(1, int(safe_number(cleanup.get("inputNotes"), len(notes)) or len(notes)))
    cleanup_artifacts = sum(
        int(safe_number(cleanup.get(field), 0) or 0)
        for field in ("removedDuplicateNotes", "shortenedSameKeyOverlaps")
    )
    cleanup_pressure = cleanup_artifacts / cleanup_input
    direct_acoustic_piano = (
        counts["acoustic_piano"] >= 24
        and acoustic_ratio >= 0.70
        and piano_ratio >= 0.90
        and density <= MAX_DIRECT_PIANO_NOTES_PER_SECOND
        and onset_cluster <= 8
        and cleanup_pressure <= MAX_DIRECT_CLEANUP_PRESSURE
    )
    return {
        "instrumentCounts": dict(sorted(counts.items())),
        "acousticPianoRatio": round_number(acoustic_ratio, 3),
        "pianoRatio": round_number(piano_ratio, 3),
        "sourceNotesPerSecond": round_number(density, 3),
        "sourceMaximumOnsetCluster": onset_cluster,
        "cleanupArtifactPressure": round_number(cleanup_pressure, 3),
        "cleanupArtifactsDetected": cleanup_artifacts,
        "maximumDirectCleanupPressure": MAX_DIRECT_CLEANUP_PRESSURE,
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


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = int(round((len(ordered) - 1) * clamp(fraction, 0.0, 1.0)))
    return ordered[position]


def average(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def expression_role(note: dict[str, Any]) -> str:
    role = note.get("arrangementRole")
    if role == "melody":
        return "melody"
    if role == "bass":
        return "bass"
    if note["midi"] < 60:
        return "source_left" if role == "source_piano" else "left_hand"
    return "source_right" if role == "source_piano" else "right_hand"


def shape_melody_forward_expression(
    notes: list[dict[str, Any]],
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

    velocity_ranges = {
        "melody": MELODY_VELOCITY_RANGE,
        "right_hand": RIGHT_HAND_VELOCITY_RANGE,
        "left_hand": LEFT_HAND_VELOCITY_RANGE,
        "bass": BASS_VELOCITY_RANGE,
        "source_right": SOURCE_RIGHT_VELOCITY_RANGE,
        "source_left": SOURCE_LEFT_VELOCITY_RANGE,
    }
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

        note["velocity"] = round_number(clamp(velocity, minimum, maximum), 3)
        note["hand"] = "left" if note["midi"] < 60 else "right"
        role_velocities[role].append(note["velocity"])

    right_velocities = [
        note["velocity"] for note in shaped if note["midi"] >= 60
    ]
    left_velocities = [
        note["velocity"] for note in shaped if note["midi"] < 60
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
    }


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


def compact_harmony(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arranged: list[dict[str, Any]] = []
    previous_voicing: list[int] = []
    for group in grouped_by_window(notes, 0.30):
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
        )[:MAX_HARMONY_PITCH_CLASSES]
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


def arrange_payload(payload: dict[str, Any], mode: str = "instrumental") -> dict[str, Any]:
    source_notes = [
        normalized
        for note in payload.get("notes", [])
        if (normalized := normalize_source_note(note)) is not None
    ]
    if not source_notes:
        raise ValueError("No notes inside the real 88-key piano range were available to arrange.")

    profile = source_profile(source_notes, payload.get("transcriptionCleanup"))
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
        density_limit = MAX_DIRECT_PIANO_NOTES_PER_SECOND
    else:
        voice_notes = [
            note for note in working if note["instrument"] in VOICE_INSTRUMENTS
        ]
        if mode == "full" and voice_notes:
            role_notes["melody"] = select_lead(
                voice_notes, window_seconds=0.08
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
        role_notes["harmony"] = compact_harmony(harmony_sources)
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

    arranged_notes, expression = shape_melody_forward_expression(arranged_notes)

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
    for note in arranged_notes:
        note["scoreDuration"] = note["duration"]
        note["visualDuration"] = note["duration"]
        note.pop("audioDuration", None)
        note.pop("releaseSeconds", None)
    output = dict(payload)
    output["instrument"] = "piano"
    output["notes"] = arranged_notes
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
            "minimumMidi": PIANO_MIN_MIDI,
            "maximumMidi": PIANO_MAX_MIDI,
            "minimumNote": "A0",
            "maximumNote": "C8",
        },
        "densityLimitNotesPerSecond": density_limit,
        "maximumOnsetCluster": MAX_ONSET_CLUSTER,
        "minimumSameKeyRetriggerMs": round(MIN_RETRIGGER_SECONDS * 1000),
        "removedPercussionNotes": percussion_removed,
        "removedForOnsetClusterLimit": cluster_removed,
        "removedForDensityLimit": density_removed,
        "removedRapidRetriggers": retriggers_removed,
        "vocalMelodyNotes": vocal_melody_notes,
        "roleCounts": dict(sorted(role_counts.items())),
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
    output['performance']['profile'] = 'polymath-piano-arranger-v5'
    output['performance']['defaultAutoplayReleaseSeconds'] = 0.62
    output['performance']['melodyForwardDynamics'] = True
    output['pianoArrangement']['version'] = 5
    output['pianoArrangement']['maximumHarmonyPitchClasses'] = (
        MAX_HARMONY_PITCH_CLASSES
    )
    output['pianoArrangement']['legatoExtendedNotes'] = legato_extended
    output['pianoArrangement']['expression'] = expression
    output['pianoArrangement']['physicalPerformance'] = {
        **(output.get('pianoPerformance') or {}),
        'profile': 'written-key-hold-damper-v1',
    }
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
    continuation_gaps = {'melody': 0.18, 'bass': 0.20, 'harmony': 0.20}
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
            onset_gap = note['time'] - previous['time']
            previous_end = previous['time'] + previous['duration']
            note_end = note['time'] + note['duration']
            same_role = note['arrangementRole'] == previous['arrangementRole']
            continuation_gap = continuation_gaps.get(note['arrangementRole'], -1.0)
            role_collision = (
                not same_role
                and onset_gap < 0.20
            )
            continues_phrase = (
                (
                    same_role
                    and continuation_gap >= 0
                    and onset_gap < continuation_gap
                )
                or role_collision
            )
            if onset_gap >= MIN_RETRIGGER_SECONDS and not continues_phrase:
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    arranged = arrange_payload(payload, args.mode)
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
