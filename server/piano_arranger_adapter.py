"""Inference for the route-specific, learned Polymath piano arranger.

MuScriptor remains responsible for detecting the notes and source instruments.
This module is only used after the request has explicitly selected Piano.  It
scores the detected events with a small supervised model, revoices the chosen
musical material onto one acoustic piano, and applies style statistics learned
from approved piano arrangements.

The inference implementation intentionally uses only Python's standard library
so the web server does not need a machine-learning runtime just to decode the
small JSON profile produced by ``ml/training/train_piano_arranger_adapter.py``.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from piano_retrigger import classify_same_key_retrigger


PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108
MIN_NOTE_SECONDS = 0.05
MAX_NOTE_SECONDS = 6.0
PERCUSSION_INSTRUMENTS = {"drums", "timpani", "percussion"}
VOICE_INSTRUMENTS = {"voice"}
BASS_INSTRUMENTS = {"acoustic_bass", "electric_bass", "contrabass"}
PIANO_INSTRUMENTS = {"acoustic_piano", "electric_piano"}
GUITAR_INSTRUMENTS = {
    "acoustic_guitar",
    "clean_electric_guitar",
    "distorted_electric_guitar",
}
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
PAD_INSTRUMENTS = {"synth_pad", "string_ensemble"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


FEATURE_NAMES = (
    "bias",
    "midi_centered",
    "midi_squared",
    "velocity",
    "duration_log",
    "onset_cluster",
    "local_density",
    "is_voice",
    "is_piano",
    "is_bass",
    "is_guitar",
    "is_lead",
    "is_pad",
    "is_other",
    "voice_high",
    "bass_low",
    "piano_mid",
    "guitar_mid",
    "extreme_low",
    "extreme_high",
)

# Version 2 appends context to the frozen v1 contract.  Keeping v1 as an exact
# prefix lets an older selector be lifted into the larger space with zero
# coefficients, preserving its logits while a new residual learns only what
# the additional musical evidence contributes.
CONTEXT_SELECTION_FEATURE_NAMES = FEATURE_NAMES + (
    "is_acoustic_guitar",
    "is_clean_electric_guitar",
    "is_distorted_electric_guitar",
    "is_acoustic_piano",
    "is_electric_piano",
    "is_electric_bass",
    "is_synth_pad",
    "is_string_ensemble",
    "role_melody",
    "role_bass",
    "role_harmony",
    "voice_proximity",
    "voice_within_060ms",
    "voice_within_180ms",
    "voice_pitch_class_within_180ms",
    "cross_family_onset_support",
    "cross_family_exact_pitch_support",
    "cross_family_pitch_class_support",
    "local_duration_percentile",
    "local_velocity_percentile",
    "same_instrument_pitch_class_recurrence",
)

# The robust ablation deliberately excludes exact instrument-subtype flags and
# the highly collinear continuous voice-distance signal.  Those fields can
# fit a four-song corpus well while learning that one particular production
# timbre is always good or bad.  The remaining signals describe transferable
# musical relationships instead of a track's sound palette.
ROBUST_CONTEXT_SELECTION_FEATURE_NAMES = FEATURE_NAMES + (
    "role_melody",
    "role_bass",
    "role_harmony",
    "voice_within_060ms",
    "voice_within_180ms",
    "voice_pitch_class_within_180ms",
    "cross_family_onset_support",
    "cross_family_exact_pitch_support",
    "cross_family_pitch_class_support",
    "local_duration_percentile",
    "local_velocity_percentile",
    "same_instrument_pitch_class_recurrence",
)

# Version 3 is dedicated to accompaniment selection.  It adds only
# transposition-invariant evidence: a note's interval above the local bass,
# its place inside the current onset, and nearby same-pitch repetition.  The
# earlier contracts stay byte-for-byte unchanged so every published v1/v2
# profile remains loadable.
LEFT_HAND_SELECTION_FEATURE_NAMES = ROBUST_CONTEXT_SELECTION_FEATURE_NAMES + tuple(
    f"interval_above_local_bass_pc_{interval:02d}" for interval in range(12)
) + (
    "is_local_onset_lowest",
    "is_local_onset_highest",
    "local_onset_pitch_percentile",
    "local_onset_unique_pitch_count",
    "previous_same_pitch_gap_log",
    "next_same_pitch_gap_log",
    "previous_same_pitch_within_180ms",
    "previous_same_pitch_within_350ms",
    "next_same_pitch_within_180ms",
    "local_bass_within_180ms",
)

# Version 4 adds a wider beat-level chroma estimate.  This is separate from
# the v3 contract so experimental v3 profiles remain reproducible.
HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES = LEFT_HAND_SELECTION_FEATURE_NAMES + (
    "wide_pitch_class_support",
    "wide_pitch_class_rank",
    "wide_pitch_class_dominance",
    "estimated_chord_is_major",
    "estimated_chord_is_minor",
    "estimated_chord_member_root",
    "estimated_chord_member_third",
    "estimated_chord_member_fifth",
    "estimated_chord_nonmember",
    "estimated_chord_confidence",
    "estimated_bass_root_match",
) + tuple(
    f"interval_above_estimated_root_pc_{interval:02d}" for interval in range(12)
)

# Version 5 scores an entire detected onset rather than pretending every note
# at that moment is an independent rhythmic decision.  Every member of an
# onset receives the same row.  The downstream decoder can therefore select a
# pianist-like gesture first, then let the chord ranker decide which pitches
# occupy that gesture.  The contract intentionally avoids song position,
# absolute pitch class, and instrument subtype so it can transfer across keys,
# songs, and different source mixes.
ONSET_GESTURE_SELECTION_FEATURE_NAMES = (
    "bias",
    "onset_note_count",
    "onset_pitch_class_count",
    "onset_family_count",
    "cross_family_onset",
    "voice_share",
    "bass_share",
    "piano_share",
    "guitar_share",
    "lead_share",
    "pad_share",
    "other_share",
    "mean_duration_log",
    "maximum_duration_log",
    "duration_spread",
    "mean_velocity",
    "maximum_velocity",
    "local_onset_density",
    "previous_onset_gap_log",
    "next_onset_gap_log",
    "adjacent_gap_similarity",
    "previous_two_gap_similarity",
    "next_two_gap_similarity",
    "pitch_class_overlap_previous",
    "pitch_class_overlap_next",
    "maximum_pitch_class_family_support",
    "bass_pitch_class_family_support",
    "triad_template_confidence",
    "triad_member_coverage",
    "pitch_span",
    "voice_onset_present",
    "voice_nearby_180ms",
)

# Version 6 adds the piece that the first onset ranker was missing: position
# inside a repeated harmonic gesture.  Pianists rarely strike every voice of a
# stable chord on every detected attack.  They alternate bass, inner tones and
# upper voices over a short cycle.  These fields describe that cycle without
# exposing a song id, absolute timestamp or absolute pitch class, so a profile
# cannot simply memorise one recording.
ONSET_GESTURE_SEQUENCE_FEATURE_NAMES = ONSET_GESTURE_SELECTION_FEATURE_NAMES + (
    "same_harmony_previous",
    "same_harmony_next",
    "harmony_change_previous",
    "harmony_change_next",
    "harmonic_run_start",
    "harmonic_run_end",
    "harmonic_run_position_capped",
    "harmonic_run_mod2_0",
    "harmonic_run_mod2_1",
) + tuple(
    f"harmonic_run_mod4_{phase}" for phase in range(4)
) + tuple(
    f"harmonic_run_mod8_{phase}" for phase in range(8)
) + (
    "previous_gap_over_local_pulse",
    "next_gap_over_local_pulse",
    "elapsed_pulses_in_harmonic_run",
    "elapsed_pulse_phase_sin",
    "elapsed_pulse_phase_cos",
)

# Duration needs different evidence from note selection.  In particular, a
# source separator may report every event with a similarly short duration even
# though the pianist should hold a chord until its next change.  These features
# expose the spacing between musical events so the supervised adapter can learn
# held notes without changing MuScriptor's instrument detector.
DURATION_FEATURE_NAMES = FEATURE_NAMES + (
    "role_melody",
    "role_bass",
    "role_harmony",
    "has_next_same_pitch",
    "next_same_pitch_gap_log",
    "next_role_onset_gap_log",
    "next_any_onset_gap_log",
    "same_pitch_gap_over_045",
    "same_pitch_gap_over_090",
    "same_pitch_gap_over_180",
    "role_gap_over_045",
    "role_gap_over_090",
    "role_gap_over_180",
)


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


def instrument_family(instrument: str) -> str:
    value = str(instrument or "other").strip().lower()
    if value in VOICE_INSTRUMENTS:
        return "voice"
    if value in PIANO_INSTRUMENTS:
        return "piano"
    if value in BASS_INSTRUMENTS:
        return "bass"
    if value in GUITAR_INSTRUMENTS:
        return "guitar"
    if value in LEAD_INSTRUMENTS:
        return "lead"
    if value in PAD_INSTRUMENTS:
        return "pad"
    return "other"


def arrangement_role(note: dict[str, Any]) -> str:
    family = instrument_family(note.get("instrument", ""))
    midi = int(note["midi"])
    if family == "voice":
        return "melody"
    if family == "bass" or midi <= 48:
        return "bass"
    if family == "lead" or midi >= 72:
        return "melody"
    return "harmony"


def normalize_source_notes(notes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source_index, item in enumerate(notes):
        midi_value = safe_number(item.get("midi", item.get("pitch")))
        time = safe_number(item.get("time", item.get("startTime", item.get("start"))))
        duration = safe_number(item.get("duration"), 0.35)
        if midi_value is None or time is None or duration is None or time < 0:
            continue
        midi = int(round(midi_value))
        if midi < PIANO_MIN_MIDI or midi > PIANO_MAX_MIDI:
            continue
        instrument = str(item.get("instrument") or "other").strip().lower()
        if instrument in PERCUSSION_INSTRUMENTS or "drum" in instrument or "percussion" in instrument:
            continue
        note = dict(item)
        note.update(
            {
                "sourceIndex": source_index,
                "midi": midi,
                "time": round_number(time),
                "duration": round_number(clamp(duration, MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)),
                "velocity": round_number(
                    clamp(safe_number(item.get("velocity"), 0.72) or 0.72, 0.05, 1.0),
                    3,
                ),
                "instrument": instrument,
            }
        )
        normalized.append(note)
    return sorted(normalized, key=lambda note: (note["time"], note["midi"], note["sourceIndex"]))


def _window_counts(times: list[float], radius: float) -> list[int]:
    counts: list[int] = []
    left = 0
    right = 0
    for time in times:
        while left < len(times) and times[left] < time - radius:
            left += 1
        while right < len(times) and times[right] <= time + radius:
            right += 1
        counts.append(max(0, right - left - 1))
    return counts


def raw_feature_rows(notes: list[dict[str, Any]]) -> list[list[float]]:
    if not notes:
        return []
    times = [float(note["time"]) for note in notes]
    onset_counts = _window_counts(times, 0.045)
    density_counts = _window_counts(times, 0.50)
    rows: list[list[float]] = []
    for note, onset_count, density_count in zip(notes, onset_counts, density_counts):
        family = instrument_family(note["instrument"])
        midi_centered = (note["midi"] - 64.0) / 24.0
        flags = {name: 1.0 if family == name else 0.0 for name in (
            "voice", "piano", "bass", "guitar", "lead", "pad", "other"
        )}
        rows.append(
            [
                1.0,
                midi_centered,
                midi_centered * midi_centered,
                float(note["velocity"]),
                math.log1p(float(note["duration"])) / math.log(7.0),
                min(1.0, onset_count / 8.0),
                min(1.0, density_count / 32.0),
                flags["voice"],
                flags["piano"],
                flags["bass"],
                flags["guitar"],
                flags["lead"],
                flags["pad"],
                flags["other"],
                flags["voice"] * (1.0 if note["midi"] >= 55 else 0.0),
                flags["bass"] * (1.0 if note["midi"] <= 55 else 0.0),
                flags["piano"] * (1.0 if 48 <= note["midi"] <= 84 else 0.0),
                flags["guitar"] * (1.0 if 45 <= note["midi"] <= 84 else 0.0),
                1.0 if note["midi"] < 33 else 0.0,
                1.0 if note["midi"] > 96 else 0.0,
            ]
        )
    return rows


def _nearest_distance(sorted_times: list[float], time: float, *, skip_exact: bool = False) -> float:
    """Return the closest onset distance without an O(n) scan."""

    if not sorted_times:
        return 6.0
    position = bisect.bisect_left(sorted_times, time)
    candidates: list[float] = []
    for index in (position - 2, position - 1, position, position + 1):
        if 0 <= index < len(sorted_times):
            distance = abs(float(sorted_times[index]) - time)
            if skip_exact and distance < 1e-7:
                continue
            candidates.append(distance)
    return min(candidates, default=6.0)


def contextual_selection_feature_rows(notes: list[dict[str, Any]]) -> list[list[float]]:
    """Append stem, vocal, agreement, and local-salience evidence to v1 rows.

    Full mixes produce many plausible pitches at the same moment.  The v1
    selector knew broad instrument families but could not distinguish an
    acoustic guitar from a dense electric-guitar stem, nor tell whether a note
    was supported by another stem or aligned with the vocal contour.  These
    features expose those facts without using a song id or a target label.
    """

    if not notes:
        return []
    base_rows = raw_feature_rows(notes)
    times = [float(note["time"]) for note in notes]
    voice_times = [
        float(note["time"])
        for note in notes
        if instrument_family(str(note.get("instrument") or "")) == "voice"
    ]
    voice_pc_times: dict[int, list[float]] = defaultdict(list)
    recurrence_times: dict[tuple[str, int], list[float]] = defaultdict(list)
    for note in notes:
        instrument = str(note.get("instrument") or "other")
        midi = int(note["midi"])
        recurrence_times[(instrument, midi % 12)].append(float(note["time"]))
        if instrument_family(instrument) == "voice":
            voice_pc_times[midi % 12].append(float(note["time"]))

    rows: list[list[float]] = []
    left = 0
    right = 0
    for index, (note, base_row) in enumerate(zip(notes, base_rows)):
        time = float(note["time"])
        midi = int(note["midi"])
        instrument = str(note.get("instrument") or "other")
        family = instrument_family(instrument)
        role = arrangement_role(note)

        while left < len(notes) and float(notes[left]["time"]) < time - 0.50:
            left += 1
        right = max(right, index + 1)
        while right < len(notes) and float(notes[right]["time"]) <= time + 0.50:
            right += 1
        local = notes[left:right]
        duration = float(note["duration"])
        velocity = float(note["velocity"])
        duration_percentile = sum(float(peer["duration"]) <= duration for peer in local) / max(1, len(local))
        velocity_percentile = sum(float(peer["velocity"]) <= velocity for peer in local) / max(1, len(local))

        onset_left = bisect.bisect_left(times, time - 0.08)
        onset_right = bisect.bisect_right(times, time + 0.08)
        onset_families: set[str] = set()
        exact_families: set[str] = set()
        pitch_class_families: set[str] = set()
        for peer_index in range(onset_left, onset_right):
            if peer_index == index:
                continue
            peer = notes[peer_index]
            peer_family = instrument_family(str(peer.get("instrument") or ""))
            if peer_family == family:
                continue
            onset_families.add(peer_family)
            peer_midi = int(peer["midi"])
            if peer_midi == midi:
                exact_families.add(peer_family)
            if peer_midi % 12 == midi % 12:
                pitch_class_families.add(peer_family)

        voice_distance = _nearest_distance(voice_times, time)
        voice_pc_distance = _nearest_distance(voice_pc_times[midi % 12], time)
        recurrence_distance = _nearest_distance(
            recurrence_times[(instrument, midi % 12)], time, skip_exact=True
        )
        rows.append(
            base_row
            + [
                1.0 if instrument == "acoustic_guitar" else 0.0,
                1.0 if instrument == "clean_electric_guitar" else 0.0,
                1.0 if instrument == "distorted_electric_guitar" else 0.0,
                1.0 if instrument == "acoustic_piano" else 0.0,
                1.0 if instrument == "electric_piano" else 0.0,
                1.0 if instrument == "electric_bass" else 0.0,
                1.0 if instrument == "synth_pad" else 0.0,
                1.0 if instrument == "string_ensemble" else 0.0,
                1.0 if role == "melody" else 0.0,
                1.0 if role == "bass" else 0.0,
                1.0 if role == "harmony" else 0.0,
                math.exp(-min(6.0, voice_distance) / 0.12),
                1.0 if voice_distance <= 0.06 else 0.0,
                1.0 if voice_distance <= 0.18 else 0.0,
                1.0 if voice_pc_distance <= 0.18 else 0.0,
                min(1.0, len(onset_families) / 4.0),
                min(1.0, len(exact_families) / 3.0),
                min(1.0, len(pitch_class_families) / 4.0),
                duration_percentile,
                velocity_percentile,
                math.exp(-min(6.0, recurrence_distance) / 0.60),
            ]
        )
    return rows


def left_hand_selection_feature_rows(notes: list[dict[str, Any]]) -> list[list[float]]:
    """Return chord-relative features for pianist-like accompaniment ranking.

    Absolute pitch-class flags would memorize the key of a training song.  A
    local-bass interval instead describes the same musical relationship after
    a song is transposed, while repetition gaps let the selector distinguish a
    useful rhythmic attack from machine-gun duplication.
    """

    if not notes:
        return []
    contextual_rows = contextual_selection_feature_rows(notes)
    contextual_indices = {
        name: index for index, name in enumerate(CONTEXT_SELECTION_FEATURE_NAMES)
    }
    robust_rows = [
        [row[contextual_indices[name]] for name in ROBUST_CONTEXT_SELECTION_FEATURE_NAMES]
        for row in contextual_rows
    ]
    times = [float(note["time"]) for note in notes]
    same_pitch_times: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        same_pitch_times[int(note["midi"])].append(float(note["time"]))

    rows: list[list[float]] = []
    log_scale = math.log(7.0)
    for index, (note, robust_row) in enumerate(zip(notes, robust_rows)):
        time = float(note["time"])
        midi = int(note["midi"])
        onset_left = bisect.bisect_left(times, time - 0.08)
        onset_right = bisect.bisect_right(times, time + 0.08)
        local_non_voice = [
            peer
            for peer in notes[onset_left:onset_right]
            if instrument_family(str(peer.get("instrument") or "")) != "voice"
        ]
        if not local_non_voice:
            local_non_voice = [note]

        local_pitches = sorted({int(peer["midi"]) for peer in local_non_voice})
        local_lowest = local_pitches[0]
        local_highest = local_pitches[-1]

        local_bass_candidates = [
            peer
            for peer in local_non_voice
            if arrangement_role(peer) == "bass"
            or instrument_family(str(peer.get("instrument") or "")) == "bass"
        ]
        bass_distance = 6.0
        if local_bass_candidates:
            bass_note = min(
                local_bass_candidates,
                key=lambda peer: (
                    abs(float(peer["time"]) - time),
                    int(peer["midi"]),
                ),
            )
            bass_distance = abs(float(bass_note["time"]) - time)
            local_bass_midi = int(bass_note["midi"])
        else:
            # Chords without a labelled bass stem still have a useful local
            # floor.  This fallback is based on relative ordering, not a song
            # key or absolute pitch class.
            local_bass_midi = local_lowest

        interval_class = (midi - local_bass_midi) % 12
        interval_flags = [
            1.0 if interval_class == candidate else 0.0
            for candidate in range(12)
        ]
        pitch_rank = sum(pitch <= midi for pitch in local_pitches) - 1
        pitch_percentile = pitch_rank / max(1, len(local_pitches) - 1)

        pitch_times = same_pitch_times[midi]
        previous_index = bisect.bisect_left(pitch_times, time - 1e-7) - 1
        next_index = bisect.bisect_right(pitch_times, time + 1e-7)
        previous_gap = (
            time - pitch_times[previous_index]
            if previous_index >= 0
            else 6.0
        )
        next_gap = (
            pitch_times[next_index] - time
            if next_index < len(pitch_times)
            else 6.0
        )
        previous_gap = clamp(previous_gap, 0.0, 6.0)
        next_gap = clamp(next_gap, 0.0, 6.0)
        rows.append(
            robust_row
            + interval_flags
            + [
                1.0 if midi == local_lowest else 0.0,
                1.0 if midi == local_highest else 0.0,
                pitch_percentile,
                min(1.0, len(local_pitches) / 8.0),
                math.log1p(previous_gap) / log_scale,
                math.log1p(next_gap) / log_scale,
                1.0 if previous_gap <= 0.18 else 0.0,
                1.0 if previous_gap <= 0.35 else 0.0,
                1.0 if next_gap <= 0.18 else 0.0,
                1.0 if bass_distance <= 0.18 else 0.0,
            ]
        )
    return rows


def harmonic_left_hand_selection_feature_rows(
    notes: list[dict[str, Any]],
) -> list[list[float]]:
    """Append beat-level harmonic evidence to the v3 accompaniment contract."""

    if not notes:
        return []
    base_rows = left_hand_selection_feature_rows(notes)
    times = [float(note["time"]) for note in notes]
    rows: list[list[float]] = []
    for note, base_row in zip(notes, base_rows):
        time = float(note["time"])
        midi = int(note["midi"])
        left = bisect.bisect_left(times, time - 0.40)
        right = bisect.bisect_right(times, time + 0.40)
        chroma = [0.0] * 12
        bass_chroma = [0.0] * 12
        for peer in notes[left:right]:
            family = instrument_family(str(peer.get("instrument") or ""))
            if family == "voice":
                continue
            distance = abs(float(peer["time"]) - time)
            family_weight = {
                "bass": 1.45,
                "piano": 1.20,
                "guitar": 1.00,
                "pad": 0.82,
                "lead": 0.65,
                "other": 0.70,
            }.get(family, 0.70)
            salience = (
                math.exp(-distance / 0.22)
                * family_weight
                * (0.35 + min(1.0, float(peer.get("velocity", 0.7))))
                * (0.35 + min(1.5, float(peer.get("duration", 0.2))))
            )
            pitch_class = int(peer["midi"]) % 12
            chroma[pitch_class] += salience
            if family == "bass" or arrangement_role(peer) == "bass":
                bass_chroma[pitch_class] += salience

        total = max(1e-9, sum(chroma))
        own_support = chroma[midi % 12]
        maximum_support = max(chroma, default=0.0)
        own_rank = sum(value < own_support for value in chroma) / 11.0
        bass_root = max(range(12), key=lambda pitch_class: bass_chroma[pitch_class])
        has_bass_evidence = max(bass_chroma, default=0.0) > 0.0

        templates: list[tuple[float, int, str]] = []
        for root in range(12):
            bass_bonus = 0.22 * bass_chroma[root]
            templates.append(
                (
                    chroma[root]
                    + 0.86 * chroma[(root + 4) % 12]
                    + 0.82 * chroma[(root + 7) % 12]
                    + bass_bonus,
                    root,
                    "major",
                )
            )
            templates.append(
                (
                    chroma[root]
                    + 0.86 * chroma[(root + 3) % 12]
                    + 0.82 * chroma[(root + 7) % 12]
                    + bass_bonus,
                    root,
                    "minor",
                )
            )
        template_score, estimated_root, quality = max(
            templates, key=lambda item: (item[0], -item[1], item[2] == "minor")
        )
        interval = (midi % 12 - estimated_root) % 12
        third = 4 if quality == "major" else 3
        root_member = interval == 0
        third_member = interval == third
        fifth_member = interval == 7
        root_interval_flags = [
            1.0 if interval == candidate else 0.0 for candidate in range(12)
        ]
        rows.append(
            base_row
            + [
                own_support / total,
                own_rank,
                own_support / max(1e-9, maximum_support),
                1.0 if quality == "major" else 0.0,
                1.0 if quality == "minor" else 0.0,
                1.0 if root_member else 0.0,
                1.0 if third_member else 0.0,
                1.0 if fifth_member else 0.0,
                1.0 if not (root_member or third_member or fifth_member) else 0.0,
                min(1.0, template_score / total),
                1.0 if has_bass_evidence and midi % 12 == bass_root else 0.0,
            ]
            + root_interval_flags
        )
    return rows


def onset_gesture_selection_feature_rows(
    notes: list[dict[str, Any]],
) -> list[list[float]]:
    """Return one transposition-invariant rhythm row per detected note.

    Source separators commonly emit several stems at exactly the same onset.
    Those events are evidence for one human gesture, not several unrelated
    timing choices.  Rows are consequently calculated per millisecond onset
    and copied onto every event in that onset.  This keeps the feature API
    compatible with existing JSON logistic profiles while allowing the
    left-hand decoder to rank complete attacks.
    """

    if not notes:
        return []

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        grouped_indices[int(round(float(note["time"]) * 1000.0))].append(index)
    onset_keys = sorted(grouped_indices)
    onset_times = [key / 1000.0 for key in onset_keys]
    onset_position = {key: position for position, key in enumerate(onset_keys)}
    voice_times = sorted(
        float(note["time"])
        for note in notes
        if instrument_family(str(note.get("instrument") or "")) == "voice"
    )

    def gap_similarity(first: float, second: float) -> float:
        if first >= 6.0 or second >= 6.0:
            return 0.0
        return math.exp(-abs(first - second) / 0.18)

    rows_by_key: dict[int, list[float]] = {}
    duration_log_scale = math.log(7.0)
    for key in onset_keys:
        position = onset_position[key]
        time = onset_times[position]
        group = [notes[index] for index in grouped_indices[key]]
        families = [
            instrument_family(str(note.get("instrument") or ""))
            for note in group
        ]
        family_counts = Counter(families)
        note_count = len(group)
        non_voice = [
            note
            for note, family in zip(group, families)
            if family != "voice"
        ]
        harmonic_group = non_voice or group
        pitch_classes = {int(note["midi"]) % 12 for note in harmonic_group}
        durations = [float(note.get("duration", 0.2)) for note in group]
        velocities = [float(note.get("velocity", 0.7)) for note in group]

        pc_families: dict[int, set[str]] = defaultdict(set)
        for note, family in zip(group, families):
            if family != "voice":
                pc_families[int(note["midi"]) % 12].add(family)
        maximum_pc_family_support = max(
            (len(value) for value in pc_families.values()), default=0
        )
        bass_notes = [
            note
            for note, family in zip(group, families)
            if family == "bass" or arrangement_role(note) == "bass"
        ]
        bass_note = min(
            bass_notes or harmonic_group,
            key=lambda note: int(note["midi"]),
        )
        bass_pitch_class = int(bass_note["midi"]) % 12
        bass_pc_family_support = len(pc_families.get(bass_pitch_class, set()))

        # Chord-template confidence is relative to the strongest possible
        # three-note template, so it is invariant to key and source density.
        chroma = [0.0] * 12
        for note, family in zip(group, families):
            if family == "voice":
                continue
            chroma[int(note["midi"]) % 12] += {
                "bass": 1.35,
                "piano": 1.15,
                "guitar": 1.0,
                "pad": 0.8,
                "lead": 0.65,
                "other": 0.7,
            }.get(family, 0.7)
        total_chroma = max(1e-9, sum(chroma))
        templates: list[tuple[float, tuple[int, int, int]]] = []
        for root in range(12):
            templates.append(
                (
                    chroma[root]
                    + 0.88 * chroma[(root + 4) % 12]
                    + 0.84 * chroma[(root + 7) % 12],
                    (root, (root + 4) % 12, (root + 7) % 12),
                )
            )
            templates.append(
                (
                    chroma[root]
                    + 0.88 * chroma[(root + 3) % 12]
                    + 0.84 * chroma[(root + 7) % 12],
                    (root, (root + 3) % 12, (root + 7) % 12),
                )
            )
        best_template_score, best_template = max(templates, key=lambda item: item[0])
        template_coverage = sum(pc in pitch_classes for pc in best_template) / 3.0

        previous_gap = (
            time - onset_times[position - 1] if position > 0 else 6.0
        )
        next_gap = (
            onset_times[position + 1] - time
            if position + 1 < len(onset_times)
            else 6.0
        )
        previous_previous_gap = (
            onset_times[position - 1] - onset_times[position - 2]
            if position > 1
            else 6.0
        )
        next_next_gap = (
            onset_times[position + 2] - onset_times[position + 1]
            if position + 2 < len(onset_times)
            else 6.0
        )
        previous_pitch_classes = (
            {
                int(notes[index]["midi"]) % 12
                for index in grouped_indices[onset_keys[position - 1]]
                if instrument_family(
                    str(notes[index].get("instrument") or "")
                )
                != "voice"
            }
            if position > 0
            else set()
        )
        next_pitch_classes = (
            {
                int(notes[index]["midi"]) % 12
                for index in grouped_indices[onset_keys[position + 1]]
                if instrument_family(
                    str(notes[index].get("instrument") or "")
                )
                != "voice"
            }
            if position + 1 < len(onset_keys)
            else set()
        )
        overlap_denominator = max(1, len(pitch_classes))
        local_left = bisect.bisect_left(onset_times, time - 0.50)
        local_right = bisect.bisect_right(onset_times, time + 0.50)
        voice_distance = _nearest_distance(voice_times, time)
        harmonic_midis = [int(note["midi"]) for note in harmonic_group]

        rows_by_key[key] = [
            1.0,
            min(1.0, note_count / 12.0),
            min(1.0, len(pitch_classes) / 8.0),
            min(1.0, len(set(families)) / 6.0),
            1.0 if len(set(families)) >= 2 else 0.0,
            family_counts.get("voice", 0) / note_count,
            family_counts.get("bass", 0) / note_count,
            family_counts.get("piano", 0) / note_count,
            family_counts.get("guitar", 0) / note_count,
            family_counts.get("lead", 0) / note_count,
            family_counts.get("pad", 0) / note_count,
            family_counts.get("other", 0) / note_count,
            math.log1p(sum(durations) / note_count) / duration_log_scale,
            math.log1p(max(durations)) / duration_log_scale,
            min(1.0, (max(durations) - min(durations)) / 2.0),
            sum(velocities) / note_count,
            max(velocities),
            min(1.0, (local_right - local_left) / 18.0),
            math.log1p(min(6.0, previous_gap)) / duration_log_scale,
            math.log1p(min(6.0, next_gap)) / duration_log_scale,
            gap_similarity(previous_gap, next_gap),
            gap_similarity(previous_gap, previous_previous_gap),
            gap_similarity(next_gap, next_next_gap),
            len(pitch_classes & previous_pitch_classes) / overlap_denominator,
            len(pitch_classes & next_pitch_classes) / overlap_denominator,
            min(1.0, maximum_pc_family_support / 4.0),
            min(1.0, bass_pc_family_support / 4.0),
            min(1.0, best_template_score / total_chroma),
            template_coverage,
            min(1.0, (max(harmonic_midis) - min(harmonic_midis)) / 48.0),
            1.0 if family_counts.get("voice", 0) else 0.0,
            1.0 if voice_distance <= 0.18 else 0.0,
        ]

    return [
        rows_by_key[int(round(float(note["time"]) * 1000.0))]
        for note in notes
    ]


def onset_gesture_sequence_feature_rows(
    notes: list[dict[str, Any]],
) -> list[list[float]]:
    """Append chord-run phase to the frozen onset-gesture feature contract.

    A run advances only on onsets containing non-vocal harmonic evidence.  It
    resets after a long silence or a material chord change.  The local pulse is
    inferred from neighbouring source gaps, so the representation remains
    tempo-relative and works when a performance is faster or slower.
    """

    if not notes:
        return []
    base_rows = onset_gesture_selection_feature_rows(notes)
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        grouped_indices[int(round(float(note["time"]) * 1000.0))].append(index)

    harmonic_keys = [
        key
        for key in sorted(grouped_indices)
        if any(
            instrument_family(str(notes[index].get("instrument") or "")) != "voice"
            for index in grouped_indices[key]
        )
    ]
    if not harmonic_keys:
        return [row + [0.0] * (len(ONSET_GESTURE_SEQUENCE_FEATURE_NAMES) - len(row)) for row in base_rows]

    times = [key / 1000.0 for key in harmonic_keys]
    pitch_classes = []
    for key in harmonic_keys:
        pitch_classes.append(
            {
                int(notes[index]["midi"]) % 12
                for index in grouped_indices[key]
                if instrument_family(str(notes[index].get("instrument") or "")) != "voice"
            }
        )

    def similarity(left: set[int], right: set[int]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        return 2.0 * overlap / (len(left) + len(right))

    gaps = [times[index] - times[index - 1] for index in range(1, len(times))]
    run_positions: list[int] = []
    run_starts: list[int] = []
    run_position = 0
    run_start = 0
    for position in range(len(harmonic_keys)):
        if position:
            previous_gap = gaps[position - 1]
            previous_similarity = similarity(
                pitch_classes[position - 1], pitch_classes[position]
            )
            if previous_gap >= 0.75 or previous_similarity < 0.40:
                run_position = 0
                run_start = position
            else:
                run_position += 1
        run_positions.append(run_position)
        run_starts.append(run_start)

    extras_by_key: dict[int, list[float]] = {}
    for position, key in enumerate(harmonic_keys):
        previous_gap = gaps[position - 1] if position else 6.0
        next_gap = gaps[position] if position < len(gaps) else 6.0
        previous_similarity = (
            similarity(pitch_classes[position - 1], pitch_classes[position])
            if position
            else 0.0
        )
        next_similarity = (
            similarity(pitch_classes[position], pitch_classes[position + 1])
            if position + 1 < len(harmonic_keys)
            else 0.0
        )
        same_previous = position > 0 and previous_gap < 0.75 and previous_similarity >= 0.40
        same_next = (
            position + 1 < len(harmonic_keys)
            and next_gap < 0.75
            and next_similarity >= 0.40
        )
        local_gaps = sorted(
            value
            for value in gaps[max(0, position - 4) : min(len(gaps), position + 4)]
            if 0.055 <= value <= 0.75
        )
        if local_gaps:
            middle = len(local_gaps) // 2
            pulse = (
                local_gaps[middle]
                if len(local_gaps) % 2
                else (local_gaps[middle - 1] + local_gaps[middle]) / 2.0
            )
        else:
            pulse = 0.30
        pulse = clamp(pulse, 0.08, 0.75)
        elapsed_pulses = (times[position] - times[run_starts[position]]) / pulse
        phase = elapsed_pulses - math.floor(elapsed_pulses)
        run_index = run_positions[position]
        extras_by_key[key] = [
            float(same_previous),
            float(same_next),
            float(position == 0 or not same_previous),
            float(position + 1 == len(harmonic_keys) or not same_next),
            float(run_index == 0),
            float(not same_next),
            min(1.0, run_index / 8.0),
            float(run_index % 2 == 0),
            float(run_index % 2 == 1),
            *[float(run_index % 4 == candidate) for candidate in range(4)],
            *[float(run_index % 8 == candidate) for candidate in range(8)],
            min(4.0, previous_gap / pulse) / 4.0,
            min(4.0, next_gap / pulse) / 4.0,
            min(2.0, elapsed_pulses / 8.0),
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
        ]

    harmonic_times = [key / 1000.0 for key in harmonic_keys]
    output: list[list[float]] = []
    for note, base in zip(notes, base_rows):
        key = int(round(float(note["time"]) * 1000.0))
        extras = extras_by_key.get(key)
        if extras is None:
            time = key / 1000.0
            nearest = bisect.bisect_left(harmonic_times, time)
            candidates = [
                index
                for index in (nearest - 1, nearest)
                if 0 <= index < len(harmonic_keys)
            ]
            nearest_index = min(
                candidates,
                key=lambda index: abs(harmonic_times[index] - time),
            )
            extras = extras_by_key[harmonic_keys[nearest_index]]
        output.append(base + extras)
    return output


def selection_feature_rows(
    notes: list[dict[str, Any]],
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> list[list[float]]:
    contract = tuple(str(name) for name in feature_names)
    if contract == FEATURE_NAMES:
        return raw_feature_rows(notes)
    if contract == CONTEXT_SELECTION_FEATURE_NAMES:
        return contextual_selection_feature_rows(notes)
    if contract == ROBUST_CONTEXT_SELECTION_FEATURE_NAMES:
        full_rows = contextual_selection_feature_rows(notes)
        full_indices = {
            name: index for index, name in enumerate(CONTEXT_SELECTION_FEATURE_NAMES)
        }
        return [
            [row[full_indices[name]] for name in ROBUST_CONTEXT_SELECTION_FEATURE_NAMES]
            for row in full_rows
        ]
    if contract == LEFT_HAND_SELECTION_FEATURE_NAMES:
        return left_hand_selection_feature_rows(notes)
    if contract == HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES:
        return harmonic_left_hand_selection_feature_rows(notes)
    if contract == ONSET_GESTURE_SELECTION_FEATURE_NAMES:
        return onset_gesture_selection_feature_rows(notes)
    if contract == ONSET_GESTURE_SEQUENCE_FEATURE_NAMES:
        return onset_gesture_sequence_feature_rows(notes)
    raise ValueError("Piano arranger profile has an unsupported selection feature contract.")


def _next_distinct_onset(onsets: list[float], time: float, minimum_gap: float = 0.04) -> float:
    index = bisect.bisect_right(onsets, time + minimum_gap)
    if index >= len(onsets):
        return 6.0
    return clamp(onsets[index] - time, MIN_NOTE_SECONDS, 6.0)


def duration_feature_rows(notes: list[dict[str, Any]]) -> list[list[float]]:
    """Return context features used by the learned key-hold model."""
    if not notes:
        return []
    raw_rows = raw_feature_rows(notes)
    all_onsets = sorted({float(note["time"]) for note in notes})
    role_onsets: dict[str, list[float]] = defaultdict(list)
    for role in ("melody", "bass", "harmony"):
        role_onsets[role] = sorted(
            {float(note["time"]) for note in notes if arrangement_role(note) == role}
        )

    pitch_onsets: dict[int, list[float]] = defaultdict(list)
    for pitch in {int(note["midi"]) for note in notes}:
        pitch_onsets[pitch] = sorted(
            {float(note["time"]) for note in notes if int(note["midi"]) == pitch}
        )
    next_same_pitch: list[float | None] = []
    for note in notes:
        time = float(note["time"])
        onsets = pitch_onsets[int(note["midi"])]
        following_index = bisect.bisect_right(onsets, time + 0.04)
        next_same_pitch.append(
            clamp(onsets[following_index] - time, MIN_NOTE_SECONDS, 6.0)
            if following_index < len(onsets)
            else None
        )

    rows: list[list[float]] = []
    log_scale = math.log(7.0)
    for index, (note, raw_row) in enumerate(zip(notes, raw_rows)):
        role = arrangement_role(note)
        same_pitch_gap = next_same_pitch[index]
        # A missing later occurrence is represented by the six-second cap, but
        # the explicit flag lets the model distinguish it from a real long gap.
        effective_same_pitch_gap = same_pitch_gap if same_pitch_gap is not None else 6.0
        role_gap = _next_distinct_onset(role_onsets[role], float(note["time"]))
        any_gap = _next_distinct_onset(all_onsets, float(note["time"]))
        rows.append(
            raw_row
            + [
                1.0 if role == "melody" else 0.0,
                1.0 if role == "bass" else 0.0,
                1.0 if role == "harmony" else 0.0,
                1.0 if same_pitch_gap is not None else 0.0,
                math.log1p(effective_same_pitch_gap) / log_scale,
                math.log1p(role_gap) / log_scale,
                math.log1p(any_gap) / log_scale,
                1.0 if effective_same_pitch_gap >= 0.45 else 0.0,
                1.0 if effective_same_pitch_gap >= 0.90 else 0.0,
                1.0 if effective_same_pitch_gap >= 1.80 else 0.0,
                1.0 if role_gap >= 0.45 else 0.0,
                1.0 if role_gap >= 0.90 else 0.0,
                1.0 if role_gap >= 1.80 else 0.0,
            ]
        )
    return rows


def selection_scores(notes: list[dict[str, Any]], profile: dict[str, Any]) -> list[float]:
    model = profile.get("selectionModel") or {}
    feature_names = tuple(str(value) for value in model.get("featureNames", FEATURE_NAMES))
    means = [float(value) for value in model.get("means", [])]
    scales = [max(1e-9, float(value)) for value in model.get("scales", [])]
    model_type = str(model.get("type") or "standardized-logistic-left-hand-ranker-v1")
    if model_type == "standardized-mlp-left-hand-ranker-v1":
        hidden_weights = [
            [float(value) for value in row]
            for row in model.get("hiddenWeights", [])
        ]
        hidden_biases = [float(value) for value in model.get("hiddenBiases", [])]
        output_weights = [float(value) for value in model.get("outputWeights", [])]
        output_bias = float(model.get("outputBias", 0.0))
        if len(means) != len(feature_names) or len(scales) != len(feature_names):
            raise ValueError("Piano arranger MLP has incompatible feature scaling.")
        if len(hidden_weights) != len(feature_names):
            raise ValueError("Piano arranger MLP has incompatible hidden weights.")
        hidden_size = len(hidden_biases)
        if hidden_size < 1 or len(output_weights) != hidden_size:
            raise ValueError("Piano arranger MLP has incompatible hidden dimensions.")
        if any(len(row) != hidden_size for row in hidden_weights):
            raise ValueError("Piano arranger MLP hidden rows have inconsistent dimensions.")
        probabilities: list[float] = []
        for row in selection_feature_rows(notes, feature_names):
            standardized = [
                (value - mean) / scale
                for value, mean, scale in zip(row, means, scales)
            ]
            hidden = [
                math.tanh(
                    hidden_biases[unit]
                    + sum(
                        standardized[feature] * hidden_weights[feature][unit]
                        for feature in range(len(standardized))
                    )
                )
                for unit in range(hidden_size)
            ]
            logit = output_bias + sum(
                value * weight for value, weight in zip(hidden, output_weights)
            )
            probabilities.append(1.0 / (1.0 + math.exp(-clamp(logit, -30.0, 30.0))))
        return probabilities

    weights = [float(value) for value in model.get("weights", [])]
    if len(weights) != len(feature_names):
        raise ValueError("Piano arranger profile has incompatible selection weights.")
    if len(means) != len(weights) or len(scales) != len(weights):
        raise ValueError("Piano arranger profile has incompatible feature scaling.")
    probabilities: list[float] = []
    for row in selection_feature_rows(notes, feature_names):
        logit = sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(weights, row, means, scales)
        )
        logit = clamp(logit, -30.0, 30.0)
        probabilities.append(1.0 / (1.0 + math.exp(-logit)))
    return probabilities


def _probability_logit(value: float) -> float:
    probability = clamp(float(value), 1e-6, 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _conditional_selection_eligible(
    note: dict[str, Any], policy: dict[str, Any]
) -> bool:
    midi = int(note["midi"])
    minimum_midi = int(policy.get("minimumSourceMidi", 0))
    maximum_midi = int(policy.get("maximumSourceMidi", 127))
    if not minimum_midi <= midi <= maximum_midi:
        return False
    families = {
        str(value).strip().lower()
        for value in policy.get("sourceFamilies", [])
        if str(value).strip()
    }
    family = instrument_family(str(note.get("instrument") or ""))
    return not families or family in families


def contextual_selection_scores(
    notes: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[float], dict[str, Any] | None]:
    """Blend an alternate selector only inside an explicit source region.

    Full-mix transcription can label vocal-harmony evidence as guitar.  A
    global selector change also disturbs the already reliable voice and bass
    paths, so research profiles may install a richer contextual model only for
    a bounded source family/register.  Each model's own threshold is removed
    before its decision margin is blended; this avoids treating two differently
    calibrated probabilities as though they were directly comparable.
    """
    base_scores = selection_scores(notes, profile)
    policy = (profile.get("decoder") or {}).get("conditionalSelectionBlend") or {}
    if not policy.get("enabled"):
        return base_scores, None

    alternative_model = policy.get("selectionModel") or {}
    if not alternative_model:
        raise ValueError("Conditional selection blend requires selectionModel.")
    alternative_scores = selection_scores(
        notes, {"selectionModel": alternative_model}
    )
    alternative_share = clamp(float(policy.get("alternativeShare", 0.5)), 0.0, 1.0)
    minimum_midi = int(policy.get("minimumSourceMidi", 0))
    maximum_midi = int(policy.get("maximumSourceMidi", 127))
    families = {
        str(value).strip().lower()
        for value in policy.get("sourceFamilies", [])
        if str(value).strip()
    }
    base_threshold = clamp(
        float((profile.get("selectionModel") or {}).get("threshold", 0.5)),
        1e-6,
        1.0 - 1e-6,
    )
    alternative_threshold = clamp(
        float(alternative_model.get("threshold", 0.5)), 1e-6, 1.0 - 1e-6
    )
    base_boundary = _probability_logit(base_threshold)
    alternative_boundary = _probability_logit(alternative_threshold)

    output = list(base_scores)
    eligible = 0
    changed = 0
    absolute_change = 0.0
    for index, note in enumerate(notes):
        if not _conditional_selection_eligible(note, policy):
            continue
        eligible += 1
        base_margin = _probability_logit(base_scores[index]) - base_boundary
        alternative_margin = (
            _probability_logit(alternative_scores[index]) - alternative_boundary
        )
        blended_logit = base_boundary + (
            (1.0 - alternative_share) * base_margin
            + alternative_share * alternative_margin
        )
        blended = 1.0 / (1.0 + math.exp(-clamp(blended_logit, -30.0, 30.0)))
        output[index] = blended
        difference = abs(blended - base_scores[index])
        absolute_change += difference
        changed += int(difference > 1e-9)

    return output, {
        "profile": "threshold-centered-conditional-logit-blend-v1",
        "eligibleNotes": eligible,
        "changedScores": changed,
        "meanAbsoluteProbabilityChange": round_number(
            absolute_change / max(1, eligible), 6
        ),
        "alternativeShare": round_number(alternative_share, 4),
        "minimumSourceMidi": minimum_midi,
        "maximumSourceMidi": maximum_midi,
        "sourceFamilies": sorted(families),
        "baseThreshold": round_number(base_threshold, 6),
        "alternativeThreshold": round_number(alternative_threshold, 6),
    }


def duration_predictions(
    notes: list[dict[str, Any]], profile: dict[str, Any]
) -> list[float | None]:
    """Predict written key-hold durations, preserving old-profile compatibility."""
    model = profile.get("durationModel") or {}
    if not model:
        return [None] * len(notes)
    weights = [float(value) for value in model.get("weights", [])]
    means = [float(value) for value in model.get("means", [])]
    scales = [max(1e-9, float(value)) for value in model.get("scales", [])]
    if len(weights) != len(DURATION_FEATURE_NAMES):
        raise ValueError("Piano arranger profile has incompatible duration weights.")
    if len(means) != len(weights) or len(scales) != len(weights):
        raise ValueError("Piano arranger profile has incompatible duration feature scaling.")
    minimum = clamp(float(model.get("minimumSeconds", MIN_NOTE_SECONDS)), MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)
    maximum = clamp(float(model.get("maximumSeconds", 4.0)), minimum, MAX_NOTE_SECONDS)
    predictions: list[float | None] = []
    for row in duration_feature_rows(notes):
        predicted_log = sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(weights, row, means, scales)
        )
        predictions.append(clamp(math.expm1(clamp(predicted_log, 0.0, math.log1p(MAX_NOTE_SECONDS))), minimum, maximum))
    return predictions


def map_octave_to_range(midi: int, minimum: int, maximum: int) -> int:
    value = int(round(midi))
    while value < minimum:
        value += 12
    while value > maximum:
        value -= 12
    return int(clamp(value, minimum, maximum))


def source_midi_band(midi: int) -> str:
    if midi < 48:
        return "low"
    if midi < 60:
        return "mid"
    return "high"


def _role_config(profile: dict[str, Any], role: str) -> dict[str, Any]:
    defaults = {
        "melody": {"minimumMidi": 55, "maximumMidi": 88, "durationScale": 1.0, "medianDuration": 0.35, "velocity": 0.86},
        "bass": {"minimumMidi": 28, "maximumMidi": 55, "durationScale": 1.0, "medianDuration": 0.55, "velocity": 0.68},
        "harmony": {"minimumMidi": 45, "maximumMidi": 79, "durationScale": 1.0, "medianDuration": 0.45, "velocity": 0.62},
    }
    return {**defaults[role], **((profile.get("roles") or {}).get(role) or {})}


def _render_note(
    source: dict[str, Any],
    probability: float,
    profile: dict[str, Any],
    predicted_duration: float | None = None,
) -> dict[str, Any]:
    role = arrangement_role(source)
    config = _role_config(profile, role)
    octave_shift = int(round(float(config.get("octaveShift", 0)) / 12.0)) * 12
    midi = map_octave_to_range(
        int(source["midi"]) + octave_shift,
        int(config["minimumMidi"]),
        int(config["maximumMidi"]),
    )
    scaled_duration = float(source["duration"]) * float(config["durationScale"])
    median_duration = float(config["medianDuration"])
    source_duration_weight = clamp(
        float(profile.get("decoder", {}).get("sourceDurationWeight", 0.85)),
        0.0,
        1.0,
    )
    fallback_duration = clamp(
        scaled_duration * source_duration_weight
        + median_duration * (1.0 - source_duration_weight),
        MIN_NOTE_SECONDS,
        MAX_NOTE_SECONDS,
    )
    if predicted_duration is None:
        duration = fallback_duration
    else:
        learned_weight = clamp(
            float(profile.get("durationModel", {}).get("predictionWeight", 0.82)),
            0.0,
            1.0,
        )
        # Blend in log-duration space so a long predicted hold is not flattened
        # by the separator's often-uniform short source duration.
        duration = math.expm1(
            math.log1p(predicted_duration) * learned_weight
            + math.log1p(fallback_duration) * (1.0 - learned_weight)
        )
        duration = clamp(duration, MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)
        if profile.get("decoder", {}).get("durationExtensionOnly", False):
            duration = max(fallback_duration, duration)
    minimum_rendered_duration = clamp(
        float(
            profile.get("decoder", {}).get(
                "minimumRenderedDurationSeconds", MIN_NOTE_SECONDS
            )
        ),
        MIN_NOTE_SECONDS,
        1.0,
    )
    role_minimums = profile.get("decoder", {}).get(
        "minimumRenderedDurationByRole", {}
    )
    if isinstance(role_minimums, dict):
        minimum_rendered_duration = max(
            minimum_rendered_duration,
            clamp(
                float(role_minimums.get(role, MIN_NOTE_SECONDS)),
                MIN_NOTE_SECONDS,
                1.0,
            ),
        )
    family_minimums = profile.get("decoder", {}).get(
        "minimumRenderedDurationBySourceFamily", {}
    )
    if isinstance(family_minimums, dict):
        family = instrument_family(str(source.get("instrument") or ""))
        minimum_rendered_duration = max(
            minimum_rendered_duration,
            clamp(
                float(family_minimums.get(family, MIN_NOTE_SECONDS)),
                MIN_NOTE_SECONDS,
                1.0,
            ),
        )
    duration = max(duration, minimum_rendered_duration)
    learned_velocity = float(config["velocity"])
    velocity = clamp(learned_velocity * 0.72 + float(source["velocity"]) * 0.28, 0.05, 1.0)
    return {
        **source,
        # Preserve the factual, audio-derived source contour separately from
        # the arranger's role velocity.  The pianist-performance pass can then
        # make one chord coherent without reverse-engineering an already mixed
        # melody/bass/harmony value.
        "sourceVelocityBeforeArrangement": round_number(
            clamp(float(source["velocity"]), 0.05, 1.0), 4
        ),
        "sourceMidiBeforeArrangement": int(source["midi"]),
        "midi": midi,
        "note": midi_to_note(midi),
        "instrument": "acoustic_piano",
        "sourceInstrument": source["instrument"],
        "arrangementRole": role,
        "hand": "left" if midi < 60 else "right",
        "selectionProbability": round_number(probability, 4),
        "duration": round_number(duration),
        "velocity": round_number(velocity, 3),
    }


def apply_learned_register_model(
    notes: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Apply a conservative octave choice learned from trusted alignments.

    A full mix often contains the right pitch class in the wrong octave.  The
    profile stores hierarchical evidence keyed by musical role, detected
    source family, and coarse source register.  Weak or unseen groups fall
    back to the profile's explicit octave policy rather than overfitting a
    handful of examples.
    """
    model = profile.get("registerModel") or {}
    if model.get("type") != "hierarchical-categorical-octave-shift-v1":
        return notes, None
    groups = model.get("groups") or {}
    fallback_shift = int(round(float(model.get("fallbackShiftSemitones", 0)) / 12.0) * 12)
    uncertain_shift = int(round(float(model.get("uncertainShiftSemitones", 0)) / 12.0) * 12)
    specific_support = max(
        0.0, float(model.get("minimumSpecificWeightedSupport", 18.0))
    )
    role_support = max(0.0, float(model.get("minimumRoleWeightedSupport", 36.0)))
    minimum_confidence = clamp(float(model.get("minimumConfidence", 0.55)), 0.0, 1.0)
    output: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    shift_counts: Counter[int] = Counter()
    adjusted = 0

    for source in notes:
        note = dict(source)
        role = str(note.get("arrangementRole") or arrangement_role(note))
        family = instrument_family(str(note.get("sourceInstrument") or note.get("instrument") or ""))
        source_midi = int(note.get("sourceMidiBeforeArrangement", note["midi"]))
        keys = (
            f"{role}|{family}|{source_midi_band(source_midi)}",
            f"{role}|{family}",
        )
        shift = fallback_shift
        selected_key = "fallback"
        saw_evidence = False
        for key in keys:
            evidence = groups.get(key)
            if not isinstance(evidence, dict):
                continue
            saw_evidence = True
            support_required = role_support if key.count("|") == 1 else specific_support
            if float(evidence.get("weightedSupport", 0.0)) < support_required:
                continue
            if float(evidence.get("confidence", 0.0)) < minimum_confidence:
                continue
            shift = int(round(float(evidence.get("shiftSemitones", fallback_shift)) / 12.0) * 12)
            selected_key = key
            break
        else:
            if saw_evidence:
                shift = uncertain_shift
                selected_key = "uncertain"
        config = _role_config(profile, role)
        # Preserve the arranger's role-aware octave placement, then apply only
        # the learned/fallback correction. Reconstructing from the raw source
        # MIDI here would undo compact harmony voicing and create duplicate
        # octave layers that the selector never requested.
        midi = map_octave_to_range(
            int(note["midi"]) + shift,
            int(config["minimumMidi"]),
            int(config["maximumMidi"]),
        )
        if midi != int(note["midi"]):
            adjusted += 1
        note["midi"] = midi
        note["note"] = midi_to_note(midi)
        note["hand"] = "left" if midi < 60 else "right"
        note["learnedRegisterShiftSemitones"] = shift
        note["learnedRegisterEvidenceKey"] = selected_key
        selection_counts[selected_key] += 1
        shift_counts[shift] += 1
        output.append(note)

    return output, {
        "profile": model.get("type"),
        "adjustedNotes": adjusted,
        "fallbackShiftSemitones": fallback_shift,
        "uncertainShiftSemitones": uncertain_shift,
        "selectionCounts": dict(sorted(selection_counts.items())),
        "shiftCounts": {
            str(shift): count for shift, count in sorted(shift_counts.items())
        },
    }


def _voice_lead_harmony(
    notes: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Place chord tones in a compact, stable acoustic-piano register.

    Full-band transcription frequently identifies the correct pitch classes in
    several octaves.  Playing those literal octaves produces harsh jumps.  A
    pianist normally chooses one compact inversion and moves as little as
    possible from the preceding chord, so we do that only for accompaniment;
    the detected vocal/top melody remains untouched.
    """
    harmony = sorted(
        (dict(note) for note in notes if note.get("arrangementRole") == "harmony"),
        key=lambda item: (item["time"], item["midi"]),
    )
    untouched = [dict(note) for note in notes if note.get("arrangementRole") != "harmony"]
    if not harmony:
        return notes, 0

    groups: list[list[dict[str, Any]]] = []
    for note in harmony:
        if not groups or note["time"] - groups[-1][0]["time"] > 0.04:
            groups.append([note])
        else:
            groups[-1].append(note)

    config = _role_config(profile, "harmony")
    minimum = int(config["minimumMidi"])
    maximum = int(config["maximumMidi"])
    preferred_center = float(config.get("preferredCenterMidi", 60.0))
    previous_voicing: list[int] = []
    voiced: list[dict[str, Any]] = []
    changed = 0
    for group in groups:
        by_pitch_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for note in group:
            by_pitch_class[int(note["midi"]) % 12].append(note)
        representatives = [
            max(
                candidates,
                key=lambda item: (
                    float(item.get("selectionProbability", 0.0)),
                    float(item.get("velocity", 0.0)),
                    float(item.get("duration", 0.0)),
                ),
            )
            for candidates in by_pitch_class.values()
        ]
        representatives.sort(key=lambda item: int(item["midi"]))
        if len(representatives) > 5:
            representatives = sorted(
                representatives,
                key=lambda item: float(item.get("selectionProbability", 0.0)),
                reverse=True,
            )[:5]
            representatives.sort(key=lambda item: int(item["midi"]))

        candidate_sets = [
            [midi for midi in range(minimum, maximum + 1) if midi % 12 == int(note["midi"]) % 12]
            for note in representatives
        ]
        if not representatives or any(not candidates for candidates in candidate_sets):
            voiced.extend(group)
            continue

        def voicing_cost(values: tuple[int, ...]) -> float:
            ordered = sorted(values)
            span = ordered[-1] - ordered[0] if len(ordered) > 1 else 0
            center = sum(ordered) / len(ordered)
            source_distance = sum(
                abs(value - int(note["midi"]))
                for value, note in zip(values, representatives)
            )
            close_intervals = sum(
                max(0, 3 - (right - left))
                for left, right in zip(ordered, ordered[1:])
            )
            previous_distance = 0.0
            if previous_voicing:
                previous_distance = sum(
                    min(abs(value - previous) for previous in previous_voicing)
                    for value in ordered
                )
            return (
                abs(center - preferred_center)
                + max(0, span - 16) * 1.8
                + close_intervals * 1.2
                + source_distance * 0.08
                + previous_distance * 0.18
            )

        choices = [values for values in itertools.product(*candidate_sets) if len(set(values)) == len(values)]
        if not choices:
            voiced.extend(group)
            continue
        best = min(choices, key=voicing_cost)
        mapped_by_pitch_class = {
            int(note["midi"]) % 12: midi
            for note, midi in zip(representatives, best)
        }
        previous_voicing = sorted(mapped_by_pitch_class.values())
        for note in group:
            original_midi = int(note["midi"])
            midi = mapped_by_pitch_class.get(original_midi % 12, original_midi)
            if midi != original_midi:
                changed += 1
            note["midi"] = midi
            note["note"] = midi_to_note(midi)
            note["hand"] = "left" if midi < 60 else "right"
            voiced.append(note)
    return sorted(untouched + voiced, key=lambda item: (item["time"], item["midi"])), changed


def _group_indices_by_window(notes: list[dict[str, Any]], seconds: float) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        groups[int(math.floor(float(note["time"]) / seconds))].append(index)
    return [groups[key] for key in sorted(groups)]


def _pick_window_notes(
    notes: list[dict[str, Any]],
    scores: list[float],
    profile: dict[str, Any],
    mode: str,
) -> list[tuple[dict[str, Any], float]]:
    window_seconds = float(profile.get("decoder", {}).get("windowSeconds", 0.5))
    target_nps = float(profile.get("style", {}).get("targetNotesPerSecond", 6.5))
    density_multiplier = float(
        profile.get("decoder", {}).get("preCleanupDensityMultiplier", 1.0)
    )
    threshold = float(profile.get("selectionModel", {}).get("threshold", 0.5))
    quota_backfill_ratio = clamp(
        float(profile.get("decoder", {}).get("quotaBackfillRatio", 1.0)),
        0.0,
        1.0,
    )
    desired_per_window = target_nps * window_seconds * density_multiplier
    maximum_per_window = max(1, int(math.ceil(desired_per_window * 1.20)))
    selected: list[tuple[dict[str, Any], float]] = []
    for indices in _group_indices_by_window(notes, window_seconds):
        eligible = [
            index for index in indices
            if mode == "full" or notes[index]["instrument"] not in VOICE_INSTRUMENTS
        ]
        if not eligible:
            continue
        activity = min(1.0, len(eligible) / max(1.0, target_nps * window_seconds))
        quota = max(1, round(desired_per_window * (0.55 + 0.45 * activity)))
        quota = min(maximum_per_window, quota)
        ranked = sorted(
            eligible,
            key=lambda index: (
                scores[index],
                notes[index]["velocity"],
                notes[index]["duration"],
            ),
            reverse=True,
        )
        confident = [index for index in ranked if scores[index] >= threshold]
        chosen = confident[:quota]
        minimum_fill = min(len(ranked), int(math.ceil(quota * quota_backfill_ratio)))
        if len(chosen) < minimum_fill:
            chosen_set = set(chosen)
            chosen.extend(index for index in ranked if index not in chosen_set)
            chosen = chosen[:minimum_fill]

        # In full mode the vocal line is a compulsory musical signal, not an
        # instrument timbre that survives into the piano output.
        if mode == "full":
            voice_candidates = [index for index in ranked if notes[index]["instrument"] in VOICE_INSTRUMENTS]
            if voice_candidates and not any(index in chosen for index in voice_candidates):
                if len(chosen) >= quota:
                    chosen[-1] = voice_candidates[0]
                else:
                    chosen.append(voice_candidates[0])
        selected.extend((notes[index], scores[index]) for index in dict.fromkeys(chosen))
    return selected


def _rerank_conditional_selection_slots(
    notes: list[dict[str, Any]],
    base_selected: list[tuple[dict[str, Any], float]],
    blended_scores: list[float],
    profile: dict[str, Any],
    mode: str,
) -> tuple[list[tuple[dict[str, Any], float]], dict[str, int]]:
    """Rerank only the conditional notes already budgeted by the base model.

    This freezes every non-target note and the number of target-family slots
    per time window.  The contextual model may replace a weak upper-guitar
    choice with a better upper-guitar choice, but it cannot make the score
    denser or displace the established voice/bass paths.
    """
    policy = (profile.get("decoder") or {}).get("conditionalSelectionBlend") or {}
    window_seconds = float(profile.get("decoder", {}).get("windowSeconds", 0.5))
    score_by_index = {
        int(note["sourceIndex"]): float(score)
        for note, score in zip(notes, blended_scores)
    }
    notes_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        key = int(math.floor(float(note["time"]) / window_seconds))
        notes_by_window[key].append(note)
    selected_by_window: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for item in base_selected:
        key = int(math.floor(float(item[0]["time"]) / window_seconds))
        selected_by_window[key].append(item)

    output: list[tuple[dict[str, Any], float]] = []
    eligible_slots = 0
    replaced_slots = 0
    for key in sorted(selected_by_window):
        frozen: list[tuple[dict[str, Any], float]] = []
        old_eligible: list[tuple[dict[str, Any], float]] = []
        for item in selected_by_window[key]:
            if _conditional_selection_eligible(item[0], policy):
                old_eligible.append(item)
            else:
                frozen.append(item)
        eligible_slots += len(old_eligible)
        candidates = [
            note
            for note in notes_by_window.get(key, [])
            if (mode == "full" or note["instrument"] not in VOICE_INSTRUMENTS)
            and _conditional_selection_eligible(note, policy)
        ]
        ranked = sorted(
            candidates,
            key=lambda note: (
                score_by_index[int(note["sourceIndex"])],
                float(note["velocity"]),
                float(note["duration"]),
            ),
            reverse=True,
        )
        replacements = ranked[: len(old_eligible)]
        old_indices = {int(item[0]["sourceIndex"]) for item in old_eligible}
        new_indices = {int(note["sourceIndex"]) for note in replacements}
        replaced_slots += len(old_indices - new_indices)
        output.extend(frozen)
        output.extend(
            (note, score_by_index[int(note["sourceIndex"])])
            for note in replacements
        )
    return sorted(output, key=lambda item: (item[0]["time"], item[0]["midi"])), {
        "preservedBaseWindowCounts": True,
        "eligibleSlots": eligible_slots,
        "replacedSlots": replaced_slots,
    }


def _rerank_conditional_onset_slots(
    notes: list[dict[str, Any]],
    base_selected: list[tuple[dict[str, Any], float]],
    blended_scores: list[float],
    profile: dict[str, Any],
    mode: str,
) -> tuple[list[tuple[dict[str, Any], float]], dict[str, int | float | bool]]:
    """Rerank eligible pitches while freezing each base onset's slot count."""
    policy = (profile.get("decoder") or {}).get("conditionalSelectionBlend") or {}
    tolerance = clamp(float(policy.get("onsetSlotWindowSeconds", 0.035)), 0.005, 0.12)
    score_by_index = {
        int(note["sourceIndex"]): float(score)
        for note, score in zip(notes, blended_scores)
    }
    frozen = [
        item
        for item in base_selected
        if not _conditional_selection_eligible(item[0], policy)
    ]
    eligible_selected = sorted(
        (
            item
            for item in base_selected
            if _conditional_selection_eligible(item[0], policy)
        ),
        key=lambda item: (float(item[0]["time"]), int(item[0]["midi"])),
    )
    onset_groups: list[list[tuple[dict[str, Any], float]]] = []
    for item in eligible_selected:
        if (
            not onset_groups
            or float(item[0]["time"]) - float(onset_groups[-1][-1][0]["time"])
            > tolerance
        ):
            onset_groups.append([item])
        else:
            onset_groups[-1].append(item)

    candidates = [
        note
        for note in notes
        if (mode == "full" or note["instrument"] not in VOICE_INSTRUMENTS)
        and _conditional_selection_eligible(note, policy)
    ]
    used_indices: set[int] = set()
    replacements: list[tuple[dict[str, Any], float]] = []
    replaced_slots = 0
    for group in onset_groups:
        center = sum(float(item[0]["time"]) for item in group) / len(group)
        local = [
            note
            for note in candidates
            if int(note["sourceIndex"]) not in used_indices
            and abs(float(note["time"]) - center) <= tolerance
        ]
        ranked = sorted(
            local,
            key=lambda note: (
                score_by_index[int(note["sourceIndex"])],
                float(note["velocity"]),
                float(note["duration"]),
            ),
            reverse=True,
        )
        chosen = ranked[: len(group)]
        # The old selections are always valid local candidates.  Keep a
        # deterministic fallback in case malformed source indices violate
        # that invariant rather than silently dropping a physical strike.
        if len(chosen) < len(group):
            chosen_indices = {int(note["sourceIndex"]) for note in chosen}
            chosen.extend(
                item[0]
                for item in group
                if int(item[0]["sourceIndex"]) not in chosen_indices
            )
            chosen = chosen[: len(group)]
        old_indices = {int(item[0]["sourceIndex"]) for item in group}
        new_indices = {int(note["sourceIndex"]) for note in chosen}
        replaced_slots += len(old_indices - new_indices)
        used_indices.update(new_indices)
        replacements.extend(
            (note, score_by_index[int(note["sourceIndex"])]) for note in chosen
        )

    return sorted(
        frozen + replacements,
        key=lambda item: (item[0]["time"], item[0]["midi"]),
    ), {
        "preservedBaseWindowCounts": False,
        "preservedBaseOnsetCounts": True,
        "onsetSlotWindowSeconds": round_number(tolerance, 4),
        "eligibleOnsetGroups": len(onset_groups),
        "eligibleSlots": len(eligible_selected),
        "replacedSlots": replaced_slots,
    }


def _expand_sparse_windows(
    notes: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    decoder = profile.get("decoder") or {}
    if not decoder.get("expandSparseHarmony", True):
        return notes, 0
    window_seconds = float(decoder.get("windowSeconds", 0.5))
    target_nps = float(profile.get("style", {}).get("targetNotesPerSecond", 6.5))
    density_multiplier = float(decoder.get("preCleanupDensityMultiplier", 1.0))
    target_per_window = max(1, round(target_nps * window_seconds * density_multiplier))
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        groups[int(math.floor(note["time"] / window_seconds))].append(note)
    expanded: list[dict[str, Any]] = list(notes)
    added = 0
    for group in groups.values():
        if len(group) >= target_per_window or not group:
            continue
        existing = {(note["midi"], round(note["time"] / 0.04)) for note in group}
        candidates = sorted(
            group,
            key=lambda note: (
                note.get("arrangementRole") == "harmony",
                note.get("arrangementRole") == "bass",
                note["selectionProbability"],
            ),
            reverse=True,
        )
        for source in candidates:
            role = source.get("arrangementRole", "harmony")
            shifts = (-12, 12) if role != "bass" else (12, -12)
            for shift in shifts:
                config = _role_config(profile, role)
                midi = source["midi"] + shift
                if midi < int(config["minimumMidi"]) or midi > int(config["maximumMidi"]):
                    continue
                key = (midi, round(source["time"] / 0.04))
                if key in existing:
                    continue
                doubled = dict(source)
                doubled.update(
                    {
                        "midi": midi,
                        "note": midi_to_note(midi),
                        "velocity": round_number(clamp(source["velocity"] * 0.82, 0.05, 1.0), 3),
                        "generatedBy": "learned-pianella-octave-doubling",
                    }
                )
                expanded.append(doubled)
                existing.add(key)
                added += 1
                break
            if len(existing) >= target_per_window:
                break
    return expanded, added


def _collapse_and_limit(
    notes: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    style = profile.get("style") or {}
    duplicate_seconds = float(style.get("duplicateOnsetSeconds", 0.075))
    retrigger_seconds = float(style.get("minimumRetriggerSeconds", 0.10))
    maximum_cluster = int(style.get("maximumOnsetCluster", 6))
    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in sorted(notes, key=lambda item: (item["time"], item["midi"])):
        by_pitch[note["midi"]].append(note)
    collapsed: list[dict[str, Any]] = []
    duplicates = 0
    retriggers = 0
    preserved_retriggers = 0
    for pitch_notes in by_pitch.values():
        kept: list[dict[str, Any]] = []
        for note in pitch_notes:
            decision = (
                classify_same_key_retrigger(
                    kept[-1],
                    note,
                    duplicate_seconds=duplicate_seconds,
                    collision_seconds=retrigger_seconds,
                )
                if kept
                else "musical-repeat"
            )
            if kept and decision != "musical-repeat":
                previous = kept[-1]
                previous_end = previous["time"] + previous["duration"]
                note_end = note["time"] + note["duration"]
                preferred = previous if previous["selectionProbability"] >= note["selectionProbability"] else note
                preferred = dict(preferred)
                preferred["time"] = min(previous["time"], note["time"])
                preferred["duration"] = round_number(max(previous_end, note_end) - preferred["time"])
                kept[-1] = preferred
                if decision == "duplicate":
                    duplicates += 1
                else:
                    retriggers += 1
                continue
            if kept and note["time"] - kept[-1]["time"] < retrigger_seconds:
                preserved_retriggers += 1
            kept.append(dict(note))
        collapsed.extend(kept)

    clustered: list[dict[str, Any]] = []
    cluster_removed = 0
    for indices in _group_indices_by_window(sorted(collapsed, key=lambda item: item["time"]), 0.04):
        ordered = sorted(collapsed, key=lambda item: item["time"])
        group = [ordered[index] for index in indices]
        ranked = sorted(
            group,
            key=lambda item: (
                item.get("arrangementRole") == "melody",
                item.get("arrangementRole") == "bass",
                item["selectionProbability"],
            ),
            reverse=True,
        )
        clustered.extend(ranked[:maximum_cluster])
        cluster_removed += max(0, len(ranked) - maximum_cluster)
    return sorted(clustered, key=lambda item: (item["time"], item["midi"])), {
        "collapsedDuplicateNotes": duplicates,
        "mergedArtificialRetriggerCollisions": retriggers,
        "mergedRapidRetriggers": retriggers,
        "preservedFastMusicalRetriggers": preserved_retriggers,
        "removedForOnsetClusterLimit": cluster_removed,
    }


def _shape_legato(notes: list[dict[str, Any]], profile: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_role[note.get("arrangementRole", "harmony")].append(dict(note))
    extended = 0
    output: list[dict[str, Any]] = []
    for role, role_notes in by_role.items():
        config = _role_config(profile, role)
        maximum_bridge = float(config.get("maximumBridgeSeconds", 1.2))
        overlap = float(config.get("legatoOverlapSeconds", 0.08))
        maximum_extension = float(config.get("maximumLegatoExtensionSeconds", 0.30))
        onset_times = sorted({note["time"] for note in role_notes})
        for note in role_notes:
            next_index = bisect.bisect_right(onset_times, note["time"] + 0.04)
            if next_index < len(onset_times):
                gap = onset_times[next_index] - note["time"]
                if gap <= maximum_bridge:
                    target = min(
                        MAX_NOTE_SECONDS,
                        note["duration"] + maximum_extension,
                        gap + overlap,
                    )
                    if target > note["duration"]:
                        note["duration"] = round_number(target)
                        extended += 1
            output.append(note)
    return sorted(output, key=lambda item: (item["time"], item["midi"])), extended


def arrange_with_profile(
    payload: dict[str, Any],
    mode: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if profile.get("schema") != "polymath-piano-arranger-profile-v1":
        raise ValueError("Unsupported learned piano arranger profile.")
    source_notes = normalize_source_notes(payload.get("notes", []))
    if not source_notes:
        raise ValueError("No non-percussive notes were available for the piano arranger.")
    scores, conditional_selection = contextual_selection_scores(source_notes, profile)
    predicted_durations = duration_predictions(source_notes, profile)
    duration_by_source_index = {
        note["sourceIndex"]: prediction
        for note, prediction in zip(source_notes, predicted_durations)
    }
    conditional_policy = (profile.get("decoder") or {}).get(
        "conditionalSelectionBlend"
    ) or {}
    if conditional_selection and (
        conditional_policy.get("preserveBaseWindowCounts")
        or conditional_policy.get("preserveBaseOnsetCounts")
    ):
        base_scores = selection_scores(source_notes, profile)
        base_selected = _pick_window_notes(source_notes, base_scores, profile, mode)
        if conditional_policy.get("preserveBaseOnsetCounts"):
            selected, rerank_diagnostics = _rerank_conditional_onset_slots(
                source_notes, base_selected, scores, profile, mode
            )
        else:
            selected, rerank_diagnostics = _rerank_conditional_selection_slots(
                source_notes, base_selected, scores, profile, mode
            )
        conditional_selection.update(rerank_diagnostics)
    else:
        selected = _pick_window_notes(source_notes, scores, profile, mode)
    rendered = [
        _render_note(
            note,
            probability,
            profile,
            duration_by_source_index.get(note["sourceIndex"]),
        )
        for note, probability in selected
    ]
    register_adjusted, register_diagnostics = apply_learned_register_model(
        rendered, profile
    )
    if profile.get("decoder", {}).get("compactHarmonyVoicing", False):
        voiced, revoiced_harmony_notes = _voice_lead_harmony(
            register_adjusted, profile
        )
    else:
        voiced, revoiced_harmony_notes = register_adjusted, 0
    expanded, generated_notes = _expand_sparse_windows(voiced, profile)
    limited, cleanup = _collapse_and_limit(expanded, profile)
    arranged, legato_extended = _shape_legato(limited, profile)
    if not arranged:
        raise ValueError("The learned piano arranger did not retain any playable notes.")

    role_counts = Counter(note.get("arrangementRole", "harmony") for note in arranged)
    source_counts = Counter(note["instrument"] for note in source_notes)
    duration = max(note["time"] + note["duration"] for note in arranged)
    vocal_melody_notes = sum(
        1 for note in arranged
        if note.get("sourceInstrument") in VOICE_INSTRUMENTS
        and note.get("arrangementRole") == "melody"
    )
    result = dict(payload)
    result.update(
        {
            "notes": arranged,
            "instrument": "piano",
            "instrumentGroups": ["acoustic_piano"],
            "vocalMelodyIncluded": vocal_melody_notes > 0,
            "arrangementProfile": profile.get("id", "learned-pianella-style"),
        }
    )
    result["performance"] = {
        **(payload.get("performance") or {}),
        "profile": "polymath-learned-piano-arranger-v1",
        "arrangerProfile": profile.get("id", "learned-pianella-style"),
        "preserveScoreDurations": True,
        "defaultAutoplayReleaseSeconds": float(profile.get("style", {}).get("defaultAutoplayReleaseSeconds", 0.62)),
    }
    result["pianoArrangement"] = {
        "version": 4,
        "profile": "learned-full-mix-piano-reduction",
        "learnedProfileId": profile.get("id"),
        "learnedProfileSha256": profile.get("profileSha256"),
        "mode": mode,
        "sourceNoteCount": len(payload.get("notes", [])),
        "normalizedNonPercussiveSourceNoteCount": len(source_notes),
        "sourceInstrumentCounts": dict(sorted(source_counts.items())),
        "outputNoteCount": len(arranged),
        "outputNotesPerSecond": round_number(len(arranged) / max(1.0, duration), 3),
        "roleCounts": dict(sorted(role_counts.items())),
        "vocalMelodyNotes": vocal_melody_notes,
        "generatedAccompanimentNotes": generated_notes,
        "compactlyRevoicedHarmonyNotes": revoiced_harmony_notes,
        "learnedRegister": register_diagnostics,
        "legatoExtendedNotes": legato_extended,
        **cleanup,
        "routingContract": "instrument-aware-transcription-then-piano-only-arrangement",
        "conditionalSelectionBlend": conditional_selection,
    }
    result["transcriptionCleanup"] = {
        **(payload.get("transcriptionCleanup") or {}),
        "outputNotes": len(arranged),
        "arrangerProfile": profile.get("id"),
    }
    return result


__all__ = [
    "CONTEXT_SELECTION_FEATURE_NAMES",
    "ROBUST_CONTEXT_SELECTION_FEATURE_NAMES",
    "LEFT_HAND_SELECTION_FEATURE_NAMES",
    "HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES",
    "ONSET_GESTURE_SELECTION_FEATURE_NAMES",
    "ONSET_GESTURE_SEQUENCE_FEATURE_NAMES",
    "DURATION_FEATURE_NAMES",
    "FEATURE_NAMES",
    "arrangement_role",
    "apply_learned_register_model",
    "arrange_with_profile",
    "duration_feature_rows",
    "duration_predictions",
    "instrument_family",
    "harmonic_left_hand_selection_feature_rows",
    "left_hand_selection_feature_rows",
    "normalize_source_notes",
    "onset_gesture_selection_feature_rows",
    "onset_gesture_sequence_feature_rows",
    "raw_feature_rows",
    "selection_feature_rows",
    "selection_scores",
    "contextual_selection_scores",
]
