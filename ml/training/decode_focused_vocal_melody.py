"""Decode a focused voice transcription into one playable melody line.

The focused MuScriptor pass is useful evidence, but its output is not yet a
score: sustained syllables are often emitted as a short note every model
frame, and accompaniment leakage can appear as four to seven simultaneous
``voice`` pitches.  Sending those events directly to the piano arranger makes
the keyboard chatter.

This module performs a deterministic, inference-only decode.  It does not
alter model weights and it does not invent timing from a reference score.  It
uses only the two transcriptions produced from the customer's own audio:

* sparse voice events in the broad pass act as high-confidence anchors;
* all pitched broad-pass events provide weaker corroboration;
* a Viterbi path chooses one continuous pitch from each credible voice frame;
* adjacent same-pitch frames are joined into one held piano note.

The function is deliberately standard-library only so the RunPod worker can
use it without adding another runtime dependency.
"""

from __future__ import annotations

import bisect
import copy
import math
from collections import defaultdict
from typing import Any, Iterable


PERCUSSION_INSTRUMENTS = {"drums", "timpani", "percussion"}


def _finite_number(value: Any, fallback: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _valid_note(note: Any) -> bool:
    if not isinstance(note, dict):
        return False
    midi = _finite_number(note.get("midi", note.get("pitch")))
    onset = _finite_number(note.get("time", note.get("startTime", note.get("start"))))
    duration = _finite_number(note.get("duration"), 0.2)
    return bool(
        midi is not None
        and onset is not None
        and duration is not None
        and 0 <= round(midi) <= 127
        and onset >= 0
        and duration > 0
    )


def _instrument(note: dict[str, Any]) -> str:
    return str(note.get("instrument") or "").strip().lower()


def _midi(note: dict[str, Any]) -> int:
    return int(round(float(note.get("midi", note.get("pitch")))))


def _onset(note: dict[str, Any]) -> float:
    return float(note.get("time", note.get("startTime", note.get("start"))))


def _quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 60.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_index(
    notes: Iterable[dict[str, Any]], *, pitch_class: bool = False
) -> dict[int, list[float]]:
    index: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        key = _midi(note) % 12 if pitch_class else _midi(note)
        index[key].append(_onset(note))
    for times in index.values():
        times.sort()
    return dict(index)


def _near(index: dict[int, list[float]], key: int, onset: float, tolerance: float) -> bool:
    times = index.get(key, [])
    if not times:
        return False
    position = bisect.bisect_left(times, onset)
    return any(
        0 <= candidate < len(times)
        and abs(times[candidate] - onset) <= tolerance
        for candidate in (position - 1, position)
    )


def _cluster_onsets(
    notes: list[dict[str, Any]], window_seconds: float
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    anchor = -math.inf
    for note in sorted(notes, key=lambda item: (_onset(item), _midi(item))):
        onset = _onset(note)
        if not clusters or onset - anchor > window_seconds:
            clusters.append([note])
            anchor = onset
        else:
            clusters[-1].append(note)
    return clusters


def _bounded_config(config: dict[str, Any] | None) -> dict[str, float | int]:
    raw = config if isinstance(config, dict) else {}

    def number(name: str, fallback: float, minimum: float, maximum: float) -> float:
        value = _finite_number(raw.get(name), fallback)
        return max(minimum, min(maximum, float(fallback if value is None else value)))

    return {
        "onset_window_seconds": number("onset_window_seconds", 0.035, 0.005, 0.10),
        "voice_anchor_tolerance_seconds": number(
            "voice_anchor_tolerance_seconds", 0.10, 0.02, 0.30
        ),
        "pitched_support_tolerance_seconds": number(
            "pitched_support_tolerance_seconds", 0.06, 0.02, 0.20
        ),
        "phrase_break_seconds": number("phrase_break_seconds", 0.75, 0.20, 3.0),
        "bridge_seconds": number("bridge_seconds", 0.45, 0.05, 1.0),
        "merge_gap_seconds": number("merge_gap_seconds", 0.06, 0.0, 0.20),
        "range_padding_semitones": number("range_padding_semitones", 5.0, 0.0, 18.0),
        "range_penalty_per_semitone": number(
            "range_penalty_per_semitone", 0.18, 0.0, 1.0
        ),
        "jump_penalty_per_semitone": number(
            "jump_penalty_per_semitone", 0.08, 0.0, 0.5
        ),
        "large_leap_threshold_semitones": number(
            "large_leap_threshold_semitones", 7.0, 1.0, 24.0
        ),
        "large_leap_penalty_per_semitone": number(
            "large_leap_penalty_per_semitone", 0.16, 0.0, 1.0
        ),
        "same_pitch_bonus": number("same_pitch_bonus", 0.24, 0.0, 2.0),
        "exact_voice_anchor_bonus": number("exact_voice_anchor_bonus", 2.5, 0.0, 10.0),
        "pitch_class_voice_anchor_bonus": number(
            "pitch_class_voice_anchor_bonus", 0.45, 0.0, 5.0
        ),
        "pitched_support_bonus": number("pitched_support_bonus", 0.25, 0.0, 3.0),
        "upper_candidate_preference": number(
            "upper_candidate_preference", 0.0, -1.0, 1.0
        ),
        "maximum_unanchored_polyphony": int(
            round(number("maximum_unanchored_polyphony", 3, 1, 12))
        ),
        "minimum_input_notes_per_second": number(
            "minimum_input_notes_per_second", 0.0, 0.0, 20.0
        ),
    }


def _credible_frames(
    frames: list[list[dict[str, Any]]],
    *,
    voice_exact: dict[int, list[float]],
    settings: dict[str, float | int],
) -> tuple[list[list[dict[str, Any]]], int, int]:
    """Reject chord-like leakage while allowing short continuity bridges."""

    tolerance = float(settings["voice_anchor_tolerance_seconds"])
    maximum_polyphony = int(settings["maximum_unanchored_polyphony"])
    core: list[bool] = []
    anchored: list[bool] = []
    for frame in frames:
        onset = min(_onset(note) for note in frame)
        has_anchor = any(
            _near(voice_exact, _midi(note), onset, tolerance) for note in frame
        )
        anchored.append(has_anchor)
        core.append(has_anchor or len(frame) <= maximum_polyphony)

    bridge_seconds = float(settings["bridge_seconds"])
    accepted = list(core)
    for index, frame in enumerate(frames):
        if core[index]:
            continue
        previous = next((i for i in range(index - 1, -1, -1) if core[i]), None)
        following = next((i for i in range(index + 1, len(frames)) if core[i]), None)
        if previous is None or following is None:
            continue
        onset = min(_onset(note) for note in frame)
        previous_onset = min(_onset(note) for note in frames[previous])
        following_onset = min(_onset(note) for note in frames[following])
        if onset - previous_onset > bridge_seconds or following_onset - onset > bridge_seconds:
            continue
        pitches = {_midi(note) for note in frame}
        previous_pitches = {_midi(note) for note in frames[previous]}
        following_pitches = {_midi(note) for note in frames[following]}
        if any(
            min(abs(pitch - other) for other in previous_pitches) <= 5
            and min(abs(pitch - other) for other in following_pitches) <= 5
            for pitch in pitches
        ):
            accepted[index] = True

    kept = [frame for frame, keep in zip(frames, accepted) if keep]
    return kept, sum(anchored), len(frames) - len(kept)


def _decode_phrase(
    frames: list[list[dict[str, Any]]],
    *,
    voice_exact: dict[int, list[float]],
    voice_pitch_class: dict[int, list[float]],
    pitched_exact: dict[int, list[float]],
    range_low: float,
    range_high: float,
    range_center: float,
    settings: dict[str, float | int],
) -> list[dict[str, Any]]:
    if not frames:
        return []

    scores: list[list[float]] = []
    backpointers: list[list[int]] = []
    voice_tolerance = float(settings["voice_anchor_tolerance_seconds"])
    pitched_tolerance = float(settings["pitched_support_tolerance_seconds"])
    leap_threshold = float(settings["large_leap_threshold_semitones"])

    for frame_index, frame in enumerate(frames):
        ordered = sorted(frame, key=lambda note: (_midi(note), -float(note.get("duration", 0.2))))
        frame[:] = ordered
        row: list[float] = []
        pointers: list[int] = []
        for candidate_index, note in enumerate(frame):
            midi = _midi(note)
            onset = _onset(note)
            rank = candidate_index / max(1, len(frame) - 1)
            outside_range = max(0.0, range_low - midi, midi - range_high)
            emission = -float(settings["range_penalty_per_semitone"]) * outside_range
            emission += float(settings["upper_candidate_preference"]) * rank
            emission += 0.04 * min(2.0, float(note.get("duration", 0.2)))
            if _near(voice_exact, midi, onset, voice_tolerance):
                emission += float(settings["exact_voice_anchor_bonus"])
            elif _near(voice_pitch_class, midi % 12, onset, voice_tolerance):
                emission += float(settings["pitch_class_voice_anchor_bonus"])
            if _near(pitched_exact, midi, onset, pitched_tolerance):
                emission += float(settings["pitched_support_bonus"])

            if frame_index == 0:
                row.append(emission - 0.02 * abs(midi - range_center))
                pointers.append(-1)
                continue

            transitions: list[tuple[float, int]] = []
            for previous_index, previous in enumerate(frames[frame_index - 1]):
                jump = abs(midi - _midi(previous))
                transition = -float(settings["jump_penalty_per_semitone"]) * jump
                transition -= float(settings["large_leap_penalty_per_semitone"]) * max(
                    0.0, jump - leap_threshold
                )
                if jump == 0:
                    transition += float(settings["same_pitch_bonus"])
                transitions.append((scores[-1][previous_index] + transition, previous_index))
            best_score, best_pointer = max(transitions, key=lambda item: (item[0], -item[1]))
            row.append(emission + best_score)
            pointers.append(best_pointer)
        scores.append(row)
        backpointers.append(pointers)

    final_index = max(
        range(len(frames[-1])), key=lambda index: (scores[-1][index], -index)
    )
    path: list[dict[str, Any]] = []
    for frame_index in range(len(frames) - 1, -1, -1):
        path.append(copy.deepcopy(frames[frame_index][final_index]))
        final_index = backpointers[frame_index][final_index]
    path.reverse()
    return path


def _merge_same_pitch_fragments(
    notes: list[dict[str, Any]], gap_seconds: float
) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    joins = 0
    for source in sorted(notes, key=lambda note: (_onset(note), _midi(note))):
        note = copy.deepcopy(source)
        if merged and _midi(merged[-1]) == _midi(note):
            previous_end = _onset(merged[-1]) + float(merged[-1].get("duration", 0.2))
            if _onset(note) <= previous_end + gap_seconds:
                merged[-1]["duration"] = round(
                    max(
                        float(merged[-1].get("duration", 0.2)),
                        _onset(note) + float(note.get("duration", 0.2)) - _onset(merged[-1]),
                    ),
                    4,
                )
                joins += 1
                continue
        merged.append(note)
    return merged, joins


def decode_focused_vocal_melody(
    focused_notes: Iterable[dict[str, Any]],
    primary_notes: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one deterministic vocal melody plus audit diagnostics."""

    settings = _bounded_config(config)
    focused = [copy.deepcopy(note) for note in focused_notes if _valid_note(note)]
    primary = [copy.deepcopy(note) for note in primary_notes if _valid_note(note)]
    if not focused:
        raise ValueError("The focused vocal melody decoder received no valid notes.")

    focused_span = max(
        (_onset(note) + float(note.get("duration", 0.2)) for note in focused),
        default=0.0,
    )
    input_density = len(focused) / max(1.0, focused_span)
    minimum_density = float(settings["minimum_input_notes_per_second"])
    if minimum_density > 0 and input_density < minimum_density:
        return focused, {
            "schema": "polymath-focused-vocal-melody-decoder-v1",
            "applied": False,
            "skipReason": "input-density-below-fragmentation-threshold",
            "inputNotes": len(focused),
            "inputDurationSeconds": round(focused_span, 4),
            "inputNotesPerSecond": round(input_density, 6),
            "minimumInputNotesPerSecond": round(minimum_density, 6),
            "outputNotes": len(focused),
            "settings": settings,
        }

    primary_voice = [note for note in primary if _instrument(note) == "voice"]
    pitched_primary = [
        note for note in primary if _instrument(note) not in PERCUSSION_INSTRUMENTS
    ]
    voice_pitches = [_midi(note) for note in primary_voice]
    focused_pitches = [_midi(note) for note in focused]
    range_source = voice_pitches if len(voice_pitches) >= 4 else focused_pitches
    padding = float(settings["range_padding_semitones"])
    range_low = _quantile(range_source, 0.05) - padding
    range_high = _quantile(range_source, 0.95) + padding
    range_center = _quantile(range_source, 0.50)

    voice_exact = _time_index(primary_voice)
    voice_pitch_class = _time_index(primary_voice, pitch_class=True)
    pitched_exact = _time_index(pitched_primary)
    all_frames = _cluster_onsets(focused, float(settings["onset_window_seconds"]))
    frames, anchored_frames, rejected_frames = _credible_frames(
        all_frames, voice_exact=voice_exact, settings=settings
    )
    if not frames:
        raise ValueError("The focused vocal melody decoder found no credible melody frames.")

    phrases: list[list[list[dict[str, Any]]]] = []
    phrase: list[list[dict[str, Any]]] = []
    previous_onset: float | None = None
    for frame in frames:
        onset = min(_onset(note) for note in frame)
        if (
            phrase
            and previous_onset is not None
            and onset - previous_onset > float(settings["phrase_break_seconds"])
        ):
            phrases.append(phrase)
            phrase = []
        phrase.append(frame)
        previous_onset = onset
    if phrase:
        phrases.append(phrase)

    path: list[dict[str, Any]] = []
    for current in phrases:
        path.extend(
            _decode_phrase(
                current,
                voice_exact=voice_exact,
                voice_pitch_class=voice_pitch_class,
                pitched_exact=pitched_exact,
                range_low=range_low,
                range_high=range_high,
                range_center=range_center,
                settings=settings,
            )
        )
    decoded, joined = _merge_same_pitch_fragments(
        path, float(settings["merge_gap_seconds"])
    )
    for note in decoded:
        note["instrument"] = "voice"
        note["focusedMelodyDecoded"] = True

    diagnostics = {
        "schema": "polymath-focused-vocal-melody-decoder-v1",
        "applied": True,
        "inputNotes": len(focused),
        "inputDurationSeconds": round(focused_span, 4),
        "inputNotesPerSecond": round(input_density, 6),
        "minimumInputNotesPerSecond": round(minimum_density, 6),
        "inputFrames": len(all_frames),
        "anchoredFrames": anchored_frames,
        "rejectedChordLikeFrames": rejected_frames,
        "credibleFrames": len(frames),
        "phrases": len(phrases),
        "pathNotesBeforeJoining": len(path),
        "samePitchFragmentsJoined": joined,
        "outputNotes": len(decoded),
        "estimatedVoiceRange": {
            "minimumMidi": round(range_low, 3),
            "centerMidi": round(range_center, 3),
            "maximumMidi": round(range_high, 3),
            "source": "primary-voice" if len(voice_pitches) >= 4 else "focused-fallback",
        },
        "settings": settings,
    }
    return decoded, diagnostics
