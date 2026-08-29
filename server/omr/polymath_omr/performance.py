"""Turn literal score events into physically plausible piano performance data.

The written duration, the time a pianist keeps a key depressed, and the time a
string continues sounding under the damper pedal are three different things.
Keeping those concepts separate prevents the clipped, typewriter-like playback
that results from treating every note value as a hard audio stop.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import floor


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _articulation_name(note: dict) -> str:
    return str(note.get("articulation", "") or "").strip().lower().replace("-", " ")


def _hold_ratio(articulation: str) -> float:
    if "staccatissimo" in articulation:
        return 0.34
    if "staccato" in articulation:
        return 0.48
    if "tenuto" in articulation:
        return 0.98
    if "legato" in articulation or "slur" in articulation:
        return 1.035
    if "accent" in articulation or "marcato" in articulation:
        return 0.84
    return 0.92


def _release_seconds(articulation: str, score_duration: float) -> float:
    if "staccatissimo" in articulation:
        return 0.24
    if "staccato" in articulation:
        return 0.30
    if "accent" in articulation or "marcato" in articulation:
        return 0.40
    return _clamp(0.42 + score_duration * 0.14, 0.44, 0.72)


def _voice_key(note: dict) -> str:
    return str(note.get("voice") or note.get("hand") or "piano")


def _shape_key_holds(notes: list[dict]) -> dict:
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for note in notes:
        by_voice[_voice_key(note)].append(note)

    shortened_for_restrike = 0
    legato_connections = 0
    for voice_notes in by_voice.values():
        voice_notes.sort(key=lambda item: (float(item.get("time", 0)), str(item.get("note", ""))))
        onset_times = sorted({float(note.get("time", 0)) for note in voice_notes})
        next_onset = {
            onset: onset_times[index + 1] if index + 1 < len(onset_times) else None
            for index, onset in enumerate(onset_times)
        }
        next_same_pitch: dict[int, float] = {}
        for index, note in enumerate(voice_notes):
            start = float(note.get("time", 0))
            same = next((
                float(candidate.get("time", 0))
                for candidate in voice_notes[index + 1:]
                if candidate.get("note") == note.get("note")
                and float(candidate.get("time", 0)) > start + 0.0001
            ), None)
            if same is not None:
                next_same_pitch[id(note)] = same

        for note in voice_notes:
            start = float(note.get("time", 0))
            score_duration = _clamp(float(
                note.get("scoreDuration", note.get("duration", 0.45)) or 0.45
            ), 0.035, 32.0)
            articulation = _articulation_name(note)
            if "marcato" in articulation:
                note["velocity"] = _rounded(_clamp(float(note.get("velocity", 0.76)) * 1.14, 0.04, 1.0), 4)
            elif "accent" in articulation:
                note["velocity"] = _rounded(_clamp(float(note.get("velocity", 0.76)) * 1.08, 0.04, 1.0), 4)
            key_hold = score_duration * _hold_ratio(articulation)
            following = next_onset.get(start)

            if following is not None:
                if "legato" in articulation or "slur" in articulation:
                    key_hold = min(max(key_hold, following - start + 0.032), following - start + 0.045)
                    legato_connections += 1
                else:
                    # A real hand normally leaves a tiny breath before the next
                    # attack. The sample's release tail supplies continuity.
                    key_hold = min(key_hold, max(0.055, following - start - 0.018))

            restrike = next_same_pitch.get(id(note))
            if restrike is not None and start + key_hold > restrike - 0.038:
                key_hold = max(0.055, restrike - start - 0.038)
                shortened_for_restrike += 1

            note["duration"] = _rounded(score_duration)
            note["scoreDuration"] = _rounded(score_duration)
            note["visualDuration"] = _rounded(score_duration)
            note["audioDuration"] = _rounded(_clamp(key_hold, 0.055, 16.0))
            note["releaseSeconds"] = _rounded(_release_seconds(articulation, score_duration), 4)

    # Printed voices may cross staves or flip stem direction, but a piano still
    # has only one physical mechanism for each pitch. Enforce the release gap
    # again across the complete keyboard, independent of notation ownership.
    by_pitch: dict[str, list[dict]] = defaultdict(list)
    for note in notes:
        by_pitch[str(note.get("note", ""))].append(note)
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda item: float(item.get("time", 0)))
        for previous, following in zip(pitch_notes, pitch_notes[1:]):
            previous_start = float(previous.get("time", 0))
            following_start = float(following.get("time", 0))
            if following_start <= previous_start + 0.0001:
                continue
            latest_hold = max(0.055, following_start - previous_start - 0.038)
            if float(previous.get("audioDuration", 0)) > latest_hold:
                previous["audioDuration"] = _rounded(latest_hold)
                shortened_for_restrike += 1

    return {
        "voices": len(by_voice),
        "restrikesGivenReleaseGap": shortened_for_restrike,
        "legatoConnections": legato_connections,
    }


def _onset_groups(notes: list[dict], tolerance: float = 0.035) -> list[dict]:
    groups: list[dict] = []
    for note in sorted(notes, key=lambda item: (float(item.get("time", 0)), str(item.get("note", "")))):
        start = float(note.get("time", 0))
        if not groups or start - groups[-1]["time"] > tolerance:
            groups.append({"time": start, "notes": [note]})
        else:
            groups[-1]["notes"].append(note)
    return groups


def _bass_signature(group: dict) -> tuple[int | None, frozenset[int]]:
    midi_values = [int(note.get("midi", -1)) for note in group["notes"] if int(note.get("midi", -1)) >= 0]
    if not midi_values:
        return None, frozenset()
    return min(midi_values) % 12, frozenset(midi % 12 for midi in midi_values)


def infer_piano_pedals(notes: list[dict], bpm: float, time_signature: dict | None = None) -> list[dict]:
    """Create conservative syncopated pedaling when the score prints none.

    This is deliberately marked as inferred. It re-pedals at measure starts and
    at strong half-measure harmony changes, never on every note. New harmony is
    struck first, the old pedal is lifted just afterwards, and the pedal returns
    after the dampers have cleared—the motion used by an actual pianist.
    """

    if not notes:
        return []
    safe_bpm = _clamp(float(bpm or 100), 20.0, 300.0)
    beat_seconds = 60.0 / safe_bpm
    signature = time_signature if isinstance(time_signature, dict) else {}
    numerator = max(1, int(signature.get("numerator", 4) or 4))
    denominator = max(1, int(signature.get("denominator", 4) or 4))
    measure_seconds = beat_seconds * numerator * 4.0 / denominator
    groups = _onset_groups(notes)
    first_time = groups[0]["time"]
    last_end = max(
        float(note.get("time", 0)) + float(note.get("scoreDuration", note.get("duration", 0.4)) or 0.4)
        for note in notes
    )
    first_measure = floor(first_time / measure_seconds)
    last_measure = floor(max(first_time, last_end - 0.0001) / measure_seconds)
    boundaries: list[float] = []

    for measure_index in range(first_measure, last_measure + 1):
        measure_start = measure_index * measure_seconds
        measure_end = measure_start + measure_seconds
        inside = [group for group in groups if measure_start - 0.02 <= group["time"] < measure_end - 0.02]
        if not inside:
            continue
        articulations = [_articulation_name(note) for group in inside for note in group["notes"]]
        detached = sum("staccato" in articulation for articulation in articulations)
        if articulations and detached / len(articulations) >= 0.45:
            continue

        first_group = inside[0]
        boundaries.append(first_group["time"])
        halfway = measure_start + measure_seconds / 2.0
        later = [
            group for group in inside
            if group["time"] >= halfway - beat_seconds * 0.30
            and group["time"] - first_group["time"] >= beat_seconds * 0.70
            and (len(group["notes"]) >= 2 or any(note.get("hand") == "left" for note in group["notes"]))
        ]
        if later:
            first_bass, first_pitch_classes = _bass_signature(first_group)
            candidate = later[0]
            next_bass, next_pitch_classes = _bass_signature(candidate)
            changed_bass = first_bass is not None and next_bass is not None and first_bass != next_bass
            shared = len(first_pitch_classes & next_pitch_classes)
            union = len(first_pitch_classes | next_pitch_classes)
            harmonic_similarity = shared / max(1, union)
            if changed_bass or harmonic_similarity < 0.34:
                boundaries.append(candidate["time"])

    compact: list[float] = []
    for boundary in sorted(set(round(value, 4) for value in boundaries)):
        if not compact or boundary - compact[-1] >= beat_seconds * 0.55:
            compact.append(boundary)
    if not compact:
        return []

    events = [{
        "id": "omr-pedal-down-0",
        "time": _rounded(compact[0] + min(0.050, beat_seconds * 0.10), 4),
        "down": True,
        "value": 96,
        "controller": 64,
        "source": "inferred-score-pedaling",
        "inferred": True,
        "confidence": 0.58,
    }]
    for index, boundary in enumerate(compact[1:], 1):
        events.extend([{
            "id": f"omr-pedal-up-{index}",
            "time": _rounded(boundary + min(0.025, beat_seconds * 0.05), 4),
            "down": False,
            "value": 0,
            "controller": 64,
            "source": "inferred-score-pedaling",
            "inferred": True,
            "confidence": 0.58,
        }, {
            "id": f"omr-pedal-down-{index}",
            "time": _rounded(boundary + min(0.065, beat_seconds * 0.13), 4),
            "down": True,
            "value": 96,
            "controller": 64,
            "source": "inferred-score-pedaling",
            "inferred": True,
            "confidence": 0.58,
        }])
    events.append({
        "id": "omr-pedal-final-up",
        "time": _rounded(last_end + min(0.14, beat_seconds * 0.28), 4),
        "down": False,
        "value": 0,
        "controller": 64,
        "source": "inferred-score-pedaling",
        "inferred": True,
        "confidence": 0.58,
    })
    return events


def shape_piano_performance(payload: dict, *, infer_pedal: bool) -> dict:
    if payload.get("instrument") != "piano" or not payload.get("notes"):
        return payload
    result = deepcopy(payload)
    notes = result["notes"]
    hold_diagnostics = _shape_key_holds(notes)
    explicit_pedals = list(result.get("pedals") or [])
    if explicit_pedals:
        pedals = explicit_pedals
        pedal_source = "printed-score"
    elif infer_pedal:
        pedals = infer_piano_pedals(notes, result.get("bpm", 100), result.get("timeSignature"))
        pedal_source = "inferred-score-pedaling" if pedals else "none"
    else:
        pedals = []
        pedal_source = "none"
    result["pedals"] = pedals
    result["pedalEvents"] = pedals
    result["performance"] = {
        **(result.get("performance") or {}),
        "profile": "polymath-score-pianist-v1",
        "preserveScoreDurations": True,
        "preserveScoreTiming": True,
        "durationFieldPolicy": "written-key-hold-damper-v1",
        "sameKeyRetriggerGapSeconds": 0.038,
        "defaultAutoplayReleaseSeconds": 0.58,
    }
    result["pianoPerformance"] = {
        **hold_diagnostics,
        "pedalSource": pedal_source,
        "pedalEvents": len(pedals),
        "writtenAndPhysicalDurationsSeparated": True,
    }
    return result
