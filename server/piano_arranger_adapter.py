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
import math
from collections import Counter, defaultdict
from typing import Any, Iterable


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


def selection_scores(notes: list[dict[str, Any]], profile: dict[str, Any]) -> list[float]:
    model = profile.get("selectionModel") or {}
    weights = [float(value) for value in model.get("weights", [])]
    means = [float(value) for value in model.get("means", [])]
    scales = [max(1e-9, float(value)) for value in model.get("scales", [])]
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError("Piano arranger profile has incompatible selection weights.")
    if len(means) != len(weights) or len(scales) != len(weights):
        raise ValueError("Piano arranger profile has incompatible feature scaling.")
    probabilities: list[float] = []
    for row in raw_feature_rows(notes):
        logit = sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(weights, row, means, scales)
        )
        logit = clamp(logit, -30.0, 30.0)
        probabilities.append(1.0 / (1.0 + math.exp(-logit)))
    return probabilities


def map_octave_to_range(midi: int, minimum: int, maximum: int) -> int:
    value = int(round(midi))
    while value < minimum:
        value += 12
    while value > maximum:
        value -= 12
    return int(clamp(value, minimum, maximum))


def _role_config(profile: dict[str, Any], role: str) -> dict[str, Any]:
    defaults = {
        "melody": {"minimumMidi": 55, "maximumMidi": 88, "durationScale": 1.0, "medianDuration": 0.35, "velocity": 0.86},
        "bass": {"minimumMidi": 28, "maximumMidi": 55, "durationScale": 1.0, "medianDuration": 0.55, "velocity": 0.68},
        "harmony": {"minimumMidi": 45, "maximumMidi": 79, "durationScale": 1.0, "medianDuration": 0.45, "velocity": 0.62},
    }
    return {**defaults[role], **((profile.get("roles") or {}).get(role) or {})}


def _render_note(source: dict[str, Any], probability: float, profile: dict[str, Any]) -> dict[str, Any]:
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
    duration = clamp(
        scaled_duration * source_duration_weight
        + median_duration * (1.0 - source_duration_weight),
        MIN_NOTE_SECONDS,
        MAX_NOTE_SECONDS,
    )
    learned_velocity = float(config["velocity"])
    velocity = clamp(learned_velocity * 0.72 + float(source["velocity"]) * 0.28, 0.05, 1.0)
    return {
        **source,
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
        if len(chosen) < min(quota, len(ranked)):
            chosen_set = set(chosen)
            chosen.extend(index for index in ranked if index not in chosen_set)
            chosen = chosen[:quota]

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
    for pitch_notes in by_pitch.values():
        kept: list[dict[str, Any]] = []
        for note in pitch_notes:
            if kept and note["time"] - kept[-1]["time"] < duplicate_seconds:
                previous = kept[-1]
                previous_end = previous["time"] + previous["duration"]
                note_end = note["time"] + note["duration"]
                preferred = previous if previous["selectionProbability"] >= note["selectionProbability"] else note
                preferred = dict(preferred)
                preferred["time"] = min(previous["time"], note["time"])
                preferred["duration"] = round_number(max(previous_end, note_end) - preferred["time"])
                kept[-1] = preferred
                duplicates += 1
                continue
            if kept and note["time"] - kept[-1]["time"] < retrigger_seconds:
                previous = kept[-1]
                previous["duration"] = round_number(
                    clamp(max(previous["time"] + previous["duration"], note["time"] + note["duration"]) - previous["time"], MIN_NOTE_SECONDS, MAX_NOTE_SECONDS)
                )
                retriggers += 1
                continue
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
        "mergedRapidRetriggers": retriggers,
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
    scores = selection_scores(source_notes, profile)
    selected = _pick_window_notes(source_notes, scores, profile, mode)
    rendered = [_render_note(note, probability, profile) for note, probability in selected]
    expanded, generated_notes = _expand_sparse_windows(rendered, profile)
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
        "legatoExtendedNotes": legato_extended,
        **cleanup,
        "routingContract": "instrument-aware-transcription-then-piano-only-arrangement",
    }
    result["transcriptionCleanup"] = {
        **(payload.get("transcriptionCleanup") or {}),
        "outputNotes": len(arranged),
        "arrangerProfile": profile.get("id"),
    }
    return result


__all__ = [
    "FEATURE_NAMES",
    "arrangement_role",
    "arrange_with_profile",
    "instrument_family",
    "normalize_source_notes",
    "raw_feature_rows",
    "selection_scores",
]
