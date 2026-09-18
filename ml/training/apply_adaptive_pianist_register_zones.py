"""Apply song-summary register zones without using title or answer-key data."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    """Return scientific pitch notation without importing research analyzers."""

    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def source_family(note: dict[str, Any]) -> str:
    value = str(note.get("sourceInstrument") or "").lower()
    if "flute" in value:
        return "flute"
    if "voice" in value or "vocal" in value or "choir" in value:
        return "voice"
    if "bass" in value or "contrabass" in value:
        return "bass"
    if "guitar" in value:
        return "guitar"
    if "piano" in value or "keyboard" in value:
        return "piano"
    return "other"


def role_summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    midis = [int(round(float(note["midi"]))) for note in notes]
    families = Counter(source_family(note) for note in notes)
    return {
        "notes": len(notes),
        "medianMidi": float(median(midis)) if midis else None,
        "sourceFamilies": dict(families),
        "dominantSourceFamily": families.most_common(1)[0][0] if families else None,
        "dominantSourceFamilyShare": (
            families.most_common(1)[0][1] / len(notes) if notes else 0.0
        ),
    }


def choose_shifts(
    payload: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, int], dict[str, Any]]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in payload.get("notes") or []:
        if not isinstance(note, dict) or note.get("midi") is None:
            continue
        by_role[str(note.get("arrangementRole") or "harmony")].append(note)
    summaries = {
        role: role_summary(by_role.get(role, []))
        for role in ("melody", "bass", "harmony")
    }
    melody = summaries["melody"]
    bass = summaries["bass"]
    harmony = summaries["harmony"]
    melody_config = profile.get("melody") or {}
    bass_config = profile.get("bass") or {}
    shifts = {"melody": 0, "bass": 0, "harmony": 0}
    reasons: dict[str, str] = {}
    blocked_reasons: dict[str, str] = {}
    conditional_minimum_midi: dict[str, int] = {}

    melody_count = int(melody["notes"])
    melody_median = melody["medianMidi"]
    flute_dominant = bool(
        melody_count
        and melody["dominantSourceFamily"] == "flute"
        and float(melody["dominantSourceFamilyShare"])
        >= float(melody_config.get("fluteDominanceThreshold", 0.5))
    )
    stable_low_melody = bool(
        melody_count >= int(melody_config.get("stableEventMinimum", 250))
        and melody_median is not None
        and float(melody_config.get("lowTessituraMinimumMidi", 66))
        <= float(melody_median)
        < float(melody_config.get("lowTessituraMaximumExclusiveMidi", 72))
    )
    harmony_count = int(harmony["notes"])
    harmony_median = harmony["medianMidi"]
    melody_harmony_gap = (
        float(melody_median) - float(harmony_median)
        if melody_median is not None and harmony_median is not None
        else None
    )
    narrow_melody_harmony_gap = bool(
        melody_count >= int(melody_config.get("separationMelodyEventMinimum", 10**9))
        and harmony_count
        >= int(melody_config.get("separationHarmonyEventMinimum", 10**9))
        and melody_median is not None
        and float(melody_config.get("separationMelodyMinimumMidi", 0))
        <= float(melody_median)
        < float(melody_config.get("separationMelodyMaximumExclusiveMidi", 128))
        and melody_harmony_gap is not None
        and float(melody_config.get("minimumCurrentSeparationSemitones", 0))
        <= melody_harmony_gap
        <= float(melody_config.get("maximumCurrentSeparationSemitones", -1))
    )
    blocked_melody_source_families = {
        str(value).strip().lower()
        for value in melody_config.get("blockedDominantSourceFamilies", [])
        if str(value).strip()
    }
    dominant_melody_source_family = str(
        melody.get("dominantSourceFamily") or ""
    ).lower()
    melody_source_blocked = bool(
        melody_count
        and dominant_melody_source_family in blocked_melody_source_families
        and float(melody["dominantSourceFamilyShare"])
        >= float(
            melody_config.get("blockedDominantSourceFamilyMinimumShare", 1.0)
        )
    )
    melody_trigger_reason = (
        "flute-derived melody"
        if flute_dominant
        else "stable low-tessitura melody"
        if stable_low_melody
        else "melody too close to harmony register"
        if narrow_melody_harmony_gap
        else None
    )
    if melody_trigger_reason:
        if melody_source_blocked:
            blocked_reasons["melody"] = (
                "dominant source family "
                f"{dominant_melody_source_family!r} is blocked for octave shifts"
            )
        else:
            shifts["melody"] = int(melody_config.get("shiftSemitones", 12))
            reasons["melody"] = melody_trigger_reason
    if flute_dominant and not melody_source_blocked and bool(
        melody_config.get("shiftHarmonyWithFluteMelody", False)
    ):
        shifts["harmony"] = int(melody_config.get("shiftSemitones", 12))
        reasons["harmony"] = "shared register zone with flute-derived melody"
        conditional_minimum_midi["harmony"] = int(
            melody_config.get("fluteHarmonyMinimumMidi", 60)
        )

    bass_count = int(bass["notes"])
    bass_median = bass["medianMidi"]
    stable_bass = bass_count >= int(bass_config.get("stableEventMinimum", 250))
    melody_absent_high_bass = bool(
        stable_bass
        and melody_count == 0
        and bass_median is not None
        and float(bass_median)
        >= float(bass_config.get("melodyAbsentMinimumMedianMidi", 53))
    )
    high_melody_low_bass = bool(
        stable_bass
        and melody_median is not None
        and bass_median is not None
        and float(melody_median)
        >= float(bass_config.get("highMelodyMinimumMedianMidi", 75))
        and float(bass_median)
        < float(bass_config.get("highMelodyLowBassMaximumExclusiveMidi", 40))
    )
    very_low_bass = bool(
        stable_bass
        and bass_median is not None
        and float(bass_median)
        <= float(bass_config.get("veryLowBassMaximumMidi", 36))
    )
    if melody_absent_high_bass or high_melody_low_bass or very_low_bass:
        shifts["bass"] = int(bass_config.get("shiftSemitones", 12))
        reasons["bass"] = (
            "melody-absent high bass-labelled arrangement"
            if melody_absent_high_bass
            else "high melody with very low bass zone"
            if high_melody_low_bass
            else "stable very-low bass zone"
        )
    return shifts, {
        "roles": summaries,
        "reasons": reasons,
        "blockedReasons": blocked_reasons,
        "melodyShiftBlockedBySourceFamily": melody_source_blocked,
        "melodyHarmonyMedianGapSemitones": melody_harmony_gap,
        "conditionalMinimumMidiByRole": conditional_minimum_midi,
    }


def fold_to_range(midi: int, minimum: int, maximum: int) -> int:
    while midi < minimum:
        midi += 12
    while midi > maximum:
        midi -= 12
    return max(minimum, min(maximum, midi))


def apply_register_zones(
    payload: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile.get("schema") != "polymath-adaptive-pianist-register-zones-v1":
        raise ValueError("Unsupported adaptive register-zone profile")
    shifts, evidence = choose_shifts(payload, profile)
    conditional_minimum_midi = evidence.get("conditionalMinimumMidiByRole") or {}
    output_range = profile.get("outputRange") or {}
    minimum = int(output_range.get("minimumMidi", 33))
    maximum = int(output_range.get("maximumMidi", 108))
    output = copy.deepcopy(payload)
    notes: list[dict[str, Any]] = []
    seen: set[tuple[float, int]] = set()
    shifted = collisions = 0
    shifted_by_role: Counter[str] = Counter()
    for note in output.get("notes") or []:
        if not isinstance(note, dict) or note.get("midi") is None:
            continue
        role = str(note.get("arrangementRole") or "harmony")
        shift = int(shifts.get(role, 0))
        original_midi = int(round(float(note["midi"])))
        if (
            role in conditional_minimum_midi
            and original_midi < int(conditional_minimum_midi[role])
        ):
            shift = 0
        midi = fold_to_range(original_midi + shift, minimum, maximum)
        time = float(note.get("time", note.get("startTime", note.get("start", 0.0))))
        key = (round(time, 4), midi)
        if key in seen:
            collisions += 1
            continue
        seen.add(key)
        updated = {
            **note,
            "midi": midi,
            "note": note_name(midi),
            "hand": "right" if midi >= 60 else "left",
        }
        if midi != original_midi:
            shifted += 1
            shifted_by_role[role] += 1
            updated["adaptiveRegisterOriginalMidi"] = original_midi
            updated["adaptiveRegisterShiftSemitones"] = midi - original_midi
            updated["adaptiveRegisterProfile"] = profile.get("id")
        notes.append(updated)
    output["notes"] = notes
    diagnostics = {
        "schema": "polymath-adaptive-pianist-register-zones-application-v1",
        "profileId": profile.get("id"),
        "shifts": shifts,
        **evidence,
        "shiftedNotes": shifted,
        "shiftedByRole": dict(shifted_by_role),
        "deduplicatedCollisions": collisions,
        "warning": (
            "Research-only development rule. It must pass a new untouched "
            "hard-song holdout and blind listening before promotion."
        ),
    }
    output.setdefault("diagnostics", {})["adaptivePianistRegisterZones"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    output, diagnostics = apply_register_zones(
        load_json(Path(args.input).resolve()),
        load_json(Path(args.profile).resolve()),
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
