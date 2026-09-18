"""Apply an inference-safe melody-register probe to a direct piano score.

This operator is deliberately candidate-only: it never reads a reference or a
song identity.  It activates only when the decoded piano has almost no upper
register, then raises eligible top-of-gesture notes by one octave.  It is a
research probe until it survives whole-song holdouts and listening tests.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def onset_groups(notes: list[dict[str, Any]], window: float) -> list[list[int]]:
    ordered = sorted(
        range(len(notes)),
        key=lambda index: (
            float(notes[index].get("time", notes[index].get("startTime", 0.0))),
            int(notes[index].get("midi", notes[index].get("pitch", 0))),
        ),
    )
    groups: list[list[int]] = []
    for index in ordered:
        time = float(notes[index].get("time", notes[index].get("startTime", 0.0)))
        if not groups:
            groups.append([index])
            continue
        anchor = float(
            notes[groups[-1][0]].get("time", notes[groups[-1][0]].get("startTime", 0.0))
        )
        if time - anchor <= window:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def apply_melody_register(
    payload: dict[str, Any],
    *,
    minimum_midi: int = 64,
    shift_semitones: int = 12,
    onset_window_seconds: float = 0.035,
    activation_maximum_midi: int = 76,
    maximum_output_midi: int = 108,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_notes = payload.get("notes")
    if not isinstance(source_notes, list):
        raise ValueError("Input payload has no notes list")
    if shift_semitones not in (-12, 12):
        raise ValueError("shift_semitones must be -12 or 12")
    output = copy.deepcopy(payload)
    notes = output["notes"]
    midi_values = [
        int(note.get("midi", note.get("pitch", 0)))
        for note in notes
        if isinstance(note, dict)
    ]
    observed_maximum = max(midi_values, default=0)
    applied = observed_maximum <= activation_maximum_midi
    shifted = collisions = eligible_groups = 0
    if applied:
        for group in onset_groups(notes, onset_window_seconds):
            highest = max(
                int(notes[index].get("midi", notes[index].get("pitch", 0)))
                for index in group
            )
            if highest < minimum_midi:
                continue
            eligible_groups += 1
            existing = {
                int(notes[index].get("midi", notes[index].get("pitch", 0)))
                for index in group
            }
            for index in group:
                old_midi = int(notes[index].get("midi", notes[index].get("pitch", 0)))
                if old_midi != highest:
                    continue
                new_midi = old_midi + shift_semitones
                if not 0 <= new_midi <= maximum_output_midi or new_midi in existing:
                    collisions += 1
                    continue
                notes[index]["midi"] = new_midi
                if "pitch" in notes[index]:
                    notes[index]["pitch"] = new_midi
                notes[index]["note"] = note_name(new_midi)
                notes[index]["hand"] = "right" if new_midi >= 60 else "left"
                notes[index]["directPianoRegisterShift"] = shift_semitones
                shifted += 1

    diagnostics = {
        "schema": "polymath-direct-piano-melody-register-v1",
        "applied": applied,
        "reason": "low-upper-register-ceiling" if applied else "upper-register-already-present",
        "observedMaximumMidi": observed_maximum,
        "activationMaximumMidi": activation_maximum_midi,
        "minimumEligibleMidi": minimum_midi,
        "shiftSemitones": shift_semitones,
        "onsetWindowSeconds": onset_window_seconds,
        "eligibleGroups": eligible_groups,
        "shiftedNotes": shifted,
        "collisionSkips": collisions,
        "warning": (
            "Research-only candidate feature. It uses no reference at inference, "
            "but still requires unseen-song and listening gates."
        ),
    }
    output.setdefault("diagnostics", {})["directPianoMelodyRegister"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--minimum-midi", type=int, default=64)
    parser.add_argument("--shift-semitones", type=int, choices=(-12, 12), default=12)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--activation-maximum-midi", type=int, default=76)
    parser.add_argument("--maximum-output-midi", type=int, default=108)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_melody_register(
        payload,
        minimum_midi=args.minimum_midi,
        shift_semitones=args.shift_semitones,
        onset_window_seconds=args.onset_window_seconds,
        activation_maximum_midi=args.activation_maximum_midi,
        maximum_output_midi=args.maximum_output_midi,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
