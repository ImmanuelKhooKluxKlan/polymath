"""Apply one reversible timing offset to an already-arranged melody role.

This research utility deliberately freezes pitch, velocity, duration, harmony,
and chord-mapper decisions.  It exists so timing can be evaluated as one
controlled variable instead of re-running a context-sensitive mapper after the
melody moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAXIMUM_ABSOLUTE_DELAY_SECONDS = 0.08


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def apply_uniform_melody_delay(
    payload: dict[str, Any], delay_seconds: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    delay = float(delay_seconds)
    if abs(delay) > MAXIMUM_ABSOLUTE_DELAY_SECONDS:
        raise ValueError(
            f"delay_seconds must be between -{MAXIMUM_ABSOLUTE_DELAY_SECONDS} "
            f"and {MAXIMUM_ABSOLUTE_DELAY_SECONDS}"
        )

    arrangement = payload.get("pianoArrangement")
    adaptive = arrangement.get("adaptiveMelodyRegisterSeparation") if isinstance(arrangement, dict) else None
    gate_applied = isinstance(adaptive, dict) and bool(adaptive.get("applied"))
    changed = 0
    eligible = 0
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []

    if gate_applied:
        for note in notes:
            if not isinstance(note, dict):
                continue
            if note.get("arrangementRole") != "melody":
                continue
            if not note.get("adaptiveMelodyRegisterShiftSemitones"):
                continue
            eligible += 1
            original = note.get("originalTimeBeforeAdaptiveMelodyDelay", note.get("time"))
            try:
                original_time = float(original)
            except (TypeError, ValueError):
                continue
            note["originalTimeBeforeAdaptiveMelodyDelay"] = round(original_time, 6)
            note["time"] = round(max(0.0, original_time + delay), 6)
            note["adaptiveMelodyOnsetDelaySeconds"] = round(delay, 6)
            changed += 1

        notes.sort(
            key=lambda note: (
                float(note.get("time", 0.0)) if isinstance(note, dict) else 0.0,
                int(note.get("midi", 0)) if isinstance(note, dict) else 0,
            )
        )

    diagnostics = {
        "schema": "polymath-uniform-melody-delay-v1",
        "researchOnly": True,
        "gateApplied": gate_applied,
        "requestedDelaySeconds": round(delay, 6),
        "eligibleNotes": eligible,
        "changedNotes": changed,
        "pitchesFrozen": True,
        "velocitiesFrozen": True,
        "durationsFrozen": True,
        "harmonyFrozen": True,
    }
    if isinstance(arrangement, dict):
        arrangement["uniformMelodyDelay"] = diagnostics
        if isinstance(adaptive, dict):
            adaptive["requestedOnsetDelaySeconds"] = round(delay, 6)
            adaptive["appliedOnsetDelaySeconds"] = round(delay, 6) if gate_applied else 0.0
    return payload, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--delay-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    output_path = Path(args.output).resolve()
    payload, diagnostics = apply_uniform_melody_delay(load(candidate_path), args.delay_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
