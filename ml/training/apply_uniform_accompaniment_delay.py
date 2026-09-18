"""Apply one reversible timing offset to non-melody accompaniment notes.

This is an offline research control.  Pitch, velocity, duration, melody, and
pedal data remain frozen so an onset-delay hypothesis can be measured without
silently changing any other musical variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import instrument_family  # noqa: E402


MAXIMUM_ABSOLUTE_DELAY_SECONDS = 0.08


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def is_accompaniment(note: dict[str, Any]) -> bool:
    if str(note.get("arrangementRole") or "").strip().lower() == "melody":
        return False
    if instrument_family(str(note.get("sourceInstrument") or "")) == "voice":
        return False
    hand = str(note.get("hand") or "").strip().lower()
    if hand:
        return hand == "left"
    try:
        return int(round(float(note.get("midi", 999)))) < 60
    except (TypeError, ValueError):
        return False


def apply_uniform_accompaniment_delay(
    payload: dict[str, Any], delay_seconds: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    delay = float(delay_seconds)
    if abs(delay) > MAXIMUM_ABSOLUTE_DELAY_SECONDS:
        raise ValueError(
            f"delay_seconds must be between -{MAXIMUM_ABSOLUTE_DELAY_SECONDS} "
            f"and {MAXIMUM_ABSOLUTE_DELAY_SECONDS}"
        )
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
    changed = 0
    for note in notes:
        if not isinstance(note, dict) or not is_accompaniment(note):
            continue
        original = note.get(
            "originalTimeBeforeUniformAccompanimentDelay", note.get("time")
        )
        try:
            original_time = float(original)
        except (TypeError, ValueError):
            continue
        note["originalTimeBeforeUniformAccompanimentDelay"] = round(
            original_time, 6
        )
        note["time"] = round(max(0.0, original_time + delay), 6)
        note["uniformAccompanimentDelaySeconds"] = round(delay, 6)
        changed += 1
    notes.sort(
        key=lambda note: (
            float(note.get("time", 0.0)) if isinstance(note, dict) else 0.0,
            int(note.get("midi", 0)) if isinstance(note, dict) else 0,
        )
    )
    diagnostics = {
        "schema": "polymath-uniform-accompaniment-delay-v1",
        "researchOnly": True,
        "requestedDelaySeconds": round(delay, 6),
        "changedNotes": changed,
        "melodyFrozen": True,
        "pitchesFrozen": True,
        "velocitiesFrozen": True,
        "durationsFrozen": True,
        "pedalsFrozen": True,
    }
    arrangement = payload.setdefault("pianoArrangement", {})
    if isinstance(arrangement, dict):
        arrangement["uniformAccompanimentDelay"] = diagnostics
    return payload, diagnostics


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, diagnostics = apply_uniform_accompaniment_delay(
        load(args.candidate.resolve()), args.delay_seconds
    )
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
