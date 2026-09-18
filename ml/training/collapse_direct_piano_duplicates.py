"""Merge decoder-fragment duplicates in a direct piano transcription.

The route-specific checkpoint can emit the same pitch more than once at the
same decoder frame.  Those attacks sound like a machine gun and inflate note
density.  This cleanup is deliberately narrower than a minimum-retrigger rule:
only same-pitch attacks inside the established sub-65 ms duplicate window are
merged.  Musical repeats at 65 ms or later are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DUPLICATE_SECONDS = 0.065


def _number(note: dict[str, Any], *names: str, fallback: float = 0.0) -> float:
    for name in names:
        try:
            return float(note[name])
        except (KeyError, TypeError, ValueError):
            continue
    return fallback


def collapse_direct_piano_duplicates(
    notes: list[dict[str, Any]],
    *,
    duplicate_seconds: float = DEFAULT_DUPLICATE_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return cleaned notes plus an auditable duplicate-only report."""

    if not 0.035 <= duplicate_seconds <= 0.075:
        raise ValueError("duplicate_seconds must be between 0.035 and 0.075")
    by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for original in notes:
        if not isinstance(original, dict):
            continue
        try:
            midi = int(round(float(original.get("midi", original.get("pitch")))))
        except (TypeError, ValueError):
            continue
        note = copy.deepcopy(original)
        note["midi"] = midi
        by_pitch[midi].append(note)

    output: list[dict[str, Any]] = []
    removed = 0
    merged_groups = 0
    minimum_preserved_gap: float | None = None
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda note: _number(note, "time", "startTime", "start"))
        merged: list[dict[str, Any]] = []
        for note in pitch_notes:
            time = _number(note, "time", "startTime", "start")
            if not merged:
                merged.append(note)
                continue
            previous = merged[-1]
            previous_time = _number(previous, "time", "startTime", "start")
            gap = time - previous_time
            # Decimal timestamps such as 1.065 can subtract to
            # 0.06499999999999995.  Keep the documented boundary inclusive.
            if gap + 1e-9 >= duplicate_seconds:
                merged.append(note)
                if gap < 0.18:
                    minimum_preserved_gap = (
                        gap
                        if minimum_preserved_gap is None
                        else min(minimum_preserved_gap, gap)
                    )
                continue

            previous_duration = max(0.0, _number(previous, "duration"))
            current_duration = max(0.0, _number(note, "duration"))
            end = max(previous_time + previous_duration, time + current_duration)
            preferred = copy.deepcopy(
                max(
                    (previous, note),
                    key=lambda item: (
                        _number(item, "velocity", fallback=0.7),
                        _number(item, "duration"),
                    ),
                )
            )
            preferred["midi"] = int(previous["midi"])
            preferred["time"] = round(min(previous_time, time), 6)
            preferred["duration"] = round(max(0.01, end - preferred["time"]), 6)
            preferred["velocity"] = round(
                max(
                    _number(previous, "velocity", fallback=0.7),
                    _number(note, "velocity", fallback=0.7),
                ),
                6,
            )
            preferred["collapsedDirectPianoDuplicates"] = int(
                previous.get("collapsedDirectPianoDuplicates", 0)
            ) + 1
            merged[-1] = preferred
            removed += 1
            merged_groups += int(previous.get("collapsedDirectPianoDuplicates", 0)) == 0
        output.extend(merged)

    output.sort(
        key=lambda note: (
            _number(note, "time", "startTime", "start"),
            int(note["midi"]),
        )
    )
    report = {
        "schema": "polymath-direct-piano-duplicate-cleanup-v1",
        "inputNotes": len(notes),
        "outputNotes": len(output),
        "duplicatesRemoved": removed,
        "duplicateGroupsMerged": merged_groups,
        "duplicateWindowSeconds": duplicate_seconds,
        "preservedFastRepeatFloorSeconds": duplicate_seconds,
        "minimumPreservedFastGapSeconds": (
            None if minimum_preserved_gap is None else round(minimum_preserved_gap, 6)
        ),
        "policy": "Merge same-pitch decoder fragments below the window; preserve faster musical repeats at or above it.",
    }
    return output, report


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--duplicate-seconds", type=float, default=DEFAULT_DUPLICATE_SECONDS)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    notes, report = collapse_direct_piano_duplicates(
        payload.get("notes") or [],
        duplicate_seconds=args.duplicate_seconds,
    )
    result = copy.deepcopy(payload)
    result["notes"] = notes
    result.setdefault("diagnostics", {})["directPianoDuplicateCleanup"] = report
    atomic_json(args.output.resolve(), result)
    if args.report:
        atomic_json(args.report.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
