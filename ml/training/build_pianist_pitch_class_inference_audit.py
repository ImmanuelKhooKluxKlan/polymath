"""Build pitch-class decoder cells without opening a reference answer.

The historical analysis report includes desired pianist gestures because it
also creates training labels.  Runtime prediction does not use those fields.
This utility builds the identical candidate/source feature cells with an empty
reference assignment, making the inference boundary explicit and auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analyze_pianist_texture_patterns import (
    PERCUSSION,
    cell_example,
    gesture_time,
    group_onsets,
    local_density,
    normalize_notes,
    pitch_set,
    set_f1,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_cells(
    candidate_payload: dict[str, Any],
    source_payload: dict[str, Any],
    *,
    song_id: str,
    onset_window: float = 0.035,
) -> list[dict[str, Any]]:
    candidate = normalize_notes(candidate_payload)
    source = [
        note
        for note in normalize_notes(source_payload, source_indices=True)
        if str(note.get("instrument") or "").lower() not in PERCUSSION
    ]
    candidate_groups = group_onsets(candidate, onset_window)
    times = [gesture_time(group) for group in candidate_groups]
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(candidate_groups):
        previous_gap = (
            times[index] - times[index - 1]
            if index
            else times[index + 1] - times[index]
            if index + 1 < len(times)
            else 0.3
        )
        next_gap = (
            times[index + 1] - times[index]
            if index + 1 < len(times)
            else times[index] - times[index - 1]
            if index
            else 0.3
        )
        next_time = times[index] + max(0.03, next_gap)
        interior_source = [
            note
            for note in source
            if times[index] + onset_window < float(note["time"])
            < next_time - onset_window
        ]
        rows.append(
            cell_example(
                song_id,
                index,
                group,
                [],
                source,
                max(0.03, next_gap),
                local_density(times, index),
                previous_gap=max(0.03, previous_gap),
                previous_pitch_class_similarity=(
                    set_f1(
                        pitch_set(group, True),
                        pitch_set(candidate_groups[index - 1], True),
                    )
                    if index
                    else 0.0
                ),
                next_pitch_class_similarity=(
                    set_f1(
                        pitch_set(group, True),
                        pitch_set(candidate_groups[index + 1], True),
                    )
                    if index + 1 < len(candidate_groups)
                    else 0.0
                ),
                interior_source=interior_source,
                previous_candidate_chord_size=(
                    len(pitch_set(candidate_groups[index - 1]))
                    if index
                    else len(pitch_set(group))
                ),
                next_candidate_chord_size=(
                    len(pitch_set(candidate_groups[index + 1]))
                    if index + 1 < len(candidate_groups)
                    else len(pitch_set(group))
                ),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    source_path = args.source.resolve()
    rows = build_cells(
        load_json(candidate_path),
        load_json(source_path),
        song_id=str(args.song),
        onset_window=max(0.005, float(args.onset_window_seconds)),
    )
    payload = {
        "schema": "polymath-pitch-class-inference-audit-v1",
        "inferenceOnly": True,
        "referenceOpened": False,
        "inputs": {
            "candidate": str(candidate_path),
            "source": str(source_path),
        },
        "songs": [
            {
                "id": str(args.song),
                "cells": rows,
                "counts": {"candidateCells": len(rows)},
            }
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
