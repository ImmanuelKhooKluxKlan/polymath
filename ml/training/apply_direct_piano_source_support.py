"""Prune a direct piano decode using simultaneous full-mix pitch evidence.

The piano-constrained pass is rhythmically strong but may emit oversized chord
clusters.  A separately decoded unconditioned pass supplies factual pitch-class
support.  This operator never reads the answer sheet: it removes unsupported
pitch classes, collapses redundant octave layers, and always preserves at least
one attack so rhythm is not erased.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ml.training.apply_direct_piano_melody_register import onset_groups


def _time(note: dict[str, Any]) -> float:
    return float(note.get("time", note.get("startTime", note.get("start", 0.0))))


def _midi(note: dict[str, Any]) -> int:
    return int(round(float(note.get("midi", note.get("pitch", 0)))))


def apply_source_support(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    radius_seconds: float = 0.12,
    onset_window_seconds: float = 0.035,
    maximum_notes_per_gesture: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_notes = candidate.get("notes")
    source_notes = source.get("notes")
    if not isinstance(candidate_notes, list) or not isinstance(source_notes, list):
        raise ValueError("Candidate and source must both contain notes lists")
    if radius_seconds <= 0:
        raise ValueError("radius_seconds must be positive")
    if maximum_notes_per_gesture < 1:
        raise ValueError("maximum_notes_per_gesture must be positive")

    output = copy.deepcopy(candidate)
    notes = output["notes"]
    normalized_source = sorted(
        [note for note in source_notes if isinstance(note, dict)], key=_time
    )
    keep_indices: set[int] = set()
    unsupported_removed = octave_layers_removed = quota_removed = fallback_groups = 0

    for group in onset_groups(notes, onset_window_seconds):
        anchor = min(_time(notes[index]) for index in group)
        local = [
            note for note in normalized_source if abs(_time(note) - anchor) <= radius_seconds
        ]
        source_by_pc: dict[int, list[int]] = {}
        for note in local:
            source_by_pc.setdefault(_midi(note) % 12, []).append(_midi(note))

        supported = [index for index in group if _midi(notes[index]) % 12 in source_by_pc]
        unsupported_removed += len(group) - len(supported)
        if not supported:
            # Keep the top voice so an uncertain source pass cannot erase a
            # valid rhythmic attack from the constrained pass.
            keep_indices.add(max(group, key=lambda index: _midi(notes[index])))
            fallback_groups += 1
            continue

        best_by_pc: dict[int, int] = {}
        for index in supported:
            midi = _midi(notes[index])
            pitch_class = midi % 12
            previous = best_by_pc.get(pitch_class)
            if previous is None:
                best_by_pc[pitch_class] = index
                continue
            source_midis = source_by_pc[pitch_class]
            old_distance = min(abs(_midi(notes[previous]) - value) for value in source_midis)
            new_distance = min(abs(midi - value) for value in source_midis)
            if (new_distance, -midi) < (old_distance, -_midi(notes[previous])):
                best_by_pc[pitch_class] = index
        octave_layers_removed += len(supported) - len(best_by_pc)
        ranked = sorted(
            best_by_pc.values(),
            key=lambda index: (
                min(
                    abs(_midi(notes[index]) - value)
                    for value in source_by_pc[_midi(notes[index]) % 12]
                ),
                -int(index == max(group, key=lambda item: _midi(notes[item]))),
                -_midi(notes[index]),
            ),
        )
        chosen = ranked[:maximum_notes_per_gesture]
        quota_removed += max(0, len(ranked) - len(chosen))
        keep_indices.update(chosen)

    output["notes"] = [
        note for index, note in enumerate(notes) if index in keep_indices
    ]
    diagnostics = {
        "schema": "polymath-direct-piano-source-support-v1",
        "inputNotes": len(notes),
        "outputNotes": len(output["notes"]),
        "gestures": len(onset_groups(notes, onset_window_seconds)),
        "radiusSeconds": radius_seconds,
        "onsetWindowSeconds": onset_window_seconds,
        "maximumNotesPerGesture": maximum_notes_per_gesture,
        "unsupportedNotesRemoved": unsupported_removed,
        "redundantOctaveLayersRemoved": octave_layers_removed,
        "quotaNotesRemoved": quota_removed,
        "fallbackGroups": fallback_groups,
        "warning": (
            "Research-only dual-pass pruning. It uses no answer sheet at inference "
            "but doubles transcription work unless the passes are batched."
        ),
    }
    output.setdefault("diagnostics", {})["directPianoSourceSupport"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--radius-seconds", type=float, default=0.12)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--maximum-notes-per-gesture", type=int, default=4)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8-sig"))
    source = json.loads(Path(args.source).read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_source_support(
        candidate,
        source,
        radius_seconds=args.radius_seconds,
        onset_window_seconds=args.onset_window_seconds,
        maximum_notes_per_gesture=args.maximum_notes_per_gesture,
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
