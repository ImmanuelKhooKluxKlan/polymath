"""Compare piano-performance shape without pretending two timelines align.

This is intentionally different from note-accuracy evaluation.  It is useful
when an authored arrangement is shorter, faster, or structurally condensed
relative to the source video: note identities cannot be scored honestly, but
register, density, voicing, hold, dynamics, and retrigger style can still teach
the arranger how a pianist tends to reduce a full mix.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _number(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _quantile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * ratio
    left = int(math.floor(position))
    right = min(len(ordered) - 1, left + 1)
    progress = position - left
    return ordered[left] * (1.0 - progress) + ordered[right] * progress


def _notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in payload.get("notes", []):
        if not isinstance(item, dict):
            continue
        midi = int(round(_number(item.get("midi", item.get("pitch")), -1)))
        time = _number(item.get("time", item.get("startTime")), -1)
        duration = _number(item.get("duration"), 0.2)
        velocity = _number(item.get("velocity"), 0.72)
        if 21 <= midi <= 108 and time >= 0 and duration > 0:
            output.append(
                {
                    "midi": midi,
                    "time": time,
                    "duration": duration,
                    "velocity": max(0.0, min(1.0, velocity)),
                    "hand": str(item.get("hand") or "").lower(),
                    "sourceTrack": item.get("sourceTrack"),
                }
            )
    track_midis: dict[str, list[int]] = defaultdict(list)
    for note in output:
        if note["sourceTrack"] is not None:
            track_midis[str(note["sourceTrack"])].append(int(note["midi"]))
    track_hands: dict[str, str] = {}
    if len(track_midis) >= 2:
        ordered_tracks = sorted(
            track_midis,
            key=lambda track: median(track_midis[track]),
        )
        track_hands[ordered_tracks[0]] = "left"
        track_hands[ordered_tracks[-1]] = "right"
    for note in output:
        if note["hand"] not in {"left", "right"}:
            note["hand"] = track_hands.get(
                str(note["sourceTrack"]),
                "left" if int(note["midi"]) < 60 else "right",
            )
    return sorted(output, key=lambda note: (float(note["time"]), int(note["midi"])))


def _hand_summary(notes: list[dict[str, Any]], hand: str) -> dict[str, Any]:
    selected = [note for note in notes if note["hand"] == hand]
    if not selected:
        return {
            "notes": 0,
            "share": 0.0,
            "medianMidi": 0.0,
            "p10Midi": 0.0,
            "p90Midi": 0.0,
            "medianHoldSeconds": 0.0,
            "p90HoldSeconds": 0.0,
        }
    midis = [float(note["midi"]) for note in selected]
    durations = [float(note["duration"]) for note in selected]
    return {
        "notes": len(selected),
        "share": round(len(selected) / len(notes), 6),
        "minimumMidi": int(min(midis)),
        "p10Midi": round(_quantile(midis, 0.10), 3),
        "medianMidi": round(median(midis), 3),
        "p90Midi": round(_quantile(midis, 0.90), 3),
        "maximumMidi": int(max(midis)),
        "medianHoldSeconds": round(median(durations), 6),
        "p90HoldSeconds": round(_quantile(durations, 0.90), 6),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    notes = _notes(payload)
    if not notes:
        raise ValueError("Piano JSON contains no playable notes")
    end = max(float(note["time"]) + float(note["duration"]) for note in notes)
    duration = max(0.001, end - min(float(note["time"]) for note in notes))
    onset_groups: list[list[dict[str, float | int]]] = []
    for note in notes:
        if (
            not onset_groups
            or float(note["time"]) - float(onset_groups[-1][0]["time"]) > 0.035
        ):
            onset_groups.append([note])
        else:
            onset_groups[-1].append(note)

    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(float(note["time"]))
    same_pitch_gaps = [
        current - previous
        for times in by_pitch.values()
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    fast_repeats = sum(0.065 <= gap < 0.18 for gap in same_pitch_gaps)
    unresolved_duplicates = sum(gap < 0.065 for gap in same_pitch_gaps)
    midis = [float(note["midi"]) for note in notes]
    durations = [float(note["duration"]) for note in notes]
    velocities = [float(note["velocity"]) for note in notes]
    return {
        "notes": len(notes),
        "durationSeconds": round(duration, 6),
        "notesPerSecond": round(len(notes) / duration, 6),
        "uniqueOnsetsPerSecond": round(len(onset_groups) / duration, 6),
        "notesPerOnset": round(len(notes) / max(1, len(onset_groups)), 6),
        "maximumOnsetCluster": max(len(group) for group in onset_groups),
        "multiNoteOnsetShare": round(
            sum(len(group) >= 2 for group in onset_groups) / max(1, len(onset_groups)),
            6,
        ),
        "register": {
            "minimumMidi": int(min(midis)),
            "p10Midi": round(_quantile(midis, 0.10), 3),
            "medianMidi": round(median(midis), 3),
            "p90Midi": round(_quantile(midis, 0.90), 3),
            "maximumMidi": int(max(midis)),
            "notesBelowC3Share": round(sum(midi < 48 for midi in midis) / len(midis), 6),
            "notesAboveA5Share": round(sum(midi > 81 for midi in midis) / len(midis), 6),
        },
        "writtenDurationSeconds": {
            "p10": round(_quantile(durations, 0.10), 6),
            "median": round(median(durations), 6),
            "p90": round(_quantile(durations, 0.90), 6),
        },
        "velocity": {
            "p10": round(_quantile(velocities, 0.10), 6),
            "median": round(median(velocities), 6),
            "p90": round(_quantile(velocities, 0.90), 6),
        },
        "sameKeyRhythm": {
            "fastMusicalRetriggers65To180ms": fast_repeats,
            "fastMusicalRetriggersPerSecond": round(fast_repeats / duration, 6),
            "unresolvedDuplicatesUnder65ms": unresolved_duplicates,
        },
        "hands": {
            "left": _hand_summary(notes, "left"),
            "right": _hand_summary(notes, "right"),
        },
    }


def _relative_distance(candidate: float, reference: float) -> float:
    if candidate <= 0 or reference <= 0:
        return 0.0 if candidate == reference else 2.0
    return min(2.0, abs(math.log(candidate / reference)))


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref = summarize(reference)
    cand = summarize(candidate)
    components = {
        "noteDensity": _relative_distance(cand["notesPerSecond"], ref["notesPerSecond"]),
        "onsetDensity": _relative_distance(
            cand["uniqueOnsetsPerSecond"], ref["uniqueOnsetsPerSecond"]
        ),
        "voicingThickness": _relative_distance(
            cand["notesPerOnset"], ref["notesPerOnset"]
        ),
        "medianRegister": abs(
            cand["register"]["medianMidi"] - ref["register"]["medianMidi"]
        )
        / 12.0,
        "registerEdges": (
            abs(cand["register"]["p10Midi"] - ref["register"]["p10Midi"])
            + abs(cand["register"]["p90Midi"] - ref["register"]["p90Midi"])
        )
        / 24.0,
        "upperHandShare": abs(
            cand["hands"]["right"]["share"] - ref["hands"]["right"]["share"]
        )
        / 0.25,
        "lowerMedianRegister": abs(
            cand["hands"]["left"]["medianMidi"]
            - ref["hands"]["left"]["medianMidi"]
        )
        / 12.0,
        "upperMedianRegister": abs(
            cand["hands"]["right"]["medianMidi"]
            - ref["hands"]["right"]["medianMidi"]
        )
        / 12.0,
        "lowerRegisterEdges": (
            abs(cand["hands"]["left"]["p10Midi"] - ref["hands"]["left"]["p10Midi"])
            + abs(cand["hands"]["left"]["p90Midi"] - ref["hands"]["left"]["p90Midi"])
        )
        / 24.0,
        "upperRegisterEdges": (
            abs(cand["hands"]["right"]["p10Midi"] - ref["hands"]["right"]["p10Midi"])
            + abs(cand["hands"]["right"]["p90Midi"] - ref["hands"]["right"]["p90Midi"])
        )
        / 24.0,
        "medianHold": _relative_distance(
            cand["writtenDurationSeconds"]["median"],
            ref["writtenDurationSeconds"]["median"],
        ),
        "longHold": _relative_distance(
            cand["writtenDurationSeconds"]["p90"],
            ref["writtenDurationSeconds"]["p90"],
        ),
        "lowerLongHold": _relative_distance(
            cand["hands"]["left"]["p90HoldSeconds"],
            ref["hands"]["left"]["p90HoldSeconds"],
        ),
        "upperLongHold": _relative_distance(
            cand["hands"]["right"]["p90HoldSeconds"],
            ref["hands"]["right"]["p90HoldSeconds"],
        ),
        "fastRepeatRate": _relative_distance(
            cand["sameKeyRhythm"]["fastMusicalRetriggersPerSecond"],
            ref["sameKeyRhythm"]["fastMusicalRetriggersPerSecond"],
        ),
        "unresolvedDuplicatePenalty": min(
            2.0,
            cand["sameKeyRhythm"]["unresolvedDuplicatesUnder65ms"]
            / max(1.0, cand["durationSeconds"] * 0.05),
        ),
    }
    weights = {
        "noteDensity": 0.10,
        "onsetDensity": 0.08,
        "voicingThickness": 0.06,
        "medianRegister": 0.04,
        "registerEdges": 0.03,
        "upperHandShare": 0.11,
        "lowerMedianRegister": 0.08,
        "upperMedianRegister": 0.08,
        "lowerRegisterEdges": 0.04,
        "upperRegisterEdges": 0.04,
        "medianHold": 0.07,
        "longHold": 0.05,
        "lowerLongHold": 0.07,
        "upperLongHold": 0.05,
        "fastRepeatRate": 0.07,
        "unresolvedDuplicatePenalty": 0.03,
    }
    distance = sum(components[name] * weights[name] for name in weights)
    return {
        "schema": "polymath-piano-style-comparison-v1",
        "warning": (
            "Structural-style similarity is not note accuracy. It is valid for a "
            "shorter or differently edited reference, but cannot prove that pitches "
            "or onsets match the source video."
        ),
        "reference": ref,
        "candidate": cand,
        "distanceComponents": {
            name: round(value, 6) for name, value in components.items()
        },
        "structuralStyleDistance": round(distance, 6),
        "structuralStyleSimilarityPercent": round(100.0 * math.exp(-distance), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    result = compare(
        json.loads(reference_path.read_text(encoding="utf-8-sig")),
        json.loads(candidate_path.read_text(encoding="utf-8-sig")),
    )
    result["referencePath"] = str(reference_path)
    result["candidatePath"] = str(candidate_path)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
