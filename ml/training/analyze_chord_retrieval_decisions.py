"""Audit whether each cross-song chord retrieval helped or hurt.

The report compares a frozen baseline and a retrieval candidate at exactly the
same onset.  It then bins the pitch-class F1 change by raw-mixture density,
retrieval distance, retrieval gain and incumbent confidence.  This separates
"the library knows a familiar chord" from "the uploaded audio supports
replacing this already-good chord".
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

from ml.training.analyze_pianist_gesture_patterns import (
    inside_ranges,
    map_reference,
    normalize_notes,
    pitch_set,
    set_f1,
    group_onsets,
)
from ml.training.evaluate_piano_arranger import (
    monotonic_anchors,
    trusted_source_ranges,
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def nearest_group(
    groups: list[list[dict[str, Any]]],
    time: float,
    tolerance: float,
) -> list[dict[str, Any]] | None:
    choices = [
        group
        for group in groups
        if abs(float(group[0]["time"]) - time) <= tolerance
    ]
    return min(
        choices,
        key=lambda group: abs(float(group[0]["time"]) - time),
        default=None,
    )


def reference_group_for_change(
    groups: list[list[dict[str, Any]]],
    time: float,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]] | None:
    choices = [
        group
        for group in groups
        if abs(float(group[0]["time"]) - time) <= tolerance
    ]
    baseline_pc = pitch_set(baseline, True)
    candidate_pc = pitch_set(candidate, True)
    return min(
        choices,
        key=lambda group: (
            -max(
                set_f1(pitch_set(group, True), baseline_pc),
                set_f1(pitch_set(group, True), candidate_pc),
            ),
            abs(float(group[0]["time"]) - time),
        ),
        default=None,
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "decisions": 0,
            "improved": 0,
            "unchanged": 0,
            "worsened": 0,
            "meanPitchClassF1Delta": 0.0,
            "netImprovement": 0.0,
        }
    deltas = [float(record["pitchClassF1Delta"]) for record in records]
    return {
        "decisions": len(records),
        "improved": sum(delta > 1e-9 for delta in deltas),
        "unchanged": sum(abs(delta) <= 1e-9 for delta in deltas),
        "worsened": sum(delta < -1e-9 for delta in deltas),
        "meanPitchClassF1Delta": round(sum(deltas) / len(deltas), 6),
        "netImprovement": round(sum(deltas), 6),
        "positiveDecisionRate": round(
            sum(delta > 1e-9 for delta in deltas) / len(deltas), 6
        ),
    }


def grouped_rows(
    records: list[dict[str, Any]],
    name: str,
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return [
        {"name": value, **summarize(rows)}
        for value, rows in sorted(groups.items(), key=lambda item: item[0])
    ]


def density_band(value: int) -> str:
    if value <= 4:
        return "00-04"
    if value <= 8:
        return "05-08"
    if value <= 12:
        return "09-12"
    if value <= 20:
        return "13-20"
    return "21+"


def distance_band(value: float) -> str:
    if value <= 0.05:
        return "00-05"
    if value <= 0.10:
        return "05-10"
    if value <= 0.20:
        return "10-20"
    return "20+"


def gain_band(value: float) -> str:
    if value < 0.05:
        return "00-05"
    if value < 0.10:
        return "05-10"
    if value < 0.20:
        return "10-20"
    return "20+"


def confidence_band(value: float) -> str:
    if value < 0.65:
        return "<065"
    if value < 0.75:
        return "065-075"
    if value < 0.85:
        return "075-085"
    return "085+"


def analyze_song(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_path = Path(str(row["reference"])).resolve()
    source_path = Path(str(row["source"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    baseline_path = Path(str(row["baseline"])).resolve()
    candidate_path = Path(str(row["candidate"])).resolve()
    alignment = load(alignment_path)
    ranges = trusted_source_ranges(alignment)
    candidate_end = row.get("candidateEndSeconds")
    candidate_end = float(candidate_end) if candidate_end is not None else None
    reference = [
        note
        for note in map_reference(
            normalize_notes(
                load(reference_path),
                transpose=int(row.get("referenceTransposeSemitones", 0)),
                hard_end=(
                    float(row["referenceEndSeconds"])
                    if row.get("referenceEndSeconds") is not None
                    else None
                ),
            ),
            monotonic_anchors(alignment),
        )
        if inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    baseline = [
        note
        for note in normalize_notes(load(baseline_path), hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    candidate = [
        note
        for note in normalize_notes(load(candidate_path), hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    source = [
        note
        for note in normalize_notes(load(source_path), hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    reference_groups = group_onsets(reference, 0.035)
    baseline_groups = group_onsets(baseline, 0.035)
    candidate_groups = group_onsets(candidate, 0.035)
    source_times = [float(note["time"]) for note in source]
    source_duration = max(
        (float(note["time"]) + float(note["duration"]) for note in source),
        default=1.0,
    )
    records: list[dict[str, Any]] = []
    for observed in candidate_groups:
        changed = [
            note
            for note in observed
            if note.get("chordGestureOriginalMidi") is not None
            or note.get("chordGestureAddedNote")
        ]
        if not changed:
            continue
        time = float(observed[0]["time"])
        incumbent = nearest_group(baseline_groups, time, 0.01)
        if incumbent is None:
            continue
        target = reference_group_for_change(
            reference_groups, time, incumbent, observed, 0.25
        )
        if target is None:
            continue
        left = bisect.bisect_left(source_times, time - 0.12)
        right = bisect.bisect_right(source_times, time + 0.12)
        local_source = source[left:right]
        baseline_f1 = set_f1(pitch_set(target, True), pitch_set(incumbent, True))
        candidate_f1 = set_f1(pitch_set(target, True), pitch_set(observed, True))
        distances = [float(note.get("chordGestureNearestDistance", 9.0)) for note in changed]
        gains = [float(note.get("chordGestureRetrievalGain", 0.0)) for note in changed]
        incumbent_probabilities = [
            float(note.get("selectionProbability", 0.5)) for note in incumbent
        ]
        records.append(
            {
                "songId": str(row["id"]),
                "time": round(time, 6),
                "baselinePitchClasses": sorted(pitch_set(incumbent, True)),
                "candidatePitchClasses": sorted(pitch_set(observed, True)),
                "referencePitchClasses": sorted(pitch_set(target, True)),
                "baselinePitchClassF1": round(baseline_f1, 6),
                "candidatePitchClassF1": round(candidate_f1, 6),
                "pitchClassF1Delta": round(candidate_f1 - baseline_f1, 6),
                "localRawNotes240ms": len(local_source),
                "localRawPitchClasses240ms": len(
                    {int(note["midi"]) % 12 for note in local_source}
                ),
                "nearestPrototypeDistance": round(median(distances), 7),
                "retrievalGain": round(median(gains), 6),
                "incumbentSelectionProbability": round(
                    median(incumbent_probabilities), 6
                ),
                "incumbentChordSize": len(pitch_set(incumbent, True)),
                "sourceNotesPerSecond": round(len(source) / max(1.0, source_duration), 6),
            }
        )
    return records, {
        "id": str(row["id"]),
        "reference": str(reference_path),
        "source": str(source_path),
        "alignment": str(alignment_path),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "changedDecisionsWithReference": len(records),
        **summarize(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load(manifest_path)
    records: list[dict[str, Any]] = []
    songs = []
    for row in manifest.get("songs") or []:
        values, summary = analyze_song(row)
        records.extend(values)
        songs.append(summary)
    report = {
        "schema": "polymath-chord-retrieval-decision-audit-v1",
        "manifest": str(manifest_path),
        "summary": summarize(records),
        "songs": songs,
        "byLocalRawDensity": grouped_rows(
            records, "density", lambda row: density_band(int(row["localRawNotes240ms"]))
        ),
        "byLocalRawPitchClasses": grouped_rows(
            records,
            "pitchClasses",
            lambda row: density_band(int(row["localRawPitchClasses240ms"])),
        ),
        "byNeighborDistance": grouped_rows(
            records,
            "distance",
            lambda row: distance_band(float(row["nearestPrototypeDistance"])),
        ),
        "byRetrievalGain": grouped_rows(
            records, "gain", lambda row: gain_band(float(row["retrievalGain"]))
        ),
        "byIncumbentConfidence": grouped_rows(
            records,
            "confidence",
            lambda row: confidence_band(float(row["incumbentSelectionProbability"])),
        ),
        "byIncumbentChordSize": grouped_rows(
            records, "size", lambda row: str(int(row["incumbentChordSize"]))
        ),
        "decisions": records,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
