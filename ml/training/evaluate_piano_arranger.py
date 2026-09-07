"""Evaluate a route-specific piano arranger against a fixed musical time warp.

The fixed warp is important: refitting a fresh alignment for every candidate can
hide timing regressions.  Each approved target is mapped once onto the original
song timeline using the pre-existing alignment anchors, then both incumbent and
candidate are measured against exactly the same reference coordinates.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_notes(payload: dict[str, Any]) -> list[dict[str, float | int]]:
    notes: list[dict[str, float | int]] = []
    for item in payload.get("notes", []):
        try:
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        if 0 <= midi <= 127 and time >= 0 and math.isfinite(time) and math.isfinite(duration):
            notes.append({"midi": midi, "time": time, "duration": duration})
    return sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))


def monotonic_anchors(report: dict[str, Any]) -> list[tuple[float, float]]:
    ordered: list[tuple[float, float]] = []
    for item in sorted(report.get("anchors", []), key=lambda anchor: float(anchor["referenceTime"])):
        try:
            reference_time = float(item["referenceTime"])
            observed_time = float(item["observedTime"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(reference_time) or not math.isfinite(observed_time):
            continue
        if ordered and reference_time <= ordered[-1][0]:
            continue
        if ordered and observed_time <= ordered[-1][1]:
            continue
        ordered.append((reference_time, observed_time))
    if len(ordered) < 2:
        coarse = report.get("coarse") or {}
        scale = float(coarse.get("scale", report.get("metrics", {}).get("coarseScale", 1.0)))
        offset = float(coarse.get("offset", report.get("metrics", {}).get("coarseOffsetSeconds", 0.0)))
        ordered = [(0.0, offset), (1.0, scale + offset)]
    return ordered


def map_time(value: float, anchors: list[tuple[float, float]]) -> float:
    references = [anchor[0] for anchor in anchors]
    index = bisect.bisect_right(references, value)
    if index <= 0:
        left, right = anchors[0], anchors[1]
    elif index >= len(anchors):
        left, right = anchors[-2], anchors[-1]
    else:
        left, right = anchors[index - 1], anchors[index]
    scale = (right[1] - left[1]) / max(1e-9, right[0] - left[0])
    return left[1] + (value - left[0]) * scale


def map_reference_notes(notes: list[dict[str, float | int]], anchors: list[tuple[float, float]]) -> list[dict[str, float | int]]:
    mapped: list[dict[str, float | int]] = []
    for note in notes:
        start = map_time(float(note["time"]), anchors)
        end = map_time(float(note["time"]) + float(note["duration"]), anchors)
        mapped.append(
            {
                "midi": int(note["midi"]),
                "time": max(0.0, start),
                "duration": max(0.01, end - start),
            }
        )
    return sorted(mapped, key=lambda note: (float(note["time"]), int(note["midi"])))


def greedy_matches(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    tolerance: float,
    *,
    octave_equivalent: bool = False,
) -> list[tuple[dict[str, float | int], dict[str, float | int]]]:
    grouped: dict[int, list[tuple[int, dict[str, float | int]]]] = defaultdict(list)
    for index, note in enumerate(observed):
        key = int(note["midi"]) % 12 if octave_equivalent else int(note["midi"])
        grouped[key].append((index, note))
    times = {key: [float(note["time"]) for _index, note in values] for key, values in grouped.items()}
    used: set[int] = set()
    matches: list[tuple[dict[str, float | int], dict[str, float | int]]] = []
    for target in reference:
        key = int(target["midi"]) % 12 if octave_equivalent else int(target["midi"])
        pool = grouped.get(key, [])
        if not pool:
            continue
        target_time = float(target["time"])
        position = bisect.bisect_left(times[key], target_time)
        candidates: list[tuple[float, int, dict[str, float | int]]] = []
        left = position - 1
        while left >= 0 and target_time - times[key][left] <= tolerance:
            index, candidate = pool[left]
            if index not in used:
                candidates.append((abs(float(candidate["time"]) - target_time), index, candidate))
            left -= 1
        right = position
        while right < len(pool) and times[key][right] - target_time <= tolerance:
            index, candidate = pool[right]
            if index not in used:
                candidates.append((abs(float(candidate["time"]) - target_time), index, candidate))
            right += 1
        if not candidates:
            continue
        _distance, observed_index, best = min(candidates, key=lambda item: (item[0], abs(int(item[2]["midi"]) - int(target["midi"]))))
        used.add(observed_index)
        matches.append((target, best))
    return matches


def f1_metrics(reference_count: int, observed_count: int, match_count: int) -> dict[str, float | int]:
    precision = match_count / max(1, observed_count)
    recall = match_count / max(1, reference_count)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "matches": match_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def rapid_retriggers(notes: list[dict[str, float | int]], threshold: float = 0.10) -> int:
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(float(note["time"]))
    return sum(
        1
        for times in by_pitch.values()
        for previous, current in zip(times, times[1:])
        if current - previous < threshold
    )


def evaluate(reference: list[dict[str, float | int]], observed: list[dict[str, float | int]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "referenceNotes": len(reference),
        "observedNotes": len(observed),
        "noteCountRatio": round(len(observed) / max(1, len(reference)), 6),
        "rapidRetriggersUnder100ms": rapid_retriggers(observed),
    }
    for milliseconds in (50, 100, 250):
        exact = greedy_matches(reference, observed, milliseconds / 1000.0)
        octave = greedy_matches(reference, observed, milliseconds / 1000.0, octave_equivalent=True)
        metrics[f"exactPitchOnset{milliseconds}ms"] = f1_metrics(len(reference), len(observed), len(exact))
        metrics[f"pitchClassOnset{milliseconds}ms"] = f1_metrics(len(reference), len(observed), len(octave))

    duration_matches = greedy_matches(reference, observed, 0.25)
    duration_errors = [
        abs(float(target["duration"]) - float(candidate["duration"]))
        for target, candidate in duration_matches
    ]
    relative_errors = [
        abs(float(target["duration"]) - float(candidate["duration"])) / max(0.05, float(target["duration"]))
        for target, candidate in duration_matches
    ]
    offset_matches = sum(
        1
        for target, candidate in duration_matches
        if abs(
            (float(target["time"]) + float(target["duration"]))
            - (float(candidate["time"]) + float(candidate["duration"]))
        ) <= 0.25
    )
    cutoffs = sum(
        1
        for target, candidate in duration_matches
        if float(candidate["duration"]) < float(target["duration"]) * 0.50
    )
    metrics["duration"] = {
        "matchedNotes": len(duration_matches),
        "medianAbsoluteErrorSeconds": round(median(duration_errors), 6) if duration_errors else None,
        "medianRelativeError": round(median(relative_errors), 6) if relative_errors else None,
        "onsetAndOffset250ms": f1_metrics(len(reference), len(observed), offset_matches),
        "severeCutoffs": cutoffs,
    }
    return metrics


def weighted_average(rows: list[tuple[float, float]]) -> float:
    total = sum(weight for _value, weight in rows)
    return sum(value * weight for value, weight in rows) / max(1e-9, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest).resolve())
    baseline_dir = Path(args.baseline_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    results: dict[str, Any] = {}
    aggregate_values: dict[str, dict[str, list[tuple[float, float]]]] = {
        "baseline": defaultdict(list),
        "candidate": defaultdict(list),
    }
    trusted_regressions: list[dict[str, Any]] = []

    for pair in manifest.get("pairs", []):
        pair_id = str(pair["id"])
        weight = float(pair.get("weight", 1.0))
        target = normalize_notes(load_json(Path(pair["target"])))
        anchors = monotonic_anchors(load_json(Path(pair["alignmentReport"])))
        mapped_target = map_reference_notes(target, anchors)
        baseline = normalize_notes(load_json(baseline_dir / f"{pair_id}-arranged.json"))
        candidate = normalize_notes(load_json(candidate_dir / f"{pair_id}.json"))
        baseline_metrics = evaluate(mapped_target, baseline)
        candidate_metrics = evaluate(mapped_target, candidate)
        results[pair_id] = {
            "weight": weight,
            "fixedAnchorCount": len(anchors),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
        for name, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
            for tolerance in (50, 100, 250):
                aggregate_values[name][f"exactF1_{tolerance}ms"].append(
                    (float(metrics[f"exactPitchOnset{tolerance}ms"]["f1"]), weight)
                )
                aggregate_values[name][f"pitchClassF1_{tolerance}ms"].append(
                    (float(metrics[f"pitchClassOnset{tolerance}ms"]["f1"]), weight)
                )
            aggregate_values[name]["durationMedianAbsoluteErrorSeconds"].append(
                (float(metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0), weight)
            )
            aggregate_values[name]["severeCutoffs"].append((float(metrics["duration"]["severeCutoffs"]), weight))
            aggregate_values[name]["rapidRetriggersUnder100ms"].append((float(metrics["rapidRetriggersUnder100ms"]), weight))
        if weight >= 0.75:
            baseline_f1 = float(baseline_metrics["exactPitchOnset100ms"]["f1"])
            candidate_f1 = float(candidate_metrics["exactPitchOnset100ms"]["f1"])
            if candidate_f1 < baseline_f1 - 0.01:
                trusted_regressions.append(
                    {"song": pair_id, "baselineExactF1_100ms": baseline_f1, "candidateExactF1_100ms": candidate_f1}
                )

    aggregate = {
        name: {
            metric: round(weighted_average(values), 6)
            for metric, values in metrics.items()
        }
        for name, metrics in aggregate_values.items()
    }
    deltas = {
        metric: round(aggregate["candidate"][metric] - aggregate["baseline"][metric], 6)
        for metric in aggregate["baseline"]
    }
    baseline = aggregate["baseline"]
    candidate = aggregate["candidate"]
    gates = {
        "exactF1_100ms_improves": candidate["exactF1_100ms"] >= baseline["exactF1_100ms"] + 0.005,
        "exactF1_250ms_does_not_regress": candidate["exactF1_250ms"] >= baseline["exactF1_250ms"] - 0.002,
        "pitchClassF1_250ms_does_not_regress": candidate["pitchClassF1_250ms"] >= baseline["pitchClassF1_250ms"] - 0.002,
        "no_trusted_song_regresses_over_1_point": not trusted_regressions,
        "duration_error_not_over_10_percent_worse": candidate["durationMedianAbsoluteErrorSeconds"] <= baseline["durationMedianAbsoluteErrorSeconds"] * 1.10,
        "severe_cutoffs_not_over_5_percent_worse": candidate["severeCutoffs"] <= baseline["severeCutoffs"] * 1.05 + 1.0,
        "rapid_retriggers_not_over_5_percent_worse": candidate["rapidRetriggersUnder100ms"] <= baseline["rapidRetriggersUnder100ms"] * 1.05 + 1.0,
    }
    decision = "PROMOTE" if all(gates.values()) else "REJECT"
    output = {
        "schema": "polymath-piano-arranger-evaluation-v1",
        "fixedWarpPolicy": "approved target-to-original anchors are frozen before candidate evaluation",
        "baselineDirectory": str(baseline_dir),
        "candidateDirectory": str(candidate_dir),
        "songs": results,
        "weightedAggregate": aggregate,
        "candidateMinusBaseline": deltas,
        "promotionGates": gates,
        "trustedSongRegressions": trusted_regressions,
        "decision": decision,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "weightedAggregate": aggregate, "candidateMinusBaseline": deltas, "promotionGates": gates}, indent=2))


if __name__ == "__main__":
    main()
