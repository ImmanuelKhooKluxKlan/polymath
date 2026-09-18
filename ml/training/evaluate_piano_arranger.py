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


GATE_EPSILON = 1e-12


def at_least(value: float, threshold: float) -> bool:
    """Compare frozen metrics without rejecting an exact decimal boundary."""

    return float(value) + GATE_EPSILON >= float(threshold)


def at_most(value: float, threshold: float) -> bool:
    """Compare frozen metrics without rejecting an exact decimal boundary."""

    return float(value) <= float(threshold) + GATE_EPSILON


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def optional_duration(item: dict[str, Any], field: str, fallback: float) -> float:
    try:
        value = float(item.get(field, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback
    return max(0.01, value) if math.isfinite(value) else fallback


def normalized_velocity(item: dict[str, Any]) -> float:
    try:
        value = float(item.get("velocity", 0.72))
    except (TypeError, ValueError):
        value = 0.72
    if value > 1.0:
        value /= 127.0
    if not math.isfinite(value):
        value = 0.72
    return max(0.01, min(1.0, value))


def normalize_notes(
    payload: dict[str, Any],
    *,
    transpose_semitones: int = 0,
) -> list[dict[str, float | int]]:
    notes: list[dict[str, float | int]] = []
    for item in payload.get("notes", []):
        try:
            midi = (
                int(round(float(item.get("midi", item.get("pitch")))))
                + int(transpose_semitones)
            )
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        if 0 <= midi <= 127 and time >= 0 and math.isfinite(time) and math.isfinite(duration):
            visual_duration = optional_duration(item, "visualDuration", duration)
            audio_duration = optional_duration(item, "audioDuration", duration)
            notes.append(
                {
                    "midi": midi,
                    "time": time,
                    "duration": duration,
                    "visualDuration": visual_duration,
                    "audioDuration": audio_duration,
                    "velocity": normalized_velocity(item),
                }
            )
    return sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))


def reference_transpose_semitones(pair: dict[str, Any]) -> int:
    """Read a predeclared octave-only reference register convention.

    Piano-route outputs use a frozen whole-score octave lift. Some trusted
    targets instead retain the singer's written register. Requiring an octave
    multiple prevents this metadata from becoming a candidate-specific pitch
    correction while making exact-register scoring comparable.
    """
    raw_value = pair.get("referenceTransposeSemitones", 0)
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError("referenceTransposeSemitones must be an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("referenceTransposeSemitones must be an integer")
    semitones = int(numeric)
    if abs(semitones) > 48 or semitones % 12 != 0:
        raise ValueError(
            "referenceTransposeSemitones must be an octave multiple between -48 and 48"
        )
    return semitones


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
                **note,
                "midi": int(note["midi"]),
                "time": max(0.0, start),
                "duration": max(0.01, end - start),
                "visualDuration": max(0.01, end - start),
                "audioDuration": max(0.01, end - start),
            }
        )
    return sorted(mapped, key=lambda note: (float(note["time"]), int(note["midi"])))


def prepare_reference_notes(
    pair: dict[str, Any],
    target_payload: dict[str, Any],
    alignment_report: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Freeze one reference in the candidate/source clock exactly once.

    Some supervised labels were already exported in source time.  Mapping
    those labels through the alignment a second time silently corrupts every
    onset and can make a worse decoder appear better.  Partial references are
    clipped in their native clock before mapping; candidate/source cutoffs are
    applied afterwards.
    """

    normalized = normalize_notes(
        target_payload,
        transpose_semitones=reference_transpose_semitones(pair),
    )
    reference_end = pair.get("referenceEndSeconds")
    if reference_end is not None:
        reference_end = float(reference_end)
        if not math.isfinite(reference_end) or reference_end <= 0:
            raise ValueError("referenceEndSeconds must be a positive finite number")
        normalized = [
            note for note in normalized if float(note["time"]) < reference_end
        ]
    mapped = (
        normalized
        if bool(pair.get("referenceAlreadyAligned"))
        else map_reference_notes(normalized, monotonic_anchors(alignment_report))
    )
    mapped = notes_inside_ranges(mapped, trusted_source_ranges(alignment_report))
    candidate_end = pair.get("candidateEndSeconds")
    if candidate_end is not None:
        candidate_end = float(candidate_end)
        if not math.isfinite(candidate_end) or candidate_end <= 0:
            raise ValueError("candidateEndSeconds must be a positive finite number")
        mapped = [note for note in mapped if float(note["time"]) < candidate_end]
    return mapped


def trusted_source_ranges(report: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Return the fixed regions that were approved before candidate scoring.

    Older reports without section decisions retain whole-song compatibility.
    An explicit report with zero approved windows returns an empty list and
    therefore cannot accidentally become a whole-song evaluation.
    """
    windows = report.get("qualityWindows")
    if not isinstance(windows, list):
        return None
    ranges: list[tuple[float, float]] = []
    for window in windows:
        if not isinstance(window, dict) or window.get("status") not in {"trusted", "accepted-manually"}:
            continue
        try:
            raw_start = window.get("sourceStartSeconds")
            raw_end = window.get("sourceEndSeconds")
            start = float(window.get("sourceStart") if raw_start is None else raw_start)
            end = float(window.get("sourceEnd") if raw_end is None else raw_end)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            ranges.append((max(0.0, start), end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def notes_inside_ranges(
    notes: list[dict[str, float | int]],
    ranges: list[tuple[float, float]] | None,
) -> list[dict[str, float | int]]:
    if ranges is None:
        return notes
    return [
        note
        for note in notes
        if any(start <= float(note["time"]) < end for start, end in ranges)
    ]


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


def duration_metrics(
    matches: list[tuple[dict[str, float | int], dict[str, float | int]]],
    observed_field: str,
) -> dict[str, Any]:
    duration_errors = [
        abs(float(target["duration"]) - float(candidate.get(observed_field, candidate["duration"])))
        for target, candidate in matches
    ]
    relative_errors = [
        abs(float(target["duration"]) - float(candidate.get(observed_field, candidate["duration"])))
        / max(0.05, float(target["duration"]))
        for target, candidate in matches
    ]
    offset_matches = sum(
        1
        for target, candidate in matches
        if abs(
            (float(target["time"]) + float(target["duration"]))
            - (
                float(candidate["time"])
                + float(candidate.get(observed_field, candidate["duration"]))
            )
        )
        <= 0.25
    )
    cutoffs = sum(
        1
        for target, candidate in matches
        if float(candidate.get(observed_field, candidate["duration"]))
        < float(target["duration"]) * 0.50
    )
    return {
        "matchedNotes": len(matches),
        "medianAbsoluteErrorSeconds": round(median(duration_errors), 6) if duration_errors else None,
        "medianRelativeError": round(median(relative_errors), 6) if relative_errors else None,
        "severeCutoffs": cutoffs,
        "severeCutoffRate": round(cutoffs / max(1, len(matches)), 6),
    }, offset_matches


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

    # Hold quality must not disappear merely because a note was voiced in the
    # wrong octave. Exact-register accuracy is already measured above. Using
    # pitch-class + onset matches here keeps the duration population stable
    # when a candidate fixes register placement and prevents a wrong-octave
    # baseline from escaping cutoff penalties entirely.
    duration_matches = greedy_matches(
        reference, observed, 0.25, octave_equivalent=True
    )
    metrics["durationMatchingPolicy"] = "pitch-class-onset-250ms"
    for name, field in (
        ("duration", "duration"),
        ("visualDuration", "visualDuration"),
        ("physicalDuration", "audioDuration"),
    ):
        values, offset_matches = duration_metrics(duration_matches, field)
        values["onsetAndOffset250ms"] = f1_metrics(len(reference), len(observed), offset_matches)
        metrics[name] = values
    velocity_errors = [
        abs(float(target.get("velocity", 0.72)) - float(candidate.get("velocity", 0.72)))
        for target, candidate in duration_matches
    ]
    velocity_bias = [
        float(candidate.get("velocity", 0.72)) - float(target.get("velocity", 0.72))
        for target, candidate in duration_matches
    ]
    metrics["velocity"] = {
        "matchingPolicy": "pitch-class-onset-250ms",
        "matchedNotes": len(duration_matches),
        "meanAbsoluteError": (
            round(sum(velocity_errors) / len(velocity_errors), 6)
            if velocity_errors
            else None
        ),
        "medianAbsoluteError": (
            round(median(velocity_errors), 6) if velocity_errors else None
        ),
        "meanBias": (
            round(sum(velocity_bias) / len(velocity_bias), 6)
            if velocity_bias
            else None
        ),
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
        reference_transpose = reference_transpose_semitones(pair)
        alignment_report = load_json(Path(pair["alignmentReport"]))
        anchors = monotonic_anchors(alignment_report)
        ranges = trusted_source_ranges(alignment_report)
        target_payload = load_json(Path(pair["target"]))
        normalized_target = normalize_notes(
            target_payload,
            transpose_semitones=reference_transpose,
        )
        all_mapped_target = (
            normalized_target
            if bool(pair.get("referenceAlreadyAligned"))
            else map_reference_notes(normalized_target, anchors)
        )
        mapped_target = prepare_reference_notes(
            pair, target_payload, alignment_report
        )
        baseline = notes_inside_ranges(
            normalize_notes(load_json(baseline_dir / f"{pair_id}-arranged.json")),
            ranges,
        )
        candidate = notes_inside_ranges(
            normalize_notes(load_json(candidate_dir / f"{pair_id}.json")),
            ranges,
        )
        if not mapped_target:
            raise ValueError(f"{pair_id}: alignment has no approved target notes for evaluation")
        baseline_metrics = evaluate(mapped_target, baseline)
        candidate_metrics = evaluate(mapped_target, candidate)
        evaluated_duration = None if ranges is None else sum(end - start for start, end in ranges)
        source_duration = float(
            alignment_report.get("metrics", {}).get("observedDurationSeconds")
            or alignment_report.get("metrics", {}).get("sourceDurationSeconds")
            or alignment_report.get("timeline", {}).get("sourceDurationSeconds")
            or 0.0
        )
        coverage = (
            None
            if ranges is None or source_duration <= 0
            else min(1.0, float(evaluated_duration or 0.0) / source_duration)
        )
        results[pair_id] = {
            "weight": weight,
            "referenceTransposeSemitones": reference_transpose,
            "fixedAnchorCount": len(anchors),
            "approvedSourceRanges": None if ranges is None else len(ranges),
            "alignmentCoverage": None if coverage is None else round(coverage, 6),
            "totalReferenceNotes": len(all_mapped_target),
            "evaluatedReferenceNotes": len(mapped_target),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
        for name, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
            for tolerance in (50, 100, 250):
                exact = metrics[f"exactPitchOnset{tolerance}ms"]
                pitch_class = metrics[f"pitchClassOnset{tolerance}ms"]
                for statistic in ("precision", "recall", "f1"):
                    aggregate_values[name][
                        f"exact{statistic.title()}_{tolerance}ms"
                    ].append((float(exact[statistic]), weight))
                    aggregate_values[name][
                        f"pitchClass{statistic.title()}_{tolerance}ms"
                    ].append((float(pitch_class[statistic]), weight))
            aggregate_values[name]["durationMedianAbsoluteErrorSeconds"].append(
                (float(metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0), weight)
            )
            aggregate_values[name]["severeCutoffs"].append((float(metrics["duration"]["severeCutoffs"]), weight))
            aggregate_values[name]["severeCutoffRate"].append((float(metrics["duration"]["severeCutoffRate"]), weight))
            aggregate_values[name]["visualDurationMedianAbsoluteErrorSeconds"].append(
                (float(metrics["visualDuration"]["medianAbsoluteErrorSeconds"] or 0.0), weight)
            )
            aggregate_values[name]["visualSevereCutoffRate"].append(
                (float(metrics["visualDuration"]["severeCutoffRate"]), weight)
            )
            aggregate_values[name]["physicalDurationMedianAbsoluteErrorSeconds"].append(
                (float(metrics["physicalDuration"]["medianAbsoluteErrorSeconds"] or 0.0), weight)
            )
            aggregate_values[name]["physicalSevereCutoffRate"].append(
                (float(metrics["physicalDuration"]["severeCutoffRate"]), weight)
            )
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
        "exactF1_100ms_improves": at_least(
            candidate["exactF1_100ms"], baseline["exactF1_100ms"] + 0.005
        ),
        # F1 can rise by deleting output notes while finding no additional
        # music.  A promoted arranger must also retain more strict-onset target
        # notes, which protects the sung/lead line from silent pruning.
        "exactRecall_100ms_improves": at_least(
            candidate["exactRecall_100ms"], baseline["exactRecall_100ms"] + 0.003
        ),
        "exactF1_250ms_does_not_regress": at_least(
            candidate["exactF1_250ms"], baseline["exactF1_250ms"] - 0.002
        ),
        "pitchClassF1_250ms_does_not_regress": at_least(
            candidate["pitchClassF1_250ms"], baseline["pitchClassF1_250ms"] - 0.002
        ),
        "pitchClassRecall_250ms_does_not_regress": at_least(
            candidate["pitchClassRecall_250ms"],
            baseline["pitchClassRecall_250ms"] - 0.002,
        ),
        "no_trusted_song_regresses_over_1_point": not trusted_regressions,
        "duration_error_not_over_10_percent_worse": at_most(
            candidate["durationMedianAbsoluteErrorSeconds"],
            baseline["durationMedianAbsoluteErrorSeconds"] * 1.10,
        ),
        # Rates are comparable when recall changes; raw counts are not. A
        # candidate that finds four times as many correct notes should not fail
        # merely because its larger matched set contains more absolute tails.
        "visual_cutoff_rate_not_worse": (
            at_most(
                candidate["visualSevereCutoffRate"],
                baseline["visualSevereCutoffRate"] * 1.05 + 0.005,
            )
        ),
        "physical_duration_error_not_over_10_percent_worse": (
            at_most(
                candidate["physicalDurationMedianAbsoluteErrorSeconds"],
                baseline["physicalDurationMedianAbsoluteErrorSeconds"] * 1.10,
            )
        ),
        "physical_cutoff_rate_not_worse": (
            at_most(
                candidate["physicalSevereCutoffRate"],
                baseline["physicalSevereCutoffRate"] * 1.05 + 0.005,
            )
        ),
        "rapid_retriggers_not_over_5_percent_worse": at_most(
            candidate["rapidRetriggersUnder100ms"],
            baseline["rapidRetriggersUnder100ms"] * 1.05 + 1.0,
        ),
        "alignment_coverage_is_sufficient": all(
            song["alignmentCoverage"] is None or float(song["alignmentCoverage"]) >= 0.60
            for song in results.values()
        ),
        "alignment_sample_size_is_sufficient": all(
            int(song["evaluatedReferenceNotes"]) >= 100 for song in results.values()
        ),
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
