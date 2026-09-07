"""Repeat-aware alignment of an approved piano score to an original-song score.

The ordinary note aligner assumes one mostly continuous time transform.  This
aligner works at musical-section scale and uses semi-global dynamic time warping
with affine gaps.  It can therefore treat an intro, removed verse, repeated
chorus, instrumental break, or early screen-recording start as an explicit
section insertion/deletion instead of stretching every later label.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


PERCUSSION_PATTERN = ("drum", "percussion", "cymbal", "kick", "snare", "hi_hat", "hi-hat")
STATE_MATCH = 0
STATE_SOURCE_GAP = 1
STATE_REFERENCE_GAP = 2
INF = np.float64(1e30)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), clamp(fraction, 0.0, 1.0)))


def normalize_notes(payload: dict[str, Any], *, drop_percussion: bool = True) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for source_index, item in enumerate(payload.get("notes", [])):
        try:
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
            velocity = clamp(float(item.get("velocity", 0.72)), 0.05, 1.0)
        except (TypeError, ValueError):
            continue
        instrument = str(item.get("instrument") or "unknown").strip().lower()
        if not (0 <= midi <= 127) or time < 0 or not math.isfinite(time) or not math.isfinite(duration):
            continue
        if drop_percussion and any(token in instrument for token in PERCUSSION_PATTERN):
            continue
        notes.append(
            {
                "sourceIndex": source_index,
                "midi": midi,
                "pitchClass": midi % 12,
                "time": time,
                "duration": duration,
                "velocity": velocity,
                "instrument": instrument,
            }
        )
    return sorted(notes, key=lambda note: (note["time"], note["midi"], note["sourceIndex"]))


def build_frames(notes: list[dict[str, Any]], frame_seconds: float) -> tuple[np.ndarray, np.ndarray, float]:
    maximum_time = max((note["time"] + note["duration"] for note in notes), default=0.0)
    frame_count = max(2, int(math.ceil(maximum_time / frame_seconds)) + 1)
    chroma = np.zeros((frame_count, 12), dtype=np.float64)
    activity = np.zeros(frame_count, dtype=np.float64)
    instrument_weights = {
        "voice": 1.15,
        "acoustic_piano": 1.05,
        "electric_piano": 1.0,
        "acoustic_bass": 0.9,
        "electric_bass": 0.9,
        "synth_pad": 0.75,
    }
    for note in notes:
        start = max(0, int(math.floor(note["time"] / frame_seconds)))
        limited_end = min(note["time"] + note["duration"], note["time"] + 2.5)
        end = min(frame_count - 1, int(math.floor(limited_end / frame_seconds)))
        instrument_weight = instrument_weights.get(note["instrument"], 1.0)
        base = (0.35 + note["velocity"] * 0.65) * instrument_weight
        for frame_index in range(start, end + 1):
            onset_weight = 1.0 if frame_index == start else 0.22
            strength = base * onset_weight
            chroma[frame_index, note["pitchClass"]] += strength
            activity[frame_index] += strength

    # Local temporal smoothing makes cover arpeggios comparable to block chords.
    padded = np.pad(chroma, ((1, 1), (0, 0)))
    smoothed = padded[1:-1] * 0.62 + padded[:-2] * 0.19 + padded[2:] * 0.19
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    normalized = np.divide(smoothed, norms, out=np.zeros_like(smoothed), where=norms > 1e-9)
    activity_scale = float(np.quantile(activity[activity > 0], 0.9)) if np.any(activity > 0) else 1.0
    normalized_activity = np.clip(activity / max(1e-9, activity_scale), 0.0, 1.0)
    return normalized, normalized_activity, maximum_time


def similarity_matrix(
    reference_chroma: np.ndarray,
    source_chroma: np.ndarray,
    reference_activity: np.ndarray,
    source_activity: np.ndarray,
    transpose: int,
) -> np.ndarray:
    shifted_reference = np.roll(reference_chroma, transpose, axis=1)
    pitch_similarity = np.clip(shifted_reference @ source_chroma.T, 0.0, 1.0)
    activity_similarity = 1.0 - np.abs(reference_activity[:, None] - source_activity[None, :])
    both_silent = (reference_activity[:, None] < 0.03) & (source_activity[None, :] < 0.03)
    similarity = pitch_similarity * 0.86 + activity_similarity * 0.14
    similarity[both_silent] = np.maximum(similarity[both_silent], 0.45)
    one_silent = (reference_activity[:, None] < 0.03) ^ (source_activity[None, :] < 0.03)
    similarity[one_silent] *= 0.45
    return np.clip(similarity, 0.0, 1.0)


def affine_semiglobal_path(
    similarity: np.ndarray,
    *,
    source_gap_open: float,
    source_gap_extend: float,
    reference_gap_open: float,
    reference_gap_extend: float,
) -> dict[str, Any]:
    reference_count, source_count = similarity.shape
    costs = np.full((3, reference_count + 1, source_count + 1), INF, dtype=np.float64)
    pointers = np.full((3, reference_count + 1, source_count + 1), 255, dtype=np.uint8)

    # The source may contain arbitrary recording noise or a long intro before
    # the approved piano target begins. Prefix and suffix source frames are free.
    costs[STATE_MATCH, 0, :] = 0.0
    costs[STATE_SOURCE_GAP, 0, :] = 0.0
    pointers[STATE_MATCH, 0, 1:] = STATE_MATCH
    pointers[STATE_SOURCE_GAP, 0, 1:] = STATE_SOURCE_GAP
    for reference_index in range(1, reference_count + 1):
        if reference_index == 1:
            costs[STATE_REFERENCE_GAP, reference_index, 0] = reference_gap_open
            pointers[STATE_REFERENCE_GAP, reference_index, 0] = STATE_MATCH
        else:
            costs[STATE_REFERENCE_GAP, reference_index, 0] = (
                costs[STATE_REFERENCE_GAP, reference_index - 1, 0] + reference_gap_extend
            )
            pointers[STATE_REFERENCE_GAP, reference_index, 0] = STATE_REFERENCE_GAP

    distance = 1.0 - similarity
    for reference_index in range(1, reference_count + 1):
        for source_index in range(1, source_count + 1):
            previous = costs[:, reference_index - 1, source_index - 1]
            previous_state = int(np.argmin(previous))
            costs[STATE_MATCH, reference_index, source_index] = (
                previous[previous_state] + distance[reference_index - 1, source_index - 1]
            )
            pointers[STATE_MATCH, reference_index, source_index] = previous_state

            source_gap_candidates = np.asarray(
                [
                    costs[STATE_MATCH, reference_index, source_index - 1] + source_gap_open,
                    costs[STATE_SOURCE_GAP, reference_index, source_index - 1] + source_gap_extend,
                    costs[STATE_REFERENCE_GAP, reference_index, source_index - 1] + source_gap_open,
                ]
            )
            source_gap_state = int(np.argmin(source_gap_candidates))
            costs[STATE_SOURCE_GAP, reference_index, source_index] = source_gap_candidates[source_gap_state]
            pointers[STATE_SOURCE_GAP, reference_index, source_index] = source_gap_state

            reference_gap_candidates = np.asarray(
                [
                    costs[STATE_MATCH, reference_index - 1, source_index] + reference_gap_open,
                    costs[STATE_SOURCE_GAP, reference_index - 1, source_index] + reference_gap_open,
                    costs[STATE_REFERENCE_GAP, reference_index - 1, source_index] + reference_gap_extend,
                ]
            )
            reference_gap_state = int(np.argmin(reference_gap_candidates))
            costs[STATE_REFERENCE_GAP, reference_index, source_index] = reference_gap_candidates[reference_gap_state]
            pointers[STATE_REFERENCE_GAP, reference_index, source_index] = reference_gap_state

    end_costs = costs[:, reference_count, :]
    flat_index = int(np.argmin(end_costs))
    state, source_index = np.unravel_index(flat_index, end_costs.shape)
    reference_index = reference_count
    total_cost = float(end_costs[state, source_index])
    operations: list[dict[str, Any]] = []
    matches: list[tuple[int, int, float]] = []
    source_gap_frames = 0
    reference_gap_frames = 0
    while reference_index > 0:
        previous_state = int(pointers[state, reference_index, source_index])
        if state == STATE_MATCH:
            reference_frame = reference_index - 1
            source_frame = source_index - 1
            frame_similarity = float(similarity[reference_frame, source_frame])
            matches.append((reference_frame, source_frame, frame_similarity))
            operations.append(
                {
                    "operation": "match",
                    "referenceFrame": int(reference_frame),
                    "sourceFrame": int(source_frame),
                    "similarity": round(frame_similarity, 6),
                }
            )
            reference_index -= 1
            source_index -= 1
        elif state == STATE_SOURCE_GAP:
            operations.append(
                {
                    "operation": "source-section-not-in-target",
                    "referenceFrame": int(reference_index - 1),
                    "sourceFrame": int(source_index - 1),
                }
            )
            source_index -= 1
            source_gap_frames += 1
        else:
            operations.append(
                {
                    "operation": "target-section-not-in-source",
                    "referenceFrame": int(reference_index - 1),
                    "sourceFrame": int(source_index - 1),
                }
            )
            reference_index -= 1
            reference_gap_frames += 1
        state = previous_state
        if source_index < 0:
            break
    matches.reverse()
    operations.reverse()
    matched_similarity = [item[2] for item in matches]
    return {
        "cost": total_cost,
        "normalizedCost": total_cost / max(1, reference_count),
        "matches": matches,
        "operations": operations,
        "sourceGapFrames": source_gap_frames,
        "referenceGapFrames": reference_gap_frames,
        "meanMatchedSimilarity": float(np.mean(matched_similarity)) if matched_similarity else 0.0,
        "medianMatchedSimilarity": float(np.median(matched_similarity)) if matched_similarity else 0.0,
    }


def choose_alignment(
    reference_chroma: np.ndarray,
    source_chroma: np.ndarray,
    reference_activity: np.ndarray,
    source_activity: np.ndarray,
    options: dict[str, float],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for transpose in range(12):
        similarity = similarity_matrix(
            reference_chroma,
            source_chroma,
            reference_activity,
            source_activity,
            transpose,
        )
        path = affine_semiglobal_path(similarity, **options)
        matched_fraction = len(path["matches"]) / max(1, reference_chroma.shape[0])
        # Prefer a well-supported path; a path cannot win merely by deleting
        # most of the target and matching a tiny coincidental phrase.
        selection_score = (
            path["normalizedCost"]
            + max(0.0, 0.70 - matched_fraction) * 2.0
            + transpose_distance(transpose) * 0.005
        )
        candidates.append(
            {
                **path,
                "transposeSemitones": transpose if transpose <= 6 else transpose - 12,
                "matchedReferenceFramePercent": round(matched_fraction * 100.0, 3),
                "selectionScore": selection_score,
            }
        )
    return min(candidates, key=lambda candidate: candidate["selectionScore"])


def transpose_distance(transpose: int) -> int:
    return min(transpose, 12 - transpose)


def group_sections(matches: list[tuple[int, int, float]], frame_seconds: float) -> list[dict[str, Any]]:
    if not matches:
        return []
    groups: list[list[tuple[int, int, float]]] = [[matches[0]]]
    for item in matches[1:]:
        previous = groups[-1][-1]
        reference_jump = item[0] - previous[0]
        source_jump = item[1] - previous[1]
        if reference_jump > 3 or source_jump > 3 or abs(source_jump - reference_jump) > 3:
            groups.append([item])
        else:
            groups[-1].append(item)
    sections: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        reference_start = group[0][0] * frame_seconds
        reference_end = (group[-1][0] + 1) * frame_seconds
        source_start = group[0][1] * frame_seconds
        source_end = (group[-1][1] + 1) * frame_seconds
        reference_span = max(frame_seconds, reference_end - reference_start)
        source_span = max(frame_seconds, source_end - source_start)
        similarities = [item[2] for item in group]
        sections.append(
            {
                "id": f"section-{index:03d}",
                "targetStartSeconds": round(reference_start, 4),
                "targetEndSeconds": round(reference_end, 4),
                "sourceStartSeconds": round(source_start, 4),
                "sourceEndSeconds": round(source_end, 4),
                "localScale": round(source_span / reference_span, 6),
                "matchedFrames": len(group),
                "meanSimilarity": round(float(np.mean(similarities)), 6),
                "medianSimilarity": round(float(np.median(similarities)), 6),
                "minimumSimilarity": round(float(np.min(similarities)), 6),
            }
        )
    return sections


def build_anchors(
    matches: list[tuple[int, int, float]],
    frame_seconds: float,
    *,
    bin_seconds: float = 4.0,
) -> list[dict[str, Any]]:
    bins: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for item in matches:
        bins[int((item[0] * frame_seconds) // bin_seconds)].append(item)
    anchors: list[dict[str, Any]] = []
    for bin_index in sorted(bins):
        group = bins[bin_index]
        strong = [item for item in group if item[2] >= 0.38] or group
        reference_time = median([(item[0] + 0.5) * frame_seconds for item in strong])
        source_time = median([(item[1] + 0.5) * frame_seconds for item in strong])
        anchors.append(
            {
                "referenceTime": round(reference_time, 6),
                "observedTime": round(source_time, 6),
                "support": len(strong),
                "structuralSimilarity": round(float(np.mean([item[2] for item in strong])), 6),
                "kind": "repeat-aware-affine-section",
            }
        )
    # Enforce strict monotonicity after bin aggregation.
    monotonic: list[dict[str, Any]] = []
    for anchor in anchors:
        if monotonic and anchor["referenceTime"] <= monotonic[-1]["referenceTime"]:
            continue
        if monotonic and anchor["observedTime"] <= monotonic[-1]["observedTime"]:
            continue
        monotonic.append(anchor)
    return monotonic


def quality_windows(
    matches: list[tuple[int, int, float]],
    frame_seconds: float,
    reference_duration: float,
    window_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    by_window: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for item in matches:
        by_window[int((item[0] * frame_seconds) // window_seconds)].append(item)
    windows: list[dict[str, Any]] = []
    count = max(1, int(math.ceil(reference_duration / window_seconds)))
    for window_index in range(count):
        group = by_window.get(window_index, [])
        expected_frames = max(1, round(window_seconds / frame_seconds))
        coverage = len(group) / expected_frames
        mean_similarity = float(np.mean([item[2] for item in group])) if group else 0.0
        if coverage >= 0.60 and mean_similarity >= 0.58:
            status = "trusted"
        elif coverage >= 0.35 and mean_similarity >= 0.42:
            status = "review"
        else:
            status = "unsafe"
        windows.append(
            {
                "id": f"window-{window_index + 1:03d}",
                "targetStartSeconds": round(window_index * window_seconds, 4),
                "targetEndSeconds": round(min(reference_duration, (window_index + 1) * window_seconds), 4),
                "matchedFrameCoverage": round(coverage, 4),
                "meanSimilarity": round(mean_similarity, 6),
                "status": status,
            }
        )
    return windows


def map_time(value: float, anchors: list[dict[str, Any]]) -> float:
    references = [float(anchor["referenceTime"]) for anchor in anchors]
    position = bisect.bisect_right(references, value)
    if len(anchors) < 2:
        return value
    if position <= 0:
        left, right = anchors[0], anchors[1]
    elif position >= len(anchors):
        left, right = anchors[-2], anchors[-1]
    else:
        left, right = anchors[position - 1], anchors[position]
    left_reference = float(left["referenceTime"])
    right_reference = float(right["referenceTime"])
    scale = (float(right["observedTime"]) - float(left["observedTime"])) / max(
        1e-9, right_reference - left_reference
    )
    return float(left["observedTime"]) + (value - left_reference) * scale


def closest_window(time: float, windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for window in windows:
        if window["targetStartSeconds"] <= time < window["targetEndSeconds"]:
            return window
    return windows[-1] if windows else None


def add_source_ranges_to_windows(
    windows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Record the source-time span represented by every target quality window.

    The trainer uses these ranges to ignore uncertain source sections completely.
    Without this mask, an omitted bridge in a cover would be mislabeled as thousands
    of negative examples and the arranger would learn to delete valid music.
    """
    enriched: list[dict[str, Any]] = []
    for window in windows:
        source_start = map_time(float(window["targetStartSeconds"]), anchors)
        source_end = map_time(float(window["targetEndSeconds"]), anchors)
        enriched.append(
            {
                **window,
                "sourceStartSeconds": round(max(0.0, min(source_start, source_end)), 6),
                "sourceEndSeconds": round(max(0.0, max(source_start, source_end)), 6),
            }
        )
    return enriched


def align_target_payload(
    target_payload: dict[str, Any],
    anchors: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    output = dict(target_payload)
    aligned_notes: list[dict[str, Any]] = []
    for item in target_payload.get("notes", []):
        try:
            start = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        mapped_start = max(0.0, map_time(start, anchors))
        mapped_end = max(mapped_start + 0.01, map_time(start + duration, anchors))
        window = closest_window(start, windows)
        aligned_notes.append(
            {
                **item,
                "originalTime": start,
                "time": round(mapped_start, 6),
                "duration": round(mapped_end - mapped_start, 6),
                "alignmentWindowId": window["id"] if window else None,
                "alignmentStatus": window["status"] if window else "unsafe",
                "trainingEligible": bool(window and window["status"] == "trusted"),
            }
        )
    output["notes"] = aligned_notes
    output["timeline"] = "original-source-audio"
    output["sectionAlignment"] = {
        "schema": "polymath-repeat-aware-section-alignment-v1",
        "trainingEligibleNotes": sum(1 for note in aligned_notes if note["trainingEligible"]),
    }
    return output


def greedy_note_metrics(
    reference: list[dict[str, Any]],
    source: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    source_by_pitch_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in source:
        source_by_pitch_class[note["pitchClass"]].append(note)
    used: set[int] = set()
    exact = 0
    octave = 0
    residuals: list[float] = []
    for target in reference:
        expected = map_time(target["time"], anchors)
        candidates = [
            note
            for note in source_by_pitch_class[target["pitchClass"]]
            if note["sourceIndex"] not in used and abs(note["time"] - expected) <= tolerance
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda note: (abs(note["time"] - expected), abs(note["midi"] - target["midi"])),
        )
        used.add(best["sourceIndex"])
        residuals.append(abs(best["time"] - expected))
        if best["midi"] == target["midi"]:
            exact += 1
        else:
            octave += 1
    matched = exact + octave
    return {
        "toleranceMs": round(tolerance * 1000),
        "matchedNotes": matched,
        "matchedReferencePercent": round(matched / max(1, len(reference)) * 100.0, 3),
        "exactPitchMatches": exact,
        "exactPitchPercentOfMatches": round(exact / max(1, matched) * 100.0, 3),
        "medianResidualMs": round((median(residuals) if residuals else 0.0) * 1000.0, 3),
        "p95ResidualMs": round((quantile(residuals, 0.95) or 0.0) * 1000.0, 3),
    }


def collect_training_matches(
    reference: list[dict[str, Any]],
    source: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    tolerance: float = 0.50,
) -> list[dict[str, Any]]:
    source_by_pitch_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in source:
        source_by_pitch_class[note["pitchClass"]].append(note)
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    for target in reference:
        window = closest_window(target["time"], windows)
        if not window or window["status"] != "trusted":
            continue
        expected = map_time(target["time"], anchors)
        candidates = [
            note
            for note in source_by_pitch_class[target["pitchClass"]]
            if note["sourceIndex"] not in used and abs(note["time"] - expected) <= tolerance
        ]
        if not candidates:
            continue

        def match_cost(note: dict[str, Any]) -> tuple[float, float, int]:
            residual = abs(note["time"] - expected) / max(1e-9, tolerance)
            exact_penalty = 0.0 if note["midi"] == target["midi"] else 0.18
            octave_penalty = abs(note["midi"] - target["midi"]) / 12.0 * 0.035
            return residual + exact_penalty + octave_penalty, residual, abs(note["midi"] - target["midi"])

        best = min(candidates, key=match_cost)
        used.add(best["sourceIndex"])
        matches.append(
            {
                "reference": target,
                "observed": best,
                "expectedTime": round(expected, 6),
                "timingResidualSeconds": round(best["time"] - expected, 6),
                "exactPitch": best["midi"] == target["midi"],
                "octaveDifference": round((best["midi"] - target["midi"]) / 12),
                "qualityWindowId": window["id"],
                "trainingEligible": True,
            }
        )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Approved piano JSON on its own timeline")
    parser.add_argument("--source", required=True, help="MuScriptor original-song JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-seconds", type=float, default=0.5)
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    source_path = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    target_payload = load_json(target_path)
    source_payload = load_json(source_path)
    target_notes = normalize_notes(target_payload)
    source_notes = normalize_notes(source_payload)
    if len(target_notes) < 24 or len(source_notes) < 24:
        raise ValueError("Both scores need at least 24 non-percussive notes.")
    frame_seconds = clamp(float(args.frame_seconds), 0.25, 2.0)
    reference_chroma, reference_activity, reference_duration = build_frames(target_notes, frame_seconds)
    source_chroma, source_activity, source_duration = build_frames(source_notes, frame_seconds)
    gap_options = {
        "source_gap_open": 0.80,
        "source_gap_extend": 0.22,
        "reference_gap_open": 1.00,
        "reference_gap_extend": 0.30,
    }
    best = choose_alignment(
        reference_chroma,
        source_chroma,
        reference_activity,
        source_activity,
        gap_options,
    )
    anchors = build_anchors(best["matches"], frame_seconds)
    if len(anchors) < 2:
        raise ValueError("The repeat-aware aligner could not build a monotonic section map.")
    sections = group_sections(best["matches"], frame_seconds)
    windows = quality_windows(best["matches"], frame_seconds, reference_duration)
    windows = add_source_ranges_to_windows(windows, anchors)
    aligned_payload = align_target_payload(target_payload, anchors, windows)
    training_matches = collect_training_matches(
        target_notes,
        source_notes,
        anchors,
        windows,
    )
    trusted_note_fraction = sum(
        1 for note in aligned_payload["notes"] if note["trainingEligible"]
    ) / max(1, len(aligned_payload["notes"]))
    note_match_fraction = len(training_matches) / max(1, len(target_notes))
    confidence = clamp(
        best["meanMatchedSimilarity"]
        * math.sqrt(max(0.0, trusted_note_fraction * note_match_fraction)),
        0.0,
        1.0,
    )
    verdict = (
        "trusted-for-reviewed-section-training"
        if confidence >= 0.70
        else "manual-review-required"
        if confidence >= 0.45
        else "weak-sections-only"
    )
    metrics = {
        "targetNotes": len(target_notes),
        "sourceNotes": len(source_notes),
        "targetDurationSeconds": round(reference_duration, 4),
        "sourceDurationSeconds": round(source_duration, 4),
        "transposeSemitones": best["transposeSemitones"],
        "matchedReferenceFramePercent": best["matchedReferenceFramePercent"],
        "meanMatchedSimilarity": round(best["meanMatchedSimilarity"], 6),
        "medianMatchedSimilarity": round(best["medianMatchedSimilarity"], 6),
        "sourceGapSeconds": round(best["sourceGapFrames"] * frame_seconds, 4),
        "referenceGapSeconds": round(best["referenceGapFrames"] * frame_seconds, 4),
        "anchorCount": len(anchors),
        "sectionCount": len(sections),
        "trustedWindows": sum(1 for window in windows if window["status"] == "trusted"),
        "reviewWindows": sum(1 for window in windows if window["status"] == "review"),
        "unsafeWindows": sum(1 for window in windows if window["status"] == "unsafe"),
        "trainingEligibleNotes": sum(1 for note in aligned_payload["notes"] if note["trainingEligible"]),
        "trainingMatchedNotes": len(training_matches),
        "trainingExactPitchMatches": sum(1 for match in training_matches if match["exactPitch"]),
        "confidence": round(confidence, 6),
        "verdict": verdict,
        "noteAgreement": [
            greedy_note_metrics(target_notes, source_notes, anchors, tolerance)
            for tolerance in (0.10, 0.25, 0.50)
        ],
    }
    report = {
        "schema": "polymath-repeat-aware-section-alignment-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "targetFile": str(target_path),
        "sourceFile": str(source_path),
        "frameSeconds": frame_seconds,
        "algorithm": {
            "name": "semi-global-chroma-dtw-with-affine-section-gaps",
            "freeSourcePrefixAndSuffix": True,
            "transposeSearchSemitones": list(range(-5, 7)),
            **gap_options,
        },
        "metrics": metrics,
        "anchors": anchors,
        "sections": sections,
        "qualityWindows": windows,
        "matches": training_matches,
    }
    write_json(output_dir / "section-alignment-report.json", report)
    write_json(output_dir / "aligned-target.json", aligned_payload)
    write_json(
        output_dir / "section-operations.json",
        {
            "schema": "polymath-repeat-aware-section-operations-v1",
            "operations": best["operations"],
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
