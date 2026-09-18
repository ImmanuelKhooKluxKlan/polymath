"""Learn one pianist phrase and replay it only when the input repeats.

This adapter is a narrow supervised-learning experiment for arrangements such
as Kiss Me, where a later verse/chorus substantially repeats an earlier one.
The approved piano reference is read only while fitting the first occurrence.
At application time the adapter detects the later recurrence from the baseline
arrangement, derives a monotonic tempo warp, estimates transposition and local
energy change, and renders the learned gesture motif into the repeated region.

The resulting profile contains a song-conditioned motif.  It is useful evidence
for a future bar-level decoder, but it is not cross-song generalisation and must
remain research-only until evaluated on licensed, held-out songs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from .analyze_pianist_gesture_patterns import (
        group_onsets,
        normalize_notes,
        pitch_set,
        set_f1,
    )
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import (
        evaluation,
        finite,
        midi_to_note,
        source_groups,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import (
        group_onsets,
        normalize_notes,
        pitch_set,
        set_f1,
    )
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import evaluation, finite, midi_to_note, source_groups


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def slice_groups(
    groups: list[list[dict[str, Any]]], start: float, end: float
) -> list[list[dict[str, Any]]]:
    return [group for group in groups if start <= float(group[0]["time"]) < end]


def shifted_groups(
    groups: list[list[dict[str, Any]]], offset: float
) -> list[list[dict[str, Any]]]:
    return [
        [{**note, "time": float(note["time"]) + offset} for note in group]
        for group in groups
    ]


def repeat_trial(
    groups: list[list[dict[str, Any]]],
    *,
    template_start: float,
    template_end: float,
    offset: float,
    maximum_time_distance: float,
    gap_cost: float,
) -> dict[str, Any]:
    template = slice_groups(groups, template_start, template_end)
    repeated = slice_groups(
        groups, template_start + offset, template_end + offset
    )
    shifted = shifted_groups(template, offset)
    matches, missing, extra, score = align_sequences(
        shifted,
        repeated,
        maximum_time_distance=maximum_time_distance,
        gap_cost=gap_cost,
    )
    similarities = [
        set_f1(pitch_set(shifted[left], True), pitch_set(repeated[right], True))
        for left, right in matches
    ]
    denominator = max(1, len(shifted) + len(repeated))
    return {
        "offsetSeconds": offset,
        "templateGestures": len(template),
        "repeatGestures": len(repeated),
        "matchedGestures": len(matches),
        "missingGestures": len(missing),
        "extraGestures": len(extra),
        "meanPitchClassF1": sum(similarities) / max(1, len(similarities)),
        "sequenceScore": score,
        "normalizedScore": score / denominator,
        "matches": matches,
        "template": template,
        "repeated": repeated,
    }


def detect_repeat(
    groups: list[list[dict[str, Any]]],
    *,
    template_start: float,
    template_end: float,
    minimum_offset: float,
    maximum_offset: float,
    coarse_step: float = 0.10,
    fine_step: float = 0.01,
    maximum_time_distance: float = 0.35,
    gap_cost: float = 0.85,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if template_end <= template_start:
        raise ValueError("template_end must be greater than template_start")
    if maximum_offset <= minimum_offset:
        raise ValueError("maximum_offset must be greater than minimum_offset")

    coarse_offsets = []
    value = minimum_offset
    while value <= maximum_offset + 1e-9:
        coarse_offsets.append(round(value, 6))
        value += coarse_step
    coarse = [
        repeat_trial(
            groups,
            template_start=template_start,
            template_end=template_end,
            offset=offset,
            maximum_time_distance=maximum_time_distance,
            gap_cost=gap_cost,
        )
        for offset in coarse_offsets
    ]
    coarse_best = min(
        coarse,
        key=lambda row: (
            float(row["normalizedScore"]),
            -float(row["meanPitchClassF1"]),
            -int(row["matchedGestures"]),
        ),
    )
    fine_minimum = max(minimum_offset, float(coarse_best["offsetSeconds"]) - coarse_step)
    fine_maximum = min(maximum_offset, float(coarse_best["offsetSeconds"]) + coarse_step)
    fine_offsets = []
    value = fine_minimum
    while value <= fine_maximum + 1e-9:
        fine_offsets.append(round(value, 6))
        value += fine_step
    fine = [
        repeat_trial(
            groups,
            template_start=template_start,
            template_end=template_end,
            offset=offset,
            maximum_time_distance=maximum_time_distance,
            gap_cost=gap_cost,
        )
        for offset in fine_offsets
    ]
    # The coarse winner also occurs in the fine sweep.  Keep only one result
    # per offset so diagnostics do not advertise duplicate runner-up settings.
    unique_trials = {
        round(float(row["offsetSeconds"]), 6): row for row in coarse + fine
    }
    ranked = sorted(
        unique_trials.values(),
        key=lambda row: (
            float(row["normalizedScore"]),
            -float(row["meanPitchClassF1"]),
            -int(row["matchedGestures"]),
        ),
    )
    return ranked[0], ranked


def group_velocity(group: list[dict[str, Any]]) -> float:
    return median(finite(note.get("velocity"), 0.7) for note in group)


def infer_transposition(
    template: list[list[dict[str, Any]]],
    repeated: list[list[dict[str, Any]]],
    matches: list[tuple[int, int]],
    shifts: Iterable[int] = range(-11, 12),
) -> tuple[int, list[dict[str, Any]]]:
    trials = []
    for shift in shifts:
        values = []
        for left, right in matches:
            shifted = {(pitch + shift) % 12 for pitch in pitch_set(template[left], True)}
            values.append(set_f1(shifted, pitch_set(repeated[right], True)))
        trials.append(
            {
                "semitones": shift,
                "meanPitchClassF1": sum(values) / max(1, len(values)),
            }
        )
    best = max(
        trials,
        key=lambda row: (
            float(row["meanPitchClassF1"]),
            -abs(int(row["semitones"])),
        ),
    )
    return int(best["semitones"]), trials


def monotonic_anchors(
    template: list[list[dict[str, Any]]],
    repeated: list[list[dict[str, Any]]],
    matches: list[tuple[int, int]],
    *,
    template_start: float,
    template_end: float,
    offset: float,
) -> list[tuple[float, float]]:
    raw = [
        (
            float(template[left][0]["time"]),
            float(repeated[right][0]["time"]),
        )
        for left, right in matches
    ]
    raw.extend(
        [
            (template_start, template_start + offset),
            (template_end, template_end + offset),
        ]
    )
    by_left: dict[float, list[float]] = {}
    for left, right in raw:
        by_left.setdefault(left, []).append(right)
    anchors = sorted((left, median(rights)) for left, rights in by_left.items())
    output: list[tuple[float, float]] = []
    for left, right in anchors:
        if output and (left <= output[-1][0] or right <= output[-1][1]):
            continue
        output.append((left, right))
    if len(output) < 2:
        raise ValueError("Repeat matching did not produce a monotonic tempo map")
    return output


def map_time(value: float, anchors: list[tuple[float, float]]) -> float:
    if value <= anchors[0][0]:
        left, right = anchors[0], anchors[1]
    elif value >= anchors[-1][0]:
        left, right = anchors[-2], anchors[-1]
    else:
        left, right = anchors[0], anchors[1]
        for index in range(1, len(anchors)):
            if value <= anchors[index][0]:
                left, right = anchors[index - 1], anchors[index]
                break
    span = max(1e-9, right[0] - left[0])
    ratio = (value - left[0]) / span
    return left[1] + ratio * (right[1] - left[1])


def local_energy_ratio(
    value: float,
    velocity_anchors: list[tuple[float, float]],
    neighbors: int = 9,
) -> float:
    closest = sorted(velocity_anchors, key=lambda row: abs(row[0] - value))[
        :neighbors
    ]
    return median(row[1] for row in closest) if closest else 1.0


def nearest_group(
    groups: list[list[dict[str, Any]]], value: float, maximum_distance: float = 0.40
) -> list[dict[str, Any]]:
    if not groups:
        return []
    result = min(groups, key=lambda group: abs(float(group[0]["time"]) - value))
    return (
        result
        if abs(float(result[0]["time"]) - value) <= maximum_distance
        else []
    )


def clean_template_note(note: dict[str, Any], template_start: float) -> dict[str, Any]:
    excluded = {
        "_payloadIndex",
        "sourceIndex",
        "originalTime",
        "originalDuration",
        "referenceTime",
        "referenceDuration",
    }
    output = {key: value for key, value in note.items() if key not in excluded}
    output["relativeTime"] = rounded(float(note["time"]) - template_start)
    output.pop("time", None)
    return output


def evaluation_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, str]:
    """Accept a large correction or a smaller strictly safe refinement.

    Once a phrase is already strong, requiring five percentage points of F1
    and exact-set gain at the same time rejects useful one-gesture fixes.  The
    refinement route therefore permits a smaller structural gain only when
    coverage, precision, hand occupancy, timing, velocity, duration, and the
    aggregate sequence score are all protected.
    """

    safety = (
        float(candidate["referenceGestureRecall"])
        >= float(baseline["referenceGestureRecall"])
        and float(candidate["candidateGesturePrecision"])
        >= float(baseline["candidateGesturePrecision"])
        and float(candidate["coverageAdjustedPitchClassF1"])
        >= float(baseline["coverageAdjustedPitchClassF1"])
        and float(candidate["coverageAdjustedExactPitchClassRate"])
        >= float(baseline["coverageAdjustedExactPitchClassRate"])
        and float(candidate["coverageAdjustedOccupancyAccuracy"])
        >= float(baseline["coverageAdjustedOccupancyAccuracy"])
        and float(candidate["onsetMaeSeconds"])
        <= float(baseline["onsetMaeSeconds"]) + 0.01
        and float(candidate["gestureVelocityMae"])
        <= float(baseline["gestureVelocityMae"]) + 0.01
        and float(candidate["exactKeyDurationMaeSeconds"])
        <= float(baseline["exactKeyDurationMaeSeconds"]) + 0.01
    )
    structural_leap = (
        float(candidate["coverageAdjustedPitchClassF1"])
        >= float(baseline["coverageAdjustedPitchClassF1"]) + 0.05
        and float(candidate["coverageAdjustedExactPitchClassRate"])
        >= float(baseline["coverageAdjustedExactPitchClassRate"]) + 0.05
        and float(candidate["sequenceScore"])
        <= 0.85 * float(baseline["sequenceScore"])
    )
    safe_refinement = (
        (
            float(candidate["coverageAdjustedPitchClassF1"])
            >= float(baseline["coverageAdjustedPitchClassF1"]) + 0.01
            or float(candidate["coverageAdjustedExactPitchClassRate"])
            >= float(baseline["coverageAdjustedExactPitchClassRate"]) + 0.05
        )
        and float(candidate["sequenceScore"])
        <= 0.90 * float(baseline["sequenceScore"])
    )
    if safety and structural_leap:
        return True, "structural-leap"
    if safety and safe_refinement:
        return True, "safe-structural-refinement"
    return False, "gate-not-met"


def fit_profile(
    reference_groups: list[list[dict[str, Any]]],
    *,
    profile_id: str,
    song_id: str,
    template_start: float,
    template_end: float,
    repeat_search_minimum: float,
    repeat_search_maximum: float,
    time_warp_blend: float = 1.0,
    velocity_scale_blend: float = 1.0,
    onset_snap_blend: float = 0.0,
    require_timing_confidence: bool = True,
    minimum_velocity_scale: float = 0.75,
    maximum_velocity_scale: float = 1.25,
    use_destination_velocity: bool = False,
    minimum_input_pitch_f1: float = 0.68,
    maximum_input_normalized_score: float = 0.30,
) -> dict[str, Any]:
    template_groups = slice_groups(reference_groups, template_start, template_end)
    notes = [
        clean_template_note(note, template_start)
        for group in template_groups
        for note in group
    ]
    profile = {
        "schema": "polymath-pianist-repetition-profile-v1",
        "id": profile_id,
        "enabled": True,
        "training": {
            "songId": song_id,
            "method": "input-detected repeated-section gesture transfer",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
        },
        "template": {
            "startSeconds": template_start,
            "endSeconds": template_end,
            "durationSeconds": template_end - template_start,
            "gestureCount": len(template_groups),
            "noteCount": len(notes),
            "notes": notes,
        },
        "detection": {
            "minimumOffsetSeconds": repeat_search_minimum,
            "maximumOffsetSeconds": repeat_search_maximum,
            "coarseStepSeconds": 0.10,
            "fineStepSeconds": 0.01,
            "maximumTimeDistanceSeconds": 0.35,
            "gapCost": 0.85,
            "minimumMeanPitchClassF1": minimum_input_pitch_f1,
            "maximumNormalizedScore": maximum_input_normalized_score,
            "minimumTimingMeanPitchClassF1": 0.60,
            "maximumTimingNormalizedScore": 0.30,
            "requireTimingConfidence": require_timing_confidence,
        },
        "application": {
            "targetReadAtInference": False,
            "tempoWarp": "piecewise-monotonic-input-gesture-anchors",
            "timeWarpBlend": time_warp_blend,
            "velocity": (
                "local-input-energy-ratio-clamped-"
                f"{minimum_velocity_scale:g}-{maximum_velocity_scale:g}"
            ),
            "velocityScaleBlend": velocity_scale_blend,
            "minimumVelocityScale": minimum_velocity_scale,
            "maximumVelocityScale": maximum_velocity_scale,
            "velocitySource": (
                "nearest-destination-input-gesture"
                if use_destination_velocity
                else "scaled-template"
            ),
            "onsetSnapBlend": onset_snap_blend,
            "transposition": "input-pitch-class-search-minus11-plus11",
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    return profile


def apply_profile(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    detection_payload: dict[str, Any] | None = None,
    timing_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timing_payload = timing_payload or candidate
    candidate_groups = group_onsets(normalize_notes(timing_payload), 0.035)
    detection_payload = detection_payload or candidate
    # A full-mix transcription may contain pitched drum events.  They help
    # rhythm, but corrupt harmonic recurrence and can make a short phrase lock
    # onto the wrong section.  ``source_groups`` retains voice/harmony/bass and
    # removes percussion using the same contract as the intro motif adapter.
    groups = source_groups(detection_payload)
    template_start = float(profile["template"]["startSeconds"])
    template_end = float(profile["template"]["endSeconds"])
    detection = profile["detection"]
    best, ranked = detect_repeat(
        groups,
        template_start=template_start,
        template_end=template_end,
        minimum_offset=float(detection["minimumOffsetSeconds"]),
        maximum_offset=float(detection["maximumOffsetSeconds"]),
        coarse_step=float(detection["coarseStepSeconds"]),
        fine_step=float(detection["fineStepSeconds"]),
        maximum_time_distance=float(detection["maximumTimeDistanceSeconds"]),
        gap_cost=float(detection["gapCost"]),
    )
    template_groups = best["template"]
    repeated_groups = best["repeated"]
    matches = best["matches"]
    offset = float(best["offsetSeconds"])
    timing_trial = repeat_trial(
        candidate_groups,
        template_start=template_start,
        template_end=template_end,
        offset=offset,
        maximum_time_distance=float(detection["maximumTimeDistanceSeconds"]),
        gap_cost=float(detection["gapCost"]),
    )
    timing_template_groups = timing_trial["template"]
    timing_repeated_groups = timing_trial["repeated"]
    timing_matches = timing_trial["matches"]
    timing_evidence_passed = (
        float(timing_trial["meanPitchClassF1"])
        >= float(detection.get("minimumTimingMeanPitchClassF1", 0.60))
        and float(timing_trial["normalizedScore"])
        <= float(detection.get("maximumTimingNormalizedScore", 0.30))
    )
    timing_confidence_required = bool(
        detection.get("requireTimingConfidence", True)
    )
    timing_confidence_passed = (
        not timing_confidence_required or timing_evidence_passed
    )
    confidence_passed = (
        float(best["meanPitchClassF1"])
        >= float(detection.get("minimumMeanPitchClassF1", 0.68))
        and float(best["normalizedScore"])
        <= float(detection.get("maximumNormalizedScore", 0.30))
        and timing_confidence_passed
    )
    anchors = monotonic_anchors(
        timing_template_groups,
        timing_repeated_groups,
        timing_matches,
        template_start=template_start,
        template_end=template_end,
        offset=offset,
    )
    transposition, transposition_trials = infer_transposition(
        template_groups, repeated_groups, matches
    )
    minimum_velocity_scale = finite(
        profile["application"].get("minimumVelocityScale"), 0.75
    )
    maximum_velocity_scale = finite(
        profile["application"].get("maximumVelocityScale"), 1.25
    )
    if not 0 < minimum_velocity_scale <= maximum_velocity_scale:
        raise ValueError("Invalid velocity-scale bounds")
    velocity_anchors = []
    for left, right in timing_matches:
        template_candidate = timing_template_groups[left]
        repeat_candidate = timing_repeated_groups[right]
        template_time = float(template_candidate[0]["time"])
        velocity_anchors.append(
            (
                template_time,
                max(
                    minimum_velocity_scale,
                    min(
                        maximum_velocity_scale,
                        group_velocity(repeat_candidate)
                        / max(0.05, group_velocity(template_candidate)),
                    ),
                ),
            )
        )
    time_warp_blend = max(
        0.0, min(1.0, finite(profile["application"].get("timeWarpBlend"), 1.0))
    )
    velocity_scale_blend = max(
        0.0,
        min(1.0, finite(profile["application"].get("velocityScaleBlend"), 1.0)),
    )
    onset_snap_blend = max(
        0.0,
        min(1.0, finite(profile["application"].get("onsetSnapBlend"), 0.0)),
    )
    use_destination_velocity = (
        profile["application"].get("velocitySource")
        == "nearest-destination-input-gesture"
    )

    def mapped(value: float) -> float:
        constant = value + offset
        warped = map_time(value, anchors)
        return constant + time_warp_blend * (warped - constant)

    replacement_start = mapped(template_start)
    replacement_end = mapped(template_end)
    relative_group_times = sorted(
        {
            rounded(float(note["relativeTime"]))
            for note in profile["template"]["notes"]
        }
    )
    planned_times = [mapped(template_start + value) for value in relative_group_times]
    destination_times = [
        float(group[0]["time"])
        for group in candidate_groups
        if replacement_start - 0.30
        <= float(group[0]["time"])
        < replacement_end + 0.30
    ]
    planned_dummy = [[{"time": value, "midi": 60}] for value in planned_times]
    destination_dummy = [[{"time": value, "midi": 60}] for value in destination_times]
    snap_matches, _, _, _ = align_sequences(
        planned_dummy,
        destination_dummy,
        maximum_time_distance=0.30,
        gap_cost=0.26,
    )
    snap_by_relative = {
        relative_group_times[left]: destination_times[right]
        for left, right in snap_matches
    }

    output_notes = []
    for template_note in profile["template"]["notes"]:
        source_time = template_start + float(template_note["relativeTime"])
        source_duration = max(
            0.03,
            finite(
                template_note.get("scoreDuration", template_note.get("duration")),
                0.2,
            ),
        )
        planned_time = mapped(source_time)
        snap_time = snap_by_relative.get(
            rounded(float(template_note["relativeTime"])), planned_time
        )
        time = planned_time + onset_snap_blend * (snap_time - planned_time)
        # Snapping moves the whole physical key gesture.  Its release must move
        # by the same amount; subtracting the snapped onset from the unsnapped
        # end silently lengthened early snaps and shortened late snaps.  Tempo
        # warping may scale articulation, but onset correction itself must not.
        duration = max(
            0.03,
            mapped(source_time + source_duration) - planned_time,
        )
        raw_velocity_scale = local_energy_ratio(source_time, velocity_anchors)
        velocity_scale = 1.0 + velocity_scale_blend * (raw_velocity_scale - 1.0)
        midi = int(template_note["midi"]) + transposition
        if not 21 <= midi <= 108:
            continue
        note = {
            key: value
            for key, value in template_note.items()
            if key != "relativeTime"
        }
        destination_group = nearest_group(candidate_groups, time, 0.35)
        if use_destination_velocity and destination_group:
            output_velocity = group_velocity(destination_group)
        else:
            output_velocity = (
                finite(template_note.get("velocity"), 0.7) * velocity_scale
            )
        note.update(
            {
                "midi": midi,
                "note": midi_to_note(midi),
                "time": rounded(time),
                "duration": rounded(duration),
                "scoreDuration": rounded(duration),
                "visualDuration": rounded(duration),
                "audioDuration": rounded(duration),
                "velocity": rounded(max(0.05, min(1.0, output_velocity))),
                "source": "polymath-pianist-repetition-adapter",
                "sourceInstrument": "learned_pianist_repetition",
                "generatedBy": "pianist-repetition-adapter-v1",
            }
        )
        output_notes.append(note)

    # Onset snapping can move the first learned gesture a few milliseconds
    # before the abstract mapped boundary.  Removing notes only from the
    # abstract interval would then retain the stale destination chord and
    # layer it underneath the learned one.  Expand the replacement interval to
    # complete onset gestures, using the same 35 ms grouping contract as the
    # evaluator.  A note this close to a generated onset would be heard and
    # evaluated as part of that same physical key strike, so it must be
    # replaced as one unit.
    onset_guard_seconds = 0.035
    generated_onsets = [float(note["time"]) for note in output_notes]
    effective_replacement_start = replacement_start
    effective_replacement_end = replacement_end
    if generated_onsets:
        effective_replacement_start = min(
            replacement_start, min(generated_onsets) - onset_guard_seconds
        )
        effective_replacement_end = max(
            replacement_end, max(generated_onsets) + onset_guard_seconds
        )
    retained = [
        dict(note)
        for note in candidate.get("notes") or []
        if not (
            effective_replacement_start
            <= finite(note.get("time", note.get("startTime")), -1)
            < effective_replacement_end
        )
    ]
    output = copy.deepcopy(candidate)
    output["notes"] = sorted(
        retained + output_notes if confidence_passed else list(candidate.get("notes") or []),
        key=lambda note: (finite(note.get("time"), 0), int(note.get("midi", 0))),
    )
    diagnostics = {
        "applied": confidence_passed,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "detectedOffsetSeconds": rounded(offset),
        "plannedReplacementStartSeconds": rounded(replacement_start),
        "plannedReplacementEndSeconds": rounded(replacement_end),
        "replacementStartSeconds": rounded(effective_replacement_start),
        "replacementEndSeconds": rounded(effective_replacement_end),
        "replacementOnsetGuardSeconds": rounded(onset_guard_seconds),
        "inputTemplateGestures": int(best["templateGestures"]),
        "inputRepeatGestures": int(best["repeatGestures"]),
        "inputMatchedGestures": int(best["matchedGestures"]),
        "inputMeanPitchClassF1": rounded(best["meanPitchClassF1"]),
        "inputNormalizedSequenceScore": rounded(best["normalizedScore"]),
        "timingInputMatchedGestures": int(timing_trial["matchedGestures"]),
        "timingInputMeanPitchClassF1": rounded(timing_trial["meanPitchClassF1"]),
        "timingInputNormalizedSequenceScore": rounded(timing_trial["normalizedScore"]),
        "confidencePassed": confidence_passed,
        "timingConfidenceRequired": timing_confidence_required,
        "timingEvidencePassed": timing_evidence_passed,
        "timingConfidencePassed": timing_confidence_passed,
        "inferredTranspositionSemitones": transposition,
        "tempoAnchorCount": len(anchors),
        "timeWarpBlend": rounded(time_warp_blend),
        "velocityScaleBlend": rounded(velocity_scale_blend),
        "minimumVelocityScale": rounded(minimum_velocity_scale),
        "maximumVelocityScale": rounded(maximum_velocity_scale),
        "velocitySource": profile["application"].get(
            "velocitySource", "scaled-template"
        ),
        "onsetSnapBlend": rounded(onset_snap_blend),
        "onsetSnapMatches": len(snap_matches),
        "generatedNotes": len(output_notes) if confidence_passed else 0,
        "retainedNotes": len(retained) if confidence_passed else len(candidate.get("notes") or []),
        "targetReadAtInference": False,
        "detectionInput": "separate-raw-input" if detection_payload is not candidate else "candidate",
        "timingInput": "separate-frozen-input" if timing_payload is not candidate else "candidate",
        "runnerUpOffsets": [
            {
                "offsetSeconds": rounded(row["offsetSeconds"]),
                "normalizedScore": rounded(row["normalizedScore"]),
            }
            for row in ranked[:5]
        ],
        "transpositionTrials": [
            {
                "semitones": int(row["semitones"]),
                "meanPitchClassF1": rounded(row["meanPitchClassF1"]),
            }
            for row in transposition_trials
        ],
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistRepetitionGrammar"] = diagnostics
    arrangement["outputNoteCount"] = len(output["notes"])
    output["arrangementProfile"] = (
        f"{candidate.get('arrangementProfile', 'piano')}-repetition-v1"
    )
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument(
        "--detection-input",
        help="Optional richer raw transcription used only to detect repeated sections.",
    )
    parser.add_argument(
        "--timing-input",
        help="Optional frozen baseline used for timing confidence and energy scaling.",
    )
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--template-start", type=float, required=True)
    parser.add_argument("--template-end", type=float, required=True)
    parser.add_argument("--search-min-offset", type=float, default=35.0)
    parser.add_argument("--search-max-offset", type=float, default=65.0)
    parser.add_argument("--time-warp-blend", type=float, default=1.0)
    parser.add_argument("--velocity-scale-blend", type=float, default=1.0)
    parser.add_argument("--minimum-velocity-scale", type=float, default=0.75)
    parser.add_argument("--maximum-velocity-scale", type=float, default=1.25)
    parser.add_argument("--minimum-input-pitch-f1", type=float, default=0.68)
    parser.add_argument("--maximum-input-normalized-score", type=float, default=0.30)
    parser.add_argument(
        "--use-destination-velocity",
        action="store_true",
        help="Use the destination input gesture's energy for transferred notes.",
    )
    parser.add_argument("--onset-snap-blend", type=float, default=0.0)
    parser.add_argument(
        "--disable-timing-confidence",
        action="store_true",
        help="Research-only: allow raw-input repeat evidence to gate application alone.",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    row = next(
        (item for item in manifest.get("songs") or [] if item.get("id") == args.song_id),
        None,
    )
    if row is None:
        raise ValueError(f"Song {args.song_id!r} was not found in the manifest")
    reference, baseline, _ = load_song_sequences(row, 0.035)
    profile = fit_profile(
        reference,
        profile_id=args.profile_id,
        song_id=args.song_id,
        template_start=args.template_start,
        template_end=args.template_end,
        repeat_search_minimum=args.search_min_offset,
        repeat_search_maximum=args.search_max_offset,
        time_warp_blend=args.time_warp_blend,
        velocity_scale_blend=args.velocity_scale_blend,
        onset_snap_blend=args.onset_snap_blend,
        require_timing_confidence=not args.disable_timing_confidence,
        minimum_velocity_scale=args.minimum_velocity_scale,
        maximum_velocity_scale=args.maximum_velocity_scale,
        use_destination_velocity=args.use_destination_velocity,
        minimum_input_pitch_f1=args.minimum_input_pitch_f1,
        maximum_input_normalized_score=args.maximum_input_normalized_score,
    )
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    input_path = Path(args.input_candidate).resolve()
    input_payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    detection_payload = (
        json.loads(Path(args.detection_input).read_text(encoding="utf-8-sig"))
        if args.detection_input
        else input_payload
    )
    timing_payload = (
        json.loads(Path(args.timing_input).read_text(encoding="utf-8-sig"))
        if args.timing_input
        else input_payload
    )
    output, diagnostics = apply_profile(
        input_payload, profile, detection_payload, timing_payload
    )
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    generated_row = {**row, "candidate": str(output_path)}
    _, generated, _ = load_song_sequences(generated_row, 0.035)
    start = float(diagnostics["replacementStartSeconds"])
    end = float(diagnostics["replacementEndSeconds"])
    input_row = {**row, "candidate": str(input_path)}
    _, input_groups, _ = load_song_sequences(input_row, 0.035)
    held_out_baseline = evaluation(
        slice_groups(reference, start, end),
        slice_groups(input_groups, start, end),
    )
    held_out_candidate = evaluation(
        slice_groups(reference, start, end),
        slice_groups(generated, start, end),
    )
    if not diagnostics["applied"]:
        decision = "NO_CHANGE_CONFIDENCE_GUARD"
    else:
        passed, gate_reason = evaluation_gate(
            held_out_baseline, held_out_candidate
        )
        decision = (
            "LISTENING_CANDIDATE_RESEARCH_ONLY"
            if passed
            else "REJECT_OR_RESEARCH_ONLY"
        )
    if not diagnostics["applied"]:
        gate_reason = "confidence-guard"
    report = {
        "schema": "polymath-pianist-repetition-training-report-v1",
        "evidenceBoundary": (
            "The declared template occurrence is supervised training data; the "
            "detected counterpart is the evaluation region. Repeat detection, tempo "
            "warp, transposition, and energy scaling use only the input candidate at "
            "application time. This remains same-song research, not cross-song proof."
        ),
        "profile": str(profile_path),
        "inputCandidate": str(input_path),
        "detectionInput": str(Path(args.detection_input).resolve()) if args.detection_input else str(input_path),
        "timingInput": str(Path(args.timing_input).resolve()) if args.timing_input else str(input_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutRepeatEvaluation": {
            "baseline": held_out_baseline,
            "repetitionCandidate": held_out_candidate,
        },
        "evaluationGate": gate_reason,
        "trustedWindowEvaluation": {
            "baseline": evaluation(reference, input_groups),
            "repetitionCandidate": evaluation(reference, generated),
        },
        "decision": decision,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
