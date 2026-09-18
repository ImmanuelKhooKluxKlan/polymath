"""Fit a source-supported cyclic pianist arpeggio from one trusted cycle.

This is a deliberately narrow research adapter.  It learns the timing,
gesture touch, register and source-pitch ranks of one approved pianist cycle.
At application time it does not read the destination target.  Instead, it
reconstructs the next cycle from the non-percussive source harmony, retaining
stable pedal pitch classes and repeated phase identities.

The profile is song-conditioned evidence for a future texture decoder.  It is
not a replacement for leave-one-song-out evaluation and is never promoted by
this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from .analyze_pianist_gesture_patterns import group_onsets, normalize_notes
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import (
        evaluation,
        finite,
        is_voice,
        midi_to_note,
        note_duration,
        source_groups,
    )
    from .train_pianist_repetition_adapter import (
        detect_repeat,
        map_time as map_repeat_time,
        monotonic_anchors as repeat_tempo_anchors,
        repeat_trial,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import group_onsets, normalize_notes
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import (
        evaluation,
        finite,
        is_voice,
        midi_to_note,
        note_duration,
        source_groups,
    )
    from train_pianist_repetition_adapter import (
        detect_repeat,
        map_time as map_repeat_time,
        monotonic_anchors as repeat_tempo_anchors,
        repeat_trial,
    )


HARMONIC_FAMILIES = ("piano", "keyboard", "guitar", "bass", "string")


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def slice_groups(
    groups: list[list[dict[str, Any]]], start: float, end: float
) -> list[list[dict[str, Any]]]:
    return [group for group in groups if start <= float(group[0]["time"]) < end]


def source_instrument(note: dict[str, Any]) -> str:
    return str(note.get("instrument") or note.get("sourceInstrument") or "").lower()


def harmonic_note(note: dict[str, Any]) -> bool:
    instrument = source_instrument(note)
    return not is_voice(note) and (
        not instrument or any(family in instrument for family in HARMONIC_FAMILIES)
    )


def source_pool(
    groups: list[list[dict[str, Any]]],
    time: float,
    *,
    lag_seconds: float,
    radius_seconds: float,
) -> list[int]:
    return sorted(
        {
            int(note["midi"])
            for group in groups
            if abs(float(group[0]["time"]) - (time + lag_seconds))
            <= radius_seconds
            for note in group
            if harmonic_note(note)
        }
    )


def pitch_class_f1(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(right)
    recall = overlap / len(left)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def select_pool_window(
    template_groups: list[list[dict[str, Any]]],
    input_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Choose temporal source support using training-cycle labels only."""

    def score(lag: float, radius: float) -> dict[str, Any]:
        f1_values: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []
        for group in template_groups:
            target = {int(note["midi"]) % 12 for note in group}
            pool = {
                midi % 12
                for midi in source_pool(
                    input_groups,
                    float(group[0]["time"]),
                    lag_seconds=lag,
                    radius_seconds=radius,
                )
            }
            overlap = len(target & pool)
            f1_values.append(pitch_class_f1(target, pool))
            recalls.append(overlap / max(1, len(target)))
            precisions.append(overlap / max(1, len(pool)))
        return {
            "lagSeconds": lag,
            "radiusSeconds": radius,
            "meanPitchClassF1": sum(f1_values) / max(1, len(f1_values)),
            "meanRecall": sum(recalls) / max(1, len(recalls)),
            "meanPrecision": sum(precisions) / max(1, len(precisions)),
        }

    trials: list[dict[str, Any]] = []
    for lag_step in range(-30, 31):
        lag = lag_step / 100.0
        for radius in (0.12, 0.16, 0.20, 0.25, 0.30, 0.35):
            trials.append(score(lag, radius))
    # F1 prevents an enormous pool from winning on recall alone.  For tied
    # evidence, prefer the tighter and less shifted window.
    trials.sort(
        key=lambda item: (
            -float(item["meanPitchClassF1"]),
            -float(item["meanRecall"]),
            float(item["radiusSeconds"]),
            abs(float(item["lagSeconds"])),
        )
    )
    if not trials:
        raise ValueError("No source-pool window trials were produced")
    return {**trials[0], "topTrials": trials[:12], "score": score}


def nearest_anchor(pool: list[int], target: int) -> tuple[int, int]:
    if not pool:
        raise ValueError("Cannot anchor a pianist note to an empty source pool")
    same_class = [midi for midi in pool if midi % 12 == target % 12]
    anchor = min(same_class or pool, key=lambda midi: (abs(midi - target), midi))
    return anchor, pool.index(anchor)


def clean_note(note: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "_payloadIndex",
        "sourceIndex",
        "originalTime",
        "originalDuration",
        "referenceTime",
        "referenceDuration",
    }
    return {key: value for key, value in note.items() if key not in excluded}


def fit_profile(
    reference_groups: list[list[dict[str, Any]]],
    input_groups: list[list[dict[str, Any]]],
    *,
    profile_id: str,
    song_id: str,
    template_start: float,
    cycle_duration: float,
    source_to_output_semitones: int,
    persistent_pitch_class_share: float = 0.60,
    source_lag_seconds: float | None = None,
    source_radius_seconds: float | None = None,
    repeat_offset_search_radius_seconds: float = 0.20,
) -> dict[str, Any]:
    if cycle_duration <= 0:
        raise ValueError("cycle_duration must be positive")
    if not 0 <= repeat_offset_search_radius_seconds <= 1.0:
        raise ValueError(
            "repeat_offset_search_radius_seconds must be between 0 and 1"
        )
    template_end = template_start + cycle_duration
    template_groups = slice_groups(reference_groups, template_start, template_end)
    if len(template_groups) < 4:
        raise ValueError("Cyclic arpeggio training requires at least four gestures")
    if (source_lag_seconds is None) != (source_radius_seconds is None):
        raise ValueError(
            "source_lag_seconds and source_radius_seconds must be supplied together"
        )
    selected_window = select_pool_window(template_groups, input_groups)
    score_pool = selected_window.pop("score")
    window_selection = "training-cycle-search"
    if source_lag_seconds is not None and source_radius_seconds is not None:
        if not -1.0 <= source_lag_seconds <= 1.0:
            raise ValueError("source_lag_seconds must be between -1 and 1")
        if not 0.03 <= source_radius_seconds <= 1.0:
            raise ValueError("source_radius_seconds must be between 0.03 and 1")
        override_score = score_pool(
            float(source_lag_seconds), float(source_radius_seconds)
        )
        selected_window = {**selected_window, **override_score}
        window_selection = "explicit-validation-setting"
    lag = float(selected_window["lagSeconds"])
    radius = float(selected_window["radiusSeconds"])
    pitch_class_counts: Counter[int] = Counter()
    for group in template_groups:
        pitch_class_counts.update({int(note["midi"]) % 12 for note in group})
    minimum_presence = max(
        2, int(math.ceil(len(template_groups) * persistent_pitch_class_share))
    )
    persistent = sorted(
        pitch_class
        for pitch_class, count in pitch_class_counts.items()
        if count >= minimum_presence
    )

    gestures: list[dict[str, Any]] = []
    for phase, group in enumerate(template_groups):
        time = float(group[0]["time"])
        pool = source_pool(
            input_groups,
            time,
            lag_seconds=lag,
            radius_seconds=radius,
        )
        if not pool:
            raise ValueError(f"No harmonic source support near training time {time:.3f}")
        selectors = []
        for note in group:
            output_midi = int(note["midi"])
            source_space_midi = output_midi - source_to_output_semitones
            anchor, rank = nearest_anchor(pool, source_space_midi)
            selectors.append(
                {
                    "sourceRank": rank,
                    "sourceRankFraction": rank / max(1, len(pool) - 1),
                    "sourceAnchorMidi": anchor,
                    "sourceDeltaSemitones": output_midi - anchor,
                    "originalPitchClass": output_midi % 12,
                    "originalMidi": output_midi,
                    "note": clean_note(note),
                }
            )
        pitch_classes = sorted({int(note["midi"]) % 12 for note in group})
        gestures.append(
            {
                "phase": phase,
                "relativeTime": rounded(time - template_start),
                "pitchClassSignature": "-".join(map(str, pitch_classes)),
                # Equal pitch-class sets can still be different pianist
                # gestures (for example a two-note dyad versus an octave-rich
                # accent).  Only genuinely identical template voicings share
                # a destination pitch-class decision.
                "equivalenceSignature": "-".join(
                    str(int(note["midi"])) for note in group
                ),
                "pitchClasses": pitch_classes,
                "velocity": rounded(
                    median(finite(note.get("velocity"), 0.70) for note in group)
                ),
                "selectors": selectors,
            }
        )

    profile = {
        "schema": "polymath-pianist-cyclic-arpeggio-profile-v1",
        "id": profile_id,
        "enabled": True,
        "training": {
            "songId": song_id,
            "method": "source-ranked cyclic pianist texture transfer",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
        },
        "cycle": {
            "templateStartSeconds": template_start,
            "durationSeconds": cycle_duration,
            "gestureCount": len(gestures),
            "sourceToOutputSemitones": source_to_output_semitones,
            "persistentPitchClassShare": persistent_pitch_class_share,
            "persistentPitchClasses": persistent,
            "gestures": gestures,
        },
        "sourceSupport": {
            "lagSeconds": lag,
            "radiusSeconds": radius,
            "trainingMeanPitchClassF1": rounded(
                selected_window["meanPitchClassF1"]
            ),
            "trainingMeanRecall": rounded(selected_window["meanRecall"]),
            "trainingMeanPrecision": rounded(selected_window["meanPrecision"]),
            # An explicit override is a validation-tuned research setting.  The
            # destination target is still never read by ``apply_profile``, but
            # recording the selection provenance prevents us from presenting a
            # tuned validation result as an independent inference result.
            "selectionUsedDestinationTarget": (
                window_selection == "explicit-validation-setting"
            ),
            "selectionProvenance": window_selection,
            "windowSelection": window_selection,
            "topTrainingOnlyTrials": [
                {
                    key: rounded(value) if isinstance(value, float) else value
                    for key, value in trial.items()
                }
                for trial in selected_window["topTrials"]
            ],
            # This is deliberately weaker than the long-form repetition gate:
            # cyclic arpeggios may change chord content while retaining a
            # texture.  It is still strong enough to reject the next Kiss Me
            # coda cycle, where the texture itself changes.
            "minimumRepeatPitchClassF1": 0.40,
            "maximumRepeatNormalizedScore": 0.48,
            "minimumTempoWarpPitchClassF1": 0.60,
            "maximumTempoWarpNormalizedScore": 0.30,
            "repeatOffsetSearchRadiusSeconds": repeat_offset_search_radius_seconds,
        },
        "application": {
            "targetReadAtInference": False,
            "copyTemplateTiming": True,
            "useInputTempoWarp": True,
            "copyTemplateVelocityContour": True,
            "oneNotePerPredictedPitchClass": True,
            "reuseEquivalentPhasePitchClassSet": True,
            "onsetMergeWindowSeconds": 0.035,
            "snapToCandidateOnsets": True,
            "maximumCandidateOnsetSnapSeconds": 0.12,
            "minimumCandidateOnsetSnapShare": 0.90,
            "maximumMedianCandidateOnsetDistanceSeconds": 0.08,
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    return profile


def detect_cycle_offset(
    groups: list[list[dict[str, Any]]],
    *,
    template_start: float,
    duration: float,
    expected_offset: float,
    search_radius: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Locate a performed repeat near its score-predicted position.

    Human performances breathe.  Multiplying a learned cycle duration by an
    integer therefore drifts even when the musical phrase really repeats.  We
    use only the raw transcription to recover the local offset, keeping the
    destination reference unavailable at inference.
    """

    if search_radius <= 0:
        evidence = repeat_trial(
            groups,
            template_start=template_start,
            template_end=template_start + duration,
            offset=expected_offset,
            maximum_time_distance=0.35,
            gap_cost=0.85,
        )
        return expected_offset, evidence, [evidence]
    if expected_offset > 0:
        minimum = max(0.01, expected_offset - search_radius)
        maximum = expected_offset + search_radius
    else:
        minimum = expected_offset - search_radius
        maximum = min(-0.01, expected_offset + search_radius)
    best, ranked = detect_repeat(
        groups,
        template_start=template_start,
        template_end=template_start + duration,
        minimum_offset=minimum,
        maximum_offset=maximum,
        coarse_step=0.02,
        fine_step=0.005,
        maximum_time_distance=0.35,
        gap_cost=0.85,
    )
    return float(best["offsetSeconds"]), best, ranked


def monotonic_onset_matches(
    generated: list[float],
    existing: list[float],
    *,
    maximum_distance: float,
) -> list[tuple[int, int]]:
    """Match attack grids in order while allowing extra existing strikes."""

    rows, columns = len(generated) + 1, len(existing) + 1
    infinity = float("inf")
    costs = [[infinity] * columns for _ in range(rows)]
    back = [[0] * columns for _ in range(rows)]
    costs[0][0] = 0.0
    generated_gap = maximum_distance * 2.0
    existing_gap = maximum_distance * 0.20
    for i in range(1, rows):
        costs[i][0] = costs[i - 1][0] + generated_gap
        back[i][0] = 1
    for j in range(1, columns):
        costs[0][j] = costs[0][j - 1] + existing_gap
        back[0][j] = 2
    for i in range(1, rows):
        for j in range(1, columns):
            distance = abs(generated[i - 1] - existing[j - 1])
            match = (
                costs[i - 1][j - 1] + distance
                if distance <= maximum_distance
                else infinity
            )
            deletion = costs[i - 1][j] + generated_gap
            insertion = costs[i][j - 1] + existing_gap
            costs[i][j], back[i][j] = min(
                (match, 0), (deletion, 1), (insertion, 2)
            )
    i, j = len(generated), len(existing)
    matches: list[tuple[int, int]] = []
    while i or j:
        action = back[i][j]
        if i and j and action == 0:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (not j or action == 1):
            i -= 1
        else:
            j -= 1
    return list(reversed(matches))


def evaluation_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, str]:
    """Accept either a structural leap or a safe expressive improvement."""

    safety = (
        float(candidate["coverageAdjustedOccupancyAccuracy"])
        >= float(baseline["coverageAdjustedOccupancyAccuracy"])
        and float(candidate["onsetMaeSeconds"])
        <= float(baseline["onsetMaeSeconds"]) + 0.03
        and float(candidate["gestureVelocityMae"])
        <= float(baseline["gestureVelocityMae"]) + 0.02
        and float(candidate["exactKeyDurationMaeSeconds"])
        <= float(baseline["exactKeyDurationMaeSeconds"]) + 0.02
    )
    structural_leap = (
        float(candidate["coverageAdjustedPitchClassF1"])
        >= float(baseline["coverageAdjustedPitchClassF1"]) + 0.05
        and float(candidate["coverageAdjustedExactPitchClassRate"])
        >= float(baseline["coverageAdjustedExactPitchClassRate"]) + 0.05
        and float(candidate["sequenceScore"])
        <= 0.85 * float(baseline["sequenceScore"])
    )
    expressive_improvement = (
        float(candidate["coverageAdjustedPitchClassF1"])
        >= float(baseline["coverageAdjustedPitchClassF1"]) + 0.01
        and float(candidate["coverageAdjustedExactPitchClassRate"])
        >= float(baseline["coverageAdjustedExactPitchClassRate"])
        and float(candidate["coverageAdjustedOccupancyAccuracy"])
        >= float(baseline["coverageAdjustedOccupancyAccuracy"]) + 0.05
        and float(candidate["sequenceScore"])
        <= 0.85 * float(baseline["sequenceScore"])
        and float(candidate["gestureVelocityMae"])
        < float(baseline["gestureVelocityMae"])
        and float(candidate["exactKeyDurationMaeSeconds"])
        < float(baseline["exactKeyDurationMaeSeconds"])
    )
    if safety and structural_leap:
        return True, "structural-leap"
    if safety and expressive_improvement:
        return True, "safe-expressive-improvement"
    return False, "gate-not-met"


def nearest_midi_for_pc(pitch_class: int, centre: float) -> int:
    choices = [midi for midi in range(21, 109) if midi % 12 == pitch_class]
    return min(choices, key=lambda midi: (abs(midi - centre), midi))


def predicted_group(
    gesture: dict[str, Any],
    pool: list[int],
    *,
    persistent_pitch_classes: set[int],
    destination_cycle_pitch_classes: set[int],
    equivalent_pitch_classes: set[int] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    template_midis = [int(item["originalMidi"]) for item in gesture["selectors"]]
    template_centre = median(template_midis)
    for selector in gesture["selectors"]:
        rank = min(len(pool) - 1, max(0, int(selector["sourceRank"])))
        concrete = int(pool[rank]) + int(selector["sourceDeltaSemitones"])
        while concrete < 21:
            concrete += 12
        while concrete > 108:
            concrete -= 12
        entries.append(
            {
                "predictedPitchClass": concrete % 12,
                "concreteMidi": concrete,
                "originPitchClass": int(selector["originalPitchClass"]),
                "originMidi": int(selector["originalMidi"]),
                "templateNote": selector["note"],
            }
        )

    predicted_pitch_classes = {int(item["predictedPitchClass"]) for item in entries}
    for pitch_class in persistent_pitch_classes:
        if (
            pitch_class in set(gesture["pitchClasses"])
            and pitch_class in destination_cycle_pitch_classes
        ):
            predicted_pitch_classes.add(pitch_class)
    if equivalent_pitch_classes is not None:
        predicted_pitch_classes = set(equivalent_pitch_classes)

    output: list[dict[str, Any]] = []
    for pitch_class in sorted(predicted_pitch_classes):
        same_prediction = [
            item for item in entries if item["predictedPitchClass"] == pitch_class
        ]
        original_midis = [
            int(item["originMidi"])
            for item in entries
            if item["originPitchClass"] == pitch_class
        ]
        if original_midis:
            midi = (
                min(original_midis)
                if pitch_class in persistent_pitch_classes
                else min(original_midis, key=lambda value: abs(value - template_centre))
            )
        elif same_prediction:
            from_persistent_origin = [
                item
                for item in same_prediction
                if item["originPitchClass"] in persistent_pitch_classes
            ]
            chosen = max(
                from_persistent_origin or same_prediction,
                key=lambda item: item["concreteMidi"],
            ) if from_persistent_origin else min(
                same_prediction, key=lambda item: item["concreteMidi"]
            )
            midi = int(chosen["concreteMidi"])
        else:
            midi = nearest_midi_for_pc(pitch_class, template_centre)
        template_note = min(
            (item["templateNote"] for item in entries),
            key=lambda note: abs(int(note.get("midi", midi)) - midi),
        )
        output.append({"midi": midi, "templateNote": template_note})
    return output


def apply_profile(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    detection_payload: dict[str, Any],
    *,
    repeat_index: int = 1,
    expected_offset_seconds: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if repeat_index < 1:
        raise ValueError("repeat_index must be at least one")
    groups = source_groups(detection_payload)
    cycle = profile["cycle"]
    template_start = float(cycle["templateStartSeconds"])
    duration = float(cycle["durationSeconds"])
    expected_offset = (
        float(expected_offset_seconds)
        if expected_offset_seconds is not None
        else repeat_index * duration
    )
    if math.isclose(expected_offset, 0.0, abs_tol=1e-9):
        raise ValueError("expected_offset_seconds must be non-zero")
    if template_start + expected_offset < 0:
        raise ValueError("expected_offset_seconds places the cycle before time zero")
    support = profile["sourceSupport"]
    search_radius = float(support.get("repeatOffsetSearchRadiusSeconds", 0.0))
    detected_offset, repeat_evidence, offset_trials = detect_cycle_offset(
        groups,
        template_start=template_start,
        duration=duration,
        expected_offset=expected_offset,
        search_radius=search_radius,
    )
    destination_start = template_start + detected_offset
    destination_end = destination_start + duration
    onset_merge_window = float(
        profile.get("application", {}).get("onsetMergeWindowSeconds", 0.035)
    )
    tempo_anchors = repeat_tempo_anchors(
        repeat_evidence["template"],
        repeat_evidence["repeated"],
        repeat_evidence["matches"],
        template_start=template_start,
        template_end=template_start + duration,
        offset=detected_offset,
    )
    tempo_warp_requested = bool(
        profile.get("application", {}).get("useInputTempoWarp", True)
    )
    tempo_warp_evidence_passed = (
        float(repeat_evidence["meanPitchClassF1"])
        >= float(support.get("minimumTempoWarpPitchClassF1", 0.60))
        and float(repeat_evidence["normalizedScore"])
        <= float(support.get("maximumTempoWarpNormalizedScore", 0.30))
    )
    use_input_tempo_warp = tempo_warp_requested and tempo_warp_evidence_passed
    lag = float(support["lagSeconds"])
    radius = float(support["radiusSeconds"])
    destination_cycle_pool = source_pool(
        groups,
        destination_start + duration / 2.0,
        lag_seconds=lag,
        radius_seconds=duration / 2.0 + radius,
    )
    destination_cycle_pcs = {midi % 12 for midi in destination_cycle_pool}
    persistent = {int(value) for value in cycle["persistentPitchClasses"]}
    by_signature: dict[str, set[int]] = {}
    output_notes: list[dict[str, Any]] = []
    supported = 0
    predicted = 0
    empty_pools = 0
    for gesture in cycle["gestures"]:
        template_time = template_start + float(gesture["relativeTime"])
        time = (
            map_repeat_time(template_time, tempo_anchors)
            if use_input_tempo_warp
            else template_time + detected_offset
        )
        pool = source_pool(
            groups,
            time,
            lag_seconds=lag,
            radius_seconds=radius,
        )
        if not pool:
            empty_pools += 1
            continue
        signature = str(
            gesture.get("equivalenceSignature", gesture["pitchClassSignature"])
        )
        existing = by_signature.get(signature)
        generated = predicted_group(
            gesture,
            pool,
            persistent_pitch_classes=persistent,
            destination_cycle_pitch_classes=destination_cycle_pcs,
            equivalent_pitch_classes=existing,
        )
        current_pcs = {int(item["midi"]) % 12 for item in generated}
        by_signature.setdefault(signature, current_pcs)
        local_source_pcs = {midi % 12 for midi in pool}
        predicted += len(current_pcs)
        supported += len(current_pcs & local_source_pcs)
        for item in generated:
            template_note = clean_note(dict(item["templateNote"]))
            midi = int(item["midi"])
            duration_seconds = note_duration(template_note)
            # A template onset just inside the cycle can round to exactly the
            # exclusive replacement boundary.  That silently drops the final
            # pianist gesture during evaluation/playback slicing.  Keep it one
            # microsecond inside the declared cycle instead.
            safe_time = min(time, rounded(destination_end) - 0.000001)
            template_note.update(
                {
                    "midi": midi,
                    "note": midi_to_note(midi),
                    "time": rounded(safe_time),
                    "duration": rounded(duration_seconds),
                    "scoreDuration": rounded(duration_seconds),
                    "visualDuration": rounded(duration_seconds),
                    "audioDuration": rounded(duration_seconds),
                    "velocity": rounded(float(gesture["velocity"])),
                    "source": "polymath-pianist-cyclic-arpeggio-adapter",
                    "sourceInstrument": "learned_pianist_cyclic_arpeggio",
                    "generatedBy": "pianist-cyclic-arpeggio-adapter-v1",
                }
            )
            output_notes.append(template_note)

    generated_onsets = sorted({float(note["time"]) for note in output_notes})
    application = profile.get("application", {})
    snap_requested = bool(application.get("snapToCandidateOnsets", False))
    maximum_snap_distance = float(
        application.get("maximumCandidateOnsetSnapSeconds", 0.12)
    )
    existing_groups = group_onsets(normalize_notes(candidate), onset_merge_window)
    existing_onsets = [
        float(group[0]["time"])
        for group in existing_groups
        if destination_start - maximum_snap_distance
        <= float(group[0]["time"])
        <= destination_end + maximum_snap_distance
    ]
    onset_matches = monotonic_onset_matches(
        generated_onsets,
        existing_onsets,
        maximum_distance=maximum_snap_distance,
    )
    onset_distances = [
        abs(generated_onsets[left] - existing_onsets[right])
        for left, right in onset_matches
    ]
    snap_share = len(onset_matches) / max(1, len(generated_onsets))
    median_snap_distance = median(onset_distances) if onset_distances else math.inf
    snap_applied = (
        snap_requested
        and snap_share
        >= float(application.get("minimumCandidateOnsetSnapShare", 0.90))
        and median_snap_distance
        <= float(
            application.get(
                "maximumMedianCandidateOnsetDistanceSeconds", 0.08
            )
        )
    )
    if snap_applied:
        snap_map = {
            generated_onsets[left]: existing_onsets[right]
            for left, right in onset_matches
        }
        for note in output_notes:
            onset = float(note["time"])
            if onset in snap_map:
                note["time"] = rounded(snap_map[onset])
        generated_onsets = sorted({float(note["time"]) for note in output_notes})
    if generated_onsets:
        # Replace complete onset gestures, including an older strike that sits
        # a few milliseconds across the nominal phrase boundary.  Deriving the
        # boundary from generated attacks avoids both duplicate hammers at the
        # entrance and deletion of the next phrase at the far edge.
        replacement_start = max(
            0.0, min(generated_onsets) - onset_merge_window
        )
        replacement_end = max(generated_onsets) + onset_merge_window + 0.000001
    else:
        replacement_start = max(0.0, destination_start - onset_merge_window)
        replacement_end = destination_end + onset_merge_window

    template_count = sum(
        1
        for group in groups
        if template_start <= float(group[0]["time"]) < template_start + duration
    )
    destination_count = sum(
        1
        for group in groups
        if destination_start <= float(group[0]["time"]) < destination_end
    )
    onset_count_ratio = destination_count / max(1, template_count)
    support_share = supported / max(1, predicted)
    confidence_passed = (
        empty_pools == 0
        and 0.55 <= onset_count_ratio <= 1.80
        and support_share >= 0.55
        and float(repeat_evidence["meanPitchClassF1"])
        >= float(support.get("minimumRepeatPitchClassF1", 0.40))
        and float(repeat_evidence["normalizedScore"])
        <= float(support.get("maximumRepeatNormalizedScore", 0.48))
        and len(output_notes) >= len(cycle["gestures"])
    )
    retained = [
        dict(note)
        for note in candidate.get("notes") or []
        if not (
            replacement_start
            <= finite(note.get("time", note.get("startTime")), -1)
            < replacement_end
        )
    ]
    output = copy.deepcopy(candidate)
    output["notes"] = sorted(
        retained + output_notes if confidence_passed else list(candidate.get("notes") or []),
        key=lambda note: (finite(note.get("time"), 0.0), int(note.get("midi", 0))),
    )
    diagnostics = {
        "applied": confidence_passed,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "repeatIndex": repeat_index,
        "expectedOffsetSource": (
            "explicit" if expected_offset_seconds is not None else "cycle-multiple"
        ),
        "repeatDirection": "forward" if detected_offset > 0 else "reverse",
        "expectedOffsetSeconds": rounded(expected_offset),
        "detectedOffsetSeconds": rounded(detected_offset),
        "offsetCorrectionSeconds": rounded(detected_offset - expected_offset),
        "repeatOffsetSearchRadiusSeconds": rounded(search_radius),
        "destinationStartSeconds": rounded(destination_start),
        "destinationEndSeconds": rounded(destination_end),
        "onsetMergeWindowSeconds": rounded(onset_merge_window),
        "inputTempoWarpRequested": tempo_warp_requested,
        "inputTempoWarpEvidencePassed": tempo_warp_evidence_passed,
        "inputTempoWarpApplied": use_input_tempo_warp,
        "inputTempoAnchorCount": len(tempo_anchors),
        "candidateOnsetSnapRequested": snap_requested,
        "candidateOnsetSnapApplied": snap_applied,
        "candidateOnsetSnapMatches": len(onset_matches),
        "candidateOnsetSnapShare": rounded(snap_share),
        "candidateOnsetMedianDistanceSeconds": (
            rounded(median_snap_distance)
            if math.isfinite(median_snap_distance)
            else None
        ),
        "replacementStartSeconds": rounded(replacement_start),
        "replacementEndSeconds": rounded(replacement_end),
        "sourceTemplateOnsets": template_count,
        "sourceDestinationOnsets": destination_count,
        "sourceOnsetCountRatio": rounded(onset_count_ratio),
        "predictedPitchClasses": predicted,
        "sourceSupportedPitchClasses": supported,
        "sourcePitchClassSupportShare": rounded(support_share),
        "sourceRepeatPitchClassF1": rounded(
            repeat_evidence["meanPitchClassF1"]
        ),
        "sourceRepeatNormalizedScore": rounded(
            repeat_evidence["normalizedScore"]
        ),
        "runnerUpOffsets": [
            {
                "offsetSeconds": rounded(row["offsetSeconds"]),
                "normalizedScore": rounded(row["normalizedScore"]),
            }
            for row in offset_trials[:5]
        ],
        "emptySourcePools": empty_pools,
        "generatedNotes": len(output_notes) if confidence_passed else 0,
        "retainedNotes": len(retained) if confidence_passed else len(candidate.get("notes") or []),
        "targetReadAtInference": False,
        "confidencePassed": confidence_passed,
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistCyclicArpeggioGrammar"] = diagnostics
    arrangement["outputNoteCount"] = len(output["notes"])
    output["arrangementProfile"] = (
        f"{candidate.get('arrangementProfile', 'piano')}-cyclic-arpeggio-v1"
    )
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument("--detection-input", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--template-start", type=float, required=True)
    parser.add_argument("--cycle-duration", type=float, required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--expected-repeat-offset-seconds", type=float)
    parser.add_argument("--source-lag-seconds", type=float)
    parser.add_argument("--source-radius-seconds", type=float)
    parser.add_argument(
        "--repeat-offset-search-radius-seconds", type=float, default=0.20
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    row = next(
        (item for item in manifest.get("songs") or [] if item.get("id") == args.song_id),
        None,
    )
    if row is None:
        raise ValueError(f"Song {args.song_id!r} was not found in the manifest")
    reference, _, _ = load_song_sequences(row, 0.035)
    detection_path = Path(args.detection_input).resolve()
    detection_payload = json.loads(detection_path.read_text(encoding="utf-8-sig"))
    detection_groups = source_groups(detection_payload)
    profile = fit_profile(
        reference,
        detection_groups,
        profile_id=args.profile_id,
        song_id=args.song_id,
        template_start=args.template_start,
        cycle_duration=args.cycle_duration,
        source_to_output_semitones=int(row.get("referenceTransposeSemitones") or 0),
        source_lag_seconds=args.source_lag_seconds,
        source_radius_seconds=args.source_radius_seconds,
        repeat_offset_search_radius_seconds=(
            args.repeat_offset_search_radius_seconds
        ),
    )
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    input_path = Path(args.input_candidate).resolve()
    input_payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_profile(
        input_payload,
        profile,
        detection_payload,
        repeat_index=args.repeat_index,
        expected_offset_seconds=args.expected_repeat_offset_seconds,
    )
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    input_row = {**row, "candidate": str(input_path)}
    _, baseline_groups, _ = load_song_sequences(input_row, 0.035)
    output_row = {**row, "candidate": str(output_path)}
    _, output_groups, _ = load_song_sequences(output_row, 0.035)
    start = float(diagnostics["replacementStartSeconds"])
    end = float(diagnostics["replacementEndSeconds"])
    held_baseline = evaluation(
        slice_groups(reference, start, end), slice_groups(baseline_groups, start, end)
    )
    held_candidate = evaluation(
        slice_groups(reference, start, end), slice_groups(output_groups, start, end)
    )
    metrics_passed, decision_basis = evaluation_gate(
        held_baseline, held_candidate
    )
    passed = diagnostics["applied"] and metrics_passed
    report = {
        "schema": "polymath-pianist-cyclic-arpeggio-training-report-v1",
        "evidenceBoundary": (
            "The declared first cycle is supervised training data. The repeated "
            "cycle is read only after generation for evaluation. Application uses "
            "the source transcription and fitted profile, never destination labels. "
            "This is same-song research, not cross-song proof."
        ),
        "profile": str(profile_path),
        "inputCandidate": str(input_path),
        "detectionInput": str(detection_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutCycleEvaluation": {
            "baseline": held_baseline,
            "cyclicCandidate": held_candidate,
        },
        "evaluationGate": {
            "passed": passed,
            "basis": decision_basis,
        },
        "trustedWindowEvaluation": {
            "baseline": evaluation(reference, baseline_groups),
            "cyclicCandidate": evaluation(reference, output_groups),
        },
        "decision": (
            "LISTENING_CANDIDATE_RESEARCH_ONLY"
            if passed
            else "NO_CHANGE_CONFIDENCE_GUARD"
            if not diagnostics["applied"]
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
