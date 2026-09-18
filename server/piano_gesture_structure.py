"""Inference-safe note scoring inside an already selected piano gesture.

The upstream selector decides *when* the piano should strike.  This module
answers a narrower question: when that strike contains several simultaneous
notes, which pitches look like stable authored chord tones rather than dense
full-mix leakage?  A conservative policy can remove low-confidence extras but
must never delete an onset or invent a target/reference coordinate.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from statistics import median
from typing import Any


GESTURE_NOTE_FEATURE_NAMES = (
    "bias",
    "midi_centered",
    "midi_squared",
    "source_midi_delta",
    "velocity",
    "source_velocity",
    "duration_expression",
    "selection_probability",
    "left_hand_selection_probability",
    "left_hand_onset_probability",
    "chord_completion_score",
    "is_melody",
    "is_bass_role",
    "is_harmony",
    "is_voice_source",
    "is_guitar_source",
    "is_bass_source",
    "is_piano_source",
    "gesture_size",
    "gesture_size_2",
    "gesture_size_3",
    "gesture_size_4_plus",
    "pitch_rank",
    "is_lowest",
    "is_highest",
    "distance_from_median",
    "interval_from_lowest",
    "interval_from_highest",
    "same_pc_inside_gesture",
    "consonant_neighbor_share",
    "same_midi_previous",
    "same_midi_next",
    "same_pc_previous",
    "same_pc_next",
    "local_pc_support",
    "previous_gap_expression",
    "next_gap_expression",
    "is_cyclic_texture_note",
    "texture_phase_0",
    "texture_phase_1",
    "texture_phase_2",
    "texture_phase_3",
    "texture_phase_other",
    "texture_action_upper",
    "texture_action_inner",
    "texture_action_preserve",
    "texture_action_other",
)


def _finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _role(note: dict[str, Any]) -> str:
    return str(note.get("arrangementRole") or note.get("role") or "").lower()


def _family(note: dict[str, Any]) -> str:
    value = str(note.get("sourceInstrument") or note.get("instrument") or "").lower()
    if any(token in value for token in ("voice", "vocal", "choir")):
        return "voice"
    if "guitar" in value:
        return "guitar"
    if "bass" in value:
        return "bass"
    if any(token in value for token in ("piano", "keyboard")):
        return "piano"
    return "other"


def gesture_groups(
    notes: list[dict[str, Any]], onset_window_seconds: float = 0.035
) -> list[list[dict[str, Any]]]:
    """Group simultaneous notes, preferring the renderer's immutable id."""

    if notes and all(note.get("gestureDynamicGroup") is not None for note in notes):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for note in notes:
            grouped[int(note["gestureDynamicGroup"])].append(note)
        return sorted(
            (
                sorted(group, key=lambda item: (_finite(item.get("time"), 0.0), int(item.get("midi", 60))))
                for group in grouped.values()
            ),
            key=lambda group: (_finite(group[0].get("time"), 0.0), int(group[0].get("midi", 60))),
        )

    ordered = sorted(
        notes,
        key=lambda note: (_finite(note.get("time"), 0.0), int(note.get("midi", 60))),
    )
    groups: list[list[dict[str, Any]]] = []
    for note in ordered:
        if (
            not groups
            or _finite(note.get("time"), 0.0)
            - _finite(groups[-1][0].get("time"), 0.0)
            > onset_window_seconds
        ):
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def gesture_note_feature_rows(
    groups: list[list[dict[str, Any]]], local_radius_seconds: float = 2.0
) -> list[list[dict[str, float]]]:
    """Return one feature mapping per note using candidate-only evidence."""

    if not groups:
        return []
    times = [_finite(group[0].get("time"), 0.0) for group in groups]
    midi_sets = [
        {int(round(_finite(note.get("midi"), 60.0))) for note in group}
        for group in groups
    ]
    pc_sets = [{midi % 12 for midi in midis} for midis in midi_sets]
    output: list[list[dict[str, float]]] = []
    for group_index, group in enumerate(groups):
        midis = [int(round(_finite(note.get("midi"), 60.0))) for note in group]
        unique_midis = sorted(set(midis))
        size = len(unique_midis)
        low = unique_midis[0]
        high = unique_midis[-1]
        center = median(unique_midis)
        previous_midis = midi_sets[group_index - 1] if group_index else set()
        next_midis = midi_sets[group_index + 1] if group_index + 1 < len(groups) else set()
        previous_pcs = pc_sets[group_index - 1] if group_index else set()
        next_pcs = pc_sets[group_index + 1] if group_index + 1 < len(groups) else set()
        left = bisect.bisect_left(times, times[group_index] - local_radius_seconds)
        right = bisect.bisect_right(times, times[group_index] + local_radius_seconds)
        local_pc_counts: dict[int, int] = defaultdict(int)
        for neighbor_index in range(left, right):
            if neighbor_index == group_index:
                continue
            for pitch_class in pc_sets[neighbor_index]:
                local_pc_counts[pitch_class] += 1
        local_denominator = max(1, right - left - 1)
        previous_gap = (
            times[group_index] - times[group_index - 1] if group_index else 1.0
        )
        next_gap = (
            times[group_index + 1] - times[group_index]
            if group_index + 1 < len(groups)
            else 1.0
        )
        rows: list[dict[str, float]] = []
        for note, midi in zip(group, midis):
            role = _role(note)
            family = _family(note)
            raw_texture_phase = note.get("pianistTexturePhase")
            try:
                texture_phase = int(raw_texture_phase)
            except (TypeError, ValueError):
                texture_phase = None
            texture_action = str(note.get("pianistTextureAction") or "").lower()
            is_cyclic_texture = bool(
                texture_phase is not None
                or texture_action
                or str(note.get("generatedBy") or "")
                == "source-supported-cyclic-harmony-v1"
            )
            source_midi = int(
                round(_finite(note.get("sourceMidiBeforeArrangement"), midi))
            )
            velocity = _clamp(_finite(note.get("velocity"), 0.7), 0.01, 1.0)
            source_velocity = _clamp(
                _finite(note.get("sourceVelocityBeforeArrangement"), velocity),
                0.01,
                1.0,
            )
            duration = max(0.03, _finite(note.get("duration"), 0.2))
            duration_expression = _clamp(
                (math.sqrt(duration) - math.sqrt(0.08))
                / (math.sqrt(1.0) - math.sqrt(0.08)),
                0.0,
                1.0,
            )
            rank = unique_midis.index(midi) / max(1, size - 1)
            pc = midi % 12
            consonant = sum(
                1
                for other in unique_midis
                if other != midi and abs(other - midi) % 12 in {3, 4, 5, 7, 8, 9}
            ) / max(1, size - 1)
            rows.append(
                {
                    "bias": 1.0,
                    "midi_centered": _clamp((midi - 66.0) / 24.0, -1.5, 1.5),
                    "midi_squared": ((midi - 66.0) / 24.0) ** 2,
                    "source_midi_delta": _clamp((midi - source_midi) / 24.0, -2.0, 2.0),
                    "velocity": velocity,
                    "source_velocity": source_velocity,
                    "duration_expression": duration_expression,
                    "selection_probability": _clamp(_finite(note.get("selectionProbability"), 0.5), 0.0, 1.0),
                    "left_hand_selection_probability": _clamp(_finite(note.get("leftHandSelectionProbability"), 0.5), 0.0, 1.0),
                    "left_hand_onset_probability": _clamp(_finite(note.get("leftHandOnsetSelectionProbability"), 0.5), 0.0, 1.0),
                    "chord_completion_score": _clamp(_finite(note.get("leftHandChordCompletionScore"), 0.5), 0.0, 1.0),
                    "is_melody": float(role == "melody"),
                    "is_bass_role": float(role == "bass"),
                    "is_harmony": float(role == "harmony"),
                    "is_voice_source": float(family == "voice"),
                    "is_guitar_source": float(family == "guitar"),
                    "is_bass_source": float(family == "bass"),
                    "is_piano_source": float(family == "piano"),
                    "gesture_size": _clamp((size - 1) / 5.0, 0.0, 2.0),
                    "gesture_size_2": float(size == 2),
                    "gesture_size_3": float(size == 3),
                    "gesture_size_4_plus": float(size >= 4),
                    "pitch_rank": rank,
                    "is_lowest": float(midi == low),
                    "is_highest": float(midi == high),
                    "distance_from_median": _clamp(abs(midi - center) / 24.0, 0.0, 2.0),
                    "interval_from_lowest": _clamp((midi - low) / 36.0, 0.0, 2.0),
                    "interval_from_highest": _clamp((high - midi) / 36.0, 0.0, 2.0),
                    "same_pc_inside_gesture": _clamp(sum(1 for value in unique_midis if value % 12 == pc) - 1, 0.0, 3.0),
                    "consonant_neighbor_share": consonant,
                    "same_midi_previous": float(midi in previous_midis),
                    "same_midi_next": float(midi in next_midis),
                    "same_pc_previous": float(pc in previous_pcs),
                    "same_pc_next": float(pc in next_pcs),
                    "local_pc_support": _clamp(local_pc_counts[pc] / local_denominator, 0.0, 1.0),
                    "previous_gap_expression": _clamp(previous_gap, 0.0, 1.0),
                    "next_gap_expression": _clamp(next_gap, 0.0, 1.0),
                    "is_cyclic_texture_note": float(is_cyclic_texture),
                    "texture_phase_0": float(texture_phase == 0),
                    "texture_phase_1": float(texture_phase == 1),
                    "texture_phase_2": float(texture_phase == 2),
                    "texture_phase_3": float(texture_phase == 3),
                    "texture_phase_other": float(
                        texture_phase is not None and texture_phase not in {0, 1, 2, 3}
                    ),
                    "texture_action_upper": float(texture_action == "compact-upper"),
                    "texture_action_inner": float(texture_action == "compact-inner"),
                    "texture_action_preserve": float(
                        texture_action == "compact-preserve"
                    ),
                    "texture_action_other": float(
                        bool(texture_action)
                        and texture_action
                        not in {"compact-upper", "compact-inner", "compact-preserve"}
                    ),
                }
            )
        output.append(rows)
    return output


def model_probability(features: dict[str, float], model: dict[str, Any]) -> float:
    names = [str(value) for value in model.get("featureNames") or []]
    weights = [_finite(value, 0.0) for value in model.get("weights") or []]
    means = [_finite(value, 0.0) for value in model.get("means") or []]
    scales = [_finite(value, 1.0) for value in model.get("scales") or []]
    if not names or not (len(names) == len(weights) == len(means) == len(scales)):
        return 1.0
    if any(name not in features for name in names):
        return 1.0
    logit = sum(
        weight * (features[name] - mean) / max(1e-9, abs(scale))
        for name, weight, mean, scale in zip(names, weights, means, scales)
    )
    return 1.0 / (1.0 + math.exp(-_clamp(logit, -30.0, 30.0)))


def selected_note_indices(
    group: list[dict[str, Any]],
    probabilities: list[float],
    config: dict[str, Any],
) -> set[int]:
    """Choose notes conservatively while guaranteeing one key per onset."""

    keep = set(range(len(group)))
    minimum_group_size = max(2, int(config.get("minimumGroupSize", 2)))
    if len(group) < minimum_group_size or len(probabilities) != len(group):
        return keep
    threshold = _clamp(_finite(config.get("threshold"), 0.25), 0.0, 1.0)
    minimum_gap = _clamp(_finite(config.get("minimumProbabilityGap"), 0.10), 0.0, 1.0)
    maximum_removals = max(0, int(config.get("maximumRemovalsPerGesture", 1)))
    minimum_remaining = max(1, int(config.get("minimumRemainingNotes", 1)))
    preserve_melody = bool(config.get("preserveMelody", True))
    best = max(probabilities)
    removable = [
        index
        for index, probability in enumerate(probabilities)
        if probability < threshold
        and best - probability >= minimum_gap
        and not (preserve_melody and _role(group[index]) == "melody")
    ]
    for index in sorted(removable, key=lambda value: probabilities[value]):
        if len(keep) <= minimum_remaining:
            break
        if len(group) - len(keep) >= maximum_removals:
            break
        keep.remove(index)
    return keep


def prune_gesture_notes(
    notes: list[dict[str, Any]], config: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config or not config.get("enabled"):
        return notes, {"applied": False, "removedNotes": 0}
    model = config.get("selectionModel") or {}
    groups = gesture_groups(
        notes, _clamp(_finite(config.get("onsetWindowSeconds"), 0.035), 0.005, 0.08)
    )
    rows_by_group = gesture_note_feature_rows(groups)
    kept: list[dict[str, Any]] = []
    removed = 0
    changed_groups = 0
    examined_groups = 0
    for group, rows in zip(groups, rows_by_group):
        probabilities = [model_probability(row, model) for row in rows]
        selected = selected_note_indices(group, probabilities, config)
        if len(group) >= int(config.get("minimumGroupSize", 2)):
            examined_groups += 1
        if len(selected) < len(group):
            changed_groups += 1
        for index, note in enumerate(group):
            if index not in selected:
                removed += 1
                continue
            value = dict(note)
            value["gestureStructureProbability"] = round(probabilities[index], 4)
            kept.append(value)
    kept.sort(key=lambda note: (_finite(note.get("time"), 0.0), int(note.get("midi", 60))))
    return kept, {
        "applied": True,
        "profile": str(config.get("id") or "gesture-note-pruner"),
        "inputNotes": len(notes),
        "outputNotes": len(kept),
        "examinedGroups": examined_groups,
        "changedGroups": changed_groups,
        "removedNotes": removed,
        "onsetsPreserved": True,
        "inventedNotes": 0,
    }


__all__ = [
    "GESTURE_NOTE_FEATURE_NAMES",
    "gesture_groups",
    "gesture_note_feature_rows",
    "model_probability",
    "prune_gesture_notes",
    "selected_note_indices",
]
