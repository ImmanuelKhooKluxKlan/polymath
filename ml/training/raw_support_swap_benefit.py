"""Inference-safe proposals for post-arranger accompaniment swaps.

The phase-47 raw-support selector answers whether one separated-stem event
resembles a pianist pitch class.  That is useful evidence, but it is not the
decision made by the decoder.  The decoder must decide whether replacing one
specific arranged left-hand note with one specific source proposal is safer.

This module owns that narrower contract.  It deliberately has no access to a
reference score.  Reference notes are used only by the offline fitting script
to label the proposals returned here.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

try:  # Package and direct-script execution.
    from server.piano_arranger_adapter import (
        HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
        instrument_family,
        normalize_source_notes,
        selection_feature_rows,
        selection_scores,
    )
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SERVER_ROOT = REPO_ROOT / "server"
    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))
    from piano_arranger_adapter import (  # type: ignore
        HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
        instrument_family,
        normalize_source_notes,
        selection_feature_rows,
        selection_scores,
    )

try:
    from .apply_raw_support_gesture_swap import (
        best_by_pitch_class,
        note_midi,
        note_time,
        onset_groups,
        raw_window,
    )
    from .apply_raw_support_recovery import fold_midi, selector_model
except ImportError:  # pragma: no cover
    from apply_raw_support_gesture_swap import (  # type: ignore
        best_by_pitch_class,
        note_midi,
        note_time,
        onset_groups,
        raw_window,
    )
    from apply_raw_support_recovery import fold_midi, selector_model  # type: ignore


SOURCE_CONTEXT_FEATURES = (
    "onset_cluster",
    "local_density",
    "cross_family_onset_support",
    "cross_family_pitch_class_support",
    "local_duration_percentile",
    "local_velocity_percentile",
    "same_instrument_pitch_class_recurrence",
    "is_local_onset_lowest",
    "is_local_onset_highest",
    "local_onset_pitch_percentile",
    "local_onset_unique_pitch_count",
    "previous_same_pitch_gap_log",
    "next_same_pitch_gap_log",
    "wide_pitch_class_support",
    "wide_pitch_class_rank",
    "wide_pitch_class_dominance",
    "estimated_chord_member_root",
    "estimated_chord_member_third",
    "estimated_chord_member_fifth",
    "estimated_chord_nonmember",
    "estimated_chord_confidence",
    "estimated_bass_root_match",
)

SWAP_BENEFIT_FEATURE_NAMES = (
    "bias",
    "proposal_selector_probability",
    "incumbent_selector_probability",
    "selector_probability_gain",
    "source_time_delta_signed",
    "source_time_delta_absolute",
    "source_velocity",
    "source_duration_log",
    "source_midi_centered",
    "incumbent_midi_centered",
    "target_midi_centered",
    "target_minus_incumbent",
    "target_minus_incumbent_absolute",
    "register_shift_octaves",
    "candidate_velocity",
    "candidate_duration_log",
    "gesture_size",
    "gesture_pitch_span",
    "incumbent_pitch_percentile",
    "incumbent_is_lowest",
    "incumbent_is_highest",
    "role_bass",
    "role_harmony",
    "family_piano",
    "family_bass",
    "family_guitar",
    "family_lead",
    "family_pad",
    "family_other",
    "candidate_minus_source_register_median",
    "target_minus_gesture_median",
    "proposal_probability_rank",
    "incumbent_has_source_support",
    "incumbent_source_time_distance",
) + tuple(f"source_{name}" for name in SOURCE_CONTEXT_FEATURES) + tuple(
    f"proposal_interval_pc_{interval:02d}" for interval in range(12)
)


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _duration_log(value: Any) -> float:
    return math.log1p(max(0.0, _finite(value, 0.35))) / math.log(7.0)


def _median(values: Iterable[float], fallback: float) -> float:
    kept = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not kept:
        return fallback
    middle = len(kept) // 2
    return kept[middle] if len(kept) % 2 else (kept[middle - 1] + kept[middle]) / 2.0


def score_feature_rows(rows: list[list[float]], model: dict[str, Any]) -> list[float]:
    """Score the small serialized MLP/logistic contract without NumPy."""

    if not rows:
        return []
    feature_names = tuple(str(value) for value in model.get("featureNames") or [])
    if feature_names != SWAP_BENEFIT_FEATURE_NAMES:
        raise ValueError("Swap-benefit model has an incompatible feature contract.")
    means = [float(value) for value in model.get("means") or []]
    scales = [max(1e-9, float(value)) for value in model.get("scales") or []]
    if len(means) != len(feature_names) or len(scales) != len(feature_names):
        raise ValueError("Swap-benefit model has incompatible feature scaling.")
    standardized = [
        [max(-8.0, min(8.0, (value - mean) / scale)) for value, mean, scale in zip(row, means, scales)]
        for row in rows
    ]
    model_type = str(model.get("type") or "")
    if model_type == "standardized-logistic-swap-benefit-v1":
        weights = [float(value) for value in model.get("weights") or []]
        if len(weights) != len(feature_names):
            raise ValueError("Swap-benefit logistic weights have incompatible dimensions.")
        bias = float(model.get("bias", 0.0))
        logits = [bias + sum(value * weight for value, weight in zip(row, weights)) for row in standardized]
    elif model_type == "standardized-mlp-swap-benefit-v1":
        hidden_weights = [[float(value) for value in row] for row in model.get("hiddenWeights") or []]
        hidden_biases = [float(value) for value in model.get("hiddenBiases") or []]
        output_weights = [float(value) for value in model.get("outputWeights") or []]
        if len(hidden_weights) != len(feature_names):
            raise ValueError("Swap-benefit MLP has incompatible input dimensions.")
        hidden_size = len(hidden_biases)
        if hidden_size < 1 or len(output_weights) != hidden_size:
            raise ValueError("Swap-benefit MLP has incompatible hidden dimensions.")
        if any(len(row) != hidden_size for row in hidden_weights):
            raise ValueError("Swap-benefit MLP hidden rows have inconsistent dimensions.")
        output_bias = float(model.get("outputBias", 0.0))
        logits = []
        for row in standardized:
            hidden = [
                math.tanh(
                    hidden_biases[unit]
                    + sum(row[index] * hidden_weights[index][unit] for index in range(len(row)))
                )
                for unit in range(hidden_size)
            ]
            logits.append(output_bias + sum(value * weight for value, weight in zip(hidden, output_weights)))
    else:
        raise ValueError(f"Unsupported swap-benefit model type: {model_type!r}")
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))) for logit in logits]


def generate_swap_proposals(
    candidate: dict[str, Any],
    source: dict[str, Any],
    selector_profile: dict[str, Any],
    *,
    source_radius: float = 0.18,
    minimum_source_midi: int = 0,
    maximum_source_midi: int = 59,
    minimum_output_midi: int = 33,
    maximum_output_midi: int = 71,
    register_shifts: tuple[int, ...] = (0, 12),
    source_families: set[str] | None = None,
    replace_hands: set[str] | None = None,
    replace_roles: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return every legal swap and its inference-safe feature vector."""

    candidate_notes = candidate.get("notes")
    if not isinstance(candidate_notes, list):
        raise ValueError("Candidate has no notes list.")
    raw_notes = normalize_source_notes(source.get("notes") or [])
    selector_scores = selection_scores(
        raw_notes, {"selectionModel": selector_model(selector_profile)}
    )
    source_rows = selection_feature_rows(
        raw_notes, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
    )
    source_feature_indices = {
        name: HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES.index(name)
        for name in SOURCE_CONTEXT_FEATURES
    }
    allowed_families = (
        {str(value).strip().lower() for value in source_families}
        if source_families
        else None
    )
    allowed_hands = (
        {str(value).strip().lower() for value in replace_hands}
        if replace_hands
        else {"left"}
    )
    allowed_roles = (
        {str(value).strip().lower() for value in replace_roles}
        if replace_roles
        else {"bass", "harmony"}
    )
    usable_global_indices = [
        index
        for index, note in enumerate(raw_notes)
        if instrument_family(str(note.get("instrument") or "")) != "voice"
        and (
            allowed_families is None
            or instrument_family(str(note.get("instrument") or "")) in allowed_families
        )
        and minimum_source_midi <= int(note["midi"]) <= maximum_source_midi
    ]
    usable_notes = [raw_notes[index] for index in usable_global_indices]
    usable_scores = [float(selector_scores[index]) for index in usable_global_indices]
    usable_rows = [source_rows[index] for index in usable_global_indices]
    usable_times = [float(note["time"]) for note in usable_notes]

    candidate_register = _median(
        (
            float(note_midi(note))
            for note in candidate_notes
            if str(note.get("hand") or "").lower() in allowed_hands
            and str(note.get("arrangementRole") or "").lower() in allowed_roles
        ),
        48.0,
    )
    source_register = _median((float(note["midi"]) for note in usable_notes), 48.0)
    register_gap = max(-2.0, min(2.0, (candidate_register - source_register) / 24.0))
    proposals: list[dict[str, Any]] = []
    groups = onset_groups(candidate_notes)
    for group_number, group in enumerate(groups):
        if not group:
            continue
        group_time = note_time(candidate_notes[group[0]])
        window_indices = raw_window(usable_notes, usable_times, group_time, source_radius)
        raw_by_pc = best_by_pitch_class(window_indices, usable_notes, usable_scores)
        if not raw_by_pc:
            continue
        current_pcs = {note_midi(candidate_notes[index]) % 12 for index in group}
        replaceable = [
            index
            for index in group
            if str(candidate_notes[index].get("hand") or "").lower() in allowed_hands
            and str(candidate_notes[index].get("arrangementRole") or "").lower() in allowed_roles
        ]
        if not replaceable:
            continue
        group_midis = sorted(note_midi(candidate_notes[index]) for index in group)
        group_low = group_midis[0]
        group_high = group_midis[-1]
        group_median = _median((float(value) for value in group_midis), float(group_low))
        group_span = min(2.0, (group_high - group_low) / 24.0)
        ranked_source = sorted(
            raw_by_pc.values(), key=lambda index: (usable_scores[index], -abs(usable_times[index] - group_time)), reverse=True
        )
        source_rank = {
            index: 1.0 - position / max(1, len(ranked_source) - 1)
            for position, index in enumerate(ranked_source)
        }
        for source_index in ranked_source:
            source_note = usable_notes[source_index]
            proposal_pc = int(source_note["midi"]) % 12
            if proposal_pc in current_pcs:
                continue
            source_family = instrument_family(str(source_note.get("instrument") or ""))
            source_score = float(usable_scores[source_index])
            source_context = [
                float(usable_rows[source_index][source_feature_indices[name]])
                for name in SOURCE_CONTEXT_FEATURES
            ]
            for incumbent_index in replaceable:
                incumbent = candidate_notes[incumbent_index]
                incumbent_midi = note_midi(incumbent)
                incumbent_source_index = raw_by_pc.get(incumbent_midi % 12)
                incumbent_score = (
                    float(usable_scores[incumbent_source_index])
                    if incumbent_source_index is not None
                    else 0.0
                )
                incumbent_time_distance = (
                    abs(float(usable_notes[incumbent_source_index]["time"]) - group_time)
                    / max(0.01, source_radius)
                    if incumbent_source_index is not None
                    else 1.5
                )
                occupied_midis = {
                    note_midi(candidate_notes[index])
                    for index in group
                    if index != incumbent_index
                }
                seen_targets: set[int] = set()
                for register_shift in register_shifts:
                    target_midi = fold_midi(
                        int(source_note["midi"]),
                        minimum_output_midi,
                        maximum_output_midi,
                        int(register_shift),
                    )
                    if target_midi in seen_targets:
                        continue
                    seen_targets.add(target_midi)
                    if target_midi in occupied_midis or target_midi % 12 in {
                        midi % 12 for midi in occupied_midis
                    }:
                        continue
                    source_delta = (float(source_note["time"]) - group_time) / max(0.01, source_radius)
                    incumbent_position = (
                        (incumbent_midi - group_low) / max(1, group_high - group_low)
                        if group_high > group_low
                        else 0.5
                    )
                    family_flags = [
                        float(source_family == family)
                        for family in ("piano", "bass", "guitar", "lead", "pad", "other")
                    ]
                    interval_pc = (target_midi - incumbent_midi) % 12
                    features = [
                        1.0,
                        source_score,
                        incumbent_score,
                        source_score - incumbent_score,
                        max(-1.5, min(1.5, source_delta)),
                        min(1.5, abs(source_delta)),
                        _finite(source_note.get("velocity"), 0.72),
                        _duration_log(source_note.get("duration")),
                        (int(source_note["midi"]) - 48.0) / 24.0,
                        (incumbent_midi - 48.0) / 24.0,
                        (target_midi - 48.0) / 24.0,
                        max(-2.0, min(2.0, (target_midi - incumbent_midi) / 24.0)),
                        min(2.0, abs(target_midi - incumbent_midi) / 24.0),
                        int(register_shift) / 12.0,
                        _finite(incumbent.get("velocity"), 0.72),
                        _duration_log(incumbent.get("duration")),
                        min(1.5, len(group) / 8.0),
                        group_span,
                        incumbent_position,
                        float(incumbent_midi == group_low),
                        float(incumbent_midi == group_high),
                        float(str(incumbent.get("arrangementRole") or "").lower() == "bass"),
                        float(str(incumbent.get("arrangementRole") or "").lower() == "harmony"),
                        *family_flags,
                        register_gap,
                        max(-2.0, min(2.0, (target_midi - group_median) / 24.0)),
                        float(source_rank[source_index]),
                        float(incumbent_source_index is not None),
                        min(1.5, incumbent_time_distance),
                        *source_context,
                        *[float(interval_pc == interval) for interval in range(12)],
                    ]
                    if len(features) != len(SWAP_BENEFIT_FEATURE_NAMES):
                        raise AssertionError("Swap-benefit feature contract drifted.")
                    proposals.append(
                        {
                            "groupNumber": group_number,
                            "groupTime": group_time,
                            "groupIndices": list(group),
                            "incumbentIndex": incumbent_index,
                            "incumbentMidi": incumbent_midi,
                            "sourceUsableIndex": source_index,
                            "sourceGlobalIndex": usable_global_indices[source_index],
                            "sourceIndex": source_note.get("sourceIndex"),
                            "sourceMidi": int(source_note["midi"]),
                            "sourceInstrument": source_note.get("instrument"),
                            "sourceFamily": source_family,
                            "sourceTime": float(source_note["time"]),
                            "sourceProbability": source_score,
                            "incumbentProbability": incumbent_score,
                            "probabilityGain": source_score - incumbent_score,
                            "registerShift": int(register_shift),
                            "targetMidi": target_midi,
                            "features": features,
                        }
                    )
    diagnostics = {
        "candidateNotes": len(candidate_notes),
        "sourceNotes": len(raw_notes),
        "usableSourceNotes": len(usable_notes),
        "gestureCount": len(groups),
        "proposalCount": len(proposals),
        "sourceRadiusSeconds": source_radius,
        "registerShifts": list(register_shifts),
        "sourceMidiRange": [minimum_source_midi, maximum_source_midi],
        "outputMidiRange": [minimum_output_midi, maximum_output_midi],
        "sourceFamilies": sorted(allowed_families or []),
        "replaceHands": sorted(allowed_hands),
        "replaceRoles": sorted(allowed_roles),
    }
    return proposals, diagnostics


def apply_scored_proposals(
    candidate: dict[str, Any],
    proposals: list[dict[str, Any]],
    scores: list[float],
    benefit_profile: dict[str, Any],
    generation: dict[str, Any],
    *,
    threshold: float,
    minimum_selector_probability: float,
    minimum_probability_gain: float,
    maximum_replacements_per_gesture: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply cached proposal scores without recomputing source features."""

    if len(proposals) != len(scores):
        raise ValueError("Every swap proposal must have exactly one benefit score.")
    output = copy.deepcopy(candidate)
    by_group: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    rejected_selector = rejected_gain = rejected_model = 0
    for score, proposal in zip(scores, proposals):
        if float(proposal["sourceProbability"]) < minimum_selector_probability:
            rejected_selector += 1
            continue
        if float(proposal["probabilityGain"]) < minimum_probability_gain:
            rejected_gain += 1
            continue
        if score < threshold:
            rejected_model += 1
            continue
        by_group.setdefault(int(proposal["groupNumber"]), []).append((score, proposal))

    notes = output.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Candidate has no notes list.")
    replacements = 0
    replaced_roles: Counter[str] = Counter()
    used_incumbents: set[int] = set()
    for group_number in sorted(by_group):
        accepted = 0
        for score, proposal in sorted(
            by_group[group_number],
            key=lambda item: (
                item[0],
                float(item[1]["probabilityGain"]),
                float(item[1]["sourceProbability"]),
                -abs(float(item[1]["sourceTime"]) - float(item[1]["groupTime"])),
            ),
            reverse=True,
        ):
            if accepted >= maximum_replacements_per_gesture:
                break
            incumbent_index = int(proposal["incumbentIndex"])
            if incumbent_index in used_incumbents:
                continue
            group_indices = [int(value) for value in proposal["groupIndices"]]
            occupied = {
                note_midi(notes[index])
                for index in group_indices
                if index != incumbent_index
            }
            target_midi = int(proposal["targetMidi"])
            if target_midi in occupied or target_midi % 12 in {midi % 12 for midi in occupied}:
                continue
            incumbent = notes[incumbent_index]
            original_midi = note_midi(incumbent)
            incumbent["rawSupportBenefitOriginalMidi"] = original_midi
            incumbent["rawSupportBenefitProbability"] = round(float(score), 6)
            incumbent["rawSupportBenefitSelectorProbability"] = round(
                float(proposal["sourceProbability"]), 6
            )
            incumbent["rawSupportBenefitIncumbentProbability"] = round(
                float(proposal["incumbentProbability"]), 6
            )
            incumbent["rawSupportBenefitProfile"] = benefit_profile.get("id")
            incumbent["rawSupportBenefitSourceIndex"] = proposal.get("sourceIndex")
            incumbent["midi"] = target_midi
            if "pitch" in incumbent:
                incumbent["pitch"] = target_midi
            incumbent["sourceInstrument"] = proposal.get("sourceInstrument")
            incumbent["sourceMidiBeforeArrangement"] = int(proposal["sourceMidi"])
            replaced_roles[str(incumbent.get("arrangementRole") or "unknown")] += 1
            used_incumbents.add(incumbent_index)
            replacements += 1
            accepted += 1

    diagnostics = {
        **generation,
        "benefitProfile": benefit_profile.get("id"),
        "threshold": threshold,
        "minimumSelectorProbability": minimum_selector_probability,
        "minimumProbabilityGain": minimum_probability_gain,
        "maximumReplacementsPerGesture": maximum_replacements_per_gesture,
        "replacements": replacements,
        "replacedRoles": dict(replaced_roles),
        "rejectedSelectorProbability": rejected_selector,
        "rejectedProbabilityGain": rejected_gain,
        "rejectedBenefitModel": rejected_model,
        "frozenProperties": ["onset", "gesture-size", "duration", "velocity"],
        "warning": "Private research operator; promotion requires unseen-song transfer and listening evidence.",
    }
    output.setdefault("diagnostics", {})["rawSupportSwapBenefit"] = diagnostics
    return output, diagnostics


def apply_benefit_swaps(
    candidate: dict[str, Any],
    source: dict[str, Any],
    selector_profile: dict[str, Any],
    benefit_profile: dict[str, Any],
    *,
    threshold: float,
    source_radius: float,
    minimum_selector_probability: float,
    minimum_probability_gain: float,
    maximum_replacements_per_gesture: int,
    minimum_source_midi: int,
    maximum_source_midi: int,
    minimum_output_midi: int,
    maximum_output_midi: int,
    register_shifts: tuple[int, ...],
    source_families: set[str] | None = None,
    replace_hands: set[str] | None = None,
    replace_roles: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only the highest-scoring legal proposals in each frozen gesture."""

    proposals, generation = generate_swap_proposals(
        candidate,
        source,
        selector_profile,
        source_radius=source_radius,
        minimum_source_midi=minimum_source_midi,
        maximum_source_midi=maximum_source_midi,
        minimum_output_midi=minimum_output_midi,
        maximum_output_midi=maximum_output_midi,
        register_shifts=register_shifts,
        source_families=source_families,
        replace_hands=replace_hands,
        replace_roles=replace_roles,
    )
    model = benefit_profile.get("model") or benefit_profile.get("swapBenefitModel")
    if not isinstance(model, dict):
        raise ValueError("Benefit profile has no model.")
    scores = score_feature_rows([item["features"] for item in proposals], model)
    return apply_scored_proposals(
        candidate,
        proposals,
        scores,
        benefit_profile,
        generation,
        threshold=threshold,
        minimum_selector_probability=minimum_selector_probability,
        minimum_probability_gain=minimum_probability_gain,
        maximum_replacements_per_gesture=maximum_replacements_per_gesture,
    )


def model_probabilities_numpy(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Vectorized scorer used only by offline reports/tests."""

    return np.asarray(score_feature_rows(features.tolist(), model), dtype=np.float64)


__all__ = [
    "SOURCE_CONTEXT_FEATURES",
    "SWAP_BENEFIT_FEATURE_NAMES",
    "apply_benefit_swaps",
    "apply_scored_proposals",
    "generate_swap_proposals",
    "model_probabilities_numpy",
    "score_feature_rows",
]
