"""Transfer pianist touch across a recurrence without replacing its notes.

This is deliberately narrower than ``train_pianist_repetition_adapter``.  A
repeated phrase may use the same pitches but breathe at different times.  In
that case replacing complete gestures damages a good transcription merely to
copy timing from the approved performance.  This adapter learns the velocity
contour of one approved occurrence, detects its recurrence from the candidate
input, and changes only the velocity shared by each destination gesture.

The approved destination is used only for evaluation.  Profiles are
same-song research artefacts until complete-song-separated validation proves
that the rule generalises.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .analyze_pianist_gesture_patterns import group_onsets, normalize_notes, pitch_set
    from .analyze_pianist_reduction_grammar import occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import evaluation, finite
    from .train_pianist_repetition_adapter import detect_repeat, slice_groups
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import group_onsets, normalize_notes, pitch_set
    from analyze_pianist_reduction_grammar import occupancy
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import evaluation, finite
    from train_pianist_repetition_adapter import detect_repeat, slice_groups


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def gesture_velocity(group: list[dict[str, Any]]) -> float:
    return median(finite(note.get("velocity"), 0.7) for note in group)


def gesture_record(group: list[dict[str, Any]], template_start: float) -> dict[str, Any]:
    return {
        "relativeTime": rounded(float(group[0]["time"]) - template_start),
        "midis": sorted({int(note["midi"]) for note in group}),
        "pitchClasses": sorted(pitch_set(group, True)),
        "occupancy": occupancy(group),
        "velocity": rounded(gesture_velocity(group)),
    }


def record_group(record: dict[str, Any], template_start: float) -> list[dict[str, Any]]:
    time = template_start + float(record["relativeTime"])
    return [
        {
            "time": time,
            "midi": midi,
            "duration": 0.2,
            "velocity": float(record["velocity"]),
            "hand": "left" if midi < 60 else "right",
        }
        for midi in record["midis"]
    ]


def fit_profile(
    reference_groups: list[list[dict[str, Any]]],
    *,
    profile_id: str,
    song_id: str,
    template_start: float,
    template_end: float,
    repeat_search_minimum: float,
    repeat_search_maximum: float,
    velocity_blend: float,
    energy_blend: float,
    minimum_energy_scale: float = 0.75,
    maximum_energy_scale: float = 1.25,
    minimum_input_pitch_f1: float = 0.90,
    maximum_input_normalized_score: float = 0.20,
) -> dict[str, Any]:
    template = slice_groups(reference_groups, template_start, template_end)
    if len(template) < 2:
        raise ValueError("A repetition dynamics template requires at least two gestures")
    profile = {
        "schema": "polymath-pianist-repetition-dynamics-profile-v1",
        "id": profile_id,
        "enabled": True,
        "training": {
            "songId": song_id,
            "method": "input-detected-recurrence-gesture-touch-transfer",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
        },
        "template": {
            "startSeconds": template_start,
            "endSeconds": template_end,
            "gestureCount": len(template),
            "gestures": [gesture_record(group, template_start) for group in template],
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
        },
        "application": {
            "targetReadAtInference": False,
            "velocityBlend": max(0.0, min(1.0, velocity_blend)),
            "energyBlend": max(0.0, min(1.0, energy_blend)),
            "minimumEnergyScale": minimum_energy_scale,
            "maximumEnergyScale": maximum_energy_scale,
            "preserveNotes": True,
            "preserveOnsets": True,
            "preserveDurations": True,
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    return profile


def apply_profile(
    candidate: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = group_onsets(normalize_notes(candidate), 0.035)
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
    confidence_passed = (
        float(best["meanPitchClassF1"])
        >= float(detection["minimumMeanPitchClassF1"])
        and float(best["normalizedScore"])
        <= float(detection["maximumNormalizedScore"])
    )

    trained = [
        record_group(record, template_start)
        for record in profile["template"]["gestures"]
    ]
    template_matches, _, _, _ = align_sequences(
        trained,
        best["template"],
        maximum_time_distance=0.55,
        gap_cost=0.85,
    )
    record_by_input_template = {
        input_index: record_index for record_index, input_index in template_matches
    }

    scale_samples = [
        gesture_velocity(best["repeated"][right])
        / max(0.05, gesture_velocity(best["template"][left]))
        for left, right in best["matches"]
    ]
    raw_energy_scale = median(scale_samples) if scale_samples else 1.0
    application = profile["application"]
    bounded_energy_scale = max(
        float(application["minimumEnergyScale"]),
        min(float(application["maximumEnergyScale"]), raw_energy_scale),
    )
    energy_blend = float(application["energyBlend"])
    energy_scale = 1.0 + energy_blend * (bounded_energy_scale - 1.0)
    velocity_blend = float(application["velocityBlend"])

    updates: dict[int, float] = {}
    changed_gestures: list[dict[str, Any]] = []
    if confidence_passed:
        for template_index, repeated_index in best["matches"]:
            record_index = record_by_input_template.get(template_index)
            if record_index is None:
                continue
            record = profile["template"]["gestures"][record_index]
            destination = best["repeated"][repeated_index]
            learned_velocity = max(
                0.05, min(1.0, float(record["velocity"]) * energy_scale)
            )
            current_velocity = gesture_velocity(destination)
            output_velocity = current_velocity + velocity_blend * (
                learned_velocity - current_velocity
            )
            for note in destination:
                updates[int(note["_payloadIndex"])] = rounded(output_velocity)
            changed_gestures.append(
                {
                    "time": rounded(float(destination[0]["time"])),
                    "templateGesture": record_index,
                    "currentVelocity": rounded(current_velocity),
                    "learnedVelocity": rounded(learned_velocity),
                    "outputVelocity": rounded(output_velocity),
                    "notes": len(destination),
                }
            )

    output = copy.deepcopy(candidate)
    for index, note in enumerate(output.get("notes") or []):
        if index in updates:
            note["velocity"] = updates[index]
            note["gestureVelocityCalibrated"] = updates[index]
            note["gestureDynamicsSource"] = "learned-pianist-recurrence"
    diagnostics = {
        "applied": confidence_passed and bool(updates),
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "detectedOffsetSeconds": rounded(best["offsetSeconds"]),
        "inputTemplateGestures": int(best["templateGestures"]),
        "inputRepeatGestures": int(best["repeatGestures"]),
        "inputMatchedGestures": int(best["matchedGestures"]),
        "inputMeanPitchClassF1": rounded(best["meanPitchClassF1"]),
        "inputNormalizedSequenceScore": rounded(best["normalizedScore"]),
        "confidencePassed": confidence_passed,
        "trainedGestureMatches": len(template_matches),
        "changedGestures": changed_gestures,
        "changedNotes": len(updates),
        "rawInputEnergyScale": rounded(raw_energy_scale),
        "appliedEnergyScale": rounded(energy_scale),
        "velocityBlend": rounded(velocity_blend),
        "targetReadAtInference": False,
        "runnerUpOffsets": [
            {
                "offsetSeconds": rounded(row["offsetSeconds"]),
                "normalizedScore": rounded(row["normalizedScore"]),
            }
            for row in ranked[:5]
        ],
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistRepetitionDynamics"] = diagnostics
    arrangement["outputNoteCount"] = len(output.get("notes") or [])
    return output, diagnostics


def dynamics_gate(
    baseline_local: dict[str, Any],
    candidate_local: dict[str, Any],
    baseline_full: dict[str, Any],
    candidate_full: dict[str, Any],
) -> tuple[bool, str]:
    structural_keys = (
        "referenceGestureRecall",
        "candidateGesturePrecision",
        "coverageAdjustedPitchClassF1",
        "coverageAdjustedExactPitchClassRate",
        "coverageAdjustedOccupancyAccuracy",
        "onsetMaeSeconds",
        "exactKeyDurationMaeSeconds",
        "sequenceScore",
    )
    structure_unchanged = all(
        abs(float(candidate_local[key]) - float(baseline_local[key])) <= 1e-6
        and abs(float(candidate_full[key]) - float(baseline_full[key])) <= 1e-6
        for key in structural_keys
    )
    local_gain = (
        float(candidate_local["gestureVelocityMae"])
        <= float(baseline_local["gestureVelocityMae"]) - 0.005
    )
    full_gain = (
        float(candidate_full["gestureVelocityMae"])
        < float(baseline_full["gestureVelocityMae"])
    )
    if structure_unchanged and local_gain and full_gain:
        return True, "safe-dynamics-refinement"
    return False, "gate-not-met"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--template-start", type=float, required=True)
    parser.add_argument("--template-end", type=float, required=True)
    parser.add_argument("--search-min-offset", type=float, required=True)
    parser.add_argument("--search-max-offset", type=float, required=True)
    parser.add_argument("--velocity-blend", type=float, default=1.0)
    parser.add_argument("--energy-blend", type=float, default=0.0)
    parser.add_argument("--minimum-energy-scale", type=float, default=0.75)
    parser.add_argument("--maximum-energy-scale", type=float, default=1.25)
    parser.add_argument("--minimum-input-pitch-f1", type=float, default=0.90)
    parser.add_argument("--maximum-input-normalized-score", type=float, default=0.20)
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
    profile = fit_profile(
        reference,
        profile_id=args.profile_id,
        song_id=args.song_id,
        template_start=args.template_start,
        template_end=args.template_end,
        repeat_search_minimum=args.search_min_offset,
        repeat_search_maximum=args.search_max_offset,
        velocity_blend=args.velocity_blend,
        energy_blend=args.energy_blend,
        minimum_energy_scale=args.minimum_energy_scale,
        maximum_energy_scale=args.maximum_energy_scale,
        minimum_input_pitch_f1=args.minimum_input_pitch_f1,
        maximum_input_normalized_score=args.maximum_input_normalized_score,
    )
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    input_path = Path(args.input_candidate).resolve()
    input_payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_profile(input_payload, profile)
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    input_row = {**row, "candidate": str(input_path)}
    output_row = {**row, "candidate": str(output_path)}
    _, baseline_groups, _ = load_song_sequences(input_row, 0.035)
    _, output_groups, _ = load_song_sequences(output_row, 0.035)
    offset = float(diagnostics["detectedOffsetSeconds"])
    repeat_start = args.template_start + offset
    repeat_end = args.template_end + offset
    local_reference = slice_groups(reference, repeat_start - 0.55, repeat_end + 0.55)
    local_baseline = slice_groups(baseline_groups, repeat_start - 0.55, repeat_end + 0.55)
    local_output = slice_groups(output_groups, repeat_start - 0.55, repeat_end + 0.55)
    baseline_local_eval = evaluation(local_reference, local_baseline)
    candidate_local_eval = evaluation(local_reference, local_output)
    baseline_full_eval = evaluation(reference, baseline_groups)
    candidate_full_eval = evaluation(reference, output_groups)
    passed, reason = dynamics_gate(
        baseline_local_eval,
        candidate_local_eval,
        baseline_full_eval,
        candidate_full_eval,
    )
    report = {
        "schema": "polymath-pianist-repetition-dynamics-training-report-v1",
        "evidenceBoundary": (
            "The source occurrence is supervised; recurrence detection and application "
            "read only the input candidate. This is same-song research, not unseen-song proof."
        ),
        "profile": str(profile_path),
        "inputCandidate": str(input_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutRepeatEvaluation": {
            "baseline": baseline_local_eval,
            "candidate": candidate_local_eval,
        },
        "trustedWindowEvaluation": {
            "baseline": baseline_full_eval,
            "candidate": candidate_full_eval,
        },
        "evaluationGate": reason,
        "decision": "LISTENING_CANDIDATE_RESEARCH_ONLY" if passed else "REJECT",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
