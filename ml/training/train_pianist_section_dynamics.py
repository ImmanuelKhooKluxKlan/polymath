"""Fit chord-size touch on one section and test it on a later section.

The stage changes velocity only.  Notes, onsets, hands and durations are frozen.
Previously generated pianist motifs are also frozen so a broad calibration
cannot erase a phrase contour that already passed its own held-out gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .analyze_pianist_gesture_patterns import group_onsets, normalize_notes
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import evaluation, finite
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import group_onsets, normalize_notes
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import evaluation, finite


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def slice_groups(
    groups: list[list[dict[str, Any]]], start: float, end: float
) -> list[list[dict[str, Any]]]:
    return [group for group in groups if start <= float(group[0]["time"]) < end]


def group_velocity(group: list[dict[str, Any]]) -> float:
    return median(finite(note.get("velocity"), 0.70) for note in group)


def chord_size(group: list[dict[str, Any]]) -> int:
    return min(6, len({int(note["midi"]) for note in group}))


def fit_profile(
    reference_groups: list[list[dict[str, Any]]],
    *,
    profile_id: str,
    song_id: str,
    training_start: float,
    training_end: float,
    application_start: float,
    application_end: float,
    minimum_examples: int = 2,
) -> dict[str, Any]:
    if not training_start < training_end <= application_start < application_end:
        raise ValueError(
            "Ranges must be ordered training_start < training_end <= "
            "application_start < application_end"
        )
    training = slice_groups(reference_groups, training_start, training_end)
    buckets: dict[int, list[float]] = defaultdict(list)
    for group in training:
        buckets[chord_size(group)].append(group_velocity(group))
    velocity_by_size = {
        str(size): rounded(median(values))
        for size, values in sorted(buckets.items())
        if len(values) >= minimum_examples
    }
    if not velocity_by_size:
        raise ValueError("No chord size met the minimum training support")
    profile = {
        "schema": "polymath-pianist-section-dynamics-profile-v1",
        "id": profile_id,
        "enabled": True,
        "training": {
            "songId": song_id,
            "method": "held-out section chord-size gesture calibration",
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
            "startSeconds": training_start,
            "endSeconds": training_end,
            "gestureCount": len(training),
            "examplesByChordSize": {
                str(size): len(values) for size, values in sorted(buckets.items())
            },
        },
        "application": {
            "startSeconds": application_start,
            "endSeconds": application_end,
            "velocityByChordSize": velocity_by_size,
            "minimumExamples": minimum_examples,
            "freezeGeneratedPianistMotifs": True,
            "targetReadAtInference": False,
            "notesFrozen": True,
            "onsetsFrozen": True,
            "durationsFrozen": True,
            "handsFrozen": True,
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    return profile


def is_frozen_generated_note(note: dict[str, Any]) -> bool:
    return str(note.get("generatedBy") or "").startswith("pianist-")


def original_groups(payload: dict[str, Any], onset_window: float = 0.035) -> list[list[int]]:
    rows = []
    for index, note in enumerate(payload.get("notes") or []):
        if not isinstance(note, dict):
            continue
        time = finite(note.get("time", note.get("startTime", note.get("start"))), -1)
        midi = int(round(finite(note.get("midi", note.get("pitch")), -1)))
        if time >= 0 and 21 <= midi <= 108:
            rows.append((time, midi, index))
    rows.sort()
    output: list[list[int]] = []
    starts: list[float] = []
    for time, _midi, index in rows:
        if not output or time - starts[-1] > onset_window:
            output.append([index])
            starts.append(time)
        else:
            output[-1].append(index)
    return output


def apply_profile(
    candidate: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(candidate)
    notes = output.get("notes") or []
    application = profile["application"]
    start = float(application["startSeconds"])
    end = float(application["endSeconds"])
    mapping = {
        int(size): float(value)
        for size, value in application["velocityByChordSize"].items()
    }
    changed_groups = 0
    changed_notes = 0
    frozen_groups = 0
    unsupported_groups = 0
    for indices in original_groups(output):
        group = [notes[index] for index in indices]
        time = finite(group[0].get("time", group[0].get("startTime")), -1)
        if not start <= time < end:
            continue
        if any(is_frozen_generated_note(note) for note in group):
            frozen_groups += 1
            continue
        size = min(
            6,
            len(
                {
                    int(round(finite(note.get("midi", note.get("pitch")), -1)))
                    for note in group
                }
            ),
        )
        target_velocity = mapping.get(size)
        if target_velocity is None:
            unsupported_groups += 1
            continue
        changed_groups += 1
        for note in group:
            if not math_is_close(finite(note.get("velocity"), 0.70), target_velocity):
                changed_notes += 1
            note["velocity"] = rounded(target_velocity)
            note["pianistSectionDynamicsProfile"] = profile["id"]
    diagnostics = {
        "applied": changed_groups > 0,
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "changedGroups": changed_groups,
        "changedNotes": changed_notes,
        "frozenGeneratedGroups": frozen_groups,
        "unsupportedGroups": unsupported_groups,
        "targetReadAtInference": False,
        "notesOnsetsDurationsHandsFrozen": True,
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistSectionDynamics"] = diagnostics
    return output, diagnostics


def math_is_close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--training-start", type=float, required=True)
    parser.add_argument("--training-end", type=float, required=True)
    parser.add_argument("--application-start", type=float, required=True)
    parser.add_argument("--application-end", type=float, required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
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
        training_start=args.training_start,
        training_end=args.training_end,
        application_start=args.application_start,
        application_end=args.application_end,
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
    _, input_groups, _ = load_song_sequences(input_row, 0.035)
    output_row = {**row, "candidate": str(output_path)}
    _, output_groups, _ = load_song_sequences(output_row, 0.035)
    start = float(profile["application"]["startSeconds"])
    end = float(profile["application"]["endSeconds"])
    held_reference = slice_groups(reference, start, end)
    held_baseline = evaluation(held_reference, slice_groups(input_groups, start, end))
    held_candidate = evaluation(held_reference, slice_groups(output_groups, start, end))
    structural_keys = (
        "referenceGestures",
        "candidateGestures",
        "alignedGestures",
        "missingReferenceGestures",
        "extraCandidateGestures",
        "meanPitchClassF1",
        "exactPitchClassSetRate",
        "occupancyAccuracy",
        "onsetMaeSeconds",
        "exactKeyDurationMaeSeconds",
        "sequenceScore",
    )
    structure_frozen = all(
        held_baseline[key] == held_candidate[key] for key in structural_keys
    )
    passed = (
        diagnostics["applied"]
        and structure_frozen
        and float(held_candidate["gestureVelocityMae"])
        <= float(held_baseline["gestureVelocityMae"]) - 0.01
    )
    report = {
        "schema": "polymath-pianist-section-dynamics-report-v1",
        "evidenceBoundary": (
            "Velocity medians are fitted only in the declared training section. "
            "The later application section is read only for evaluation. Notes, "
            "timing, duration, and hands remain immutable."
        ),
        "profile": str(profile_path),
        "inputCandidate": str(input_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutSectionEvaluation": {
            "baseline": held_baseline,
            "dynamicsCandidate": held_candidate,
            "structureFrozen": structure_frozen,
        },
        "trustedWindowEvaluation": {
            "baseline": evaluation(reference, input_groups),
            "dynamicsCandidate": evaluation(reference, output_groups),
        },
        "decision": (
            "LISTENING_CANDIDATE_RESEARCH_ONLY"
            if passed
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
