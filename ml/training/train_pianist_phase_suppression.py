"""Learn a beat-phase hand-separation rule and test it on a later section.

The adapter targets one specific failure mode: a selected left-hand accompaniment
strike incorrectly retains a simultaneous right-hand layer.  It learns reliable
phase slots from an approved training section, then applies them only inside a
long instrumental gap and only to strong gestures.  It never removes the left
hand, never removes an onset, and never consults target notes at application.
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
    from .analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from .analyze_pianist_sequence_alignment import align_sequences
    from .fit_pianist_hand_occupancy import load_song_sequences, rounded
    from .train_pianist_intro_motif import (
        estimate_source_pulse,
        evaluation,
        finite,
        is_percussion,
        is_voice,
        source_groups,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from analyze_pianist_gesture_patterns import group_onsets, normalize_notes
    from analyze_pianist_reduction_grammar import explicit_hand, occupancy
    from analyze_pianist_sequence_alignment import align_sequences
    from fit_pianist_hand_occupancy import load_song_sequences, rounded
    from train_pianist_intro_motif import (
        estimate_source_pulse,
        evaluation,
        finite,
        is_percussion,
        is_voice,
        source_groups,
    )


def sha256_json(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def phase_slot(time: float, origin: float, pulse: float, subdivisions: int) -> int:
    return int(round((time - origin) / pulse)) % subdivisions


def learn_phase_slots(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    *,
    origin: float,
    pulse: float,
    subdivisions: int,
    training_start: float,
    training_end: float,
    minimum_examples: int = 3,
    minimum_precision: float = 0.90,
) -> tuple[list[int], list[dict[str, Any]]]:
    matches, _, _, _ = align_sequences(
        reference, candidate, maximum_time_distance=0.55, gap_cost=0.85
    )
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for reference_index, candidate_index in matches:
        observed = candidate[candidate_index]
        time = float(observed[0]["time"])
        if not training_start <= time < training_end or occupancy(observed) != "both":
            continue
        slot = phase_slot(time, origin, pulse, subdivisions)
        counts[slot][0] += int(occupancy(reference[reference_index]) == "left-only")
        counts[slot][1] += 1
    rows = [
        {
            "slot": slot,
            "targetLeftOnly": positives,
            "candidateBoth": total,
            "precision": rounded(positives / max(1, total)),
        }
        for slot, (positives, total) in sorted(counts.items())
    ]
    selected = [
        int(row["slot"])
        for row in rows
        if int(row["candidateBoth"]) >= minimum_examples
        and float(row["precision"]) >= minimum_precision
    ]
    return selected, rows


def long_voice_gaps(
    payload: dict[str, Any], minimum_gap_seconds: float
) -> list[tuple[float, float]]:
    times = sorted(
        {
            finite(note.get("time", note.get("startTime")), -1)
            for note in payload.get("notes") or []
            if isinstance(note, dict) and is_voice(note)
        }
    )
    return [
        (left, right)
        for left, right in zip(times, times[1:])
        if right - left >= minimum_gap_seconds
    ]


def apply_profile(
    candidate: dict[str, Any],
    source: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    origin = float(profile["clock"]["originSeconds"])
    pulse = float(profile["clock"]["pulseSeconds"])
    subdivisions = int(profile["clock"]["subdivisions"])
    selected_slots = {int(value) for value in profile["selectedSlots"]}
    minimum_velocity = float(profile["application"]["minimumGestureVelocity"])
    gaps = long_voice_gaps(
        source, float(profile["application"]["minimumInstrumentalGapSeconds"])
    )
    normalized = normalize_notes(candidate)
    groups = group_onsets(normalized, 0.035)
    removed_indices: set[int] = set()
    selected_gestures = []
    for group in groups:
        time = float(group[0]["time"])
        if not any(start < time < end for start, end in gaps):
            continue
        maximum_time = finite(
            profile["application"].get("maximumApplicationTimeSeconds"),
            float("inf"),
        )
        if time >= maximum_time:
            continue
        if occupancy(group) != "both":
            continue
        velocity = median(finite(note.get("velocity"), 0.7) for note in group)
        slot = phase_slot(time, origin, pulse, subdivisions)
        if velocity < minimum_velocity or slot not in selected_slots:
            continue
        right = [note for note in group if explicit_hand(note) == "right"]
        left = [note for note in group if explicit_hand(note) == "left"]
        if not right or not left:
            continue
        removed_indices.update(int(note["_payloadIndex"]) for note in right)
        selected_gestures.append(
            {
                "time": rounded(time),
                "slot": slot,
                "velocity": rounded(velocity),
                "rightNotesRemoved": len(right),
                "leftNotesRetained": len(left),
            }
        )

    output = copy.deepcopy(candidate)
    output["notes"] = [
        note
        for index, note in enumerate(candidate.get("notes") or [])
        if index not in removed_indices
    ]
    diagnostics = {
        "applied": bool(removed_indices),
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "instrumentalGaps": [
            [rounded(start), rounded(end)] for start, end in gaps
        ],
        "selectedGestures": selected_gestures,
        "rightNotesRemoved": len(removed_indices),
        "leftNotesRemoved": 0,
        "onsetsRemoved": 0,
        "targetReadAtInference": False,
    }
    arrangement = output.setdefault("pianoArrangement", {})
    arrangement["pianistPhaseSuppression"] = diagnostics
    arrangement["outputNoteCount"] = len(output["notes"])
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input-candidate", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--training-start", type=float, required=True)
    parser.add_argument("--training-end", type=float, required=True)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--minimum-gesture-velocity", type=float, default=0.75)
    parser.add_argument("--minimum-instrumental-gap", type=float, default=20.0)
    parser.add_argument("--application-end", type=float)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    row = next(
        (item for item in manifest.get("songs") or [] if item.get("id") == args.song_id),
        None,
    )
    if row is None:
        raise ValueError(f"Song {args.song_id!r} was not found in the manifest")
    reference, sealed_baseline, _ = load_song_sequences(row, 0.035)
    source_path = Path(args.source).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    raw_groups = source_groups(source)
    non_voice = [
        group
        for group in raw_groups
        if not all(is_voice(note) or is_percussion(note) for note in group)
    ]
    origin = min(float(group[0]["time"]) for group in non_voice)
    pulse = estimate_source_pulse(non_voice, max(float(group[0]["time"]) for group in non_voice) + 1)
    slots, slot_rows = learn_phase_slots(
        reference,
        sealed_baseline,
        origin=origin,
        pulse=pulse,
        subdivisions=args.subdivisions,
        training_start=args.training_start,
        training_end=args.training_end,
    )
    if not slots:
        raise ValueError("No hand-suppression phase passed the training gate")
    profile = {
        "schema": "polymath-pianist-phase-suppression-profile-v1",
        "id": args.profile_id,
        "enabled": True,
        "training": {
            "songId": args.song_id,
            "startSeconds": args.training_start,
            "endSeconds": args.training_end,
            "sameSongStyleConditioned": True,
            "commercialUseAllowed": False,
            "decision": "RESEARCH_ONLY",
        },
        "clock": {
            "originSeconds": rounded(origin),
            "pulseSeconds": rounded(pulse),
            "subdivisions": args.subdivisions,
        },
        "slotEvidence": slot_rows,
        "selectedSlots": slots,
        "application": {
            "minimumGestureVelocity": args.minimum_gesture_velocity,
            "minimumInstrumentalGapSeconds": args.minimum_instrumental_gap,
            "maximumApplicationTimeSeconds": args.application_end,
            "removeHand": "right",
            "preserveLeftHand": True,
            "preserveOnsets": True,
            "targetReadAtInference": False,
        },
    }
    profile["profileSha256"] = sha256_json(profile)
    profile_path = Path(args.output_profile).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    input_path = Path(args.input_candidate).resolve()
    input_payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    output, diagnostics = apply_profile(input_payload, source, profile)
    output_path = Path(args.output_candidate).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    input_row = {**row, "candidate": str(input_path)}
    output_row = {**row, "candidate": str(output_path)}
    _, input_groups, _ = load_song_sequences(input_row, 0.035)
    _, output_groups, _ = load_song_sequences(output_row, 0.035)
    gaps = [(float(a), float(b)) for a, b in diagnostics["instrumentalGaps"]]
    held_reference = [
        group
        for group in reference
        if any(start < float(group[0]["time"]) < end for start, end in gaps)
    ]
    held_input = [
        group
        for group in input_groups
        if any(start < float(group[0]["time"]) < end for start, end in gaps)
    ]
    held_output = [
        group
        for group in output_groups
        if any(start < float(group[0]["time"]) < end for start, end in gaps)
    ]
    baseline_eval = evaluation(held_reference, held_input)
    candidate_eval = evaluation(held_reference, held_output)
    passed = (
        diagnostics["rightNotesRemoved"] >= 3
        and candidate_eval["sequenceScore"] < baseline_eval["sequenceScore"]
        and candidate_eval["meanPitchClassF1"] >= baseline_eval["meanPitchClassF1"]
        and candidate_eval["exactPitchClassSetRate"] >= baseline_eval["exactPitchClassSetRate"]
        and candidate_eval["occupancyAccuracy"] > baseline_eval["occupancyAccuracy"]
        # This adapter never changes onset or velocity values; tiny metric
        # movement can still occur because sequence alignment rematches a
        # neighboring gesture after pitch removal.
        and candidate_eval["onsetMaeSeconds"]
        <= baseline_eval["onsetMaeSeconds"] + 0.002
        and candidate_eval["gestureVelocityMae"]
        <= baseline_eval["gestureVelocityMae"] + 0.002
    )
    report = {
        "schema": "polymath-pianist-phase-suppression-report-v1",
        "evidenceBoundary": (
            "Phase slots were fit on the declared training section. The long "
            "instrumental gap is held-out evaluation. Application reads only source "
            "voice gaps, candidate beat phase, and candidate velocity."
        ),
        "profile": str(profile_path),
        "candidate": str(output_path),
        "application": diagnostics,
        "heldOutInstrumentalEvaluation": {
            "baseline": baseline_eval,
            "phaseCandidate": candidate_eval,
        },
        "trustedWindowEvaluation": {
            "baseline": evaluation(reference, input_groups),
            "phaseCandidate": evaluation(reference, output_groups),
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
