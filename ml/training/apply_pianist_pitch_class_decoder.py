"""Apply a frozen Pianella pitch-class decoder to one arranged score.

The operator is deliberately conservative: it starts from the current
arrangement, removes only very-low-probability pitch classes, and adds only
pitch classes that are supported by the local full-mix transcription.  It does
not use reference notes, song identity, or absolute song position.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .analyze_pianist_gesture_patterns import group_onsets, normalize_notes, note_name
from .analyze_pianist_texture_patterns import (
    family,
    is_pitch_class_decoder_note,
    nearest_source_notes,
)
from .fit_pianist_pitch_class_decoder import examples_from_song, predict


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def valid_midi_for_pitch_class(
    pitch_class: int, minimum: int, maximum: int
) -> list[int]:
    return [
        midi
        for midi in range(max(0, minimum), min(127, maximum) + 1)
        if midi % 12 == pitch_class % 12
    ]


def closest_pitch_class_midi(
    pitch_class: int,
    centre: float,
    *,
    minimum: int,
    maximum: int,
) -> int:
    options = valid_midi_for_pitch_class(pitch_class, minimum, maximum)
    if not options:
        raise ValueError("No MIDI pitch exists inside the configured output range")
    return min(options, key=lambda midi: (abs(midi - centre), midi))


def choose_source_evidence(
    notes: list[dict[str, Any]],
    *,
    pitch_class: int,
    time: float,
    centre: float,
) -> dict[str, Any] | None:
    options = [note for note in notes if int(note["midi"]) % 12 == pitch_class]
    if not options:
        return None
    return min(
        options,
        key=lambda note: (
            abs(float(note["time"]) - time) > 0.14,
            abs(float(note["time"]) - time),
            abs(float(note["midi"]) - centre),
            -float(note.get("velocity", 0.0)),
            -float(note.get("duration", 0.0)),
        ),
    )


def choose_added_midi(
    pitch_class: int,
    evidence: dict[str, Any],
    group: list[dict[str, Any]],
    *,
    strategy: str,
    minimum: int,
    maximum: int,
) -> int:
    centre = median(int(note["midi"]) for note in group)
    source_family = family(evidence.get("instrument"))
    if strategy == "source":
        desired = float(evidence["midi"])
    elif strategy == "nearest-candidate":
        desired = float(centre)
    elif strategy == "family":
        if source_family == "voice":
            desired = max(64.0, float(centre))
        elif source_family == "bass":
            desired = min(52.0, float(centre))
        else:
            desired = float(centre)
    else:
        raise ValueError(f"Unknown register strategy: {strategy}")
    return closest_pitch_class_midi(
        pitch_class, desired, minimum=minimum, maximum=maximum
    )


def validate_probability_override(
    audit_song: dict[str, Any], payload: dict[str, Any]
) -> dict[int, dict[int, float]]:
    """Validate externally-computed probabilities against one frozen audit.

    Tree/boosting experiments are serialized with joblib rather than the small
    JSON coefficient format used by the original linear decoder.  Keeping the
    score interchange format explicit lets the note operator stay deterministic
    and testable without teaching it how to unpickle arbitrary estimators.
    """

    expected_song = str(audit_song.get("id") or "")
    score_song = str(payload.get("songId") or "")
    if score_song and score_song != expected_song:
        raise ValueError(
            f"Probability score song {score_song!r} does not match audit "
            f"song {expected_song!r}"
        )
    expected_cells = len(list(audit_song.get("cells") or []))
    rows = payload.get("cells")
    if not isinstance(rows, list) or len(rows) != expected_cells:
        raise ValueError(
            "Probability scores must contain exactly one row per audit cell"
        )
    output: dict[int, dict[int, float]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Every probability score row must be an object")
        cell_index = int(row.get("cellIndex", position))
        values = row.get("probabilities")
        if cell_index in output or not isinstance(values, list) or len(values) != 12:
            raise ValueError(
                "Probability scores need unique cell indexes and 12 values per cell"
            )
        parsed = [float(value) for value in values]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in parsed):
            raise ValueError("Probability scores must be finite values from 0 to 1")
        output[cell_index] = {
            pitch_class: probability
            for pitch_class, probability in enumerate(parsed)
        }
    if set(output) != set(range(expected_cells)):
        raise ValueError("Probability score cell indexes must be contiguous from zero")
    return output


def cell_probabilities(
    audit_song: dict[str, Any],
    model: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> dict[int, dict[int, float]]:
    if override is not None:
        return validate_probability_override(audit_song, override)
    examples = examples_from_song(audit_song)
    probabilities = predict(examples, model)
    output: dict[int, dict[int, float]] = {}
    for example, probability in zip(examples, probabilities):
        output.setdefault(int(example["cellIndex"]), {})[
            int(example["pitchClass"])
        ] = float(probability)
    return output


def find_candidate_group(
    groups: list[list[dict[str, Any]]],
    time: float,
    *,
    used: set[int],
    tolerance: float = 0.04,
) -> tuple[int, list[dict[str, Any]]]:
    choices = [
        (abs(float(group[0]["time"]) - time), index, group)
        for index, group in enumerate(groups)
        if index not in used and abs(float(group[0]["time"]) - time) <= tolerance
    ]
    if not choices:
        raise ValueError(
            f"No candidate gesture matches audit cell at {time:.6f}s"
        )
    _distance, index, group = min(choices, key=lambda item: (item[0], item[1]))
    used.add(index)
    return index, group


def new_note(
    group: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    midi: int,
    time: float,
    probability: float,
) -> dict[str, Any]:
    durations = [float(note.get("duration", 0.3)) for note in group]
    score_durations = [
        float(note.get("scoreDuration", note.get("duration", 0.3)))
        for note in group
    ]
    audio_durations = [
        float(note.get("audioDuration", note.get("duration", 0.3)))
        for note in group
    ]
    velocities = [float(note.get("velocity", 0.6)) for note in group]
    duration = clamp(float(median(durations)), 0.08, 2.6)
    score_duration = clamp(float(median(score_durations)), 0.08, 2.6)
    audio_duration = clamp(float(median(audio_durations)), 0.08, 3.1)
    velocity = clamp(float(median(velocities)), 0.18, 0.96)
    source_family = family(evidence.get("instrument"))
    if source_family == "voice":
        role = "melody"
    elif source_family == "bass" or midi < 48:
        role = "bass"
    else:
        role = "harmony"
    return {
        "midi": midi,
        "note": note_name(midi),
        "time": round(time, 6),
        "duration": round(duration, 6),
        "scoreDuration": round(score_duration, 6),
        "visualDuration": round(score_duration, 6),
        "audioDuration": round(audio_duration, 6),
        "releaseSeconds": round(clamp(audio_duration - duration + 0.16, 0.12, 0.8), 6),
        "velocity": round(velocity, 6),
        "hand": "right" if midi >= 60 else "left",
        "instrument": "acoustic_piano",
        "source": "polymath-pianist-pitch-class-decoder-v1",
        "sourceInstrument": evidence.get("instrument"),
        "sourceIndex": evidence.get("sourceIndex"),
        "sourceMidiBeforeArrangement": int(evidence["midi"]),
        "arrangementRole": role,
        "generatedBy": "pianist-pitch-class-decoder-v1",
        "pitchClassProbability": round(probability, 6),
    }


def apply_decoder(
    candidate: dict[str, Any],
    source: dict[str, Any],
    audit_song: dict[str, Any],
    model: dict[str, Any],
    *,
    register_strategy: str = "family",
    addition_timing: str = "hybrid",
    preserve_melody_removals: bool = False,
    remove_generated_only: bool = False,
    maximum_generated_probability_for_removal: float | None = None,
    require_exact_source_midi_for_removal: bool = False,
    minimum_midi: int = 33,
    maximum_midi: int = 108,
    add_threshold_override: float | None = None,
    remove_threshold_override: float | None = None,
    probabilities_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(candidate)
    payload_notes = output.get("notes")
    if not isinstance(payload_notes, list):
        raise ValueError("Candidate has no notes list")
    candidate_notes = normalize_notes(candidate)
    candidate_groups = group_onsets(candidate_notes, 0.035)
    source_notes = normalize_notes(source, source_indices=True)
    rows = list(audit_song.get("cells") or [])
    probabilities = cell_probabilities(
        audit_song, model, override=probabilities_override
    )
    add_threshold = float(
        model.get("addThreshold", 0.65)
        if add_threshold_override is None
        else add_threshold_override
    )
    remove_threshold = float(
        model.get("removeThreshold", 0.125)
        if remove_threshold_override is None
        else remove_threshold_override
    )
    if not 0.0 <= add_threshold <= 1.0:
        raise ValueError("add threshold must be between 0 and 1")
    if not 0.0 <= remove_threshold <= 1.0:
        raise ValueError("remove threshold must be between 0 and 1")
    source_only = bool(model.get("sourceSupportedAdditionsOnly", True))
    used_groups: set[int] = set()
    remove_indices: set[int] = set()
    additions: list[dict[str, Any]] = []
    attempted_additions = rejected_no_evidence = protected_last_note = 0
    protected_melody = protected_non_generated = protected_upstream_confidence = 0
    protected_register_evidence = 0
    removals_by_pitch_class: dict[int, int] = {}
    additions_by_pitch_class: dict[int, int] = {}

    for cell_index, row in enumerate(rows):
        time = float(row.get("sourceTime", 0.0))
        _group_index, group = find_candidate_group(
            candidate_groups, time, used=used_groups
        )
        scores = probabilities[cell_index]
        current_pcs = {int(note["midi"]) % 12 for note in group}
        source_pcs = {int(value) % 12 for value in row.get("sourcePitchClasses") or []}
        removal_pcs = {
            pitch_class
            for pitch_class in current_pcs
            if scores[pitch_class] <= remove_threshold
        }
        if remove_generated_only:
            generated_only = {
                pitch_class
                for pitch_class in removal_pcs
                if all(
                    is_pitch_class_decoder_note(note)
                    for note in group
                    if int(note["midi"]) % 12 == pitch_class
                )
            }
            protected_non_generated += len(removal_pcs - generated_only)
            removal_pcs = generated_only
        if maximum_generated_probability_for_removal is not None:
            if not 0.0 <= maximum_generated_probability_for_removal <= 1.0:
                raise ValueError(
                    "maximum generated probability for removal must be between 0 and 1"
                )
            below_upstream_ceiling = {
                pitch_class
                for pitch_class in removal_pcs
                if all(
                    float(note.get("pitchClassProbability", 1.0))
                    <= maximum_generated_probability_for_removal
                    for note in group
                    if int(note["midi"]) % 12 == pitch_class
                )
            }
            protected_upstream_confidence += len(
                removal_pcs - below_upstream_ceiling
            )
            removal_pcs = below_upstream_ceiling
        if require_exact_source_midi_for_removal:
            evidence_rows = {
                int(item.get("pitchClass", -1)): item
                for item in row.get("sourcePitchClassEvidence") or []
                if isinstance(item, dict)
            }
            exact_register_evidence = {
                pitch_class
                for pitch_class in removal_pcs
                if bool(
                    evidence_rows.get(pitch_class, {}).get(
                        "exactCandidateMidiMatch", False
                    )
                )
            }
            protected_register_evidence += len(
                removal_pcs - exact_register_evidence
            )
            removal_pcs = exact_register_evidence
        if preserve_melody_removals:
            melody_pcs = {
                int(note["midi"]) % 12
                for note in group
                if str(note.get("arrangementRole") or "") == "melody"
            }
            protected_melody += len(removal_pcs & melody_pcs)
            removal_pcs -= melody_pcs
        remaining = [
            note for note in group if int(note["midi"]) % 12 not in removal_pcs
        ]
        if not remaining and group:
            keep_pc = max(
                current_pcs,
                key=lambda pitch_class: (
                    scores[pitch_class],
                    any(
                        str(note.get("arrangementRole") or "") == "melody"
                        and int(note["midi"]) % 12 == pitch_class
                        for note in group
                    ),
                ),
            )
            removal_pcs.discard(keep_pc)
            protected_last_note += 1
        for note in group:
            pitch_class = int(note["midi"]) % 12
            if pitch_class in removal_pcs:
                remove_indices.add(int(note["_payloadIndex"]))
                removals_by_pitch_class[pitch_class] = (
                    removals_by_pitch_class.get(pitch_class, 0) + 1
                )

        local_source = nearest_source_notes(source_notes, time)
        centre = float(median(int(note["midi"]) for note in group))
        desired_additions = {
            pitch_class
            for pitch_class in range(12)
            if pitch_class not in current_pcs
            and scores[pitch_class] >= add_threshold
            and (not source_only or pitch_class in source_pcs)
        }
        attempted_additions += len(desired_additions)
        for pitch_class in sorted(desired_additions):
            evidence = choose_source_evidence(
                local_source,
                pitch_class=pitch_class,
                time=time,
                centre=centre,
            )
            if evidence is None:
                rejected_no_evidence += 1
                continue
            midi = choose_added_midi(
                pitch_class,
                evidence,
                group,
                strategy=register_strategy,
                minimum=minimum_midi,
                maximum=maximum_midi,
            )
            if any(int(note["midi"]) == midi for note in group):
                continue
            evidence_time = float(evidence["time"])
            if addition_timing == "candidate":
                added_time = time
            elif addition_timing == "source":
                added_time = evidence_time
            elif addition_timing == "hybrid":
                added_time = evidence_time if abs(evidence_time - time) <= 0.14 else time
            else:
                raise ValueError(f"Unknown addition timing: {addition_timing}")
            additions.append(
                new_note(
                    group,
                    evidence,
                    midi=midi,
                    time=max(0.0, added_time),
                    probability=scores[pitch_class],
                )
            )
            additions_by_pitch_class[pitch_class] = (
                additions_by_pitch_class.get(pitch_class, 0) + 1
            )

    output["notes"] = [
        note for index, note in enumerate(payload_notes) if index not in remove_indices
    ] + additions
    output["notes"].sort(
        key=lambda note: (
            float(note.get("time", note.get("startTime", note.get("start", 0.0)))),
            int(round(float(note.get("midi", note.get("pitch", 0))))),
        )
    )
    diagnostics = {
        "schema": "polymath-pianist-pitch-class-application-v1",
        "modelId": (
            probabilities_override.get("modelId")
            if probabilities_override is not None
            else model.get("id")
        ),
        "modelSha256": (
            probabilities_override.get("modelSha256")
            if probabilities_override is not None
            else model.get("modelSha256")
        ),
        "probabilitySource": (
            "external-frozen-scores"
            if probabilities_override is not None
            else "json-linear-model"
        ),
        "songId": audit_song.get("id"),
        "cells": len(rows),
        "matchedCandidateGroups": len(used_groups),
        "addThreshold": add_threshold,
        "removeThreshold": remove_threshold,
        "registerStrategy": register_strategy,
        "additionTiming": addition_timing,
        "preserveMelodyRemovals": preserve_melody_removals,
        "removeGeneratedOnly": remove_generated_only,
        "maximumGeneratedProbabilityForRemoval": (
            maximum_generated_probability_for_removal
        ),
        "requireExactSourceMidiForRemoval": require_exact_source_midi_for_removal,
        "attemptedAdditions": attempted_additions,
        "acceptedAdditions": len(additions),
        "removedNotes": len(remove_indices),
        "rejectedNoEvidence": rejected_no_evidence,
        "protectedLastNote": protected_last_note,
        "protectedMelodyPitchClasses": protected_melody,
        "protectedNonGeneratedPitchClasses": protected_non_generated,
        "protectedUpstreamConfidencePitchClasses": protected_upstream_confidence,
        "protectedRegisterEvidencePitchClasses": protected_register_evidence,
        "additionsByPitchClass": {
            str(key): value for key, value in sorted(additions_by_pitch_class.items())
        },
        "removalsByPitchClass": {
            str(key): value for key, value in sorted(removals_by_pitch_class.items())
        },
        "warning": (
            "Research-only chord membership operator. Exact-note, duration, "
            "retrigger, and blind-listening gates remain mandatory."
        ),
    }
    output.setdefault("diagnostics", {})["pianistPitchClassDecoder"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scores",
        default="",
        help=(
            "Optional frozen 12-probability-per-cell JSON generated by an "
            "external research estimator."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--register-strategy",
        choices=("source", "nearest-candidate", "family"),
        default="family",
    )
    parser.add_argument(
        "--addition-timing",
        choices=("candidate", "source", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--preserve-melody-removals", action="store_true")
    parser.add_argument("--remove-generated-only", action="store_true")
    parser.add_argument("--maximum-generated-probability-for-removal", type=float)
    parser.add_argument("--require-exact-source-midi-for-removal", action="store_true")
    parser.add_argument("--minimum-midi", type=int, default=33)
    parser.add_argument("--maximum-midi", type=int, default=108)
    parser.add_argument(
        "--add-threshold",
        type=float,
        help="Research-only operating-point override; the model file stays immutable.",
    )
    parser.add_argument(
        "--remove-threshold",
        type=float,
        help="Research-only operating-point override; the model file stays immutable.",
    )
    args = parser.parse_args()
    audit = load_json(Path(args.audit).resolve())
    matches = [
        song
        for song in audit.get("songs") or []
        if str(song.get("id")) == str(args.song)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one audit song named {args.song!r}")
    output, diagnostics = apply_decoder(
        load_json(Path(args.candidate).resolve()),
        load_json(Path(args.source).resolve()),
        matches[0],
        load_json(Path(args.model).resolve()),
        register_strategy=str(args.register_strategy),
        addition_timing=str(args.addition_timing),
        preserve_melody_removals=bool(args.preserve_melody_removals),
        remove_generated_only=bool(args.remove_generated_only),
        maximum_generated_probability_for_removal=(
            args.maximum_generated_probability_for_removal
        ),
        require_exact_source_midi_for_removal=bool(
            args.require_exact_source_midi_for_removal
        ),
        minimum_midi=int(args.minimum_midi),
        maximum_midi=int(args.maximum_midi),
        add_threshold_override=args.add_threshold,
        remove_threshold_override=args.remove_threshold,
        probabilities_override=(
            load_json(Path(args.scores).resolve()) if args.scores else None
        ),
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
