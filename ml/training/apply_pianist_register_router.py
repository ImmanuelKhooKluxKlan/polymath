"""Apply a frozen conservative octave router to an arranged piano score."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package and direct-script execution.
    from .analyze_pianist_gesture_patterns import normalize_notes
    from .analyze_pianist_register_errors import decorate_gestures, family, pitch_band
    from .fit_pianist_register_router import (
        FEATURE_NAMES,
        SHIFT_CLASSES,
        model_probabilities,
        record_features,
        route_predictions,
    )
except ImportError:  # pragma: no cover
    from analyze_pianist_gesture_patterns import normalize_notes  # type: ignore
    from analyze_pianist_register_errors import decorate_gestures, family, pitch_band  # type: ignore
    from fit_pianist_register_router import (  # type: ignore
        FEATURE_NAMES,
        SHIFT_CLASSES,
        model_probabilities,
        record_features,
        route_predictions,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def feature_records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    notes = normalize_notes(payload)
    decorations = decorate_gestures(notes)
    records: list[dict[str, Any]] = []
    indices: list[int] = []
    for note in notes:
        payload_index = int(note["_payloadIndex"])
        midi = int(note["midi"])
        source_midi = int(
            round(
                float(
                    note.get(
                        "sourceMidiBeforeArrangement",
                        note.get("originalMidiBeforeRangeShift", midi),
                    )
                )
            )
        )
        records.append(
            {
                "candidateMidi": midi,
                "sourceMidi": source_midi,
                "candidateHand": str(note.get("hand") or "unknown"),
                "candidateRole": str(note.get("arrangementRole") or "unknown"),
                "sourceFamily": family(note),
                "candidatePitchBand": pitch_band(midi),
                **decorations[payload_index],
            }
        )
        indices.append(payload_index)
    return records, indices


def apply_router(
    payload: dict[str, Any],
    profile: dict[str, Any],
    *,
    only_generated_by: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile.get("schema") != "polymath-pianist-register-router-v1":
        raise ValueError("Unsupported register-router profile.")
    model = profile.get("model") or {}
    if tuple(model.get("featureNames") or ()) != FEATURE_NAMES:
        raise ValueError("Register-router feature contract mismatch.")
    policy = profile.get("policy") or {}
    records, indices = feature_records(payload)
    features = np.asarray([record_features(record) for record in records], dtype=float)
    probabilities = model_probabilities(model, features)
    predictions = route_predictions(
        probabilities,
        threshold=float(policy.get("minimumConfidence", 1.0)),
        margin=float(policy.get("minimumMargin", 1.0)),
    )
    minimum_midi = int(policy.get("minimumMidi", 33))
    maximum_midi = int(policy.get("maximumMidi", 108))
    output = copy.deepcopy(payload)
    output_notes = output.get("notes") or []
    normalized = normalize_notes(payload)
    onset_midis: dict[int, set[int]] = {}
    for note in normalized:
        onset = int(round(float(note["time"]) * 1000.0))
        onset_midis.setdefault(onset, set()).add(int(note["midi"]))
    accepted = 0
    skipped_outside_scope = 0
    rejected_range = 0
    rejected_collision = 0
    accepted_shifts: dict[str, int] = Counter()
    reserved: dict[int, set[int]] = {key: set() for key in onset_midis}
    order = sorted(
        range(len(records)),
        key=lambda index: float(np.max(probabilities[index])),
        reverse=True,
    )
    for index in order:
        shift = int(predictions[index])
        if shift == 0 or shift not in SHIFT_CLASSES:
            continue
        payload_index = indices[index]
        item = output_notes[payload_index]
        if only_generated_by and str(item.get("generatedBy") or "") != only_generated_by:
            skipped_outside_scope += 1
            continue
        original_midi = int(round(float(item.get("midi", item.get("pitch")))))
        target_midi = original_midi + shift
        if not minimum_midi <= target_midi <= maximum_midi:
            rejected_range += 1
            continue
        onset = int(
            round(
                float(item.get("time", item.get("startTime", item.get("start", 0.0))))
                * 1000.0
            )
        )
        occupied = onset_midis.get(onset, set()) - {original_midi}
        if target_midi in occupied or target_midi in reserved.setdefault(onset, set()):
            rejected_collision += 1
            continue
        item["registerRouterOriginalMidi"] = original_midi
        item["registerRouterShiftSemitones"] = shift
        item["registerRouterProfile"] = profile.get("id")
        item["midi"] = target_midi
        if "pitch" in item:
            item["pitch"] = target_midi
        reserved[onset].add(target_midi)
        accepted_shifts[str(shift)] += 1
        accepted += 1
    diagnostics = {
        "profile": profile.get("id"),
        "profileSha256": profile.get("profileSha256"),
        "candidateNotes": len(records),
        "proposedChanges": int(np.sum(predictions != 0)),
        "acceptedChanges": accepted,
        "onlyGeneratedBy": only_generated_by,
        "skippedOutsideScope": skipped_outside_scope,
        "acceptedShifts": dict(accepted_shifts),
        "rejectedOutOfRange": rejected_range,
        "rejectedCollision": rejected_collision,
        "minimumMidi": minimum_midi,
        "maximumMidi": maximum_midi,
    }
    output.setdefault("diagnostics", {})["pianistRegisterRouter"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--only-generated-by",
        default="",
        help="Optionally route only notes carrying this generatedBy value.",
    )
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    profile_path = Path(args.profile).resolve()
    output, diagnostics = apply_router(
        load_json(input_path),
        load_json(profile_path),
        only_generated_by=str(args.only_generated_by or "") or None,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
