"""Apply a frozen audio-plus-symbolic octave router to one piano candidate."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_pianist_gesture_patterns import normalize_notes
from .analyze_pianist_register_errors import decorate_gestures, family, pitch_band
from .audio_register_evidence import (
    AUDIO_FEATURE_NAMES,
    AudioRegisterEvidence,
    audio_feature_values,
    audio_gate,
)
from .fit_pianist_register_router import (
    FEATURE_NAMES as SYMBOLIC_FEATURE_NAMES,
    SHIFT_CLASSES,
    model_probabilities,
    record_features,
    route_predictions,
)


FEATURE_NAMES = (*SYMBOLIC_FEATURE_NAMES, *AUDIO_FEATURE_NAMES)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def feature_records(
    payload: dict[str, Any], evidence_reader: AudioRegisterEvidence
) -> tuple[list[dict[str, Any]], list[int]]:
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
        record = {
            "candidateMidi": midi,
            "sourceMidi": source_midi,
            "candidateHand": str(note.get("hand") or "unknown"),
            "candidateRole": str(note.get("arrangementRole") or "unknown"),
            "sourceFamily": family(note),
            "candidatePitchBand": pitch_band(midi),
            **decorations[payload_index],
        }
        evidence = evidence_reader.score(float(note["time"]), midi)
        records.append(
            {
                **record,
                "audioEvidence": evidence,
                "features": [
                    *record_features(record),
                    *audio_feature_values(evidence),
                ],
            }
        )
        indices.append(payload_index)
    return records, indices


def apply_router(
    payload: dict[str, Any],
    profile: dict[str, Any],
    audio_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile.get("schema") != "polymath-audio-pianist-register-router-v1":
        raise ValueError("Unsupported audio register-router profile")
    if not bool(profile.get("enabled")):
        raise ValueError("Audio register-router profile did not pass its training gates")
    model = profile.get("model") or {}
    if tuple(model.get("featureNames") or ()) != FEATURE_NAMES:
        raise ValueError("Audio register-router feature contract mismatch")
    policy = profile.get("policy") or {}
    evidence_reader = AudioRegisterEvidence.from_wav(audio_path)
    records, indices = feature_records(payload, evidence_reader)
    features = np.asarray([record["features"] for record in records], dtype=float)
    probabilities = model_probabilities(model, features)
    proposed = route_predictions(
        probabilities,
        threshold=float(policy.get("minimumConfidence", 1.0)),
        margin=float(policy.get("minimumMargin", 1.0)),
    )
    accepted_predictions = np.asarray(
        [
            int(prediction)
            if audio_gate(
                int(prediction),
                record["audioEvidence"],
                agreement=str(policy.get("audioAgreement") or "both"),
                minimum_gain=float(policy.get("minimumAudioLogGain", 0.0)),
            )
            else 0
            for record, prediction in zip(records, proposed)
        ],
        dtype=np.int64,
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

    accepted = rejected_range = rejected_collision = 0
    audio_rejected = int(np.sum((proposed != 0) & (accepted_predictions == 0)))
    accepted_shifts: Counter[str] = Counter()
    reserved: dict[int, set[int]] = {key: set() for key in onset_midis}
    order = sorted(
        range(len(records)),
        key=lambda index: float(np.max(probabilities[index])),
        reverse=True,
    )
    for index in order:
        shift = int(accepted_predictions[index])
        if shift == 0 or shift not in SHIFT_CLASSES:
            continue
        payload_index = indices[index]
        item = output_notes[payload_index]
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
        item["audioRegisterRouterOriginalMidi"] = original_midi
        item["audioRegisterRouterShiftSemitones"] = shift
        item["audioRegisterRouterProfile"] = profile.get("id")
        item["midi"] = target_midi
        if "pitch" in item:
            item["pitch"] = target_midi
        if "note" in item:
            item["note"] = midi_name(target_midi)
        reserved[onset].add(target_midi)
        accepted_shifts[str(shift)] += 1
        accepted += 1

    diagnostics = {
        "schema": "polymath-audio-pianist-register-router-application-v1",
        "profile": profile.get("id"),
        "profileSha256": profile.get("profileSha256"),
        "audio": str(audio_path),
        "candidateNotes": len(records),
        "modelProposedChanges": int(np.sum(proposed != 0)),
        "audioRejectedChanges": audio_rejected,
        "acceptedChanges": accepted,
        "acceptedShifts": dict(accepted_shifts),
        "rejectedOutOfRange": rejected_range,
        "rejectedCollision": rejected_collision,
        "audioAgreement": policy.get("audioAgreement"),
        "minimumAudioLogGain": policy.get("minimumAudioLogGain"),
        "minimumMidi": minimum_midi,
        "maximumMidi": maximum_midi,
    }
    output.setdefault("diagnostics", {})["audioPianistRegisterRouter"] = diagnostics
    return output, diagnostics


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    output, diagnostics = apply_router(
        load_json(Path(args.input).resolve()),
        load_json(Path(args.profile).resolve()),
        Path(args.audio).resolve(),
    )
    output_path = Path(args.output).resolve()
    atomic_json(output_path, output)
    if args.report:
        atomic_json(Path(args.report).resolve(), diagnostics)
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
