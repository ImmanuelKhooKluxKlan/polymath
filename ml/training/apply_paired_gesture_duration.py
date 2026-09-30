"""Apply a frozen paired-gesture duration model to an arranged piano score."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPOSITORY_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from omr.polymath_omr.performance import shape_piano_performance  # noqa: E402
from piano_arranger import synchronize_generated_visual_holds  # noqa: E402
from piano_gesture_calibration import (  # noqa: E402
    calibrated_gesture_duration,
    gesture_sequence_contexts,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onset_groups(
    notes: list[dict[str, Any]], onset_window_seconds: float
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in sorted(
        notes, key=lambda item: (float(item.get("time", 0.0)), int(item.get("midi", 60)))
    ):
        if (
            not groups
            or float(note.get("time", 0.0))
            - float(groups[-1][0].get("time", 0.0))
            > onset_window_seconds
        ):
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def apply_paired_duration(
    payload: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(payload)
    gesture_config = ((profile.get("decoder") or {}).get("gestureDynamics") or {})
    config = gesture_config.get("pairedDurationCalibration") or {}
    if not config.get("enabled") or not output.get("notes"):
        return output, {"applied": False, "reason": "profile-disabled-or-no-notes"}

    onset_window = min(
        0.08, max(0.005, float(gesture_config.get("onsetWindowSeconds", 0.035)))
    )
    groups = onset_groups(list(output["notes"]), onset_window)
    velocities = [
        statistics.median(float(note.get("velocity", 0.72)) for note in group)
        for group in groups
    ]
    contexts = gesture_sequence_contexts(groups, velocities)
    minimum_duration = max(0.05, float(config.get("minimumDurationSeconds", 0.05)))
    maximum_duration = min(
        6.0, max(minimum_duration, float(config.get("maximumDurationSeconds", 4.0)))
    )
    changes: list[float] = []
    ratios: list[float] = []
    changed_notes = 0
    for group, context in zip(groups, contexts, strict=True):
        base_duration = statistics.median(
            float(note.get("scoreDuration", note.get("duration", 0.2)))
            for note in group
        )
        calibrated, change = calibrated_gesture_duration(
            group,
            base_duration,
            config,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
            context=context,
        )
        ratio = calibrated / max(0.05, base_duration)
        changes.append(change)
        ratios.append(ratio)
        for note in group:
            original = float(note.get("scoreDuration", note.get("duration", 0.2)))
            value = min(maximum_duration, max(minimum_duration, original * ratio))
            changed_notes += int(abs(value - original) >= 0.005)
            note["durationBeforeGestureCalibration"] = round(original, 6)
            note["duration"] = round(value, 6)
            note["scoreDuration"] = round(value, 6)
            note["visualDuration"] = round(value, 6)
            note.pop("audioDuration", None)
            note.pop("releaseSeconds", None)

    output["notes"] = [note for group in groups for note in group]
    output = shape_piano_performance(output, infer_pedal=True)
    output = synchronize_generated_visual_holds(output)
    diagnostics = {
        "applied": True,
        "profile": str(config.get("id") or "paired-gesture-duration"),
        "groups": len(groups),
        "changedNotes": changed_notes,
        "meanSignedChangeSeconds": round(sum(changes) / max(1, len(changes)), 6),
        "meanAbsoluteChangeSeconds": round(
            sum(abs(value) for value in changes) / max(1, len(changes)), 6
        ),
        "medianDurationRatio": round(statistics.median(ratios), 6),
        "minimumDurationSeconds": minimum_duration,
        "maximumDurationSeconds": maximum_duration,
    }
    output.setdefault("pianoArrangement", {})["pairedDurationCalibration"] = diagnostics
    output["pianoArrangement"]["pairedDurationResearchOnly"] = True
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    profile_path = Path(args.profile).resolve()
    output, diagnostics = apply_paired_duration(
        load_json(input_path), load_json(profile_path)
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "polymath-paired-gesture-duration-application-v1",
        "input": str(input_path),
        "inputSha256": sha256_file(input_path),
        "profile": str(profile_path),
        "profileSha256": sha256_file(profile_path),
        "output": str(output_path),
        "notes": len(output.get("notes") or []),
        "diagnostics": diagnostics,
        "invariants": {
            "pitchesFrozen": True,
            "onsetsFrozen": True,
            "velocitiesFrozen": True,
            "noteCountFrozen": True,
        },
    }
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
