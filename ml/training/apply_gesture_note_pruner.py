"""Apply a validated research gesture-note pruner to one arranged JSON."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_gesture_structure import prune_gesture_notes  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def profile_with_overrides(
    profile: dict[str, Any],
    *,
    threshold: float | None = None,
    minimum_probability_gap: float | None = None,
    maximum_removals_per_gesture: int | None = None,
    minimum_group_size: int | None = None,
) -> dict[str, Any]:
    output = copy.deepcopy(profile)
    values = {
        "threshold": threshold,
        "minimumProbabilityGap": minimum_probability_gap,
        "maximumRemovalsPerGesture": maximum_removals_per_gesture,
        "minimumGroupSize": minimum_group_size,
    }
    for key, value in values.items():
        if value is not None:
            output[key] = value
    if not 0.0 <= float(output.get("threshold", 0.5)) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if float(output.get("minimumProbabilityGap", 0.0)) < 0.0:
        raise ValueError("minimum probability gap cannot be negative")
    if int(output.get("maximumRemovalsPerGesture", 1)) < 0:
        raise ValueError("maximum removals per gesture cannot be negative")
    if int(output.get("minimumGroupSize", 2)) < 2:
        raise ValueError("minimum group size must be at least 2")
    output["operatingPointOverrides"] = {
        key: value for key, value in values.items() if value is not None
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--minimum-probability-gap", type=float)
    parser.add_argument("--maximum-removals-per-gesture", type=int)
    parser.add_argument("--minimum-group-size", type=int)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    profile_path = Path(args.profile).resolve()
    output_path = Path(args.output).resolve()
    payload = load(candidate_path)
    profile = profile_with_overrides(
        load(profile_path),
        threshold=args.threshold,
        minimum_probability_gap=args.minimum_probability_gap,
        maximum_removals_per_gesture=args.maximum_removals_per_gesture,
        minimum_group_size=args.minimum_group_size,
    )
    notes, diagnostics = prune_gesture_notes(payload.get("notes") or [], profile)
    payload["notes"] = notes
    arrangement = payload.setdefault("pianoArrangement", {})
    arrangement["gestureNotePruning"] = {
        **diagnostics,
        "profileSha256": profile.get("profileSha256"),
        "operatingPointOverrides": profile.get("operatingPointOverrides", {}),
        "rightHandMelodyProtected": bool(profile.get("preserveMelody", True)),
    }
    arrangement["outputNoteCount"] = len(notes)
    if isinstance(payload.get("transcriptionCleanup"), dict):
        payload["transcriptionCleanup"]["outputNotes"] = len(notes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
