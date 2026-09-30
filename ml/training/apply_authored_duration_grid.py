"""Apply a frozen authored hold distribution to an arranged piano score.

This research operator changes only written/physical key-hold durations.  It
never changes pitch, onset, velocity, role, or note count.  The piano physical
performance layer is then recomputed so same-key restrikes retain a safe key-up
gap and the browser does not reuse stale release envelopes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPOSITORY_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from omr.polymath_omr.performance import shape_piano_performance  # noqa: E402
from piano_arranger import (  # noqa: E402
    apply_authored_duration_style,
    synchronize_generated_visual_holds,
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


def apply_duration_grid(
    payload: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(payload)
    style = ((profile.get("decoder") or {}).get("authoredTwoHandStyle") or {})
    notes, diagnostics = apply_authored_duration_style(
        list(output.get("notes") or []), style
    )
    if not diagnostics.get("applied"):
        return output, diagnostics

    for note in notes:
        note["scoreDuration"] = note["duration"]
        note["visualDuration"] = note["duration"]
        note.pop("audioDuration", None)
        note.pop("releaseSeconds", None)
    output["notes"] = notes
    output = shape_piano_performance(output, infer_pedal=True)
    output = synchronize_generated_visual_holds(output)
    output.setdefault("pianoArrangement", {})["authoredDurationStyle"] = diagnostics
    output["pianoArrangement"]["durationGridResearchOnly"] = True
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
    output, diagnostics = apply_duration_grid(
        load_json(input_path), load_json(profile_path)
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema": "polymath-authored-duration-grid-application-v1",
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
