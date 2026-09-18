"""Production wrapper for the listener-approved Polymath piano pipeline.

The wrapper keeps the three independently tested stages in one atomic command:
the frozen two-hand arranger, conservative low accompaniment recovery, and the
source-aware v003 melody-register safety gate.  A final, deliberately small
register-balance curve keeps the upper hand from masking the lower hand without
changing hammer velocity (and therefore without changing the note's character).
The web server writes only the final JSON, so a failed later stage can never
expose a half-processed score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SERVER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piano_arranger import arrange_payload  # noqa: E402
from ml.training.apply_adaptive_pianist_register_zones import (  # noqa: E402
    apply_register_zones,
)
from ml.training.apply_raw_support_recovery import supplement_score  # noqa: E402


MODEL_ROOT = SERVER_ROOT / "models" / "piano-arranger"
DEFAULT_ARRANGER_PROFILE = MODEL_ROOT / "polymath-pianella-v003.json"
DEFAULT_RECOVERY_PROFILE = MODEL_ROOT / "raw-support-selector-v001.json"
DEFAULT_REGISTER_PROFILE = MODEL_ROOT / "adaptive-register-v003.json"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_acoustic_hand_balance(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a subtle, continuous upper-register output trim.

    C4 starts two percent lower and the curve reaches five percent at C5.  The
    smooth transition avoids an audible level step at middle C.  MIDI velocity
    remains frozen so the selected piano sample and hammer colour do not change;
    only ``performanceGain`` is adjusted.
    """

    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
    changed = 0
    left_notes = 0
    right_notes = 0
    minimum_multiplier = 1.0
    gain_before = 0.0
    gain_after = 0.0
    for note in notes:
        if not isinstance(note, dict):
            continue
        try:
            midi = int(round(float(note.get("midi", note.get("pitch")))))
        except (TypeError, ValueError):
            continue
        if midi < 60:
            left_notes += 1
            continue
        right_notes += 1
        # C4 = 0.98, C5 and above = 0.95.  This is only -0.18 to -0.45 dB.
        multiplier = 0.98 - (0.03 * _clamp((midi - 60) / 12.0, 0.0, 1.0))
        try:
            previous = float(note.get("performanceGain", 1.0))
        except (TypeError, ValueError):
            previous = 1.0
        if not (previous > 0):
            previous = 1.0
        adjusted = _clamp(previous * multiplier, 0.25, 1.5)
        note["performanceGainBeforeHandBalance"] = round(previous, 6)
        note["performanceGain"] = round(adjusted, 6)
        minimum_multiplier = min(minimum_multiplier, multiplier)
        gain_before += previous
        gain_after += adjusted
        changed += int(abs(adjusted - previous) > 1e-9)

    diagnostics = {
        "applied": bool(changed),
        "profile": "acoustic-hand-balance-v1",
        "crossoverMidi": 60,
        "middleCMultiplier": 0.98,
        "upperRegisterMultiplier": 0.95,
        "maximumReductionDb": -0.4455,
        "leftNotesUnchanged": left_notes,
        "rightNotesAdjusted": changed,
        "rightNotesSeen": right_notes,
        "minimumAppliedMultiplier": round(minimum_multiplier, 6),
        "meanRightPerformanceGainBefore": (
            round(gain_before / right_notes, 6) if right_notes else None
        ),
        "meanRightPerformanceGainAfter": (
            round(gain_after / right_notes, 6) if right_notes else None
        ),
        "pitchesFrozen": True,
        "onsetsFrozen": True,
        "durationsFrozen": True,
        "hammerVelocitiesFrozen": True,
    }
    payload.setdefault("pianoArrangement", {})["acousticHandBalance"] = diagnostics
    return payload, diagnostics


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def run_pipeline(
    source: dict[str, Any],
    mode: str,
    arranger_profile: dict[str, Any],
    recovery_profile: dict[str, Any],
    register_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arranged = arrange_payload(source, mode, arranger_profile)
    recovered, recovery = supplement_score(
        arranged,
        source,
        recovery_profile,
        threshold=0.90,
        coverage_radius=0.15,
        cluster_seconds=0.08,
        onset_window=0.10,
        maximum_additions_per_onset=1,
        gesture_anchor_radius=0.18,
        maximum_source_midi=59,
        maximum_piano_source_midi=52,
        register_shift=0,
        minimum_output_midi=33,
        maximum_output_midi=71,
        style_velocity_blend=0.75,
        style_duration_blend=0.65,
    )
    output, register = apply_register_zones(recovered, register_profile)
    output, hand_balance = apply_acoustic_hand_balance(output)
    diagnostics = {
        "schema": "polymath-production-piano-pipeline-v1",
        "id": "polymath-pianella-v003",
        "arrangerProfile": arranger_profile.get("id"),
        "rawSupportRecovery": recovery,
        "adaptiveRegister": register,
        "acousticHandBalance": hand_balance,
        "blanketGlobalRegisterShiftEnabled": False,
    }
    output["arrangementProfile"] = "polymath-pianella-v003"
    output.setdefault("pianoArrangement", {})["productionPipeline"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "instrumental"), default="instrumental")
    parser.add_argument("--profile", default=str(DEFAULT_ARRANGER_PROFILE))
    parser.add_argument("--recovery-profile", default=str(DEFAULT_RECOVERY_PROFILE))
    parser.add_argument("--register-profile", default=str(DEFAULT_REGISTER_PROFILE))
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output, diagnostics = run_pipeline(
        load_json(Path(args.input).resolve()),
        args.mode,
        load_json(Path(args.profile).resolve()),
        load_json(Path(args.recovery_profile).resolve()),
        load_json(Path(args.register_profile).resolve()),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
