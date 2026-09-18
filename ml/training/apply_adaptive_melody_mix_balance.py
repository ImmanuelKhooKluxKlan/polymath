"""Apply the inference-safe Iowa sample-pack melody balance to a JSON score.

This offline research utility mirrors the production arranger's final mix
stage.  It changes only ``performanceGain`` and records its diagnostics; note
selection, pitch, onset, written duration, audio duration, and velocity remain
frozen.  No pianist reference or answer MIDI is read at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_mix_balance import adaptive_melody_mix_balance  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def apply_balance(
    payload: dict[str, Any],
    *,
    target_melody_share: float = 0.59,
) -> tuple[dict[str, Any], dict[str, Any]]:
    notes = payload.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Candidate payload must contain a notes list")
    balanced, diagnostics = adaptive_melody_mix_balance(
        notes,
        {
            "enabled": True,
            "targetEstimatedMelodyShare": target_melody_share,
        },
    )
    payload["notes"] = balanced
    payload.setdefault("pianoArrangement", {})[
        "adaptiveMelodyMixBalance"
    ] = diagnostics
    return payload, diagnostics


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-melody-share", type=float, default=0.59)
    args = parser.parse_args()

    payload, diagnostics = apply_balance(
        load(args.candidate.resolve()),
        target_melody_share=args.target_melody_share,
    )
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
