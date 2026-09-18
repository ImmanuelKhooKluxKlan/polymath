"""Extract a direct piano-route prediction from an unconditioned transcription.

A route-fine-tuned checkpoint can emit its learned acoustic-piano reduction next
to residual source-family predictions.  Feeding that mixed payload back through
the rule-based arranger destroys the learned reduction.  This research-only
utility keeps the direct piano stream when it is sufficiently dominant and
records the routing evidence in the output for later promotion audits.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ml.training.muscriptor_tokens import canonical_instrument_name


def extract_direct_piano(
    payload: dict[str, Any],
    *,
    minimum_share: float = 0.5,
    minimum_notes: int = 64,
    include_electric_piano: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    notes = payload.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Input payload has no notes list")
    if not 0.0 <= minimum_share <= 1.0:
        raise ValueError("minimum_share must be between 0 and 1")
    if minimum_notes < 1:
        raise ValueError("minimum_notes must be positive")

    selected: list[dict[str, Any]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        try:
            instrument = canonical_instrument_name(note.get("instrument"))
        except ValueError:
            # Unpitched families such as the decoder's ``drums`` label are
            # valid residual output but are outside the individual-instrument
            # tokenizer. They simply are not part of the direct piano route.
            continue
        accepted = {"acoustic_piano"}
        if include_electric_piano:
            accepted.add("electric_piano")
        if instrument not in accepted:
            continue
        copied = copy.deepcopy(note)
        copied["instrument"] = "acoustic_piano"
        selected.append(copied)

    source_count = sum(isinstance(note, dict) for note in notes)
    piano_share = len(selected) / source_count if source_count else 0.0
    if len(selected) < minimum_notes or piano_share < minimum_share:
        raise ValueError(
            "Direct piano route is not dominant enough: "
            f"{len(selected)} notes, {piano_share:.3f} share"
        )

    selected.sort(
        key=lambda note: (
            float(note.get("time", note.get("startTime", note.get("start", 0.0)))),
            int(round(float(note.get("midi", note.get("pitch", 0))))),
        )
    )
    diagnostics = {
        "schema": "polymath-direct-piano-route-extraction-v1",
        "sourceNotes": source_count,
        "selectedPianoNotes": len(selected),
        "pianoShare": round(piano_share, 6),
        "minimumShare": minimum_share,
        "minimumNotes": minimum_notes,
        "includedPianoFamilies": (
            ["acoustic_piano", "electric_piano"]
            if include_electric_piano
            else ["acoustic_piano"]
        ),
        "decision": "DIRECT_PIANO_ROUTE",
        "warning": (
            "Research-only routing decision. Complete-song score, duration, "
            "retrigger, and blind-listening gates remain mandatory."
        ),
    }
    output = copy.deepcopy(payload)
    output["notes"] = selected
    output["instrument"] = "piano"
    output["instrumentGroups"] = ["acoustic_piano"]
    output.setdefault("diagnostics", {})["directPianoRoute"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--minimum-share", type=float, default=0.5)
    parser.add_argument("--minimum-notes", type=int, default=64)
    parser.add_argument(
        "--include-electric-piano",
        action="store_true",
        help="Treat acoustic and electric piano decoder streams as one piano-family route.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    output, diagnostics = extract_direct_piano(
        payload,
        minimum_share=args.minimum_share,
        minimum_notes=args.minimum_notes,
        include_electric_piano=args.include_electric_piano,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
