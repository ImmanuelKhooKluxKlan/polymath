"""Route around chord expansion when the input is already a piano performance.

The pitch-class decoder is useful for reducing a full mix to a playable piano
arrangement.  It is the wrong operator for an input that the upstream arranger
has already classified as an acoustic-piano performance: in that route it can
only invent extra chord tones.  This module makes that distinction explicit
and keeps the decision independent of reference/target notes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BYPASS_PROFILES = ("acoustic-piano-preserve",)
DEFAULT_MINIMUM_SOURCE_NOTES = 24
DEFAULT_MINIMUM_SOURCE_PIANO_SHARE = 0.98
DEFAULT_MINIMUM_SOURCE_ACOUSTIC_PIANO_SHARE = 0.70
PIANO_SOURCE_INSTRUMENTS = {
    "acoustic_piano",
    "electric_piano",
    "keyboard",
    "piano",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def note_count(payload: dict[str, Any], *, label: str) -> int:
    notes = payload.get("notes")
    if not isinstance(notes, list):
        raise ValueError(f"{label} payload has no notes list")
    return len(notes)


def source_provenance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize the arranger's immutable source-instrument provenance.

    The arranger renders every selected output note as ``acoustic_piano``.
    Therefore the output ``instrument`` field cannot distinguish a piano
    recording from a full-mix reduction.  ``sourceInstrument`` preserves the
    transcriber's original instrument assignment and is the safe routing
    signal.  Missing provenance is never treated as piano evidence.
    """

    notes = payload.get("notes") or []
    values = [
        str(note.get("sourceInstrument") or "").strip().lower()
        for note in notes
        if isinstance(note, dict)
        and str(note.get("sourceInstrument") or "").strip()
    ]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    piano = sum(counts.get(name, 0) for name in PIANO_SOURCE_INSTRUMENTS)
    acoustic = counts.get("acoustic_piano", 0)
    return {
        "notesWithSourceInstrument": total,
        "sourceInstrumentCounts": dict(sorted(counts.items())),
        "sourcePianoShare": round(piano / max(1, total), 6),
        "sourceAcousticPianoShare": round(acoustic / max(1, total), 6),
    }


def route_decoder(
    pre_decoder: dict[str, Any],
    decoded: dict[str, Any],
    *,
    bypass_profiles: Iterable[str] = DEFAULT_BYPASS_PROFILES,
    minimum_source_notes: int = DEFAULT_MINIMUM_SOURCE_NOTES,
    minimum_source_piano_share: float = DEFAULT_MINIMUM_SOURCE_PIANO_SHARE,
    minimum_source_acoustic_piano_share: float = (
        DEFAULT_MINIMUM_SOURCE_ACOUSTIC_PIANO_SHARE
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose a candidate using source-side routing metadata only.

    ``pre_decoder`` is the arranged/register-normalized score immediately
    before pitch-class chord expansion.  ``decoded`` is the score after that
    expansion.  No reference score or song identity is accepted by this API,
    which makes accidental target leakage structurally impossible here.
    """

    pre_count = note_count(pre_decoder, label="Pre-decoder")
    decoded_count = note_count(decoded, label="Decoded")
    profiles = tuple(sorted({str(value) for value in bypass_profiles if str(value)}))
    if not profiles:
        raise ValueError("At least one bypass profile is required")
    if minimum_source_notes < 1:
        raise ValueError("minimum_source_notes must be positive")
    for label, value in (
        ("minimum_source_piano_share", minimum_source_piano_share),
        (
            "minimum_source_acoustic_piano_share",
            minimum_source_acoustic_piano_share,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label} must be between zero and one")

    arrangement = pre_decoder.get("pianoArrangement")
    if arrangement is not None and not isinstance(arrangement, dict):
        raise ValueError("pianoArrangement must be an object when present")
    profile = str((arrangement or {}).get("profile") or "unknown")
    provenance = source_provenance_summary(pre_decoder)
    provenance_bypass = bool(
        int(provenance["notesWithSourceInstrument"]) >= minimum_source_notes
        and float(provenance["sourcePianoShare"])
        >= float(minimum_source_piano_share)
        and float(provenance["sourceAcousticPianoShare"])
        >= float(minimum_source_acoustic_piano_share)
    )
    profile_bypass = profile in profiles
    bypassed = profile_bypass or provenance_bypass
    reason = (
        "arrangement-profile"
        if profile_bypass
        else "source-instrument-provenance"
        if provenance_bypass
        else "decoder-required"
    )
    selected_name = "pre-decoder" if bypassed else "decoded"
    selected = pre_decoder if bypassed else decoded
    output = copy.deepcopy(selected)

    report = {
        "schema": "polymath-pianist-decoder-router-report-v1",
        "decisionInputs": {
            "pianoArrangementProfile": profile,
            "bypassProfiles": list(profiles),
            **provenance,
            "minimumSourceNotes": minimum_source_notes,
            "minimumSourcePianoShare": minimum_source_piano_share,
            "minimumSourceAcousticPianoShare": (
                minimum_source_acoustic_piano_share
            ),
        },
        "targetOrReferenceDataUsed": False,
        "bypassedPitchClassDecoder": bypassed,
        "decisionReason": reason,
        "selectedCandidate": selected_name,
        "preDecoderNotes": pre_count,
        "decodedNotes": decoded_count,
        "outputNotes": note_count(output, label="Output"),
        "avoidedGeneratedNotes": max(0, decoded_count - pre_count) if bypassed else 0,
    }
    output["pianistPitchClassDecoderRouting"] = copy.deepcopy(report)
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-decoder", type=Path, required=True)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--bypass-profile",
        action="append",
        default=[],
        help="Profile that must bypass chord expansion (repeatable)",
    )
    args = parser.parse_args()
    profiles = tuple(args.bypass_profile) or DEFAULT_BYPASS_PROFILES
    output, report = route_decoder(
        load_json(args.pre_decoder.resolve()),
        load_json(args.decoded.resolve()),
        bypass_profiles=profiles,
    )
    atomic_json(args.output.resolve(), output)
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
