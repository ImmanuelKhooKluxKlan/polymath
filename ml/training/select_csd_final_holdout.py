#!/usr/bin/env python3
"""Select a CSD holdout using names and MIDI note-event counts only.

This utility deliberately exposes no pitch, velocity, or timing coordinates.  It
exists so a final evaluation target can be committed before audio inference or
reference conversion is performed.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import zipfile

import mido


ENGLISH_A_PATTERN = re.compile(r"^en\d{3}a$")


def note_event_count(midi_bytes: bytes) -> int:
    """Return sounding note-on event count without retaining note coordinates."""

    midi = mido.MidiFile(file=io.BytesIO(midi_bytes))
    return sum(
        1
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and int(message.velocity) > 0
    )


def select_from_archive(
    archive_path: Path,
    *,
    excluded: set[str],
    minimum_notes: int,
) -> tuple[str, list[dict[str, object]]]:
    """Apply the locked filename/count-only selection rule to a CSD archive."""

    with zipfile.ZipFile(archive_path) as archive:
        midi_members: dict[str, str] = {}
        for member in archive.namelist():
            path = PurePosixPath(member)
            if (
                len(path.parts) >= 4
                and path.parts[-3:-1] == ("english", "mid")
                and path.suffix.lower() == ".mid"
                and ENGLISH_A_PATTERN.fullmatch(path.stem)
            ):
                midi_members[path.stem] = member

        scan: list[dict[str, object]] = []
        for basename in sorted(midi_members):
            if basename in excluded:
                continue
            count = note_event_count(archive.read(midi_members[basename]))
            eligible = count >= minimum_notes
            scan.append(
                {
                    "basename": basename,
                    "targetMidiNoteEvents": count,
                    "eligible": eligible,
                }
            )
            if eligible:
                return basename, scan

    raise RuntimeError("No CSD English 'a' MIDI satisfied the locked policy")


def build_selection(
    *,
    basename: str,
    scan: list[dict[str, object]],
    parent_policy: Path,
    candidate_sha256: str,
    reference_transpose: int,
    minimum_notes: int,
) -> dict[str, object]:
    prefix = f"CSD/english"
    return {
        "schema": "polymath-final-holdout-supplement-selection-v2",
        "selectedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selectedBeforeAudioExtractionOrInference": True,
        "targetCoordinatesInspectedDuringSelection": False,
        "parentPolicy": str(parent_policy.resolve()),
        "selectionScan": scan,
        "selectedBasename": basename,
        "members": {
            "audio": f"{prefix}/wav/{basename}.wav",
            "midi": f"{prefix}/mid/{basename}.mid",
            "annotationCsv": f"{prefix}/csv/{basename}.csv",
            "annotationText": f"{prefix}/txt/{basename}.txt",
            "lyrics": f"{prefix}/lyric/{basename}.txt",
        },
        "evaluationProtocol": {
            "candidateFileSha256": candidate_sha256.lower(),
            "referenceTransposeSemitones": reference_transpose,
            "candidateMayBeRetunedAfterScoring": False,
            "minimumReferenceNotes": minimum_notes,
            "purpose": "one-shot independent supplemental research evaluation",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--minimum-notes", type=int, default=100)
    parser.add_argument("--reference-transpose", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basename, scan = select_from_archive(
        args.archive,
        excluded=set(args.exclude),
        minimum_notes=args.minimum_notes,
    )
    selection = build_selection(
        basename=basename,
        scan=scan,
        parent_policy=args.policy,
        candidate_sha256=args.candidate_sha256,
        reference_transpose=args.reference_transpose,
        minimum_notes=args.minimum_notes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selectedBasename": basename, "scan": scan}, indent=2))


if __name__ == "__main__":
    main()
