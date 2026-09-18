"""Audit octave routing and duplicate layers against an authored pianist.

Pitch-class accuracy can hide a very audible failure: the arranger finds the
right note name but places it in the wrong octave.  This report pairs notes by
pitch class on the frozen musical clock, then explains the required octave
move using candidate-only fields (hand, role, source family, register and
position inside the current gesture).

The script is evaluation-only.  It never edits a profile and never exposes a
reference coordinate to runtime inference.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # Package and direct CLI execution.
    from .analyze_full_mix_arranger import greedy_match_indices
    from .analyze_pianist_gesture_patterns import (
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover
    from analyze_full_mix_arranger import greedy_match_indices  # type: ignore
    from analyze_pianist_gesture_patterns import (  # type: ignore
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        trusted_source_ranges,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def family(note: dict[str, Any]) -> str:
    value = str(
        note.get("sourceInstrument") or note.get("instrument") or "other"
    ).lower()
    if "voice" in value or "vocal" in value or "choir" in value:
        return "voice"
    if "guitar" in value:
        return "guitar"
    if "bass" in value or "contrabass" in value:
        return "bass"
    if "piano" in value or "keyboard" in value:
        return "piano"
    if any(token in value for token in ("violin", "viola", "cello", "string")):
        return "strings"
    return "other"


def pitch_band(midi: int) -> str:
    if midi < 48:
        return "bass-<C3"
    if midi < 60:
        return "lower-C3-B3"
    if midi < 72:
        return "middle-C4-B4"
    if midi < 84:
        return "upper-C5-B5"
    return "high->=C6"


def shift_band(semitones: int) -> str:
    if semitones <= -24:
        return "down-2+-octaves"
    if semitones == -12:
        return "down-1-octave"
    if semitones == 0:
        return "exact-octave"
    if semitones == 12:
        return "up-1-octave"
    if semitones >= 24:
        return "up-2+-octaves"
    return "non-octave-error"


def proportion_rows(
    records: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    output: list[dict[str, Any]] = []
    for name, values in groups.items():
        shifts = Counter(str(item["shiftBand"]) for item in values)
        exact = shifts["exact-octave"]
        output.append(
            {
                "name": name,
                "pairedNotes": len(values),
                "exactOctave": exact,
                "exactOctaveRate": round(exact / max(1, len(values)), 6),
                "requiredShiftBands": dict(shifts.most_common()),
            }
        )
    return sorted(output, key=lambda item: (-int(item["pairedNotes"]), str(item["name"])))


def decorate_gestures(notes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    decorated: dict[int, dict[str, Any]] = {}
    for group_index, group in enumerate(group_onsets(notes, 0.035)):
        ordered_midis = sorted({int(item["midi"]) for item in group})
        for item in group:
            midi = int(item["midi"])
            payload_index = int(item["_payloadIndex"])
            decorated[payload_index] = {
                "gestureIndex": group_index,
                "gestureSize": len(ordered_midis),
                "pitchRank": ordered_midis.index(midi) / max(1, len(ordered_midis) - 1),
                "isLowest": midi == ordered_midis[0],
                "isHighest": midi == ordered_midis[-1],
                "samePitchClassLayers": sum(
                    other % 12 == midi % 12 for other in ordered_midis
                ),
            }
    return decorated


def duplicate_layer_summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        try:
            source_index = int(note["sourceIndex"])
        except (KeyError, TypeError, ValueError):
            continue
        groups[(int(round(float(note["time"]) * 1000)), source_index)].append(note)
    duplicates = [values for values in groups.values() if len(values) > 1]
    cross_hand = [
        values
        for values in duplicates
        if len({str(note.get("hand") or "unknown") for note in values}) > 1
    ]
    octave_layers = [
        values
        for values in duplicates
        if len({int(note["midi"]) for note in values}) > 1
        and len({int(note["midi"]) % 12 for note in values}) == 1
    ]
    return {
        "sameSourceSameOnsetGroups": len(duplicates),
        "notesInsideThoseGroups": sum(len(values) for values in duplicates),
        "crossHandGroups": len(cross_hand),
        "samePitchClassOctaveLayerGroups": len(octave_layers),
        "examples": [
            {
                "time": round(float(values[0]["time"]), 6),
                "sourceIndex": int(values[0]["sourceIndex"]),
                "midis": sorted(int(note["midi"]) for note in values),
                "hands": sorted({str(note.get("hand") or "unknown") for note in values}),
                "roles": sorted(
                    {str(note.get("arrangementRole") or "unknown") for note in values}
                ),
            }
            for values in octave_layers[:30]
        ],
    }


def analyze_song(
    row: dict[str, Any],
    tolerance: float,
    candidate_dir: Path | None = None,
    reference_transpose_override: int | None = None,
) -> dict[str, Any]:
    reference_path = Path(str(row["reference"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = (
        (candidate_dir / f'{str(row.get("id") or "candidate")}.json').resolve()
        if candidate_dir is not None
        else Path(str(row["candidate"])).resolve()
    )
    transpose = (
        int(reference_transpose_override)
        if reference_transpose_override is not None
        else int(row.get("referenceTransposeSemitones") or 0)
    )
    reference_end = row.get("referenceEndSeconds")
    candidate_end = row.get("candidateEndSeconds")
    reference_end = float(reference_end) if reference_end is not None else None
    candidate_end = float(candidate_end) if candidate_end is not None else None
    alignment = load_json(alignment_path)
    ranges = trusted_source_ranges(alignment)
    normalized_reference = normalize_notes(
        load_json(reference_path),
        transpose=transpose,
        hard_end=reference_end,
    )
    mapped_reference = (
        normalized_reference
        if bool(row.get("referenceAlreadyAligned"))
        else map_reference(normalized_reference, monotonic_anchors(alignment))
    )
    reference = [
        note
        for note in mapped_reference
        if inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    candidate = [
        note
        for note in normalize_notes(load_json(candidate_path), hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    decorations = decorate_gestures(candidate)
    pairs = greedy_match_indices(
        reference, candidate, tolerance, octave_equivalent=True
    )
    records: list[dict[str, Any]] = []
    for reference_index, candidate_index in pairs:
        target = reference[reference_index]
        observed = candidate[candidate_index]
        decoration = decorations[int(observed["_payloadIndex"])]
        shift = int(target["midi"]) - int(observed["midi"])
        source_midi = int(
            round(
                finite(
                    observed.get(
                        "sourceMidiBeforeArrangement",
                        observed.get("originalMidiBeforeRangeShift", observed["midi"]),
                    ),
                    float(observed["midi"]),
                )
            )
        )
        records.append(
            {
                "referenceIndex": reference_index,
                "candidateIndex": candidate_index,
                "time": round(float(observed["time"]), 6),
                "onsetErrorSeconds": round(
                    float(observed["time"]) - float(target["time"]), 6
                ),
                "referenceMidi": int(target["midi"]),
                "candidateMidi": int(observed["midi"]),
                "sourceMidi": source_midi,
                "requiredShiftSemitones": shift,
                "shiftBand": shift_band(shift),
                "candidateHand": str(observed.get("hand") or "unknown"),
                "candidateRole": str(observed.get("arrangementRole") or "unknown"),
                "sourceFamily": family(observed),
                "candidatePitchBand": pitch_band(int(observed["midi"])),
                "sourcePitchBand": pitch_band(source_midi),
                **decoration,
            }
        )
    exact = sum(int(item["requiredShiftSemitones"] == 0) for item in records)
    octave_moves = sum(
        int(item["requiredShiftSemitones"] % 12 == 0) for item in records
    )
    return {
        "id": str(row.get("id") or candidate_path.stem),
        "inputs": {
            "reference": str(reference_path),
            "alignment": str(alignment_path),
            "candidate": str(candidate_path),
        },
        "referenceAlreadyAligned": bool(row.get("referenceAlreadyAligned")),
        "counts": {
            "referenceNotes": len(reference),
            "candidateNotes": len(candidate),
            "pitchClassPairedNotes": len(records),
            "exactOctaveBefore": exact,
            "exactOctaveRateBefore": round(exact / max(1, len(records)), 6),
            "octaveOnlyPairs": octave_moves,
            "oracleExactOctaveCeilingWithinPairedPopulation": round(
                octave_moves / max(1, len(records)), 6
            ),
            "unpairedReferenceNotes": len(reference) - len(records),
            "unpairedCandidateNotes": len(candidate) - len(records),
        },
        "requiredShiftBands": dict(
            Counter(str(item["shiftBand"]) for item in records).most_common()
        ),
        "byCandidateHand": proportion_rows(
            records, lambda item: str(item["candidateHand"])
        ),
        "byCandidateRole": proportion_rows(
            records, lambda item: str(item["candidateRole"])
        ),
        "bySourceFamily": proportion_rows(
            records, lambda item: str(item["sourceFamily"])
        ),
        "byCandidatePitchBand": proportion_rows(
            records, lambda item: str(item["candidatePitchBand"])
        ),
        "byGesturePosition": proportion_rows(
            records,
            lambda item: (
                "singleton"
                if int(item["gestureSize"]) == 1
                else "lowest"
                if bool(item["isLowest"])
                else "highest"
                if bool(item["isHighest"])
                else "inner"
            ),
        ),
        "duplicateLayers": duplicate_layer_summary(candidate),
        "pairs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate-dir",
        help="Optional directory containing one current-code candidate named <song-id>.json.",
    )
    parser.add_argument(
        "--reference-transpose-override",
        type=int,
        help=(
            "Use one declared octave convention for every reference. The value must "
            "be an octave multiple; this is intended for register audits only."
        ),
    )
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    candidate_dir = Path(args.candidate_dir).resolve() if args.candidate_dir else None
    transpose_override = args.reference_transpose_override
    if transpose_override is not None and (
        abs(int(transpose_override)) > 48 or int(transpose_override) % 12 != 0
    ):
        raise ValueError("reference transpose override must be an octave multiple from -48 to 48")
    rows = manifest.get("songs") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("The manifest must contain at least one row in songs")
    songs = [
        analyze_song(
            row,
            max(0.01, float(args.tolerance_seconds)),
            candidate_dir,
            transpose_override,
        )
        for row in rows
    ]
    all_pairs = [item for song in songs for item in song["pairs"]]
    payload = {
        "schema": "polymath-pianist-register-audit-v1",
        "evidenceBoundary": (
            "Evaluation only; fixed alignments and approved windows. Kiss Me at/after "
            "02:30 is excluded. Oracle values are ceilings, not runtime claims."
        ),
        "manifest": str(manifest_path),
        "toleranceSeconds": float(args.tolerance_seconds),
        "aggregate": {
            "songs": len(songs),
            "pitchClassPairedNotes": len(all_pairs),
            "requiredShiftBands": dict(
                Counter(str(item["shiftBand"]) for item in all_pairs).most_common()
            ),
            "byCandidateHand": proportion_rows(
                all_pairs, lambda item: str(item["candidateHand"])
            ),
            "byCandidateRole": proportion_rows(
                all_pairs, lambda item: str(item["candidateRole"])
            ),
            "byGesturePosition": proportion_rows(
                all_pairs,
                lambda item: (
                    "singleton"
                    if int(item["gestureSize"]) == 1
                    else "lowest"
                    if bool(item["isLowest"])
                    else "highest"
                    if bool(item["isHighest"])
                    else "inner"
                ),
            ),
        },
        "songs": songs,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "aggregate": payload["aggregate"],
                "songs": [
                    {"id": song["id"], **song["counts"]} for song in songs
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
