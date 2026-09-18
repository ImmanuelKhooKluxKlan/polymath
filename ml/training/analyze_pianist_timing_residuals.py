"""Measure source-clock onset residuals for aligned pianist candidates.

This is a diagnostic only.  It may use the answer key to explain errors, but
the reported offsets must not be copied into production unless a separate
song-held-out rule can predict them from inference-time evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
from statistics import median
from typing import Any

from .evaluate_piano_arranger import (
    greedy_matches,
    load_json,
    prepare_reference_notes,
)
from .search_default_piano_pipeline import clip_candidate
from .train_pianist_intro_motif import (
    estimate_source_pulse,
    is_voice,
    source_groups,
)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    blend = position - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"matches": 0, "medianMs": None, "p10Ms": None, "p90Ms": None}
    return {
        "matches": len(values),
        "medianMs": round(median(values) * 1000.0, 3),
        "p10Ms": round(float(quantile(values, 0.10)) * 1000.0, 3),
        "p90Ms": round(float(quantile(values, 0.90)) * 1000.0, 3),
    }


def candidate_role(note: dict[str, Any]) -> str:
    role = str(note.get("_arrangementRole") or note.get("arrangementRole") or "").strip().lower()
    source_instrument = str(
        note.get("_sourceInstrument") or note.get("sourceInstrument") or ""
    ).lower()
    if role == "melody" or source_instrument == "voice":
        return "melody"
    return "accompaniment"


def attach_candidate_roles(
    normalized: list[dict[str, Any]], payload: dict[str, Any]
) -> None:
    roles: dict[tuple[float, int], deque[tuple[str, str]]] = defaultdict(deque)
    for note in payload.get("notes") or []:
        try:
            key = (round(float(note.get("time")), 6), int(round(float(note.get("midi")))))
        except (TypeError, ValueError):
            continue
        roles[key].append(
            (
                str(note.get("arrangementRole") or ""),
                str(note.get("sourceInstrument") or ""),
            )
        )
    for note in normalized:
        key = (round(float(note["time"]), 6), int(note["midi"]))
        if roles[key]:
            role, instrument = roles[key].popleft()
            note["_arrangementRole"] = role
            note["_sourceInstrument"] = instrument


def analyze_song(
    row: dict[str, Any], reference_payload: dict[str, Any], alignment: dict[str, Any],
    candidate_payload: dict[str, Any], source_payload: dict[str, Any]
) -> dict[str, Any]:
    reference = prepare_reference_notes(row, reference_payload, alignment)
    if row.get("candidateEndSeconds") is not None:
        end = float(row["candidateEndSeconds"])
        reference = [note for note in reference if float(note["time"]) < end]
    candidate = clip_candidate(candidate_payload, row, alignment)
    attach_candidate_roles(candidate, candidate_payload)
    matches = greedy_matches(reference, candidate, 0.25, octave_equivalent=True)
    errors = [float(observed["time"]) - float(target["time"]) for target, observed in matches]
    raw_source_groups = source_groups(source_payload)
    harmonic_source_groups = [
        group for group in raw_source_groups if not all(is_voice(note) for note in group)
    ]
    maximum_time = max(
        (float(note["time"]) for note in candidate), default=0.0
    ) + 1.0
    try:
        pulse = float(estimate_source_pulse(harmonic_source_groups, maximum_time))
    except ValueError:
        pulse = 0.0
    phase_errors = Counter()
    if pulse > 0:
        for error in errors:
            phase_errors[str(max(-3, min(3, round(error / pulse))))] += 1
    by_role: dict[str, list[float]] = {"melody": [], "accompaniment": []}
    by_section: dict[int, list[float]] = {}
    for target, observed in matches:
        error = float(observed["time"]) - float(target["time"])
        by_role[candidate_role(observed)].append(error)
        section = int(float(observed["time"]) // 30.0)
        by_section.setdefault(section, []).append(error)
    return {
        "referenceNotes": len(reference),
        "candidateNotes": len(candidate),
        "pitchClassMatchesAt250ms": len(matches),
        "all": summarize(errors),
        "sourcePulseSeconds": round(pulse, 6) if pulse > 0 else None,
        "observedMinusReferencePulseMultiples": dict(
            sorted(phase_errors.items(), key=lambda item: int(item[0]))
        ),
        "roles": {role: summarize(values) for role, values in by_role.items()},
        "sections": [
            {
                "startSeconds": section * 30,
                "endSeconds": (section + 1) * 30,
                **summarize(by_section[section]),
            }
            for section in sorted(by_section)
        ],
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--song", action="append", default=[])
    args = parser.parse_args()
    manifest = load_json(args.manifest.resolve())
    selected = {str(value) for value in args.song if str(value)}
    songs: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row.get("id") or "")
        if selected and song_id not in selected:
            continue
        if not row.get("baseline"):
            continue
        alignment = load_json(Path(str(row["alignment"])).resolve())
        songs.append(
            {
                "id": song_id,
                **analyze_song(
                    row,
                    load_json(Path(str(row["reference"])).resolve()),
                    alignment,
                    load_json(Path(str(row["candidate"])).resolve()),
                    load_json(Path(str(row["source"])).resolve()),
                ),
            }
        )
    result = {
        "schema": "polymath-pianist-timing-residual-audit-v1",
        "answerKeyUsedForDiagnosisOnly": True,
        "songs": songs,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
