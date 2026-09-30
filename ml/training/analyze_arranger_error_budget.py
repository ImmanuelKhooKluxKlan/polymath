"""Measure where a full-mix piano arrangement loses factual accuracy.

This audit deliberately separates four stages that are often collapsed into
one score:

1. onset availability;
2. pitch-class selection;
3. exact octave/register placement;
4. final candidate precision.

The raw-source ceilings are optimistic *oracles*.  They answer whether useful
evidence existed before the arranger, not whether the runtime knew which event
to choose.  No reference coordinate is exposed to inference and this script
never edits a model profile.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Any

try:  # Package and direct-script execution.
    from .evaluate_piano_arranger import (
        f1_metrics,
        greedy_matches,
        load_json,
        normalize_notes,
        notes_inside_ranges,
        prepare_reference_notes,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover
    from evaluate_piano_arranger import (  # type: ignore
        f1_metrics,
        greedy_matches,
        load_json,
        normalize_notes,
        notes_inside_ranges,
        prepare_reference_notes,
        trusted_source_ranges,
    )


PERCUSSION_TOKENS = (
    "drum",
    "percussion",
    "cymbal",
    "hi_hat",
    "hihat",
    "snare",
    "timpani",
)


def is_percussion(item: dict[str, Any]) -> bool:
    instrument = str(item.get("instrument") or "").strip().lower()
    return any(token in instrument for token in PERCUSSION_TOKENS)


def pitched_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "notes": [
            item
            for item in payload.get("notes", [])
            if isinstance(item, dict) and not is_percussion(item)
        ],
    }


def clip_observed_notes(
    notes: list[dict[str, float | int]],
    pair: dict[str, Any],
    ranges: list[tuple[float, float]] | None,
) -> list[dict[str, float | int]]:
    clipped = notes_inside_ranges(notes, ranges)
    candidate_end = pair.get("candidateEndSeconds")
    if candidate_end is None:
        return clipped
    end = float(candidate_end)
    if not math.isfinite(end) or end <= 0:
        raise ValueError("candidateEndSeconds must be a positive finite number")
    return [note for note in clipped if float(note["time"]) < end]


def greedy_onset_matches(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    tolerance: float,
) -> list[tuple[dict[str, float | int], dict[str, float | int]]]:
    """Pair nearest onsets one-to-one while deliberately ignoring pitch."""

    ordered = sorted(
        enumerate(observed), key=lambda item: (float(item[1]["time"]), item[0])
    )
    times = [float(note["time"]) for _index, note in ordered]
    used: set[int] = set()
    matches: list[tuple[dict[str, float | int], dict[str, float | int]]] = []
    for target in reference:
        target_time = float(target["time"])
        position = bisect.bisect_left(times, target_time)
        candidates: list[tuple[float, int, dict[str, float | int]]] = []
        left = position - 1
        while left >= 0 and target_time - times[left] <= tolerance:
            original_index, note = ordered[left]
            if original_index not in used:
                candidates.append(
                    (abs(float(note["time"]) - target_time), original_index, note)
                )
            left -= 1
        right = position
        while right < len(ordered) and times[right] - target_time <= tolerance:
            original_index, note = ordered[right]
            if original_index not in used:
                candidates.append(
                    (abs(float(note["time"]) - target_time), original_index, note)
                )
            right += 1
        if not candidates:
            continue
        _distance, original_index, note = min(
            candidates,
            key=lambda item: (
                item[0],
                abs(int(item[2]["midi"]) - int(target["midi"])),
                item[1],
            ),
        )
        used.add(original_index)
        matches.append((target, note))
    return matches


def perfect_pruned_metrics(reference_count: int, supported_count: int) -> dict[str, Any]:
    """F1 ceiling if an oracle retained only supported source events."""

    return f1_metrics(reference_count, supported_count, supported_count)


def stage_metrics(
    reference: list[dict[str, float | int]],
    raw: list[dict[str, float | int]],
    candidate: list[dict[str, float | int]],
    tolerance: float,
) -> dict[str, Any]:
    reference_count = len(reference)
    raw_onset = len(greedy_onset_matches(reference, raw, tolerance))
    raw_pitch_class = len(
        greedy_matches(reference, raw, tolerance, octave_equivalent=True)
    )
    raw_exact = len(greedy_matches(reference, raw, tolerance))
    candidate_onset = len(greedy_onset_matches(reference, candidate, tolerance))
    candidate_pitch_class = len(
        greedy_matches(reference, candidate, tolerance, octave_equivalent=True)
    )
    candidate_exact = len(greedy_matches(reference, candidate, tolerance))

    return {
        "counts": {
            "referenceNotes": reference_count,
            "rawPitchedNotes": len(raw),
            "candidateNotes": len(candidate),
        },
        "rawEvidence": {
            "onsetRecallCeiling": round(raw_onset / max(1, reference_count), 6),
            "pitchClassRecallCeiling": round(
                raw_pitch_class / max(1, reference_count), 6
            ),
            "exactPitchRecallCeiling": round(raw_exact / max(1, reference_count), 6),
            "perfectPrunedOnsetF1Ceiling": perfect_pruned_metrics(
                reference_count, raw_onset
            )["f1"],
            "perfectPrunedPitchClassF1Ceiling": perfect_pruned_metrics(
                reference_count, raw_pitch_class
            )["f1"],
            "perfectPrunedExactPitchF1Ceiling": perfect_pruned_metrics(
                reference_count, raw_exact
            )["f1"],
        },
        "candidate": {
            "onset": f1_metrics(reference_count, len(candidate), candidate_onset),
            "pitchClass": f1_metrics(
                reference_count, len(candidate), candidate_pitch_class
            ),
            "exactPitch": f1_metrics(
                reference_count, len(candidate), candidate_exact
            ),
        },
        "recoverableHeadroom": {
            "registerF1Points": round(
                f1_metrics(reference_count, len(candidate), candidate_pitch_class)["f1"]
                - f1_metrics(reference_count, len(candidate), candidate_exact)["f1"],
                6,
            ),
            "candidatePitchSelectionF1Points": round(
                f1_metrics(reference_count, len(candidate), candidate_onset)["f1"]
                - f1_metrics(reference_count, len(candidate), candidate_pitch_class)["f1"],
                6,
            ),
            "rawSelectionAndGenerationF1Points": round(
                perfect_pruned_metrics(reference_count, raw_pitch_class)["f1"]
                - f1_metrics(reference_count, len(candidate), candidate_pitch_class)["f1"],
                6,
            ),
        },
        "approximateReferenceErrorCounts": {
            "candidateExactMatches": candidate_exact,
            "candidatePitchClassButWrongRegister": max(
                0, candidate_pitch_class - candidate_exact
            ),
            "candidateOnsetButWrongPitchClass": max(
                0, candidate_onset - candidate_pitch_class
            ),
            "candidateMissingOnset": max(0, reference_count - candidate_onset),
            "rawPitchClassSupportedButNotRetained": max(
                0, raw_pitch_class - candidate_pitch_class
            ),
            "upstreamMissingPitchClass": max(0, reference_count - raw_pitch_class),
        },
    }


def candidate_path(candidate_directory: Path, song_id: str) -> Path:
    alternatives = (
        candidate_directory / f"{song_id}-arranged.json",
        candidate_directory / f"{song_id}.json",
    )
    for path in alternatives:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No candidate for {song_id!r}; checked "
        + ", ".join(str(path) for path in alternatives)
    )


def weighted_average(
    rows: list[tuple[float, dict[str, Any]]], path: tuple[str, ...]
) -> float:
    numerator = 0.0
    denominator = 0.0
    for weight, row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        numerator += weight * float(value)
        denominator += weight
    return round(numerator / max(1e-9, denominator), 6)


def analyze_manifest(
    manifest: dict[str, Any],
    candidate_directory: Path,
    tolerance: float,
) -> dict[str, Any]:
    songs: dict[str, Any] = {}
    weighted_rows: list[tuple[float, dict[str, Any]]] = []
    for pair in manifest.get("pairs") or []:
        song_id = str(pair["id"])
        alignment = load_json(Path(pair["alignmentReport"]))
        ranges = trusted_source_ranges(alignment)
        reference = prepare_reference_notes(
            pair,
            load_json(Path(pair["target"])),
            alignment,
        )
        raw = clip_observed_notes(
            normalize_notes(pitched_payload(load_json(Path(pair["source"])))),
            pair,
            ranges,
        )
        candidate = clip_observed_notes(
            normalize_notes(load_json(candidate_path(candidate_directory, song_id))),
            pair,
            ranges,
        )
        metrics = stage_metrics(reference, raw, candidate, tolerance)
        weight = max(0.0, float(pair.get("weight", 1.0)))
        songs[song_id] = {"weight": weight, **metrics}
        weighted_rows.append((weight, metrics))

    metric_paths = {
        "rawOnsetRecallCeiling": ("rawEvidence", "onsetRecallCeiling"),
        "rawPitchClassRecallCeiling": (
            "rawEvidence",
            "pitchClassRecallCeiling",
        ),
        "rawExactPitchRecallCeiling": (
            "rawEvidence",
            "exactPitchRecallCeiling",
        ),
        "perfectPrunedPitchClassF1Ceiling": (
            "rawEvidence",
            "perfectPrunedPitchClassF1Ceiling",
        ),
        "candidateOnsetF1": ("candidate", "onset", "f1"),
        "candidatePitchClassF1": ("candidate", "pitchClass", "f1"),
        "candidateExactPitchF1": ("candidate", "exactPitch", "f1"),
        "registerF1Headroom": ("recoverableHeadroom", "registerF1Points"),
        "rawSelectionAndGenerationF1Headroom": (
            "recoverableHeadroom",
            "rawSelectionAndGenerationF1Points",
        ),
    }
    aggregate = {
        name: weighted_average(weighted_rows, path)
        for name, path in metric_paths.items()
    }
    aggregate["largestMeasuredHeadroom"] = max(
        ("register", aggregate["registerF1Headroom"]),
        (
            "selection-and-generation",
            aggregate["rawSelectionAndGenerationF1Headroom"],
        ),
        key=lambda item: item[1],
    )[0]
    return {
        "schema": "polymath-arranger-error-budget-v1",
        "evidenceBoundary": (
            "Evaluation only. All reference warps and trusted windows are frozen; "
            "raw-source values are optimistic oracles, not runtime accuracy."
        ),
        "toleranceSeconds": tolerance,
        "songs": songs,
        "weightedAggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    args = parser.parse_args()

    tolerance = max(0.01, min(1.0, float(args.tolerance_seconds)))
    manifest_path = Path(args.manifest).resolve()
    candidate_directory = Path(args.candidate_dir).resolve()
    report = analyze_manifest(load_json(manifest_path), candidate_directory, tolerance)
    report["inputs"] = {
        "manifest": str(manifest_path),
        "candidateDirectory": str(candidate_directory),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["weightedAggregate"]}, indent=2))


if __name__ == "__main__":
    main()
