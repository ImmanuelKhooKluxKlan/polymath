"""Explain which continuous-song notes a checkpoint recovered or regressed.

Aggregate F1 can hide the mechanism of a change. This audit pairs both sides of
one frozen checkpoint comparison with the same reviewed song timelines, then
profiles reference notes that only the candidate found, only the baseline
found, or neither found. The result is development evidence for choosing the
next hypothesis; it is not a new promotion metric.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ml.training.evaluate_checkpoint import stitch_clip_notes
from ml.training.evaluate_predictions import match_notes, normalize_notes
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    evaluation_view,
    predictions_for_records,
    sha256_file,
)
from ml.training.train_muscriptor_piano import read_jsonl


def velocity_band(note: dict[str, Any]) -> str:
    velocity = float(note.get("velocity", 0.75))
    if velocity < 0.20:
        return "under_0.20"
    if velocity < 0.35:
        return "0.20_to_0.34"
    if velocity < 0.55:
        return "0.35_to_0.54"
    return "0.55_and_up"


def pitch_band(note: dict[str, Any]) -> str:
    midi = int(note["midi"])
    if midi < 48:
        return "low_A0_to_B2"
    if midi < 72:
        return "middle_C3_to_B4"
    return "high_C5_and_up"


def duration_band(note: dict[str, Any]) -> str:
    duration = float(note.get("duration", 0.1))
    if duration < 0.15:
        return "under_150ms"
    if duration < 0.50:
        return "150_to_499ms"
    if duration < 1.50:
        return "500_to_1499ms"
    return "1500ms_and_up"


def boundary_band(note: dict[str, Any]) -> str:
    remainder = float(note["time"]) % 5.0
    distance = min(remainder, 5.0 - remainder)
    return "within_150ms" if distance <= 0.15 else "away_from_boundary"


def chord_sizes(notes: list[dict[str, Any]], seconds: float = 0.04) -> dict[int, int]:
    result: dict[int, int] = {}
    ordered = sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))
    start = 0
    while start < len(ordered):
        anchor = float(ordered[start]["time"])
        end = start + 1
        while end < len(ordered) and float(ordered[end]["time"]) - anchor <= seconds:
            end += 1
        size = end - start
        label = 1 if size == 1 else (3 if size <= 3 else 4)
        for note in ordered[start:end]:
            result[int(note["index"])] = label
        start = end
    return result


def density_sizes(notes: list[dict[str, Any]], radius: float = 0.25) -> dict[int, int]:
    ordered = sorted(notes, key=lambda note: float(note["time"]))
    result: dict[int, int] = {}
    left = 0
    right = 0
    for note in ordered:
        center = float(note["time"])
        while left < len(ordered) and float(ordered[left]["time"]) < center - radius:
            left += 1
        while right < len(ordered) and float(ordered[right]["time"]) <= center + radius:
            right += 1
        result[int(note["index"])] = right - left
    return result


def profile_notes(
    notes: list[dict[str, Any]],
    context_notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Profile selected notes inside the complete reference texture."""

    context = context_notes if context_notes is not None else notes
    chords = chord_sizes(context)
    densities = density_sizes(context)

    def counts(classifier: Callable[[dict[str, Any]], str]) -> dict[str, int]:
        return dict(sorted(Counter(classifier(note) for note in notes).items()))

    return {
        "notes": len(notes),
        "velocity": counts(velocity_band),
        "pitch": counts(pitch_band),
        "duration": counts(duration_band),
        "clipBoundary": counts(boundary_band),
        "chordSize": dict(sorted(Counter(
            "single" if chords[int(note["index"])] == 1 else (
                "two_or_three" if chords[int(note["index"])] == 3 else "four_or_more"
            )
            for note in notes
        ).items())),
        "localDensity": dict(sorted(Counter(
            "sparse_1_to_3" if densities[int(note["index"])] <= 3 else (
                "medium_4_to_7" if densities[int(note["index"])] <= 7 else "dense_8_plus"
            )
            for note in notes
        ).items())),
    }


def merge_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = ("velocity", "pitch", "duration", "clipBoundary", "chordSize", "localDensity")
    merged: dict[str, Any] = {"notes": sum(int(profile["notes"]) for profile in profiles)}
    for dimension in dimensions:
        counts: Counter[str] = Counter()
        for profile in profiles:
            counts.update({key: int(value) for key, value in profile[dimension].items()})
        merged[dimension] = dict(sorted(counts.items()))
    return merged


def profile_rates(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
) -> dict[str, Any]:
    dimensions = ("velocity", "pitch", "duration", "clipBoundary", "chordSize", "localDensity")
    result: dict[str, Any] = {
        "notes": round(
            int(numerator["notes"]) / max(1, int(denominator["notes"])), 6,
        )
    }
    for dimension in dimensions:
        result[dimension] = {
            key: round(
                int(numerator[dimension].get(key, 0))
                / max(1, int(denominator[dimension].get(key, 0))),
                6,
            )
            for key in sorted(denominator[dimension])
        }
    return result


def _match_sets(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    tolerance: float,
) -> tuple[set[int], set[int]]:
    matches = match_notes(references, predictions, tolerance, instrument_aware=True)
    return (
        {int(match["reference"]["index"]) for match in matches},
        {int(match["predicted"]["index"]) for match in matches},
    )


def analyze_song_change(
    references: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float = 0.10,
) -> dict[str, Any]:
    references = normalize_notes(references)
    baseline = normalize_notes(baseline)
    candidate = normalize_notes(candidate)
    baseline_reference, baseline_prediction = _match_sets(references, baseline, tolerance)
    candidate_reference, candidate_prediction = _match_sets(references, candidate, tolerance)

    recovered_ids = candidate_reference - baseline_reference
    regressed_ids = baseline_reference - candidate_reference
    persistent_ids = set(range(len(references))) - baseline_reference - candidate_reference
    common_ids = baseline_reference & candidate_reference

    def selected(indices: set[int]) -> list[dict[str, Any]]:
        return [note for note in references if int(note["index"]) in indices]

    return {
        "referenceNotes": len(references),
        "baselinePredictedNotes": len(baseline),
        "candidatePredictedNotes": len(candidate),
        "baselineMatchedNotes": len(baseline_reference),
        "candidateMatchedNotes": len(candidate_reference),
        "baselineFalsePositives": len(baseline) - len(baseline_prediction),
        "candidateFalsePositives": len(candidate) - len(candidate_prediction),
        "commonMatches": len(common_ids),
        "recoveredReferenceNotes": len(recovered_ids),
        "regressedReferenceNotes": len(regressed_ids),
        "persistentMisses": len(persistent_ids),
        "netMatchChange": len(candidate_reference) - len(baseline_reference),
        "profiles": {
            "reference": profile_notes(references, references),
            "recovered": profile_notes(selected(recovered_ids), references),
            "regressed": profile_notes(selected(regressed_ids), references),
            "persistentMisses": profile_notes(selected(persistent_ids), references),
        },
        "examples": {
            "recovered": selected(recovered_ids)[:20],
            "regressed": selected(regressed_ids)[:20],
        },
    }


def analyze_comparison(
    comparison_path: Path,
    manifest_path: Path,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("schema") != "polymath-checkpoint-comparison-v1":
        raise TimelineScoreError("Expected a paired checkpoint comparison")
    records = read_jsonl(manifest_path)
    baseline_predictions = predictions_for_records(
        evaluation_view(comparison, "baseline"), records,
    )
    candidate_predictions = predictions_for_records(
        evaluation_view(comparison, "candidate"), records,
    )
    reference_clips = [list(record.get("notes") or []) for record in records]
    references, reference_merges = stitch_clip_notes(records, reference_clips, reference=True)
    baseline, baseline_merges = stitch_clip_notes(records, baseline_predictions, reference=False)
    candidate, candidate_merges = stitch_clip_notes(records, candidate_predictions, reference=False)

    per_song = {
        song_id: analyze_song_change(
            references.get(song_id, []),
            baseline.get(song_id, []),
            candidate.get(song_id, []),
            tolerance,
        )
        for song_id in sorted(set(references) | set(baseline) | set(candidate))
    }
    totals = {
        key: sum(int(song[key]) for song in per_song.values())
        for key in (
            "referenceNotes",
            "baselinePredictedNotes",
            "candidatePredictedNotes",
            "baselineMatchedNotes",
            "candidateMatchedNotes",
            "baselineFalsePositives",
            "candidateFalsePositives",
            "commonMatches",
            "recoveredReferenceNotes",
            "regressedReferenceNotes",
            "persistentMisses",
            "netMatchChange",
        )
    }
    profiles = {
        category: merge_profiles([
            song["profiles"][category] for song in per_song.values()
        ])
        for category in ("reference", "recovered", "regressed", "persistentMisses")
    }
    return {
        "schema": "polymath-checkpoint-change-attribution-v1",
        "onsetToleranceSeconds": tolerance,
        "comparison": str(comparison_path.resolve()),
        "comparisonSha256": sha256_file(comparison_path),
        "validationManifest": str(manifest_path.resolve()),
        "validationManifestSha256": sha256_file(manifest_path),
        "totals": totals,
        "profiles": profiles,
        "profileRates": {
            "persistentMisses": profile_rates(
                profiles["persistentMisses"], profiles["reference"],
            ),
            "recovered": profile_rates(profiles["recovered"], profiles["reference"]),
            "regressed": profile_rates(profiles["regressed"], profiles["reference"]),
        },
        "perSong": per_song,
        "boundaryAccounting": {
            "referenceContinuationMerges": reference_merges,
            "baselinePredictedMerges": baseline_merges,
            "candidatePredictedMerges": candidate_merges,
        },
        "interpretation": (
            "Recovered and regressed counts are reference-note membership changes at "
            "the same exact-pitch onset tolerance; they do not score style."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--onset-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.onset_tolerance_ms <= 1000:
        parser.error("--onset-tolerance-ms must be within (0, 1000]")
    result = analyze_comparison(
        args.comparison.resolve(),
        args.validation_manifest.resolve(),
        args.onset_tolerance_ms / 1000.0,
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), **result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
