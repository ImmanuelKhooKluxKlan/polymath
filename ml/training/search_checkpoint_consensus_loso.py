"""Search an inference-safe multi-checkpoint consensus under whole-song LOSO.

The target MIDI is used only after a candidate has been generated.  Candidate
construction sees the incumbent transcription plus two independently trained
checkpoint outputs.  This lets us test whether checkpoint agreement can repair
onsets or recover missing events without leaking the answer key into inference.

This is a research/distillation tool.  Running three 5.5 GB checkpoints for a
live request is intentionally not treated as the intended production design.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

from .evaluate_piano_arranger import (
    greedy_matches,
    load_json,
    normalize_notes,
    prepare_reference_notes,
)
from .search_default_piano_pipeline import clip_candidate, measure
from .transcription_accuracy_scorecard import atomic_json, f1_from_counts


def one_to_one_matches(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    radius_seconds: float,
) -> list[tuple[int, int]]:
    """Return deterministic same-pitch nearest pairs within one time radius."""

    candidates: list[tuple[float, int, int]] = []
    right_by_pitch: dict[int, list[int]] = defaultdict(list)
    for right_index, note in enumerate(right):
        right_by_pitch[int(note["midi"])].append(right_index)
    for left_index, left_note in enumerate(left):
        for right_index in right_by_pitch.get(int(left_note["midi"]), []):
            distance = abs(float(left_note["time"]) - float(right[right_index]["time"]))
            if distance <= radius_seconds:
                candidates.append((distance, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, left_index, right_index in sorted(
        candidates, key=lambda item: (item[0], item[1], item[2])
    ):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append((left_index, right_index))
    return sorted(pairs)


def consensus_notes(
    incumbent: list[dict[str, Any]],
    checkpoint_a: list[dict[str, Any]],
    checkpoint_b: list[dict[str, Any]],
    *,
    match_radius_seconds: float,
    onset_blend: float,
    require_both_for_onset: bool,
    add_corroborated_missing: bool,
    corroboration_radius_seconds: float,
    incumbent_exclusion_radius_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, int | float | bool]]:
    """Fuse three outputs using model agreement only.

    Existing incumbent events are retained.  Their onsets may move toward the
    median candidate-checkpoint onset.  New events are admitted only when both
    candidate checkpoints agree and the incumbent has no nearby same-pitch
    event.  Durations and velocities remain incumbent-authored for incumbent
    events so this experiment isolates event/onset accuracy.
    """

    if not 0.0 <= onset_blend <= 1.0:
        raise ValueError("onset_blend must be between 0 and 1")
    output = [deepcopy(note) for note in incumbent]
    incumbent_to_a = dict(one_to_one_matches(incumbent, checkpoint_a, match_radius_seconds))
    incumbent_to_b = dict(one_to_one_matches(incumbent, checkpoint_b, match_radius_seconds))
    shifted = 0
    supported_by_both = 0
    for incumbent_index, note in enumerate(output):
        supporting_times: list[float] = []
        if incumbent_index in incumbent_to_a:
            supporting_times.append(float(checkpoint_a[incumbent_to_a[incumbent_index]]["time"]))
        if incumbent_index in incumbent_to_b:
            supporting_times.append(float(checkpoint_b[incumbent_to_b[incumbent_index]]["time"]))
        if len(supporting_times) == 2:
            supported_by_both += 1
        if not supporting_times or (require_both_for_onset and len(supporting_times) < 2):
            continue
        source_time = float(note["time"])
        destination = float(median(supporting_times))
        updated_time = source_time + onset_blend * (destination - source_time)
        if abs(updated_time - source_time) > 1e-12:
            note["time"] = max(0.0, updated_time)
            shifted += 1

    additions = 0
    if add_corroborated_missing:
        matched_a = set(incumbent_to_a.values())
        matched_b = set(incumbent_to_b.values())
        unmatched_a = [note for index, note in enumerate(checkpoint_a) if index not in matched_a]
        unmatched_b = [note for index, note in enumerate(checkpoint_b) if index not in matched_b]
        for a_index, b_index in one_to_one_matches(
            unmatched_a, unmatched_b, corroboration_radius_seconds
        ):
            a_note = unmatched_a[a_index]
            b_note = unmatched_b[b_index]
            candidate_time = float(median([float(a_note["time"]), float(b_note["time"])]))
            candidate_midi = int(a_note["midi"])
            covered = any(
                int(existing["midi"]) == candidate_midi
                and abs(float(existing["time"]) - candidate_time)
                <= incumbent_exclusion_radius_seconds
                for existing in incumbent
            )
            if covered:
                continue
            duration = float(median([float(a_note["duration"]), float(b_note["duration"])]))
            velocity = float(median([float(a_note["velocity"]), float(b_note["velocity"])]))
            output.append(
                {
                    **deepcopy(a_note),
                    "midi": candidate_midi,
                    "time": max(0.0, candidate_time),
                    "duration": max(0.01, duration),
                    "visualDuration": max(0.01, duration),
                    "audioDuration": max(0.01, duration),
                    "velocity": max(0.01, min(1.0, velocity)),
                    "consensusRecovery": True,
                }
            )
            additions += 1
    output.sort(key=lambda note: (float(note["time"]), int(note["midi"])))
    return output, {
        "incumbentNotes": len(incumbent),
        "outputNotes": len(output),
        "shiftedIncumbentNotes": shifted,
        "incumbentEventsSupportedByBoth": supported_by_both,
        "corroboratedAdditions": additions,
        "requireBothForOnset": require_both_for_onset,
    }


def compact_metrics(reference: list[dict[str, Any]], observed: list[dict[str, Any]]) -> dict[str, Any]:
    notes = measure(reference, observed)["notes"]
    return {
        "referenceNotes": len(reference),
        "observedNotes": len(observed),
        "exactPitchOnset100ms": notes["exactPitchOnset100ms"],
        "exactPitchOnset250ms": notes["exactPitchOnset250ms"],
        "pitchClassOnset250ms": notes["pitchClassOnset250ms"],
        "rapidRetriggersUnder100ms": notes["rapidRetriggersUnder100ms"],
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = sum(int(row["referenceNotes"]) for row in rows)
    observed = sum(int(row["observedNotes"]) for row in rows)

    def metric(name: str) -> dict[str, float | int]:
        matches = sum(int(row[name]["matches"]) for row in rows)
        return f1_from_counts(reference, observed, matches)

    return {
        "songs": len(rows),
        "referenceNotes": reference,
        "observedNotes": observed,
        "exactPitchOnset100ms": metric("exactPitchOnset100ms"),
        "exactPitchOnset250ms": metric("exactPitchOnset250ms"),
        "pitchClassOnset250ms": metric("pitchClassOnset250ms"),
        "rapidRetriggersUnder100ms": sum(
            int(row["rapidRetriggersUnder100ms"]) for row in rows
        ),
    }


def parameter_key(parameters: dict[str, Any]) -> tuple[Any, ...]:
    return (
        parameters["matchRadiusSeconds"],
        parameters["onsetBlend"],
        parameters["requireBothForOnset"],
        parameters["addCorroboratedMissing"],
        parameters["corroborationRadiusSeconds"],
        parameters["incumbentExclusionRadiusSeconds"],
    )


def selection_key(aggregate: dict[str, Any], parameters: dict[str, Any]) -> tuple[Any, ...]:
    """Prefer strict accuracy, then broad pitch evidence, then less mutation."""

    return (
        float(aggregate["exactPitchOnset100ms"]["f1"]),
        float(aggregate["exactPitchOnset100ms"]["recall"]),
        float(aggregate["exactPitchOnset250ms"]["f1"]),
        float(aggregate["pitchClassOnset250ms"]["f1"]),
        -float(parameters["onsetBlend"]),
        -int(bool(parameters["addCorroboratedMissing"])),
        tuple(-float(value) if isinstance(value, (int, float)) else value for value in parameter_key(parameters)),
    )


def oracle_union(
    reference: list[dict[str, Any]], variants: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, tolerance, octave_equivalent in (
        ("exactPitchOnset100ms", 0.1, False),
        ("exactPitchOnset250ms", 0.25, False),
        ("pitchClassOnset250ms", 0.25, True),
    ):
        sets = {
            variant: {
                id(target)
                for target, _candidate in greedy_matches(
                    reference,
                    notes,
                    tolerance,
                    octave_equivalent=octave_equivalent,
                )
            }
            for variant, notes in variants.items()
        }
        union = set().union(*sets.values())
        intersection = set.intersection(*sets.values()) if sets else set()
        output[name] = {
            "matchesByVariant": {key: len(value) for key, value in sets.items()},
            "oracleUnionMatches": len(union),
            "oracleUnionRecall": round(len(union) / max(1, len(reference)), 6),
            "allVariantAgreementMatches": len(intersection),
        }
    return output


def load_song(row: dict[str, Any]) -> dict[str, Any]:
    alignment = load_json(Path(row["alignment"]))
    reference = prepare_reference_notes(row, load_json(Path(row["reference"])), alignment)
    payloads = {name: load_json(Path(path)) for name, path in row["variants"].items()}
    variants = {
        name: clip_candidate(payload, row, alignment) for name, payload in payloads.items()
    }
    required = {"incumbent", "checkpointA", "checkpointB"}
    if set(variants) != required:
        raise ValueError(f"{row.get('id')}: variants must be exactly {sorted(required)}")
    return {
        "id": row["id"],
        "row": row,
        "alignment": alignment,
        "reference": reference,
        "payloads": payloads,
        "variants": variants,
    }


def parameter_grid(search: dict[str, Any]) -> list[dict[str, Any]]:
    names = (
        "matchRadiusSeconds",
        "onsetBlend",
        "requireBothForOnset",
        "addCorroboratedMissing",
        "corroborationRadiusSeconds",
        "incumbentExclusionRadiusSeconds",
    )
    values = [search[name] for name in names]
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def run(manifest: dict[str, Any]) -> dict[str, Any]:
    songs = [load_song(row) for row in manifest["songs"]]
    grid = parameter_grid(manifest["search"])
    evaluations: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    diagnostics: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for parameters in grid:
        key = parameter_key(parameters)
        evaluations[key] = {}
        diagnostics[key] = {}
        for song in songs:
            notes, diagnostic = consensus_notes(
                song["variants"]["incumbent"],
                song["variants"]["checkpointA"],
                song["variants"]["checkpointB"],
                match_radius_seconds=float(parameters["matchRadiusSeconds"]),
                onset_blend=float(parameters["onsetBlend"]),
                require_both_for_onset=bool(parameters["requireBothForOnset"]),
                add_corroborated_missing=bool(parameters["addCorroboratedMissing"]),
                corroboration_radius_seconds=float(parameters["corroborationRadiusSeconds"]),
                incumbent_exclusion_radius_seconds=float(parameters["incumbentExclusionRadiusSeconds"]),
            )
            evaluations[key][song["id"]] = compact_metrics(song["reference"], notes)
            diagnostics[key][song["id"]] = diagnostic

    baseline_by_song = {
        song["id"]: compact_metrics(song["reference"], song["variants"]["incumbent"])
        for song in songs
    }
    variants_by_song = {
        song["id"]: {
            name: compact_metrics(song["reference"], notes)
            for name, notes in song["variants"].items()
        }
        for song in songs
    }
    loso_rows: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    for holdout in songs:
        training_ids = [song["id"] for song in songs if song["id"] != holdout["id"]]
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
        for parameters in grid:
            key = parameter_key(parameters)
            training_aggregate = aggregate_metrics(
                [evaluations[key][song_id] for song_id in training_ids]
            )
            ranked.append((selection_key(training_aggregate, parameters), parameters, training_aggregate))
        _rank, selected, training_aggregate = max(ranked, key=lambda item: item[0])
        selected_key = parameter_key(selected)
        heldout_metrics = evaluations[selected_key][holdout["id"]]
        selected_metrics.append(heldout_metrics)
        loso_rows.append(
            {
                "heldOutSong": holdout["id"],
                "trainingSongs": training_ids,
                "selectedParameters": selected,
                "trainingAggregate": training_aggregate,
                "heldOutBaseline": baseline_by_song[holdout["id"]],
                "heldOutCandidate": heldout_metrics,
                "diagnostics": diagnostics[selected_key][holdout["id"]],
            }
        )

    baseline = aggregate_metrics(list(baseline_by_song.values()))
    loso = aggregate_metrics(selected_metrics)
    oracle_rows = {
        song["id"]: oracle_union(song["reference"], song["variants"])
        for song in songs
    }
    exact_union_matches = sum(
        row["exactPitchOnset100ms"]["oracleUnionMatches"] for row in oracle_rows.values()
    )
    total_reference = sum(len(song["reference"]) for song in songs)
    delta = round(
        float(loso["exactPitchOnset100ms"]["f1"])
        - float(baseline["exactPitchOnset100ms"]["f1"]),
        6,
    )
    no_song_regressed = all(
        float(row["heldOutCandidate"]["exactPitchOnset100ms"]["f1"])
        + 1e-12
        >= float(row["heldOutBaseline"]["exactPitchOnset100ms"]["f1"])
        for row in loso_rows
    )
    return {
        "schema": "polymath-checkpoint-consensus-loso-report-v1",
        "manifestId": manifest.get("id"),
        "candidateGenerationUsesTarget": False,
        "warning": (
            "All songs are opened development evidence. A fresh sealed song and blind listening "
            "remain mandatory before production promotion."
        ),
        "gridCandidates": len(grid),
        "variantMetricsBySong": variants_by_song,
        "baselineAggregate": baseline,
        "losoAggregate": loso,
        "strictExact100F1Delta": delta,
        "noHeldOutSongRegressed": no_song_regressed,
        "oracleComplementBySong": oracle_rows,
        "oracleStrictExact100Recall": round(exact_union_matches / max(1, total_reference), 6),
        "folds": loso_rows,
        "decision": (
            "CANDIDATE_FOR_NEW_SEALED_HOLDOUT"
            if delta > 0.001 and no_song_regressed
            else "REJECT_NO_SAFE_CROSS_SONG_GAIN"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_json(Path(args.manifest))
    report = run(manifest)
    atomic_json(Path(args.output), report)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "baselineExact100F1": report["baselineAggregate"]["exactPitchOnset100ms"]["f1"],
        "losoExact100F1": report["losoAggregate"]["exactPitchOnset100ms"]["f1"],
        "delta": report["strictExact100F1Delta"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
