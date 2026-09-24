"""Search a boundary replacement policy that preserves primary note releases.

The first shifted-window experiment proved that centred context recovers onset
events near artificial five-second cuts, but its highest-onset-F1 radius made
too many notes overlong.  This follow-up remains inference-safe: it never sees
reference notes while merging predictions.  When a shifted-pass note has a
same-pitch primary-pass partner nearby, the shifted onset is retained while the
primary absolute release time is reused.  Unmatched recovered notes retain the
shifted duration.

Unlike the earlier search, candidates must pass the pre-frozen full validation
safety gate before onset metrics may rank them.  If none passes, there is no
winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.apply_full_validation_gate import evaluate_gate
from ml.training.evaluate_checkpoint import evaluate_decoded_predictions, stitch_clip_notes
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    predictions_for_records,
    score_song_timelines,
    sha256_file,
)
from ml.training.search_overlap_boundary_policy import (
    coverage_edge_distance,
    deduplicate_notes,
    primary_boundary_distance,
    split_song_predictions,
)
from ml.training.train_muscriptor_piano import read_jsonl


def _note_key(note: dict[str, Any]) -> tuple[str, int]:
    return (
        str(note.get("instrument") or "acoustic_piano"),
        int(note["midi"]),
    )


def pair_shifted_to_primary(
    shifted: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    tolerance_seconds: float,
) -> dict[int, int]:
    """Return deterministic one-to-one same-pitch onset partners."""

    if tolerance_seconds < 0:
        raise ValueError("Onset match tolerance cannot be negative")
    candidates: list[tuple[float, float, float, int, int]] = []
    for shifted_index, shifted_note in enumerate(shifted):
        shifted_time = float(shifted_note["time"])
        for primary_index, primary_note in enumerate(primary):
            if _note_key(shifted_note) != _note_key(primary_note):
                continue
            primary_time = float(primary_note["time"])
            distance = abs(shifted_time - primary_time)
            if distance <= tolerance_seconds + 1e-12:
                candidates.append((
                    distance,
                    shifted_time,
                    primary_time,
                    shifted_index,
                    primary_index,
                ))
    matched_shifted: set[int] = set()
    matched_primary: set[int] = set()
    assignments: dict[int, int] = {}
    for _, _, _, shifted_index, primary_index in sorted(candidates):
        if shifted_index in matched_shifted or primary_index in matched_primary:
            continue
        assignments[shifted_index] = primary_index
        matched_shifted.add(shifted_index)
        matched_primary.add(primary_index)
    return assignments


def merge_song_predictions_release_preserving(
    primary: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    primary_windows: list[dict[str, Any]],
    overlap_windows: list[dict[str, Any]],
    radius_seconds: float,
    *,
    onset_match_tolerance_seconds: float,
    window_seconds: float = 5.0,
    minimum_duration_seconds: float = 0.01,
) -> list[dict[str, Any]]:
    """Replace boundary onsets while retaining primary releases when paired."""

    if not 0 <= radius_seconds <= window_seconds / 2:
        raise ValueError("Boundary radius must be within half a window")
    if minimum_duration_seconds <= 0:
        raise ValueError("Minimum duration must be positive")

    primary_near_boundary = [
        note for note in primary
        if primary_boundary_distance(float(note["time"]), window_seconds)
        <= radius_seconds
        and coverage_edge_distance(float(note["time"]), overlap_windows) is not None
    ]
    shifted_near_boundary = [
        note for note in overlap
        if primary_boundary_distance(float(note["time"]), window_seconds)
        <= radius_seconds
        and coverage_edge_distance(float(note["time"]), overlap_windows) is not None
    ]
    partners = pair_shifted_to_primary(
        shifted_near_boundary,
        primary_near_boundary,
        onset_match_tolerance_seconds,
    )

    selected: list[dict[str, Any]] = []
    for source in primary:
        time = float(source["time"])
        replace = (
            coverage_edge_distance(time, overlap_windows) is not None
            and primary_boundary_distance(time, window_seconds) <= radius_seconds
        )
        if replace:
            continue
        note = dict(source)
        note["_edgeDistance"] = coverage_edge_distance(time, primary_windows) or 0.0
        note["_pass"] = "primary"
        note["_durationSource"] = "primary"
        selected.append(note)

    for shifted_index, source in enumerate(shifted_near_boundary):
        note = dict(source)
        time = float(note["time"])
        primary_index = partners.get(shifted_index)
        if primary_index is not None:
            primary_note = primary_near_boundary[primary_index]
            primary_time = float(primary_note["time"])
            primary_duration = max(
                minimum_duration_seconds,
                float(primary_note.get("duration") or minimum_duration_seconds),
            )
            primary_release = primary_time + primary_duration
            release_preserving_duration = primary_release - time
            note["duration"] = (
                release_preserving_duration
                if release_preserving_duration >= minimum_duration_seconds
                else primary_duration
            )
            note["_durationSource"] = "primary-release"
        else:
            note["duration"] = max(
                minimum_duration_seconds,
                float(note.get("duration") or minimum_duration_seconds),
            )
            note["_durationSource"] = "shifted-unmatched"
        note["_edgeDistance"] = coverage_edge_distance(time, overlap_windows) or 0.0
        note["_pass"] = "overlap"
        selected.append(note)

    result = deduplicate_notes(selected)
    for note in result:
        note.pop("_durationSource", None)
    return result


def safe_selection_rank(row: dict[str, Any]) -> tuple[float, float, float]:
    metrics = row["continuousSong"]["metrics"]
    return (
        float(metrics["100ms"]["microF1"]),
        float(metrics["100ms"]["recall"]),
        float(metrics["250ms"]["microF1"]),
    )


def rank_safe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = [row for row in rows if row["safetyGate"]["researchGatePassed"]]
    return sorted(safe, key=safe_selection_rank, reverse=True)


def search_policy(
    primary_evaluation: dict[str, Any],
    overlap_evaluation: dict[str, Any],
    primary_records: list[dict[str, Any]],
    overlap_records: list[dict[str, Any]],
    radii: list[float],
    *,
    onset_match_tolerance_seconds: float,
    baseline_timeline: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[float, list[list[dict[str, Any]]]],
    dict[float, dict[str, Any]],
    dict[float, dict[str, Any]],
]:
    primary_predictions = predictions_for_records(primary_evaluation, primary_records)
    overlap_predictions = predictions_for_records(overlap_evaluation, overlap_records)
    primary_songs, _ = stitch_clip_notes(primary_records, primary_predictions, reference=False)
    overlap_songs, _ = stitch_clip_notes(overlap_records, overlap_predictions, reference=False)
    windows_primary: dict[str, list[dict[str, Any]]] = {}
    windows_overlap: dict[str, list[dict[str, Any]]] = {}
    for record in primary_records:
        windows_primary.setdefault(str(record.get("songId") or "unknown"), []).append(record)
    for record in overlap_records:
        windows_overlap.setdefault(str(record.get("songId") or "unknown"), []).append(record)

    predictions_by_radius: dict[float, list[list[dict[str, Any]]]] = {}
    timelines_by_radius: dict[float, dict[str, Any]] = {}
    gates_by_radius: dict[float, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for radius in radii:
        merged_songs = {
            song_id: merge_song_predictions_release_preserving(
                primary_songs.get(song_id, []),
                overlap_songs.get(song_id, []),
                windows_primary.get(song_id, []),
                windows_overlap.get(song_id, []),
                radius,
                onset_match_tolerance_seconds=onset_match_tolerance_seconds,
            )
            for song_id in sorted(set(primary_songs) | set(overlap_songs))
        }
        predictions = split_song_predictions(primary_records, merged_songs)
        predictions_by_radius[radius] = predictions
        metrics = evaluate_decoded_predictions(primary_records, predictions)
        timeline = score_song_timelines(primary_records, predictions)
        safety = evaluate_gate(baseline_timeline, timeline, gate)
        timelines_by_radius[radius] = timeline
        gates_by_radius[radius] = safety
        rows.append({
            "radiusSeconds": radius,
            "clipLocal": {key: metrics[key] for key in ("50ms", "100ms", "250ms")},
            "continuousSong": {
                "metrics": timeline["metrics"],
                "perSong": timeline["perSong"],
                "songClusterBootstrap95": timeline["songClusterBootstrap95"],
                "diagnostics100ms": timeline["diagnostics100ms"],
            },
            "safetyGate": safety,
        })
    safe_rows = rank_safe_candidates(rows)
    return ({
        "schema": "polymath-overlap-release-preserving-search-v1",
        "selectionPolicy": "filter by complete research safety gate, then rank",
        "selectionRanking": [
            "continuousSong.100ms.microF1",
            "continuousSong.100ms.recall",
            "continuousSong.250ms.microF1",
        ],
        "onsetMatchToleranceSeconds": onset_match_tolerance_seconds,
        "winnerRadiusSeconds": safe_rows[0]["radiusSeconds"] if safe_rows else None,
        "safeCandidateCount": len(safe_rows),
        "candidates": rows,
    }, predictions_by_radius, timelines_by_radius, gates_by_radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--overlap-evaluation", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--overlap-manifest", type=Path, required=True)
    parser.add_argument("--baseline-timeline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--radii", default="0.10,0.15,0.25,0.40,0.60,0.80")
    parser.add_argument("--onset-match-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--winner-evaluation", type=Path, required=True)
    parser.add_argument("--winner-timeline", type=Path, required=True)
    parser.add_argument("--winner-gate", type=Path, required=True)
    args = parser.parse_args()

    radii = sorted({float(value) for value in args.radii.split(",") if value.strip()})
    if not radii:
        parser.error("At least one radius is required")
    paths = {
        "primaryEvaluation": args.primary_evaluation.resolve(),
        "overlapEvaluation": args.overlap_evaluation.resolve(),
        "primaryManifest": args.primary_manifest.resolve(),
        "overlapManifest": args.overlap_manifest.resolve(),
        "baselineTimeline": args.baseline_timeline.resolve(),
        "gate": args.gate.resolve(),
    }
    primary_evaluation = json.loads(paths["primaryEvaluation"].read_text(encoding="utf-8"))
    overlap_evaluation = json.loads(paths["overlapEvaluation"].read_text(encoding="utf-8"))
    if primary_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Primary input is not a single checkpoint evaluation")
    if overlap_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Overlap input is not a single checkpoint evaluation")
    primary_records = read_jsonl(paths["primaryManifest"])
    overlap_records = read_jsonl(paths["overlapManifest"])
    baseline_timeline = json.loads(paths["baselineTimeline"].read_text(encoding="utf-8"))
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    report, predictions, timelines, gates = search_policy(
        primary_evaluation,
        overlap_evaluation,
        primary_records,
        overlap_records,
        radii,
        onset_match_tolerance_seconds=args.onset_match_tolerance_seconds,
        baseline_timeline=baseline_timeline,
        gate=gate,
    )
    report["sources"] = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    winner_radius = report["winnerRadiusSeconds"]
    if winner_radius is None:
        print(json.dumps({
            "report": str(destination),
            "winnerRadiusSeconds": None,
            "safeCandidateCount": 0,
        }, indent=2))
        return
    winner_radius = float(winner_radius)
    winner_metrics = evaluate_decoded_predictions(
        primary_records,
        predictions[winner_radius],
        include_raw_predictions=True,
    )
    winner = {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": primary_evaluation.get("checkpoint"),
        "validationManifest": str(paths["primaryManifest"]),
        "clips": len(primary_records),
        "instrumentConstraint": primary_evaluation.get("instrumentConstraint"),
        "metrics": winner_metrics,
        "inferencePolicy": {
            "schema": "polymath-overlap-release-preserving-v1",
            "radiusSeconds": winner_radius,
            "overlapOffsetSeconds": 2.5,
            "onsetMatchToleranceSeconds": args.onset_match_tolerance_seconds,
            "searchReport": str(destination),
        },
    }
    outputs = {
        args.winner_evaluation.resolve(): winner,
        args.winner_timeline.resolve(): timelines[winner_radius],
        args.winner_gate.resolve(): gates[winner_radius],
    }
    for output, payload in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(destination),
        "winnerRadiusSeconds": winner_radius,
        "safeCandidateCount": report["safeCandidateCount"],
        "winner100msF1": timelines[winner_radius]["metrics"]["100ms"]["microF1"],
        "winnerCertificationPassed": gates[winner_radius]["certification"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
