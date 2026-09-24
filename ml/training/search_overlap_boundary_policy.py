"""Search a frozen boundary replacement radius using a 2.5-second decode pass.

The policy uses no answer-key feature at inference. Primary predictions remain
everywhere except a narrow zone around each five-second primary cut, where the
shifted pass heard the same audio near the centre of its window. Validation
labels select one radius; the sealed split remains untouched until final freeze.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.evaluate_checkpoint import (
    evaluate_decoded_predictions,
    stitch_clip_notes,
)
from ml.training.rescore_song_timelines import (
    TimelineScoreError,
    predictions_for_records,
    score_song_timelines,
    sha256_file,
)
from ml.training.train_muscriptor_piano import read_jsonl


def coverage_edge_distance(
    time: float,
    records: list[dict[str, Any]],
) -> float | None:
    distances = []
    for record in records:
        start = float(record.get("sourceStart") or 0)
        end = start + float(record.get("durationSeconds") or 0)
        if start - 1e-9 <= time <= end + 1e-9:
            distances.append(max(0.0, min(time - start, end - time)))
    return max(distances) if distances else None


def primary_boundary_distance(time: float, window_seconds: float = 5.0) -> float:
    remainder = time % window_seconds
    return min(remainder, window_seconds - remainder)


def deduplicate_notes(
    notes: list[dict[str, Any]],
    tolerance: float = 0.03,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for note in notes:
        key = (str(note.get("instrument") or "acoustic_piano"), int(note["midi"]))
        grouped.setdefault(key, []).append(dict(note))
    result: list[dict[str, Any]] = []
    for group in grouped.values():
        merged: list[dict[str, Any]] = []
        for note in sorted(group, key=lambda row: float(row["time"])):
            if merged and abs(float(note["time"]) - float(merged[-1]["time"])) <= tolerance:
                previous = merged[-1]
                if float(note.get("_edgeDistance") or 0) > float(previous.get("_edgeDistance") or 0):
                    merged[-1] = note
                else:
                    previous["duration"] = max(
                        float(previous.get("duration") or 0.1),
                        float(note.get("duration") or 0.1),
                    )
            else:
                merged.append(note)
        result.extend(merged)
    for note in result:
        note.pop("_edgeDistance", None)
        note.pop("_pass", None)
    return sorted(result, key=lambda note: (float(note["time"]), int(note["midi"])))


def merge_song_predictions(
    primary: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    primary_windows: list[dict[str, Any]],
    overlap_windows: list[dict[str, Any]],
    radius_seconds: float,
    *,
    window_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    if not 0 <= radius_seconds <= window_seconds / 2:
        raise ValueError("Boundary radius must be within half a window")
    selected: list[dict[str, Any]] = []
    for source_name, notes, source_windows in (
        ("primary", primary, primary_windows),
        ("overlap", overlap, overlap_windows),
    ):
        for source in notes:
            note = dict(source)
            time = float(note["time"])
            overlap_distance = coverage_edge_distance(time, overlap_windows)
            near_primary_cut = primary_boundary_distance(time, window_seconds) <= radius_seconds
            use_overlap = overlap_distance is not None and near_primary_cut
            if (source_name == "overlap") != use_overlap:
                continue
            edge_distance = coverage_edge_distance(time, source_windows)
            note["_edgeDistance"] = edge_distance if edge_distance is not None else 0.0
            note["_pass"] = source_name
            selected.append(note)
    return deduplicate_notes(selected)


def split_song_predictions(
    records: list[dict[str, Any]],
    songs: dict[str, list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for record in records:
        song_id = str(record.get("songId") or "unknown")
        start = float(record.get("sourceStart") or 0)
        end = start + float(record.get("durationSeconds") or 0)
        local = []
        for source in songs.get(song_id, []):
            onset = float(source["time"])
            if not (start <= onset < end or (start == end and onset == start)):
                continue
            note = dict(source)
            note["time"] = onset - start
            local.append(note)
        result.append(sorted(local, key=lambda note: (note["time"], note["midi"])))
    return result


def search_policy(
    primary_evaluation: dict[str, Any],
    overlap_evaluation: dict[str, Any],
    primary_records: list[dict[str, Any]],
    overlap_records: list[dict[str, Any]],
    radii: list[float],
) -> tuple[dict[str, Any], dict[float, list[list[dict[str, Any]]]]]:
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
    rows = []
    for radius in radii:
        merged_songs = {
            song_id: merge_song_predictions(
                primary_songs.get(song_id, []),
                overlap_songs.get(song_id, []),
                windows_primary.get(song_id, []),
                windows_overlap.get(song_id, []),
                radius,
            )
            for song_id in sorted(set(primary_songs) | set(overlap_songs))
        }
        predictions = split_song_predictions(primary_records, merged_songs)
        predictions_by_radius[radius] = predictions
        metrics = evaluate_decoded_predictions(primary_records, predictions)
        timeline = score_song_timelines(primary_records, predictions)
        rows.append({
            "radiusSeconds": radius,
            "clipLocal": {
                key: metrics[key] for key in ("50ms", "100ms", "250ms")
            },
            "continuousSong": {
                "metrics": timeline["metrics"],
                "perSong": timeline["perSong"],
                "songClusterBootstrap95": timeline["songClusterBootstrap95"],
                "diagnostics100ms": timeline["diagnostics100ms"],
            },
        })
    rows.sort(
        key=lambda row: (
            float(row["continuousSong"]["metrics"]["100ms"]["microF1"]),
            float(row["continuousSong"]["metrics"]["100ms"]["recall"]),
            float(row["continuousSong"]["metrics"]["250ms"]["microF1"]),
        ),
        reverse=True,
    )
    return {
        "schema": "polymath-overlap-boundary-policy-search-v1",
        "selectionRanking": [
            "continuousSong.100ms.microF1",
            "continuousSong.100ms.recall",
            "continuousSong.250ms.microF1",
        ],
        "winnerRadiusSeconds": rows[0]["radiusSeconds"] if rows else None,
        "candidates": rows,
    }, predictions_by_radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-evaluation", type=Path, required=True)
    parser.add_argument("--overlap-evaluation", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--overlap-manifest", type=Path, required=True)
    parser.add_argument("--radii", default="0.10,0.15,0.25,0.40,0.60,0.80")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--winner-evaluation", type=Path, required=True)
    args = parser.parse_args()
    radii = sorted({float(value) for value in args.radii.split(",") if value.strip()})
    if not radii:
        parser.error("At least one radius is required")
    primary_path = args.primary_evaluation.resolve()
    overlap_path = args.overlap_evaluation.resolve()
    primary_manifest = args.primary_manifest.resolve()
    overlap_manifest = args.overlap_manifest.resolve()
    primary_evaluation = json.loads(primary_path.read_text(encoding="utf-8"))
    overlap_evaluation = json.loads(overlap_path.read_text(encoding="utf-8"))
    if primary_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Primary input is not a single checkpoint evaluation")
    if overlap_evaluation.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TimelineScoreError("Overlap input is not a single checkpoint evaluation")
    primary_records = read_jsonl(primary_manifest)
    overlap_records = read_jsonl(overlap_manifest)
    report, predictions = search_policy(
        primary_evaluation,
        overlap_evaluation,
        primary_records,
        overlap_records,
        radii,
    )
    report.update({
        "primaryEvaluation": str(primary_path),
        "primaryEvaluationSha256": sha256_file(primary_path),
        "overlapEvaluation": str(overlap_path),
        "overlapEvaluationSha256": sha256_file(overlap_path),
        "primaryManifest": str(primary_manifest),
        "primaryManifestSha256": sha256_file(primary_manifest),
        "overlapManifest": str(overlap_manifest),
        "overlapManifestSha256": sha256_file(overlap_manifest),
    })
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    winner_radius = float(report["winnerRadiusSeconds"])
    winner_metrics = evaluate_decoded_predictions(
        primary_records,
        predictions[winner_radius],
        include_raw_predictions=True,
    )
    winner = {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": primary_evaluation.get("checkpoint"),
        "validationManifest": str(primary_manifest),
        "clips": len(primary_records),
        "instrumentConstraint": primary_evaluation.get("instrumentConstraint"),
        "metrics": winner_metrics,
        "inferencePolicy": {
            "schema": "polymath-overlap-boundary-replacement-v1",
            "radiusSeconds": winner_radius,
            "overlapOffsetSeconds": 2.5,
            "searchReport": str(destination),
        },
    }
    winner_path = args.winner_evaluation.resolve()
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(destination),
        "winnerEvaluation": str(winner_path),
        "winnerRadiusSeconds": winner_radius,
        "winner100msF1": report["candidates"][0]["continuousSong"]["metrics"]["100ms"]["microF1"],
    }, indent=2))


if __name__ == "__main__":
    main()
