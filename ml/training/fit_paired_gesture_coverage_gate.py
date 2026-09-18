"""Fit a gate for whether an arranger gesture has a pianist counterpart.

Paired chord models are trained only on gestures that align to an authored
target.  Applying them to every inference gesture creates a distribution
shift: extra/unmatched arranger gestures were never represented in training.
This gate learns that coverage boundary from all candidate gestures with
whole-song holdouts, using the same source/candidate context available online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from .fit_chord_gesture_library import (
    aligned_target_left_notes,
    group_nearby_notes,
    load_json,
)
from .fit_paired_chord_gesture_mapper import (
    candidate_left_notes,
    canonical_manifest_pairs,
    match_group_sequences,
    paired_context,
)
from .fit_onset_gesture_ranker import trusted_windows
from .fit_pianist_texture_decoder import (
    classification_metrics,
    fit_logistic,
    sigmoid,
    standardize,
)
from piano_arranger_adapter import normalize_source_notes  # noqa: E402


def build_song_examples(
    pair: dict[str, Any],
    *,
    reference_hand_split: int,
    match_tolerance: float,
    context_radius: float,
) -> list[dict[str, Any]]:
    source = load_json(Path(pair["source"]).resolve())
    target = load_json(Path(pair["target"]).resolve())
    candidate = load_json(Path(pair["candidate"]).resolve())
    report = load_json(Path(pair["alignmentReport"]).resolve())
    windows = trusted_windows(report)
    candidate_end = (
        float(pair["candidateEndSeconds"])
        if pair.get("candidateEndSeconds") is not None
        else None
    )
    source_notes = normalize_source_notes(source.get("notes", []))
    source_times = [float(note["time"]) for note in source_notes]
    candidate_groups = group_nearby_notes(
        candidate_left_notes(candidate, windows, candidate_end)
    )
    target_groups = group_nearby_notes(
        [
            note
            for note in aligned_target_left_notes(target, report, reference_hand_split)
            if candidate_end is None or float(note["time"]) < candidate_end
        ]
    )
    matched = {
        candidate_index
        for candidate_index, _target_index in match_group_sequences(
            candidate_groups, target_groups, match_tolerance
        )
    }
    return [
        {
            "songId": str(pair["id"]),
            "time": round(float(group[0]["time"]), 6),
            "label": float(index in matched),
            "features": paired_context(
                source_notes,
                source_times,
                candidate_groups,
                index,
                context_radius,
            )[0],
        }
        for index, group in enumerate(candidate_groups)
    ]


def choose_policy(trials: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [
        row
        for row in trials
        if float(row["aggregate"]["precision"]) >= 0.80
        and float(row["worstFoldPrecision"]) >= 0.70
        and int(row["minimumFoldPredictions"]) >= 3
    ]
    pool = safe or trials
    return max(
        pool,
        key=lambda row: (
            bool(row in safe),
            float(row["aggregate"]["f0_5"]),
            float(row["worstFoldPrecision"]),
            float(row["aggregate"]["recall"]),
            -int(row["aggregate"]["predictedPositive"]),
        ),
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--match-tolerance", type=float, default=0.14)
    parser.add_argument("--context-radius", type=float, default=0.35)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    pairs = canonical_manifest_pairs(load_json(manifest_path))
    by_song = {
        str(pair["id"]): build_song_examples(
            pair,
            reference_hand_split=args.reference_hand_split,
            match_tolerance=args.match_tolerance,
            context_radius=args.context_radius,
        )
        for pair in pairs
    }
    rows = [row for values in by_song.values() for row in values]
    if len(by_song) < 3 or not rows:
        raise ValueError("Coverage fitting requires at least three non-empty songs")
    feature_count = len(rows[0]["features"])
    trials: list[dict[str, Any]] = []
    for ridge in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        for positive_weight in (0.5, 0.75, 1.0, 1.5):
            predictions: list[dict[str, Any]] = []
            for heldout in by_song:
                training = [row for song, values in by_song.items() if song != heldout for row in values]
                validation = by_song[heldout]
                matrix = np.asarray([row["features"] for row in training], dtype=float)
                labels = np.asarray([row["label"] for row in training], dtype=float)
                standardized, means, scales = standardize(matrix)
                weights = fit_logistic(
                    standardized,
                    labels,
                    ridge=ridge,
                    positive_weight=positive_weight,
                )
                validation_matrix = np.asarray([row["features"] for row in validation], dtype=float)
                probabilities = sigmoid(((validation_matrix - means) / scales) @ weights)
                predictions.append(
                    {
                        "songId": heldout,
                        "labels": np.asarray([row["label"] for row in validation], dtype=float),
                        "probabilities": probabilities,
                    }
                )
            all_labels = np.concatenate([item["labels"] for item in predictions])
            all_probabilities = np.concatenate([item["probabilities"] for item in predictions])
            for threshold in np.arange(0.50, 0.951, 0.025):
                aggregate = classification_metrics(all_labels, all_probabilities, float(threshold))
                folds = [
                    {
                        "heldOutSong": item["songId"],
                        **classification_metrics(item["labels"], item["probabilities"], float(threshold)),
                    }
                    for item in predictions
                ]
                active = [fold for fold in folds if int(fold["predictedPositive"]) > 0]
                trials.append(
                    {
                        "ridge": ridge,
                        "positiveWeight": positive_weight,
                        "threshold": round(float(threshold), 6),
                        "aggregate": aggregate,
                        "worstFoldPrecision": round(
                            min((float(fold["precision"]) for fold in active), default=0.0), 6
                        ),
                        "minimumFoldPredictions": min(
                            (int(fold["predictedPositive"]) for fold in folds), default=0
                        ),
                        "folds": folds,
                    }
                )
    winner = choose_policy(trials)
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=float)
    standardized, means, scales = standardize(matrix)
    weights = fit_logistic(
        standardized,
        labels,
        ridge=float(winner["ridge"]),
        positive_weight=float(winner["positiveWeight"]),
    )
    gate = {
        "id": args.gate_id,
        "type": "paired-gesture-coverage-logistic-v1",
        "featureNames": [f"paired_context_{index:03d}" for index in range(feature_count)],
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "threshold": winner["threshold"],
        "contextRadiusSeconds": args.context_radius,
        "training": {
            "schema": "polymath-paired-gesture-coverage-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "policy": "whole-song leave-one-out; all candidate gestures included",
            "commercialUseAllowed": False,
        },
    }
    canonical = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    gate["gateSha256"] = hashlib.sha256(canonical).hexdigest()
    safe_count = sum(
        float(row["aggregate"]["precision"]) >= 0.80
        and float(row["worstFoldPrecision"]) >= 0.70
        and int(row["minimumFoldPredictions"]) >= 3
        for row in trials
    )
    report = {
        "schema": "polymath-paired-gesture-coverage-report-v1",
        "sampleCounts": {song: len(values) for song, values in sorted(by_song.items())},
        "positiveCounts": {
            song: int(sum(row["label"] for row in values))
            for song, values in sorted(by_song.items())
        },
        "safePolicyCount": safe_count,
        "winner": winner,
        "leaderboard": sorted(
            trials,
            key=lambda row: (
                float(row["aggregate"]["f0_5"]),
                float(row["worstFoldPrecision"]),
                float(row["aggregate"]["recall"]),
            ),
            reverse=True,
        )[:30],
        "decision": "RESEARCH_ONLY",
    }
    atomic_json(args.output_gate.resolve(), gate)
    atomic_json(args.report.resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "leaderboard"}, indent=2))


if __name__ == "__main__":
    main()
