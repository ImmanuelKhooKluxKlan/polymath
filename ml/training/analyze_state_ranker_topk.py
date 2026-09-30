"""Audit how much pitch-class accuracy remains in a ranker's top choices.

The report is deliberately retrospective: references may choose the best state
*after* an inference-safe model has ranked the lattice, but never contribute to
the states or their scores.  A large top-2 gap is evidence for a focused
reranker; a small gap says that more candidate evidence is needed instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge,
        predict,
        prepare_song,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1  # type: ignore
    from fit_pitch_class_state_ranker import (  # type: ignore
        canonical_song_id,
        finite,
        fit_ridge,
        predict,
        prepare_song,
    )


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Return deterministic descending-score indices."""

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    limit = max(1, min(len(scores), int(k)))
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:limit]


def evaluate_top_k(song: dict[str, Any], model: dict[str, Any], k: int) -> dict[str, Any]:
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    selected_rank = {rank: 0 for rank in range(1, k + 1)}
    top_one_optimal = 0
    exact = 0
    macro_reward = 0.0
    top_one_reward = 0.0
    for cell in song["cells"]:
        scores = predict(cell["features"], model)
        indices = top_k_indices(scores, k)
        rewards = [float(cell["rewards"][index]) for index in indices]
        best_offset = max(
            range(len(indices)),
            key=lambda offset: (rewards[offset], -offset),
        )
        selected_index = indices[best_offset]
        selected_rank[best_offset + 1] += 1
        selected = cell["states"][selected_index]
        target = cell["target"]
        counts["truePositive"] += len(target & selected)
        counts["falsePositive"] += len(selected - target)
        counts["falseNegative"] += len(target - selected)
        macro_reward += set_f1(target, selected)
        top_one_reward += float(cell["rewards"][indices[0]])
        top_one_optimal += int(best_offset == 0)
        exact += int(target == selected)
    cells = len(song["cells"])
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": cells,
        "topK": k,
        "oracle": {
            **micro_metrics(counts),
            "macroF1": round(macro_reward / max(1, cells), 6),
            "exactSetRate": round(exact / max(1, cells), 6),
        },
        "topOneMacroF1": round(top_one_reward / max(1, cells), 6),
        "topOneAlreadyBestShare": round(top_one_optimal / max(1, cells), 6),
        "selectedRankCounts": {str(rank): count for rank, count in selected_rank.items()},
    }


def weighted(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(max(0.0, float(row["weight"])) for row in rows)
    return sum(
        max(0.0, float(row["weight"])) * float(row["oracle"][field])
        for row in rows
    ) / max(1e-12, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--best-state-weight", type=float, default=1.0)
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    maximum_states = max(4, min(256, int(args.maximum_states)))
    songs: list[dict[str, Any]] = []
    for raw in audit.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(prepare_song(copied, maximum_states=maximum_states))
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    values = sorted({max(1, int(value)) for value in args.top_k})
    rows_by_k: dict[int, list[dict[str, Any]]] = {value: [] for value in values}
    for heldout in evaluation:
        training = [song for song in songs if song["groupId"] != heldout["groupId"]]
        model = fit_ridge(
            training,
            ridge=max(0.0, args.ridge),
            best_state_weight=max(0.0, args.best_state_weight),
        )
        for value in values:
            rows_by_k[value].append(evaluate_top_k(heldout, model, value))
    trials = [
        {
            "topK": value,
            "weightedPitchClassF1": round(weighted(rows_by_k[value], "f1"), 6),
            "worstSongPitchClassF1": round(
                min(float(row["oracle"]["f1"]) for row in rows_by_k[value]), 6
            ),
            "songs": rows_by_k[value],
        }
        for value in values
    ]
    report = {
        "schema": "polymath-state-ranker-top-k-audit-v1",
        "status": "retrospective-oracle-not-a-deployable-model",
        "audit": str(audit_path),
        "evaluationSongIds": [song["id"] for song in evaluation],
        "settings": {
            "maximumStates": maximum_states,
            "ridge": max(0.0, args.ridge),
            "bestStateWeight": max(0.0, args.best_state_weight),
            "foldBoundary": "whole-composition-held-out",
        },
        "trials": trials,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "topK": [
                    {
                        key: row[key]
                        for key in ("topK", "weightedPitchClassF1", "worstSongPitchClassF1")
                    }
                    for row in trials
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
