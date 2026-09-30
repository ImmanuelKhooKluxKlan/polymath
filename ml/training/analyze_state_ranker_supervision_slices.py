"""Slice held-out state-ranker errors by reference/alignment properties.

Reference-derived fields are used only for retrospective diagnosis.  They are
never exposed as inference features.  The report helps distinguish learnable
model error from questionable or structurally different supervision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .analyze_state_ranker_topk import top_k_indices
    from .fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        predict,
        prepare_song,
        ridge_group_statistics,
    )
except ImportError:  # pragma: no cover
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from analyze_state_ranker_topk import top_k_indices
    from fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        predict,
        prepare_song,
        ridge_group_statistics,
    )


def numeric_band(value: float, cuts: tuple[float, ...]) -> str:
    lower = 0.0
    for upper in cuts:
        if value < upper:
            return f"{lower:.2f}-{upper:.2f}"
        lower = upper
    return f"{lower:.2f}+"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    oracle = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    for row in rows:
        target = row["target"]
        for bucket, state in ((base, row["base"]), (oracle, row["oracle"])):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
    return {
        "cells": len(rows),
        "basePitchClass": micro_metrics(base),
        "topKOraclePitchClass": micro_metrics(oracle),
        "meanBaseSetF1": round(float(np.mean([row["baseReward"] for row in rows])), 6),
        "meanTopKOracleSetF1": round(
            float(np.mean([row["oracleReward"] for row in rows])), 6
        ),
    }


def slice_rows(
    rows: list[dict[str, Any]], labeler: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[labeler(row["raw"])].append(row)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    raw_by_id: dict[str, list[dict[str, Any]]] = {}
    songs: list[dict[str, Any]] = []
    for raw_song in payload.get("songs") or []:
        copied = dict(raw_song)
        song_id = str(raw_song.get("id") or "")
        copied["weight"] = max(0.0, finite(raw_song.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(song_id)
        raw_by_id[song_id] = [dict(cell) for cell in raw_song.get("cells") or []]
        songs.append(
            prepare_song(copied, maximum_states=max(4, int(args.maximum_states)))
        )
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    groups = set(statistics)
    all_rows: list[dict[str, Any]] = []
    song_reports: list[dict[str, Any]] = []
    for song in evaluation:
        model = fit_ridge_from_group_statistics(
            statistics,
            included_groups=sorted(groups - {song["groupId"]}),
            ridge=max(0.0, args.base_ridge),
            best_state_weight=max(0.0, args.base_best_state_weight),
        )
        rows: list[dict[str, Any]] = []
        raw_cells = raw_by_id[song["id"]]
        for cell in song["cells"]:
            scores = predict(cell["features"], model)
            indices = top_k_indices(scores, max(2, int(args.top_k)))
            base = cell["states"][indices[0]]
            oracle_index = max(indices, key=lambda idx: float(cell["rewards"][idx]))
            oracle = cell["states"][oracle_index]
            rows.append(
                {
                    "target": cell["target"],
                    "base": base,
                    "oracle": oracle,
                    "baseReward": set_f1(cell["target"], base),
                    "oracleReward": set_f1(cell["target"], oracle),
                    "raw": raw_cells[int(cell["index"])],
                }
            )
        all_rows.extend(rows)
        song_reports.append({"id": song["id"], **summarize(rows)})

    slices = {
        "textureLabel": slice_rows(all_rows, lambda row: str(row.get("textureLabel") or "unknown")),
        "targetGestureCount": slice_rows(
            all_rows, lambda row: str(min(3, int(row.get("targetGestureCount", 0))))
        ),
        "sourcePitchClassSupport": slice_rows(
            all_rows,
            lambda row: numeric_band(
                min(1.0, max(0.0, finite(row.get("sourcePitchClassSupport")))),
                (0.5, 0.75, 0.9, 0.999999),
            ),
        ),
        "candidateToReferencePitchClassF1": slice_rows(
            all_rows,
            lambda row: numeric_band(
                min(1.0, max(0.0, finite(row.get("candidateToReferencePitchClassF1")))),
                (0.25, 0.5, 0.75, 0.999999),
            ),
        ),
        "offCentreRawAttackMatchShare": slice_rows(
            all_rows,
            lambda row: (
                "centered-no-off-centre"
                if int(row.get("offCentreReferenceGestures", 0)) == 0
                else numeric_band(
                    min(
                        1.0,
                        max(0.0, finite(row.get("offCentreReferenceRawAttackMatchShare"))),
                    ),
                    (0.01, 0.5, 0.999999),
                )
            ),
        ),
    }
    report = {
        "schema": "polymath-state-ranker-supervision-slices-v1",
        "audit": str(audit_path),
        "evidenceBoundary": (
            "Reference-derived fields are retrospective slices only and are not "
            "available to the inference model."
        ),
        "settings": {
            "topK": max(2, int(args.top_k)),
            "validationBoundary": "whole-composition-held-out",
        },
        "aggregate": summarize(all_rows),
        "songs": song_reports,
        "slices": slices,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
