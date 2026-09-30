"""Diagnose why a held-out state ranker misses better top-k candidates.

This is deliberately an analysis tool, not another trainer.  It measures the
remaining opportunity by edit type and asks whether the better state has a
different relationship to its temporal neighbours.  Whole compositions are
excluded before their base scores are produced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Package and direct-script execution.
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
except ImportError:  # pragma: no cover - direct script fallback.
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


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def edit_type(base: frozenset[int], better: frozenset[int]) -> str:
    if better == base:
        return "same"
    if better < base:
        return "remove-only"
    if base < better:
        return "add-only"
    return "replace-mixed"


def reward_band(value: float) -> str:
    if value < 0.25:
        return "00-24"
    if value < 0.50:
        return "25-49"
    if value < 0.75:
        return "50-74"
    if value < 1.0 - 1e-9:
        return "75-99"
    return "100"


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    material = list(rows)
    if not material:
        return {"cells": 0}
    return {
        "cells": len(material),
        "meanBaseReward": round(float(np.mean([r["baseReward"] for r in material])), 6),
        "meanTopKReward": round(float(np.mean([r["bestReward"] for r in material])), 6),
        "meanAvailableGain": round(float(np.mean([r["gain"] for r in material])), 6),
        "meanBaseScoreMargin": round(
            float(np.mean([r["baseScoreMargin"] for r in material])), 6
        ),
        "betterHasHigherNeighbourAffinityShare": round(
            float(np.mean([r["affinityDelta"] > 1e-9 for r in material])), 6
        ),
        "betterHasLowerNeighbourAffinityShare": round(
            float(np.mean([r["affinityDelta"] < -1e-9 for r in material])), 6
        ),
        "meanNeighbourAffinityDelta": round(
            float(np.mean([r["affinityDelta"] for r in material])), 6
        ),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    return {key: summarize_rows(value) for key, value in sorted(buckets.items())}


def analyze_song(song: dict[str, Any], model: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for cell in song["cells"]:
        scores = predict(cell["features"], model)
        indices = top_k_indices(scores, top_k)
        ranked.append(
            {
                "cell": cell,
                "scores": scores,
                "indices": indices,
                "base": cell["states"][indices[0]],
                "betterIndex": max(indices, key=lambda idx: float(cell["rewards"][idx])),
            }
        )

    opportunities: list[dict[str, Any]] = []
    rank_counts: Counter[str] = Counter()
    size_confusion: Counter[str] = Counter()
    size_deltas: list[int] = []
    topk_exact_size = 0
    base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    size_oracle_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    for position, item in enumerate(ranked):
        cell = item["cell"]
        indices = item["indices"]
        better_index = item["betterIndex"]
        base = item["base"]
        better = cell["states"][better_index]
        base_reward = float(cell["rewards"][indices[0]])
        better_reward = float(cell["rewards"][better_index])
        best_rank = indices.index(better_index) + 1
        rank_counts[str(best_rank)] += 1
        target_size = len(cell["target"])
        base_size = len(base)
        size_confusion[f"{target_size}->{base_size}"] += 1
        size_deltas.append(base_size - target_size)
        topk_exact_size += int(any(len(cell["states"][idx]) == target_size for idx in indices))
        exact_size_indices = [
            idx for idx in indices if len(cell["states"][idx]) == target_size
        ]
        if exact_size_indices:
            size_oracle_index = max(exact_size_indices, key=lambda idx: float(item["scores"][idx]))
        else:
            size_oracle_index = min(
                indices,
                key=lambda idx: (
                    abs(len(cell["states"][idx]) - target_size),
                    -float(item["scores"][idx]),
                ),
            )
        size_oracle = cell["states"][size_oracle_index]
        for bucket, state in ((base_counts, base), (size_oracle_counts, size_oracle)):
            bucket["truePositive"] += len(cell["target"] & state)
            bucket["falsePositive"] += len(state - cell["target"])
            bucket["falseNegative"] += len(cell["target"] - state)
        if better_reward <= base_reward + 1e-9:
            continue

        neighbours: list[frozenset[int]] = []
        if position:
            neighbours.append(ranked[position - 1]["base"])
        if position + 1 < len(ranked):
            neighbours.append(ranked[position + 1]["base"])
        base_affinity = float(np.mean([jaccard(base, n) for n in neighbours])) if neighbours else 0.0
        better_affinity = (
            float(np.mean([jaccard(better, n) for n in neighbours])) if neighbours else 0.0
        )
        opportunities.append(
            {
                "index": int(cell["index"]),
                "baseReward": base_reward,
                "bestReward": better_reward,
                "gain": better_reward - base_reward,
                "bestRank": best_rank,
                "editType": edit_type(base, better),
                "baseSize": base_size,
                "betterSize": len(better),
                "targetSize": target_size,
                "rewardBand": reward_band(base_reward),
                "baseScoreMargin": float(item["scores"][indices[0]] - item["scores"][better_index]),
                "baseAffinity": base_affinity,
                "betterAffinity": better_affinity,
                "affinityDelta": better_affinity - base_affinity,
            }
        )

    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": len(ranked),
        "topKBestRankCounts": dict(sorted(rank_counts.items())),
        "basePitchClass": micro_metrics(base_counts),
        "oracleTargetSizeTopKPitchClass": micro_metrics(size_oracle_counts),
        "cardinality": {
            "meanTargetSize": round(
                float(np.mean([len(item["cell"]["target"]) for item in ranked])), 6
            ),
            "meanBaseSize": round(float(np.mean([len(item["base"]) for item in ranked])), 6),
            "meanBaseMinusTarget": round(float(np.mean(size_deltas)), 6),
            "exactShare": round(float(np.mean([delta == 0 for delta in size_deltas])), 6),
            "overSelectedShare": round(float(np.mean([delta > 0 for delta in size_deltas])), 6),
            "underSelectedShare": round(float(np.mean([delta < 0 for delta in size_deltas])), 6),
            "topKContainsExactSizeShare": round(topk_exact_size / max(1, len(ranked)), 6),
            "confusionTargetToBase": dict(sorted(size_confusion.items())),
        },
        "opportunities": summarize_rows(opportunities),
        "byEditType": grouped(opportunities, "editType"),
        "byBaseRewardBand": grouped(opportunities, "rewardBand"),
        "byTargetSize": grouped(opportunities, "targetSize"),
        "rawOpportunities": opportunities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(
            prepare_song(copied, maximum_states=max(4, int(args.maximum_states)))
        )
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    reports: list[dict[str, Any]] = []
    for song in evaluation:
        model = fit_ridge_from_group_statistics(
            statistics,
            included_groups=sorted(set(statistics) - {song["groupId"]}),
            ridge=max(0.0, args.base_ridge),
            best_state_weight=max(0.0, args.base_best_state_weight),
        )
        reports.append(analyze_song(song, model, top_k=max(2, int(args.top_k))))

    all_rows = [row for song in reports for row in song.pop("rawOpportunities")]
    report = {
        "schema": "polymath-state-ranker-topk-error-analysis-v1",
        "audit": str(audit_path),
        "settings": {
            "topK": max(2, int(args.top_k)),
            "maximumStates": max(4, int(args.maximum_states)),
            "baseRidge": max(0.0, args.base_ridge),
            "baseBestStateWeight": max(0.0, args.base_best_state_weight),
            "validationBoundary": "whole-composition-held-out",
        },
        "aggregateOpportunities": summarize_rows(all_rows),
        "aggregateByEditType": grouped(all_rows, "editType"),
        "aggregateByBaseRewardBand": grouped(all_rows, "rewardBand"),
        "aggregateByTargetSize": grouped(all_rows, "targetSize"),
        "songs": reports,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["aggregateOpportunities"]}, indent=2))


if __name__ == "__main__":
    main()
