"""Combine the top-k reward reranker with an independent cardinality prior.

Both components are fitted inside each whole-composition-held-out fold.  The
cardinality classifier contributes only log-probability by state size; it
cannot invent notes or inspect labels at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package and direct-script execution.
    from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from .fit_pitch_class_cardinality_reranker import (
        fit_cardinality_logistic,
        predict_cardinality_probabilities,
    )
    from .fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        prepare_song,
        ridge_group_statistics,
    )
    from .fit_pitch_class_state_topk_reranker import (
        fit_meta_ridge,
        make_meta_song,
        meta_predict,
        select_meta_index,
    )
except ImportError:  # pragma: no cover - direct script fallback.
    from analyze_pitch_class_state_lattice import micro_metrics, set_f1
    from fit_pitch_class_cardinality_reranker import (
        fit_cardinality_logistic,
        predict_cardinality_probabilities,
    )
    from fit_pitch_class_state_ranker import (
        canonical_song_id,
        finite,
        fit_ridge_from_group_statistics,
        prepare_song,
        ridge_group_statistics,
    )
    from fit_pitch_class_state_topk_reranker import (
        fit_meta_ridge,
        make_meta_song,
        meta_predict,
        select_meta_index,
    )


def joint_scores(
    cell: dict[str, Any],
    meta_model: dict[str, Any],
    cardinality_model: dict[str, Any],
    *,
    cardinality_strength: float,
) -> np.ndarray:
    scores = meta_predict(cell["features"], meta_model)
    probabilities = predict_cardinality_probabilities(cell, cardinality_model)
    if probabilities is None or cardinality_strength <= 0.0:
        return scores
    return scores + max(0.0, cardinality_strength) * np.asarray(
        [
            np.log(max(1e-6, float(probabilities[min(6, max(1, len(state))) - 1])))
            for state in cell["states"]
        ]
    )


def evaluate_song(
    song: dict[str, Any],
    meta_model: dict[str, Any],
    cardinality_model: dict[str, Any],
    *,
    cardinality_strength: float,
    minimum_override_margin: float,
) -> dict[str, Any]:
    base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    selected_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    changes = improvements = regressions = 0
    for cell in song["cells"]:
        selected_index = select_meta_index(
            joint_scores(
                cell,
                meta_model,
                cardinality_model,
                cardinality_strength=cardinality_strength,
            ),
            minimum_override_margin,
        )
        target = cell["target"]
        base = cell["states"][0]
        selected = cell["states"][selected_index]
        for bucket, state in ((base_counts, base), (selected_counts, selected)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        base_reward = set_f1(target, base)
        selected_reward = set_f1(target, selected)
        changes += int(selected_index != 0)
        improvements += int(selected_reward > base_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < base_reward)
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": len(song["cells"]),
        "base": micro_metrics(base_counts),
        "jointReranker": micro_metrics(selected_counts),
        "changes": changes,
        "improvements": improvements,
        "regressions": regressions,
    }


def weighted(rows: list[dict[str, Any]], section: str) -> float:
    total = sum(max(0.0, float(row["weight"])) for row in rows)
    return sum(
        max(0.0, float(row["weight"])) * float(row[section]["f1"])
        for row in rows
    ) / max(1e-12, total)


def load_songs(path: Path, maximum_states: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(prepare_song(copied, maximum_states=maximum_states))
    return songs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--evaluation-audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    parser.add_argument("--meta-ridge", type=float, default=0.01)
    parser.add_argument("--meta-best-state-weight", type=float, default=1.0)
    parser.add_argument("--cardinality-c", type=float, default=0.01)
    parser.add_argument(
        "--cardinality-strength-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02],
    )
    parser.add_argument(
        "--override-margin-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08],
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    maximum_states = max(4, min(256, int(args.maximum_states)))
    top_k = max(2, min(maximum_states, int(args.top_k)))
    songs = load_songs(audit_path, maximum_states)
    evaluation_path = Path(args.evaluation_audit).resolve() if args.evaluation_audit else None
    evaluation_source = (
        load_songs(evaluation_path, maximum_states) if evaluation_path else songs
    )
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in evaluation_source if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    all_groups = set(statistics)
    model_cache: dict[frozenset[str], dict[str, Any]] = {}

    def base_model_excluding(groups: set[str]) -> dict[str, Any]:
        key = frozenset(groups)
        if key not in model_cache:
            model_cache[key] = fit_ridge_from_group_statistics(
                statistics,
                included_groups=sorted(all_groups - groups),
                ridge=max(0.0, args.base_ridge),
                best_state_weight=max(0.0, args.base_best_state_weight),
            )
        return model_cache[key]

    folds: list[dict[str, Any]] = []
    for heldout in evaluation:
        outer = heldout["groupId"]
        meta_training: list[dict[str, Any]] = []
        for inner in sorted(all_groups - {outer}):
            inner_base = base_model_excluding({outer, inner})
            meta_training.extend(
                make_meta_song(song, inner_base, top_k=top_k)
                for song in songs
                if song["groupId"] == inner
            )
        heldout_base = base_model_excluding({outer})
        folds.append(
            {
                "heldout": make_meta_song(heldout, heldout_base, top_k=top_k),
                "meta": fit_meta_ridge(
                    meta_training,
                    ridge=max(0.0, args.meta_ridge),
                    best_state_weight=max(0.0, args.meta_best_state_weight),
                ),
                "cardinality": fit_cardinality_logistic(
                    meta_training,
                    top_k=top_k,
                    regularization_c=max(1e-5, args.cardinality_c),
                ),
            }
        )

    trials: list[dict[str, Any]] = []
    for strength in sorted(
        {max(0.0, float(value)) for value in args.cardinality_strength_grid}
    ):
        for margin in sorted(
            {max(0.0, float(value)) for value in args.override_margin_grid}
        ):
            rows = [
                evaluate_song(
                    fold["heldout"],
                    fold["meta"],
                    fold["cardinality"],
                    cardinality_strength=strength,
                    minimum_override_margin=margin,
                )
                for fold in folds
            ]
            trials.append(
                {
                    "cardinalityStrength": strength,
                    "minimumOverrideMargin": margin,
                    "baseWeightedPitchClassF1": round(weighted(rows, "base"), 6),
                    "weightedPitchClassF1": round(weighted(rows, "jointReranker"), 6),
                    "worstSongPitchClassF1": round(
                        min(float(row["jointReranker"]["f1"]) for row in rows), 6
                    ),
                    "folds": rows,
                }
            )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            -float(row["cardinalityStrength"]),
            -float(row["minimumOverrideMargin"]),
        ),
    )
    report = {
        "schema": "polymath-pitch-class-joint-reranker-training-v1",
        "status": "opened-development-only",
        "audit": str(audit_path),
        "evaluationAudit": str(evaluation_path) if evaluation_path else None,
        "evaluationSongIds": args.evaluation_song_ids,
        "settings": {
            "topK": top_k,
            "maximumStates": maximum_states,
            "baseRidge": args.base_ridge,
            "baseBestStateWeight": args.base_best_state_weight,
            "metaRidge": args.meta_ridge,
            "metaBestStateWeight": args.meta_best_state_weight,
            "cardinalityC": args.cardinality_c,
            "validationBoundary": "outer-and-inner-composition-excluded",
        },
        "best": best,
        "trials": trials,
        "decision": {
            "promotable": False,
            "reason": "Opened-song blend selection requires independent transfer validation.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                **{key: value for key, value in best.items() if key != "folds"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
