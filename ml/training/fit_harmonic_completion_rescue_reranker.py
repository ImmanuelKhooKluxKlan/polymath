"""Cross-fit a selector over accepted top-k plus implied-triad rescue states.

Unlike the rejected mixed-lattice ablation, this preserves the accepted base
top-k ordering and appends completion states in a separate lane.  Whole-song
outer/inner exclusions prevent base-score leakage into the meta learner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    from .fit_pitch_class_state_topk_reranker import (
        evaluate_meta_song,
        fit_meta_ridge,
        make_meta_song,
        meta_predict,
        meta_feature_matrix,
        select_meta_index,
        weighted,
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
    from fit_pitch_class_state_topk_reranker import (
        evaluate_meta_song,
        fit_meta_ridge,
        make_meta_song,
        meta_predict,
        meta_feature_matrix,
        select_meta_index,
        weighted,
    )


def prepare_rescue_bundle(
    raw: dict[str, Any],
    *,
    maximum_states: int,
    rescue_harmonic_context: bool = False,
) -> dict[str, Any]:
    """Materialize only rescue differences, releasing the large full lattices."""

    base = prepare_song(raw, maximum_states=maximum_states)
    original = prepare_song(
        raw,
        maximum_states=256,
        include_harmonic_context=rescue_harmonic_context,
    )
    expanded = prepare_song(
        raw,
        maximum_states=256,
        include_harmonic_context=rescue_harmonic_context,
        allow_harmonic_completion=True,
    )
    original_by_index = {cell["index"]: cell for cell in original["cells"]}
    expanded_by_index = {cell["index"]: cell for cell in expanded["cells"]}
    rescues: dict[int, dict[str, Any]] = {}
    for cell in base["cells"]:
        index = cell["index"]
        original_states = set(original_by_index[index]["states"])
        original_features = {
            state: original_by_index[index]["features"][position]
            for position, state in enumerate(original_by_index[index]["states"])
        }
        expanded_cell = expanded_by_index[index]
        rescue_indices = [
            position
            for position, state in enumerate(expanded_cell["states"])
            if state not in original_states
        ]
        rescues[index] = {
            "states": [expanded_cell["states"][position] for position in rescue_indices],
            "features": np.asarray(
                [expanded_cell["features"][position] for position in rescue_indices],
                dtype=np.float32,
            ),
            "rewards": np.asarray(
                [expanded_cell["rewards"][position] for position in rescue_indices],
                dtype=np.float32,
            ),
            "originalFeatures": original_features,
        }
    base["rescues"] = rescues
    return base


def make_rescue_meta_song(
    song: dict[str, Any],
    base_model: dict[str, Any],
    *,
    top_k: int,
    maximum_rescues: int,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell in song["cells"]:
        base_scores = predict(cell["features"], base_model)
        accepted_indices = top_k_indices(base_scores, top_k)
        accepted_states = [cell["states"][index] for index in accepted_indices]
        accepted_features = [
            rescue_feature
            for state, index in zip(accepted_states, accepted_indices, strict=True)
            for rescue_feature in [
                song["rescues"][cell["index"]]
                .get("originalFeatures", {})
                .get(state, cell["features"][index])
            ]
        ]
        accepted_rewards = [float(cell["rewards"][index]) for index in accepted_indices]
        accepted_scores = [float(base_scores[index]) for index in accepted_indices]

        rescue = song["rescues"][cell["index"]]
        rescue_features = np.asarray(rescue["features"], dtype=float)
        if len(rescue_features):
            rescue_scores = predict(rescue_features, base_model)
            rescue_order = np.argsort(-rescue_scores, kind="stable")[
                : max(0, maximum_rescues)
            ]
        else:
            rescue_scores = np.asarray([], dtype=float)
            rescue_order = np.asarray([], dtype=int)
        states = accepted_states + [rescue["states"][index] for index in rescue_order]
        state_features = np.asarray(
            accepted_features + [rescue_features[index] for index in rescue_order],
            dtype=float,
        )
        scores = np.asarray(
            accepted_scores + [float(rescue_scores[index]) for index in rescue_order],
            dtype=float,
        )
        rewards = np.asarray(
            accepted_rewards + [float(rescue["rewards"][index]) for index in rescue_order],
            dtype=float,
        )
        cells.append(
            {
                "target": cell["target"],
                "candidate": cell["candidate"],
                "states": states,
                "features": meta_feature_matrix(
                    state_features, scores, list(range(len(states)))
                ),
                "rewards": rewards,
                "baseScores": scores,
                "acceptedCount": len(accepted_states),
                "rescueCount": len(states) - len(accepted_states),
            }
        )
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "weight": float(song["weight"]),
        "cells": cells,
    }


def evaluate_cascade_song(
    accepted_song: dict[str, Any],
    rescue_song: dict[str, Any],
    accepted_model: dict[str, Any],
    rescue_model: dict[str, Any],
    *,
    accepted_override_margin: float,
    rescue_override_margin: float,
) -> dict[str, Any]:
    """Preserve the accepted reranker, then allow specialist rescue overrides."""

    base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    accepted_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    cascade_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    oracle_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    rescue_changes = improvements = regressions = 0
    for accepted_cell, rescue_cell in zip(
        accepted_song["cells"], rescue_song["cells"], strict=True
    ):
        accepted_index = select_meta_index(
            meta_predict(accepted_cell["features"], accepted_model),
            accepted_override_margin,
        )
        rescue_scores = meta_predict(rescue_cell["features"], rescue_model)
        accepted_count = int(rescue_cell["acceptedCount"])
        selected_index = accepted_index
        if len(rescue_scores) > accepted_count:
            best_rescue = accepted_count + int(np.argmax(rescue_scores[accepted_count:]))
            if (
                float(rescue_scores[best_rescue] - rescue_scores[accepted_index])
                >= rescue_override_margin
            ):
                selected_index = best_rescue
        target = rescue_cell["target"]
        base = rescue_cell["states"][0]
        accepted = rescue_cell["states"][accepted_index]
        selected = rescue_cell["states"][selected_index]
        oracle = rescue_cell["states"][int(np.argmax(rescue_cell["rewards"]))]
        for bucket, state in (
            (base_counts, base),
            (accepted_counts, accepted),
            (cascade_counts, selected),
            (oracle_counts, oracle),
        ):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        rescue_changes += int(selected_index != accepted_index)
        improvements += int(set_f1(target, selected) > set_f1(target, accepted) + 1e-9)
        regressions += int(set_f1(target, selected) + 1e-9 < set_f1(target, accepted))
    return {
        "id": rescue_song["id"],
        "groupId": rescue_song["groupId"],
        "weight": float(rescue_song["weight"]),
        "cells": len(rescue_song["cells"]),
        "base": micro_metrics(base_counts),
        "acceptedReranker": micro_metrics(accepted_counts),
        "reranker": micro_metrics(cascade_counts),
        "topKOracle": micro_metrics(oracle_counts),
        "rescueChanges": rescue_changes,
        "improvementsOverAccepted": improvements,
        "regressionsFromAccepted": regressions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-rescues", type=int, default=99)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument(
        "--rescue-harmonic-context",
        action="store_true",
        help="Give the rescue selector target-blind global/local key features.",
    )
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--cascade-accepted-reranker",
        action="store_true",
        help="Run the proven top-k ridge reranker first, then gate rescue overrides.",
    )
    parser.add_argument("--accepted-meta-ridge", type=float, default=0.01)
    parser.add_argument("--accepted-meta-best-state-weight", type=float, default=1.0)
    parser.add_argument("--accepted-override-margin", type=float, default=0.02)
    parser.add_argument("--meta-ridge-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--meta-best-state-weight-grid", type=float, nargs="+", default=[0.0, 1.0, 3.0, 8.0])
    parser.add_argument(
        "--override-margin-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2],
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    maximum_states = max(4, min(256, int(args.maximum_states)))
    top_k = max(2, min(maximum_states, int(args.top_k)))
    maximum_rescues = max(1, int(args.maximum_rescues))
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(raw.get("id"))
        songs.append(
            prepare_rescue_bundle(
                copied,
                maximum_states=maximum_states,
                rescue_harmonic_context=bool(args.rescue_harmonic_context),
            )
        )

    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")
    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    all_groups = set(statistics)
    base_cache: dict[frozenset[str], dict[str, Any]] = {}

    def base_model_excluding(groups: set[str]) -> dict[str, Any]:
        key = frozenset(groups)
        if key not in base_cache:
            base_cache[key] = fit_ridge_from_group_statistics(
                statistics,
                included_groups=sorted(all_groups - groups),
                ridge=max(0.0, args.base_ridge),
                best_state_weight=max(0.0, args.base_best_state_weight),
            )
        return base_cache[key]

    fold_material: list[dict[str, Any]] = []
    for heldout in evaluation:
        outer = heldout["groupId"]
        meta_training: list[dict[str, Any]] = []
        accepted_training: list[dict[str, Any]] = []
        for inner in sorted(all_groups - {outer}):
            inner_model = base_model_excluding({outer, inner})
            for song in songs:
                if song["groupId"] != inner:
                    continue
                meta_training.append(
                    make_rescue_meta_song(
                        song,
                        inner_model,
                        top_k=top_k,
                        maximum_rescues=maximum_rescues,
                    )
                )
                accepted_training.append(
                    make_meta_song(song, inner_model, top_k=top_k)
                )
        outer_model = base_model_excluding({outer})
        fold_material.append(
            {
                "training": meta_training,
                "acceptedTraining": accepted_training,
                "heldout": make_rescue_meta_song(
                    heldout,
                    outer_model,
                    top_k=top_k,
                    maximum_rescues=maximum_rescues,
                ),
                "acceptedHeldout": make_meta_song(
                    heldout, outer_model, top_k=top_k
                ),
            }
        )

    accepted_models = [
        fit_meta_ridge(
            fold["acceptedTraining"],
            ridge=max(0.0, args.accepted_meta_ridge),
            best_state_weight=max(0.0, args.accepted_meta_best_state_weight),
        )
        for fold in fold_material
    ]

    trials: list[dict[str, Any]] = []
    for ridge in sorted({max(0.0, float(value)) for value in args.meta_ridge_grid}):
        for best_weight in sorted(
            {max(0.0, float(value)) for value in args.meta_best_state_weight_grid}
        ):
            fitted = [
                fit_meta_ridge(
                    fold["training"], ridge=ridge, best_state_weight=best_weight
                )
                for fold in fold_material
            ]
            for margin in sorted(
                {max(0.0, float(value)) for value in args.override_margin_grid}
            ):
                if args.cascade_accepted_reranker:
                    rows = [
                        evaluate_cascade_song(
                            fold["acceptedHeldout"],
                            fold["heldout"],
                            accepted_model,
                            rescue_model,
                            accepted_override_margin=max(
                                0.0, args.accepted_override_margin
                            ),
                            rescue_override_margin=margin,
                        )
                        for fold, accepted_model, rescue_model in zip(
                            fold_material, accepted_models, fitted, strict=True
                        )
                    ]
                else:
                    rows = [
                        evaluate_meta_song(
                            fold["heldout"],
                            model,
                            minimum_override_margin=margin,
                        )
                        for fold, model in zip(fold_material, fitted, strict=True)
                    ]
                trials.append(
                    {
                        "ridge": ridge,
                        "bestStateWeight": best_weight,
                        "minimumOverrideMargin": margin,
                        "baseWeightedPitchClassF1": round(weighted(rows, "base"), 6),
                        "weightedPitchClassF1": round(weighted(rows, "reranker"), 6),
                        "worstSongPitchClassF1": round(
                            min(float(row["reranker"]["f1"]) for row in rows), 6
                        ),
                        "rescueOracleWeightedPitchClassF1": round(
                            weighted(rows, "topKOracle"), 6
                        ),
                        "folds": rows,
                    }
                )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            -float(row["bestStateWeight"]),
            -float(row["ridge"]),
            -float(row["minimumOverrideMargin"]),
        ),
    )
    report = {
        "schema": "polymath-harmonic-completion-rescue-reranker-v1",
        "status": "opened-development-only",
        "audit": str(audit_path),
        "evaluationSongIds": args.evaluation_song_ids,
        "settings": {
            "topK": top_k,
            "maximumRescues": maximum_rescues,
            "rescueHarmonicContext": bool(args.rescue_harmonic_context),
            "cascadeAcceptedReranker": bool(args.cascade_accepted_reranker),
            "acceptedMetaRidge": max(0.0, args.accepted_meta_ridge),
            "acceptedMetaBestStateWeight": max(
                0.0, args.accepted_meta_best_state_weight
            ),
            "acceptedOverrideMargin": max(0.0, args.accepted_override_margin),
            "acceptedLatticePreserved": True,
            "validationBoundary": "outer-and-inner-composition-excluded",
        },
        "best": best,
        "trials": trials,
        "decision": {
            "promotable": False,
            "reason": "Opened-song selection; independent transfer validation required.",
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
