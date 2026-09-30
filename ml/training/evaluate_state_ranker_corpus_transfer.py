"""Measure whether an auxiliary corpus transfers or merely creates a specialist.

This experiment fits two otherwise identical whole-gesture rankers inside each
leave-one-composition-out fold:

* ``base`` sees only the frozen training audit;
* ``augmented`` also sees one or more auxiliary audits.

Their state scores can then be blended without ever reading the held-out
reference.  Reporting the complete alpha curve makes a tempting average gain
visible when it is paid for by a weaker worst-song result.  The script is a
research gate; it does not publish a model or modify production weights.
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


def parse_grid(value: str) -> list[float]:
    values = sorted(
        {
            max(0.0, min(1.0, float(item.strip())))
            for item in value.split(",")
            if item.strip()
        }
    )
    if not values:
        raise argparse.ArgumentTypeError("the alpha grid cannot be empty")
    return values


def load_audit_songs(
    path: Path,
    *,
    maximum_states: int,
    weight_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    audit = json.loads(path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for raw_song in audit.get("songs") or []:
        copied = dict(raw_song)
        copied["weight"] = (
            max(0.0, finite(raw_song.get("weight"), 1.0))
            * max(0.0, weight_multiplier)
        )
        copied["auditSource"] = str(path)
        songs.append(prepare_song(copied, maximum_states=maximum_states))
    return songs


def blended_argmax(
    base_scores: np.ndarray,
    augmented_scores: np.ndarray,
    alpha: float,
) -> int:
    """Select a state from an inference-safe convex score blend."""

    if base_scores.shape != augmented_scores.shape:
        raise ValueError("base and augmented scores must have identical shapes")
    blend = max(0.0, min(1.0, float(alpha)))
    scores = (1.0 - blend) * base_scores + blend * augmented_scores
    return int(np.argmax(scores))


def evaluate_blend(
    song: dict[str, Any],
    base_model: dict[str, Any],
    augmented_model: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    reward = 0.0
    exact = 0
    changed_from_base = 0
    changed_from_augmented = 0
    for cell in song["cells"]:
        base_scores = predict(cell["features"], base_model)
        augmented_scores = predict(cell["features"], augmented_model)
        selected_index = blended_argmax(base_scores, augmented_scores, alpha)
        selected = cell["states"][selected_index]
        target = cell["target"]
        counts["truePositive"] += len(target & selected)
        counts["falsePositive"] += len(selected - target)
        counts["falseNegative"] += len(target - selected)
        reward += set_f1(target, selected)
        exact += int(target == selected)
        changed_from_base += int(selected_index != int(np.argmax(base_scores)))
        changed_from_augmented += int(
            selected_index != int(np.argmax(augmented_scores))
        )
    cells = len(song["cells"])
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": cells,
        "weight": float(song["weight"]),
        "ranker": {
            **micro_metrics(counts),
            "macroF1": round(reward / max(1, cells), 6),
            "exactSetRate": round(exact / max(1, cells), 6),
        },
        "changedFromBase": changed_from_base,
        "changedFromAugmented": changed_from_augmented,
    }


def effective_evaluation_weights(
    songs: list[dict[str, Any]], *, group_balanced: bool
) -> dict[str, float]:
    group_counts: dict[str, int] = {}
    for song in songs:
        group_counts[song["groupId"]] = group_counts.get(song["groupId"], 0) + 1
    return {
        song["id"]: (
            max(0.0, float(song["weight"]))
            / (group_counts[song["groupId"]] if group_balanced else 1)
        )
        for song in songs
    }


def summarize(
    rows: list[dict[str, Any]], weights: dict[str, float]
) -> dict[str, Any]:
    total = sum(weights.get(row["id"], 0.0) for row in rows)
    weighted = sum(
        weights.get(row["id"], 0.0) * float(row["ranker"]["f1"])
        for row in rows
    ) / max(1e-12, total)
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(row["groupId"], []).append(float(row["ranker"]["f1"]))
    group_scores = {
        group: sum(values) / len(values) for group, values in by_group.items()
    }
    return {
        "weightedPitchClassF1": round(weighted, 6),
        "worstSongPitchClassF1": round(
            min((float(row["ranker"]["f1"]) for row in rows), default=0.0), 6
        ),
        "worstCompositionPitchClassF1": round(
            min(group_scores.values(), default=0.0), 6
        ),
        "changesFromBase": sum(int(row["changedFromBase"]) for row in rows),
        "changesFromAugmented": sum(
            int(row["changedFromAugmented"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--extra-audit", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-audit-weight", type=float, default=0.5)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--best-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--alpha-grid",
        type=parse_grid,
        default=parse_grid("0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"),
    )
    parser.add_argument("--evaluation-song-ids", nargs="+")
    parser.add_argument(
        "--group-balanced-evaluation",
        action="store_true",
        help="Divide each song weight by the number of variants in its composition.",
    )
    args = parser.parse_args()

    maximum_states = max(4, min(256, int(args.maximum_states)))
    base_path = Path(args.base_audit).resolve()
    extra_paths = [Path(value).resolve() for value in args.extra_audit]
    base_songs = load_audit_songs(
        base_path, maximum_states=maximum_states, weight_multiplier=1.0
    )
    extra_songs = [
        song
        for path in extra_paths
        for song in load_audit_songs(
            path,
            maximum_states=maximum_states,
            weight_multiplier=max(0.0, args.extra_audit_weight),
        )
    ]
    requested_ids = set(args.evaluation_song_ids or [])
    evaluation_songs = (
        [song for song in base_songs if song["id"] in requested_ids]
        if requested_ids
        else base_songs
    )
    missing = requested_ids - {song["id"] for song in evaluation_songs}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")

    rows_by_alpha: dict[float, list[dict[str, Any]]] = {
        alpha: [] for alpha in args.alpha_grid
    }
    groups = sorted({song["groupId"] for song in evaluation_songs})
    for heldout_group in groups:
        heldout_songs = [
            song for song in evaluation_songs if song["groupId"] == heldout_group
        ]
        base_training = [
            song for song in base_songs if song["groupId"] != heldout_group
        ]
        augmented_training = base_training + [
            song for song in extra_songs if song["groupId"] != heldout_group
        ]
        base_model = fit_ridge(
            base_training,
            ridge=max(0.0, args.ridge),
            best_state_weight=max(0.0, args.best_state_weight),
        )
        augmented_model = fit_ridge(
            augmented_training,
            ridge=max(0.0, args.ridge),
            best_state_weight=max(0.0, args.best_state_weight),
        )
        for alpha in args.alpha_grid:
            rows_by_alpha[alpha].extend(
                evaluate_blend(
                    song,
                    base_model,
                    augmented_model,
                    alpha=alpha,
                )
                for song in heldout_songs
            )

    weights = effective_evaluation_weights(
        evaluation_songs, group_balanced=bool(args.group_balanced_evaluation)
    )
    trials = [
        {
            "alpha": alpha,
            **summarize(rows_by_alpha[alpha], weights),
            "songs": rows_by_alpha[alpha],
        }
        for alpha in args.alpha_grid
    ]
    best_weighted = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstCompositionPitchClassF1"]),
            -abs(float(row["alpha"])),
        ),
    )
    best_floor = max(
        trials,
        key=lambda row: (
            float(row["worstCompositionPitchClassF1"]),
            float(row["weightedPitchClassF1"]),
            -abs(float(row["alpha"])),
        ),
    )
    report = {
        "schema": "polymath-state-ranker-corpus-transfer-v1",
        "status": "research-only-no-production-promotion",
        "baseAudit": str(base_path),
        "extraAudits": [str(path) for path in extra_paths],
        "settings": {
            "extraAuditWeight": max(0.0, args.extra_audit_weight),
            "maximumStates": maximum_states,
            "ridge": max(0.0, args.ridge),
            "bestStateWeight": max(0.0, args.best_state_weight),
            "groupBalancedEvaluation": bool(args.group_balanced_evaluation),
            "foldBoundary": "whole-composition-held-out",
        },
        "evaluationSongIds": [song["id"] for song in evaluation_songs],
        "bestWeighted": best_weighted,
        "bestFloor": best_floor,
        "trials": trials,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "bestWeighted": {
                    key: best_weighted[key]
                    for key in (
                        "alpha",
                        "weightedPitchClassF1",
                        "worstCompositionPitchClassF1",
                    )
                },
                "bestFloor": {
                    key: best_floor[key]
                    for key in (
                        "alpha",
                        "weightedPitchClassF1",
                        "worstCompositionPitchClassF1",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
