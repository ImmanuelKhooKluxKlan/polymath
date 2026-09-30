"""Measure inferred-triad candidates without displacing the accepted lattice.

The accepted top-k list is frozen.  Complete implied triads are generated in a
separate 256-state diagnostic lattice, differenced against the same lattice
without completion, and appended only as rescue candidates.  This establishes
whether harmonic inference adds answer coverage before any selector is fitted.
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


def add_counts(bucket: dict[str, int], target: frozenset[int], state: frozenset[int]) -> None:
    bucket["truePositive"] += len(target & state)
    bucket["falsePositive"] += len(state - target)
    bucket["falseNegative"] += len(target - state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-song-ids", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rescue-counts", type=int, nargs="+", default=[1, 2, 4, 99])
    parser.add_argument("--base-ridge", type=float, default=0.01)
    parser.add_argument("--base-best-state-weight", type=float, default=1.0)
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    raw_by_id: dict[str, dict[str, Any]] = {}
    songs: list[dict[str, Any]] = []
    for raw in payload.get("songs") or []:
        copied = dict(raw)
        song_id = str(raw.get("id") or "")
        copied["weight"] = max(0.0, finite(raw.get("weight"), 1.0))
        copied["groupId"] = canonical_song_id(song_id)
        raw_by_id[song_id] = copied
        songs.append(prepare_song(copied, maximum_states=64))

    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation song ids: {sorted(missing)}")
    statistics = ridge_group_statistics(
        songs, best_state_weight=max(0.0, args.base_best_state_weight)
    )
    all_groups = set(statistics)
    rescue_counts = sorted({max(1, int(value)) for value in args.rescue_counts})
    song_reports: list[dict[str, Any]] = []

    for song in evaluation:
        model = fit_ridge_from_group_statistics(
            statistics,
            included_groups=sorted(all_groups - {song["groupId"]}),
            ridge=max(0.0, args.base_ridge),
            best_state_weight=max(0.0, args.base_best_state_weight),
        )
        raw = raw_by_id[song["id"]]
        full_original = prepare_song(raw, maximum_states=256)
        full_expanded = prepare_song(
            raw, maximum_states=256, allow_harmonic_completion=True
        )
        original_by_index = {cell["index"]: cell for cell in full_original["cells"]}
        expanded_by_index = {cell["index"]: cell for cell in full_expanded["cells"]}
        base_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
        topk_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
        rescue_metrics = {
            count: {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
            for count in rescue_counts
        }
        cells_with_rescue = cells_improved = 0
        rescue_candidates = 0
        low_support_rows: list[dict[str, Any]] = []
        raw_cells = raw.get("cells") or []
        for cell in song["cells"]:
            scores = predict(cell["features"], model)
            indices = top_k_indices(scores, max(2, int(args.top_k)))
            accepted = [cell["states"][idx] for idx in indices]
            base = accepted[0]
            topk_oracle = max(accepted, key=lambda state: set_f1(cell["target"], state))
            add_counts(base_counts, cell["target"], base)
            add_counts(topk_counts, cell["target"], topk_oracle)

            original_states = set(original_by_index[cell["index"]]["states"])
            expanded_cell = expanded_by_index[cell["index"]]
            extra_rows = [
                (state, expanded_cell["features"][index])
                for index, state in enumerate(expanded_cell["states"])
                if state not in original_states
            ]
            if extra_rows:
                extra_scores = predict(
                    np.asarray([features for _state, features in extra_rows]), model
                )
                ordered_extra = [
                    extra_rows[index][0]
                    for index in np.argsort(-extra_scores, kind="stable")
                ]
                cells_with_rescue += 1
                rescue_candidates += len(ordered_extra)
            else:
                ordered_extra = []
            best_rescue_reward = max(
                [set_f1(cell["target"], state) for state in ordered_extra],
                default=-1.0,
            )
            cells_improved += int(best_rescue_reward > set_f1(cell["target"], topk_oracle) + 1e-9)
            for count in rescue_counts:
                pool = accepted + ordered_extra[:count]
                chosen = max(pool, key=lambda state: set_f1(cell["target"], state))
                add_counts(rescue_metrics[count], cell["target"], chosen)
            support = finite(raw_cells[int(cell["index"])].get("sourcePitchClassSupport"))
            if support < 0.5:
                low_support_rows.append(
                    {
                        "target": cell["target"],
                        "accepted": accepted,
                        "extra": ordered_extra,
                    }
                )

        low_reports: dict[str, Any] = {}
        for count in rescue_counts:
            counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
            for row in low_support_rows:
                pool = row["accepted"] + row["extra"][:count]
                chosen = max(pool, key=lambda state: set_f1(row["target"], state))
                add_counts(counts, row["target"], chosen)
            low_reports[str(count)] = micro_metrics(counts)
        song_reports.append(
            {
                "id": song["id"],
                "cells": len(song["cells"]),
                "cellsWithRescueCandidates": cells_with_rescue,
                "meanRescueCandidatesWhenAvailable": round(
                    rescue_candidates / max(1, cells_with_rescue), 6
                ),
                "cellsWhereRescueImprovesTopKOracle": cells_improved,
                "base": micro_metrics(base_counts),
                "acceptedTopKOracle": micro_metrics(topk_counts),
                "rescueOracleByMaximumExtras": {
                    str(count): micro_metrics(metrics)
                    for count, metrics in rescue_metrics.items()
                },
                "lowSourceSupportCells": len(low_support_rows),
                "lowSourceSupportRescueOracle": low_reports,
            }
        )

    report = {
        "schema": "polymath-harmonic-completion-rescue-analysis-v1",
        "audit": str(audit_path),
        "evidenceBoundary": (
            "Accepted top-k states remain untouched. Rescue states are target-blind "
            "implied-triad completions; reference is used only for oracle measurement."
        ),
        "settings": {"topK": max(2, int(args.top_k)), "rescueCounts": rescue_counts},
        "songs": song_reports,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "songs": [
                    {
                        "id": row["id"],
                        "acceptedTopKOracleF1": row["acceptedTopKOracle"]["f1"],
                        "allRescueOracleF1": row["rescueOracleByMaximumExtras"][str(rescue_counts[-1])]["f1"],
                        "improvedCells": row["cellsWhereRescueImprovesTopKOracle"],
                    }
                    for row in song_reports
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
