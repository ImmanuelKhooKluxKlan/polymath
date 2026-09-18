"""Blend complementary whole-song out-of-fold proposal scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ablate_raw_support_swap_benefit_models import summarize_model
from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / max(1, len(values))
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument(
        "--right-report",
        type=Path,
        help="Optional second out-of-fold report containing the right model.",
    )
    parser.add_argument("--left-model", default="current-tiny-mlp-v001")
    parser.add_argument("--right-model", default="extra-trees-d8-l10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest.resolve())
    selector_root = args.selector_profiles_root.resolve()
    songs = [
        build_song_examples(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            source_radius=0.18,
            minimum_source_midi=0,
            maximum_source_midi=59,
            minimum_output_midi=33,
            maximum_output_midi=71,
            register_shifts=(0, 12),
        )
        for row in manifest.get("songs") or []
    ]
    left_source = load_json(args.ablation_report.resolve())
    right_source = (
        load_json(args.right_report.resolve()) if args.right_report else left_source
    )
    left_by_name = {
        str(row.get("name")): row for row in left_source.get("models") or []
    }
    right_by_name = {
        str(row.get("name")): row for row in right_source.get("models") or []
    }
    if args.left_model not in left_by_name:
        raise ValueError(f"Left model is absent from its report: {args.left_model}")
    if args.right_model not in right_by_name:
        raise ValueError(f"Right model is absent from its report: {args.right_model}")
    left = left_by_name[args.left_model]["probabilities"]
    right = right_by_name[args.right_model]["probabilities"]
    blends: list[dict[str, Any]] = []
    for mode in ("raw", "rank"):
        for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
            probabilities: dict[str, np.ndarray] = {}
            for song in songs:
                song_id = song["id"]
                left_values = np.asarray(left[song_id], dtype=np.float64)
                right_values = np.asarray(right[song_id], dtype=np.float64)
                if mode == "rank":
                    left_values = percentile_ranks(left_values)
                    right_values = percentile_ranks(right_values)
                probabilities[song_id] = alpha * left_values + (1.0 - alpha) * right_values
            blends.append(
                summarize_model(
                    f"{mode}-blend-{args.left_model}-{args.right_model}-a{alpha:.2f}",
                    songs,
                    probabilities,
                )
            )
    for name, combine in (
        ("rank-geometric", lambda left_values, right_values: np.sqrt(left_values * right_values)),
        ("rank-minimum", np.minimum),
        ("rank-maximum", np.maximum),
    ):
        probabilities = {}
        for song in songs:
            song_id = song["id"]
            left_values = percentile_ranks(np.asarray(left[song_id], dtype=np.float64))
            right_values = percentile_ranks(np.asarray(right[song_id], dtype=np.float64))
            probabilities[song_id] = combine(left_values, right_values)
        blends.append(
            summarize_model(
                f"{name}-{args.left_model}-{args.right_model}", songs, probabilities
            )
        )
    # A veto preserves the calibrated left score while allowing the second
    # model to reject only its least convincing proposals.  Unlike a blend,
    # this cannot promote a proposal the established model scored poorly.
    for minimum_rank in (0.30, 0.50, 0.65, 0.75, 0.85, 0.90):
        probabilities = {}
        for song in songs:
            song_id = song["id"]
            left_values = np.asarray(left[song_id], dtype=np.float64)
            right_ranks = percentile_ranks(
                np.asarray(right[song_id], dtype=np.float64)
            )
            probabilities[song_id] = np.where(
                right_ranks >= minimum_rank, left_values, 0.0
            )
        blends.append(
            summarize_model(
                f"right-rank-veto-q{minimum_rank:.2f}-{args.left_model}-{args.right_model}",
                songs,
                probabilities,
            )
        )
    blends.sort(
        key=lambda row: (
            float(row["aggregate"]["minimumSongTop10Precision"]),
            float(row["aggregate"]["meanSongTop10Precision"]),
            float(row["aggregate"]["minimumSongAveragePrecision"]),
            float(row["aggregate"]["meanSongAveragePrecision"]),
        ),
        reverse=True,
    )
    report = {
        "schema": "polymath-swap-benefit-oof-score-blends-v1",
        "evidenceBoundary": "Every input score is whole-song out-of-fold; rank normalization uses only inference-time proposal scores from that song.",
        "manifest": str(args.manifest.resolve()),
        "sourceReports": [
            str(args.ablation_report.resolve()),
            str((args.right_report or args.ablation_report).resolve()),
        ],
        "sourceModels": [args.left_model, args.right_model],
        "ranking": [
            {"rank": index + 1, "name": row["name"], **row["aggregate"]}
            for index, row in enumerate(blends)
        ],
        "models": blends,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "top": report["ranking"][:5]}, indent=2))


if __name__ == "__main__":
    main()
