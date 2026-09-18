"""Compose two whole-song OOF rankers into register-specific branch scores.

Each branch keeps its own calibrated threshold and safety gates.  Scores are
mapped to a common margin around 0.8 so the existing decoder can resolve a
gesture conflict while still applying at most one replacement.  This script
does not fit on references; it only combines already out-of-fold scores and
inference-time proposal fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def find_model(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in report.get("models") or [] if str(row.get("name")) == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one model named {name!r}")
    return matches[0]


def branch_score(
    raw_score: float,
    proposal: dict[str, Any],
    *,
    threshold: float,
    selector_minimum: float,
    gain_minimum: float,
    scale: float,
) -> float:
    if float(proposal["sourceProbability"]) < selector_minimum:
        return 0.0
    if float(proposal["probabilityGain"]) < gain_minimum:
        return 0.0
    return float(np.clip(0.8 + scale * (raw_score - threshold), 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--left-register", type=int, required=True)
    parser.add_argument("--left-threshold", type=float, required=True)
    parser.add_argument("--left-selector-minimum", type=float, required=True)
    parser.add_argument("--left-gain-minimum", type=float, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--right-register", type=int, required=True)
    parser.add_argument("--right-threshold", type=float, required=True)
    parser.add_argument("--right-selector-minimum", type=float, required=True)
    parser.add_argument("--right-gain-minimum", type=float, required=True)
    parser.add_argument("--right-scales", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.left_register == args.right_register:
        raise ValueError("Branch registers must differ")
    scales = tuple(float(value) for value in args.right_scales.split(",") if value.strip())
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("Right scales must be positive")

    manifest = load_json(args.manifest.resolve())
    selector_root = args.selector_profiles_root.resolve()
    left = find_model(load_json(args.left_report.resolve()), args.left_model)
    right = find_model(load_json(args.right_report.resolve()), args.right_model)
    songs = [
        build_song_examples(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            source_radius=0.18,
            minimum_source_midi=0,
            maximum_source_midi=59,
            minimum_output_midi=33,
            maximum_output_midi=71,
            register_shifts=(args.left_register, args.right_register),
        )
        for row in manifest.get("songs") or []
    ]

    models: list[dict[str, Any]] = []
    for right_scale in scales:
        probabilities: dict[str, list[float]] = {}
        for song in songs:
            song_id = song["id"]
            left_values = left.get("probabilities", {}).get(song_id)
            right_values = right.get("probabilities", {}).get(song_id)
            if not isinstance(left_values, list) or len(left_values) != len(song["proposals"]):
                raise ValueError(f"{song_id} left score count mismatch")
            if not isinstance(right_values, list) or len(right_values) != len(song["proposals"]):
                raise ValueError(f"{song_id} right score count mismatch")
            combined: list[float] = []
            for index, proposal in enumerate(song["proposals"]):
                register = int(proposal["registerShift"])
                if register == args.left_register:
                    value = branch_score(
                        float(left_values[index]),
                        proposal,
                        threshold=float(args.left_threshold),
                        selector_minimum=float(args.left_selector_minimum),
                        gain_minimum=float(args.left_gain_minimum),
                        scale=1.0,
                    )
                elif register == args.right_register:
                    value = branch_score(
                        float(right_values[index]),
                        proposal,
                        threshold=float(args.right_threshold),
                        selector_minimum=float(args.right_selector_minimum),
                        gain_minimum=float(args.right_gain_minimum),
                        scale=right_scale,
                    )
                else:
                    value = 0.0
                combined.append(round(value, 8))
            probabilities[song_id] = combined
        models.append(
            {
                "name": f"register-branch-right-scale-{right_scale:g}",
                "wholeSongOutOfFold": True,
                "probabilities": probabilities,
            }
        )

    report = {
        "schema": "polymath-swap-benefit-register-branch-scores-v1",
        "evidenceBoundary": "Both source score sets are whole-song out-of-fold; combination uses inference-time proposal fields only.",
        "manifest": str(args.manifest.resolve()),
        "leftBranch": {
            "report": str(args.left_report.resolve()),
            "model": args.left_model,
            "register": args.left_register,
            "threshold": args.left_threshold,
            "selectorMinimum": args.left_selector_minimum,
            "gainMinimum": args.left_gain_minimum,
        },
        "rightBranch": {
            "report": str(args.right_report.resolve()),
            "model": args.right_model,
            "register": args.right_register,
            "threshold": args.right_threshold,
            "selectorMinimum": args.right_selector_minimum,
            "gainMinimum": args.right_gain_minimum,
            "scales": list(scales),
        },
        "models": models,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "models": [row["name"] for row in models]}, indent=2))


if __name__ == "__main__":
    main()
