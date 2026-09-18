"""Compose conservative benefit scores by vetoing predicted harmful swaps.

The established benefit model remains the only component allowed to promote a
proposal.  A harm model may replace that score with zero, which makes the
proposal ineligible, but it may never raise a score or introduce a proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .apply_raw_support_recovery import load_json


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def comma_separated(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def numeric_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in comma_separated(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harm-report", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=comma_separated,
        default=(
            "harm-logistic-c0.1",
            "harm-random-forest-d8-l10",
            "harm-extra-trees-d8-l10",
        ),
    )
    parser.add_argument(
        "--veto-thresholds",
        type=numeric_grid,
        default=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.harm_report.resolve()
    source = load_json(source_path)
    base_by_song = source.get("baseScores") or {}
    models_by_name = {
        str(row.get("name")): row for row in source.get("models") or []
    }
    missing = [name for name in args.models if name not in models_by_name]
    if missing:
        raise ValueError(f"Harm models are absent from the report: {missing}")

    composed: list[dict[str, Any]] = []
    for model_name in args.models:
        harm_by_song = models_by_name[model_name].get("probabilities") or {}
        for threshold in args.veto_thresholds:
            probabilities: dict[str, list[float]] = {}
            veto_counts: dict[str, int] = {}
            for song_id, raw_base in base_by_song.items():
                base = np.asarray(raw_base, dtype=np.float64)
                harm = np.asarray(harm_by_song.get(song_id), dtype=np.float64)
                if base.shape != harm.shape:
                    raise ValueError(
                        f"{song_id} score shape mismatch: {base.shape} != {harm.shape}"
                    )
                vetoed = harm >= float(threshold)
                safe = np.where(vetoed, 0.0, base)
                probabilities[str(song_id)] = [
                    round(float(value), 8) for value in safe
                ]
                veto_counts[str(song_id)] = int(vetoed.sum())
            name = f"base-with-{model_name}-veto-ge-{threshold:.2f}"
            composed.append(
                {
                    "name": name,
                    "harmModel": model_name,
                    "vetoThreshold": float(threshold),
                    "direction": "veto when P(harm) >= threshold",
                    "cannotPromote": True,
                    "vetoCounts": veto_counts,
                    "probabilities": probabilities,
                }
            )

    report = {
        "schema": "polymath-swap-harm-veto-scores-v1",
        "evidenceBoundary": source.get("evidenceBoundary"),
        "sourceReport": str(source_path),
        "manifest": source.get("manifest"),
        "invariant": "Each output score is either the established base score or zero.",
        "models": composed,
    }
    output = args.output.resolve()
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "models": len(composed),
                "songs": len(base_by_song),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
