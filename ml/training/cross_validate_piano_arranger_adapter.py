"""Run leave-one-song-out validation for the learned piano arranger."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "ml" / "training" / "train_piano_arranger_adapter.py"
ARRANGER = REPO_ROOT / "server" / "piano_arranger.py"
EVALUATOR = REPO_ROOT / "ml" / "training" / "evaluate_piano_arranger.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(*arguments: str | Path) -> None:
    subprocess.run([str(argument) for argument in arguments], cwd=REPO_ROOT, check=True)


def weighted_average(values: list[tuple[float, float]]) -> float:
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / max(total, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cleaned-source-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--density-multiplier",
        type=float,
        help="Validation-only override for decoder.preCleanupDensityMultiplier.",
    )
    parser.add_argument(
        "--disable-sparse-expansion",
        action="store_true",
        help="Validation-only override that disables generated octave doublings.",
    )
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest).resolve())
    pairs = manifest.get("pairs") or []
    if len(pairs) < 3:
        raise ValueError("Leave-one-song-out validation needs at least three songs.")
    cleaned_source_dir = Path(args.cleaned_source_dir).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    folds: dict[str, Any] = {}

    for held_out in pairs:
        held_out_id = str(held_out["id"])
        fold_root = output_dir / held_out_id
        training_manifest = {
            **manifest,
            "profileId": f"pianella-loso-without-{held_out_id}",
            "pairs": [pair for pair in pairs if pair["id"] != held_out_id],
        }
        evaluation_manifest = {
            **manifest,
            "profileId": f"pianella-loso-held-out-{held_out_id}",
            "pairs": [held_out],
        }
        training_manifest_path = fold_root / "training-manifest.json"
        evaluation_manifest_path = fold_root / "evaluation-manifest.json"
        profile_path = fold_root / "profile.json"
        training_report_path = fold_root / "training-report.json"
        candidate_dir = fold_root / "candidate"
        evaluation_path = fold_root / "evaluation.json"
        write_json(training_manifest_path, training_manifest)
        write_json(evaluation_manifest_path, evaluation_manifest)
        run(
            sys.executable,
            TRAINER,
            "--manifest",
            training_manifest_path,
            "--output",
            profile_path,
            "--report",
            training_report_path,
        )
        profile = load_json(profile_path)
        decoder = profile.setdefault("decoder", {})
        if args.density_multiplier is not None:
            decoder["preCleanupDensityMultiplier"] = max(
                0.5, min(2.0, float(args.density_multiplier))
            )
        if args.disable_sparse_expansion:
            decoder["expandSparseHarmony"] = False
        write_json(profile_path, profile)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        run(
            sys.executable,
            ARRANGER,
            "--input",
            cleaned_source_dir / f"{held_out_id}.json",
            "--output",
            candidate_dir / f"{held_out_id}.json",
            "--mode",
            "full",
            "--profile",
            profile_path,
        )
        run(
            sys.executable,
            EVALUATOR,
            "--manifest",
            evaluation_manifest_path,
            "--baseline-dir",
            baseline_dir,
            "--candidate-dir",
            candidate_dir,
            "--output",
            evaluation_path,
        )
        evaluation = load_json(evaluation_path)
        song = evaluation["songs"][held_out_id]
        folds[held_out_id] = {
            "trainingSongIds": [pair["id"] for pair in training_manifest["pairs"]],
            "heldOutWeight": float(held_out.get("weight", 1.0)),
            "profile": str(profile_path),
            "baseline": song["baseline"],
            "candidate": song["candidate"],
            "candidateMinusBaseline": evaluation["candidateMinusBaseline"],
            "foldDecision": evaluation["decision"],
        }

    aggregate: dict[str, dict[str, float]] = {"baseline": {}, "candidate": {}}
    metric_paths = {
        "exactF1_50ms": ("exactPitchOnset50ms", "f1"),
        "exactF1_100ms": ("exactPitchOnset100ms", "f1"),
        "exactF1_250ms": ("exactPitchOnset250ms", "f1"),
        "pitchClassF1_250ms": ("pitchClassOnset250ms", "f1"),
    }
    for side in ("baseline", "candidate"):
        for name, path in metric_paths.items():
            aggregate[side][name] = round(
                weighted_average(
                    [
                        (float(fold[side][path[0]][path[1]]), float(fold["heldOutWeight"]))
                        for fold in folds.values()
                    ]
                ),
                6,
            )
        aggregate[side]["durationMedianAbsoluteErrorSeconds"] = round(
            weighted_average(
                [
                    (
                        float(fold[side]["duration"]["medianAbsoluteErrorSeconds"] or 0.0),
                        float(fold["heldOutWeight"]),
                    )
                    for fold in folds.values()
                ]
            ),
            6,
        )
        aggregate[side]["severeCutoffs"] = round(
            weighted_average(
                [
                    (float(fold[side]["duration"]["severeCutoffs"]), float(fold["heldOutWeight"]))
                    for fold in folds.values()
                ]
            ),
            6,
        )

    deltas = {
        name: round(aggregate["candidate"][name] - aggregate["baseline"][name], 6)
        for name in aggregate["baseline"]
    }
    baseline = aggregate["baseline"]
    candidate = aggregate["candidate"]
    promotion_gates = {
        "unseen_exact_f1_100ms_improves": (
            candidate["exactF1_100ms"] >= baseline["exactF1_100ms"] + 0.005
        ),
        "unseen_exact_f1_250ms_does_not_regress": (
            candidate["exactF1_250ms"] >= baseline["exactF1_250ms"] - 0.002
        ),
        "unseen_pitch_class_f1_250ms_does_not_regress": (
            candidate["pitchClassF1_250ms"] >= baseline["pitchClassF1_250ms"] - 0.002
        ),
        "unseen_duration_error_not_over_10_percent_worse": (
            candidate["durationMedianAbsoluteErrorSeconds"]
            <= baseline["durationMedianAbsoluteErrorSeconds"] * 1.10
        ),
        "unseen_severe_cutoffs_not_over_5_percent_worse": (
            candidate["severeCutoffs"] <= baseline["severeCutoffs"] * 1.05 + 1.0
        ),
    }
    decision = "PROMOTE" if all(promotion_gates.values()) else "REJECT"
    summary = {
        "schema": "polymath-piano-arranger-loso-v1",
        "policy": "each candidate is trained without the song used for that fold's evaluation",
        "folds": folds,
        "weightedAggregate": aggregate,
        "candidateMinusBaseline": deltas,
        "promotionGates": promotion_gates,
        "decision": decision,
        "decoderValidationOverrides": {
            "preCleanupDensityMultiplier": args.density_multiplier,
            "expandSparseHarmony": False if args.disable_sparse_expansion else None,
        },
        "warning": (
            "Only four songs exist and two have weak source-to-target alignment. "
            "This is a stress test, not production-scale validation."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "weightedAggregate": aggregate,
        "candidateMinusBaseline": deltas,
        "promotionGates": promotion_gates,
        "decision": decision,
        "decoderValidationOverrides": summary["decoderValidationOverrides"],
    }, indent=2))


if __name__ == "__main__":
    main()
