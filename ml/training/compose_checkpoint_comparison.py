"""Compose two immutable single-checkpoint evaluations without decoding again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.rescore_song_timelines import TimelineScoreError, sha256_file


def clip_signature(evaluation: dict[str, Any]) -> list[tuple[str, str, float]]:
    decoded = (evaluation.get("metrics") or {}).get("decodedClips")
    if not isinstance(decoded, list):
        raise TimelineScoreError("Single-checkpoint evaluation is missing decodedClips")
    signature: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for item in decoded:
        if not isinstance(item, dict) or not str(item.get("clipId") or ""):
            raise TimelineScoreError("Single-checkpoint evaluation contains an invalid clip")
        clip_id = str(item["clipId"])
        if clip_id in seen:
            raise TimelineScoreError(f"Single-checkpoint evaluation repeats clip {clip_id}")
        seen.add(clip_id)
        signature.append((
            clip_id,
            str(item.get("songId") or "unknown"),
            float(item.get("sourceStart") or 0),
        ))
    return signature


def compose_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_source: str | None = None,
    candidate_source: str | None = None,
) -> dict[str, Any]:
    for label, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("schema") != "polymath-checkpoint-evaluation-v1":
            raise TimelineScoreError(f"{label} is not a single-checkpoint evaluation")
    shared_fields = ("validationManifest", "clips", "instrumentConstraint")
    for field in shared_fields:
        if baseline.get(field) != candidate.get(field):
            raise TimelineScoreError(f"Evaluation mismatch: {field}")
    if clip_signature(baseline) != clip_signature(candidate):
        raise TimelineScoreError("Evaluation decoded-clip identities or timings differ")
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    deltas = {
        key: round(
            float(candidate_metrics[key]["microF1"])
            - float(baseline_metrics[key]["microF1"]),
            6,
        )
        for key in ("50ms", "100ms", "250ms")
    }
    return {
        "schema": "polymath-checkpoint-comparison-v1",
        "baseCheckpoint": baseline.get("checkpoint"),
        "candidateCheckpoint": candidate.get("checkpoint"),
        "validationManifest": baseline.get("validationManifest"),
        "clips": baseline.get("clips"),
        "instrumentConstraint": baseline.get("instrumentConstraint"),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidateMinusBaselineMicroF1": deltas,
        "composition": {
            "schema": "polymath-saved-evaluation-composition-v1",
            "baselineSource": baseline_source,
            "candidateSource": candidate_source,
            "note": "No model was decoded during composition; both frozen raw evaluations were reused.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline_evaluation.resolve()
    candidate_path = args.candidate_evaluation.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = compose_comparison(
        baseline,
        candidate,
        baseline_source=str(baseline_path),
        candidate_source=str(candidate_path),
    )
    result["composition"]["baselineSourceSha256"] = sha256_file(baseline_path)
    result["composition"]["candidateSourceSha256"] = sha256_file(candidate_path)
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "candidateMinusBaselineMicroF1": result["candidateMinusBaselineMicroF1"],
    }, indent=2))


if __name__ == "__main__":
    main()
