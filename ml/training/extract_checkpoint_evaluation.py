"""Extract one auditable side of a frozen paired checkpoint evaluation.

Older Phase 94 comparisons contain complete raw decoded clips for both models,
but some predate explicit ``baseCheckpoint`` and ``candidateCheckpoint``
fields.  This utility converts one side into the current single-checkpoint
schema without decoding again.  When the source omitted checkpoint identity,
the caller must provide it explicitly; the tool never guesses.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ml.training.compose_checkpoint_comparison import clip_signature
from ml.training.rescore_song_timelines import TimelineScoreError, sha256_file


SIDE_FIELDS = {
    "baseline": ("baseline", "baseCheckpoint"),
    "candidate": ("candidate", "candidateCheckpoint"),
}


def extract_evaluation(
    comparison: dict[str, Any],
    side: str,
    *,
    checkpoint: str | None = None,
    comparison_source: str | None = None,
    comparison_sha256: str | None = None,
) -> dict[str, Any]:
    """Return one comparison side as a single-checkpoint evaluation."""

    if comparison.get("schema") != "polymath-checkpoint-comparison-v1":
        raise TimelineScoreError("Input is not a paired checkpoint comparison")
    if side not in SIDE_FIELDS:
        raise TimelineScoreError("Side must be baseline or candidate")
    metrics_field, checkpoint_field = SIDE_FIELDS[side]
    metrics = comparison.get(metrics_field)
    if not isinstance(metrics, dict):
        raise TimelineScoreError(f"Comparison is missing {metrics_field} metrics")

    resolved_checkpoint = str(
        checkpoint or comparison.get(checkpoint_field) or ""
    ).strip()
    if not resolved_checkpoint:
        raise TimelineScoreError(
            f"Comparison omits {checkpoint_field}; provide an explicit checkpoint identity"
        )
    try:
        clip_count = int(comparison["clips"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimelineScoreError("Comparison has an invalid clip count") from exc

    result = {
        "schema": "polymath-checkpoint-evaluation-v1",
        "checkpoint": resolved_checkpoint,
        "validationManifest": comparison.get("validationManifest"),
        "clips": clip_count,
        "instrumentConstraint": comparison.get("instrumentConstraint") or [],
        "metrics": copy.deepcopy(metrics),
        "extraction": {
            "schema": "polymath-paired-evaluation-extraction-v1",
            "side": side,
            "source": comparison_source,
            "sourceSha256": comparison_sha256,
            "note": "No model was decoded; raw frozen predictions were reused.",
        },
    }
    signature = clip_signature(result)
    if len(signature) != clip_count:
        raise TimelineScoreError(
            f"Comparison declares {clip_count} clips but {side} contains {len(signature)}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--side", choices=sorted(SIDE_FIELDS), required=True)
    parser.add_argument(
        "--checkpoint",
        help=(
            "Exact checkpoint identity. Required when the older comparison "
            "does not embed the corresponding checkpoint field."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.comparison.resolve()
    comparison = json.loads(source.read_text(encoding="utf-8"))
    result = extract_evaluation(
        comparison,
        args.side,
        checkpoint=args.checkpoint,
        comparison_source=str(source),
        comparison_sha256=sha256_file(source),
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "side": args.side,
        "checkpoint": result["checkpoint"],
        "clips": result["clips"],
        "sourceSha256": result["extraction"]["sourceSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
