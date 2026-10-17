"""Create an exact-order manifest control by reweighting one clip variant.

The output preserves every record, clip ID, audio path, and row position. Only
the selected examples' positive loss weight changes. This lets two deterministic
training runs differ by one supervision signal while keeping shuffle order and
gradient-accumulation boundaries identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reweight_variant(
    rows: list[dict[str, Any]],
    *,
    clip_prefix: str,
    replacement_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not clip_prefix:
        raise ValueError("clip_prefix is required")
    if not math.isfinite(replacement_weight) or not 0 < replacement_weight <= 1:
        raise ValueError("replacement_weight must be within (0, 1]")

    output = copy.deepcopy(rows)
    matching_indices = [
        index
        for index, row in enumerate(output)
        if str(row.get("clipId") or "").startswith(clip_prefix)
    ]
    if not matching_indices:
        raise ValueError(f"No clip IDs matched prefix: {clip_prefix}")

    original_weights: set[float] = set()
    for index in matching_indices:
        record = output[index]
        original_weight = float(record.get("exampleWeight", 1.0))
        if not math.isfinite(original_weight) or not 0 < original_weight <= 1:
            raise ValueError(f"{record.get('clipId')}: invalid source exampleWeight")
        original_weights.add(original_weight)
        record["exampleWeight"] = replacement_weight
        record["weightAblation"] = {
            "schema": "polymath-variant-weight-ablation-v1",
            "clipPrefix": clip_prefix,
            "originalWeight": original_weight,
            "replacementWeight": replacement_weight,
            "purpose": "deterministic supervision-signal control",
        }

    input_ids = [str(row.get("clipId") or "") for row in rows]
    output_ids = [str(row.get("clipId") or "") for row in output]
    if input_ids != output_ids:
        raise AssertionError("Ablation changed record identity or order")
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Manifest contains duplicate clip IDs")

    summary = {
        "schema": "polymath-variant-weight-ablation-receipt-v1",
        "clipPrefix": clip_prefix,
        "replacementWeight": replacement_weight,
        "matchingClips": len(matching_indices),
        "matchingRowIndicesZeroBased": matching_indices,
        "originalWeights": sorted(original_weights),
        "weightMass": {
            "selectedBefore": sum(float(rows[index].get("exampleWeight", 1.0)) for index in matching_indices),
            "selectedAfter": replacement_weight * len(matching_indices),
            "manifestBefore": sum(float(row.get("exampleWeight", 1.0)) for row in rows),
            "manifestAfter": sum(float(row.get("exampleWeight", 1.0)) for row in output),
        },
        "totalClips": len(output),
        "recordOrderPreserved": True,
        "interpretation": (
            "Only selected example weights change; row order, audio, labels, and batch "
            "boundaries are preserved. Near-zero weight suppresses the supervision "
            "signal, while a larger weight increases its loss contribution. Weight "
            "mass is an exposure proxy, not a measurement of actual gradient norm."
        ),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--clip-prefix", required=True)
    parser.add_argument("--replacement-weight", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    rows = read_jsonl(train_path)
    output, summary = reweight_variant(
        rows,
        clip_prefix=args.clip_prefix.strip(),
        replacement_weight=args.replacement_weight,
    )

    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    train_output = destination / "prepared-train.jsonl"
    validation_output = destination / "prepared-validation.jsonl"
    write_jsonl(train_output, output)
    shutil.copyfile(validation_path, validation_output)
    summary["sources"] = {
        "trainManifest": {"path": str(train_path), "sha256": sha256(train_path)},
        "validationManifest": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
        },
    }
    summary["outputs"] = {
        "trainManifest": {"path": str(train_output), "sha256": sha256(train_output)},
        "validationManifest": {
            "path": str(validation_output),
            "sha256": sha256(validation_output),
        },
    }
    receipt = destination / "variant-weight-ablation-receipt.json"
    receipt.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
