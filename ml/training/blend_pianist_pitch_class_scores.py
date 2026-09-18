"""Blend frozen inference-only pitch-class score grids.

This utility never opens a reference.  It is intended for precision-first
ensembles where independently trained whole-song-excluded models must agree
before the existing note operator may add a chord member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_matrix(payload: dict[str, Any]) -> list[list[float]]:
    cells = list(payload.get("cells") or [])
    matrix: list[list[float]] = []
    for expected_index, cell in enumerate(cells):
        if int(cell.get("cellIndex", -1)) != expected_index:
            raise ValueError("Score cells must be contiguous and ordered")
        values = [float(value) for value in cell.get("probabilities") or []]
        if len(values) != 12 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
        ):
            raise ValueError("Each score cell needs twelve finite probabilities")
        matrix.append(values)
    if not matrix:
        raise ValueError("Score payload contains no cells")
    return matrix


def blend_values(values: list[float], mode: str) -> float:
    if mode == "minimum":
        return min(values)
    if mode == "arithmetic-mean":
        return sum(values) / len(values)
    if mode == "geometric-mean":
        product = 1.0
        for value in values:
            product *= max(0.0, value)
        return product ** (1.0 / len(values))
    raise ValueError(f"Unknown blend mode: {mode}")


def blend_payloads(payloads: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("At least two score payloads are required")
    song_ids = {str(payload.get("songId") or "") for payload in payloads}
    scopes = {str(payload.get("decisionScope") or "") for payload in payloads}
    if len(song_ids) != 1 or not next(iter(song_ids)):
        raise ValueError("All score payloads must describe the same song")
    if len(scopes) != 1:
        raise ValueError("All score payloads must use the same decision scope")
    if any(payload.get("referenceFieldsRead") is not False for payload in payloads):
        raise ValueError("Refusing to blend scores without a blind-inference proof")
    matrices = [score_matrix(payload) for payload in payloads]
    shapes = {(len(matrix), len(matrix[0])) for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError("Score grids do not have matching dimensions")
    cell_count = len(matrices[0])
    hashes = [str(payload.get("modelSha256") or "") for payload in payloads]
    canonical = json.dumps(
        {"mode": mode, "modelSha256": hashes},
        sort_keys=True,
        separators=(",", ":"),
    )
    training_ids = sorted(
        {
            str(song_id)
            for payload in payloads
            for song_id in payload.get("trainingSongIds") or []
            if str(song_id)
        }
    )
    return {
        "schema": "polymath-pianist-pitch-class-scores-v1",
        "evidenceBoundary": "blind-inference-ensemble-no-reference-fields",
        "songId": next(iter(song_ids)),
        "modelId": f"pitch-class-{mode}-ensemble",
        "modelSha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "trainingSongIds": training_ids,
        "featureContractSha256": "ensemble-not-a-feature-contract",
        "cells": [
            {
                "cellIndex": cell_index,
                "probabilities": [
                    round(
                        blend_values(
                            [matrix[cell_index][pitch_class] for matrix in matrices],
                            mode,
                        ),
                        10,
                    )
                    for pitch_class in range(12)
                ],
            }
            for cell_index in range(cell_count)
        ],
        "referenceFieldsRead": False,
        "decisionScope": next(iter(scopes)),
        "ensemble": {
            "mode": mode,
            "modelIds": [str(payload.get("modelId") or "") for payload in payloads],
            "modelSha256": hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", action="append", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("minimum", "arithmetic-mean", "geometric-mean"),
        default="minimum",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.scores]
    payload = blend_payloads([load_json(path) for path in paths], mode=args.mode)
    payload["inputs"] = [str(path) for path in paths]
    atomic_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "songId": payload["songId"],
                "cells": len(payload["cells"]),
                "mode": args.mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
