"""Evaluate a frozen broken-chord gate on complete held-out songs.

The fitter deliberately stores only inference-safe feature weights.  This
utility applies those weights to a separate texture audit and reports both the
frozen threshold and fixed top-share operating points.  The latter diagnoses
probability-calibration drift without pretending that the desired expansion
rate is available during inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .fit_pianist_texture_decoder import (
    FEATURE_NAMES,
    classification_metrics,
    feature_vector,
    sigmoid,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def probabilities_for_rows(
    rows: list[dict[str, Any]], model: dict[str, Any]
) -> np.ndarray:
    names = list(model.get("featureNames") or [])
    if names != list(FEATURE_NAMES):
        raise ValueError("Texture model feature contract does not match this evaluator")
    matrix = np.asarray([feature_vector(row) for row in rows], dtype=float)
    weights = np.asarray(model.get("weights") or [], dtype=float)
    means = np.asarray(model.get("means") or [], dtype=float)
    scales = np.asarray(model.get("scales") or [], dtype=float)
    if not (
        matrix.shape[1] == len(weights) == len(means) == len(scales)
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(np.abs(scales) > 1e-9)
    ):
        raise ValueError("Texture model contains invalid coefficient arrays")
    return sigmoid(((matrix - means) / scales) @ weights)


def top_share_metrics(
    labels: np.ndarray, probabilities: np.ndarray, share: float
) -> dict[str, Any]:
    count = min(len(labels), max(0, int(round(len(labels) * share))))
    if not count:
        threshold = 1.0 + 1e-9
    else:
        threshold = float(np.partition(probabilities, len(probabilities) - count)[-count])
    return {
        "requestedShare": round(float(share), 6),
        "threshold": round(threshold, 9),
        **classification_metrics(labels, probabilities, threshold),
    }


def evaluate(
    audit: dict[str, Any], model: dict[str, Any], song_ids: set[str] | None = None
) -> dict[str, Any]:
    songs: list[dict[str, Any]] = []
    for song in audit.get("songs") or []:
        song_id = str(song.get("id") or "")
        if song_ids and song_id not in song_ids:
            continue
        rows = list(song.get("cells") or [])
        if not rows:
            continue
        labels = np.asarray(
            [float(int(row.get("targetGestureCount", 0)) >= 2) for row in rows],
            dtype=float,
        )
        probabilities = probabilities_for_rows(rows, model)
        threshold = float(model.get("threshold", 0.5))
        songs.append(
            {
                "id": song_id,
                "frozenThreshold": classification_metrics(
                    labels, probabilities, threshold
                ),
                "topShareDiagnostics": [
                    top_share_metrics(labels, probabilities, share)
                    for share in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35)
                ],
                "rankedCells": [
                    {
                        "cellIndex": int(row.get("cellIndex", index)),
                        "time": float(row.get("sourceTime", 0.0)),
                        "probability": round(float(probabilities[index]), 9),
                        "actualUnfold": bool(labels[index]),
                        "actualGestureCount": int(row.get("targetGestureCount", 0)),
                        "candidateChordSize": int(row.get("candidateChordSize", 0)),
                        "nextGapSeconds": float(row.get("nextGapSeconds", 0.0)),
                    }
                    for index, row in sorted(
                        enumerate(rows),
                        key=lambda item: (-float(probabilities[item[0]]), item[0]),
                    )
                ],
            }
        )
    if not songs:
        raise ValueError("The audit has no requested songs with texture cells")
    return {
        "schema": "polymath-pianist-texture-heldout-evaluation-v1",
        "modelId": model.get("id"),
        "modelSha256": model.get("modelSha256"),
        "evidenceBoundary": (
            "Complete-song diagnostic. Top-share rows are oracle-free operating "
            "point diagnostics, not permission to infer a target song's true rate."
        ),
        "songs": songs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--song", action="append", default=[])
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    model_path = Path(args.model).resolve()
    report = evaluate(
        load_json(audit_path),
        load_json(model_path),
        set(args.song) or None,
    )
    report["audit"] = str(audit_path)
    report["model"] = str(model_path)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "songs": [
                    {"id": song["id"], **song["frozenThreshold"]}
                    for song in report["songs"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
