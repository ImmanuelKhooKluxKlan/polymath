"""Score one inference audit with frozen two-stage pitch-class models.

The output intentionally uses the probability-override contract consumed by
``apply_pianist_pitch_class_decoder``.  A selected pitch class receives 0.99
and an omitted class receives 0.01, so the existing note operator can
materialize a whole-gesture decision without learning about the rankers.

Reference fields are needed only because the historical audit contains the
cell boundary.  They never enter state generation or model scoring.  Cells
with no aligned target were excluded when the frozen rankers were fitted; the
scorer therefore preserves their incumbent candidate set instead of making an
unsupported drop decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_pitch_class_state_lattice import pitch_classes
from .fit_pitch_class_state_ranker import (
    FEATURE_NAMES,
    canonical_song_id,
    predict,
    prepare_song,
)
from .fit_pitch_class_state_topk_reranker import (
    META_FEATURE_NAMES,
    meta_feature_matrix,
    meta_predict,
    ranked_state_indices,
    select_meta_index,
)
from .fit_pitch_class_cardinality_reranker import (
    cardinality_feature_names,
    cardinality_features,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_models(
    base_model: dict[str, Any], meta_model: dict[str, Any]
) -> tuple[int, int, float]:
    if base_model.get("schema") != "polymath-pitch-class-state-ranker-v1":
        raise ValueError("Base model is not a serialized pitch-class state ranker")
    if list(base_model.get("featureNames") or []) != list(FEATURE_NAMES):
        raise ValueError("Base model feature contract does not match this scorer")
    if (
        meta_model.get("schema")
        != "polymath-pitch-class-state-topk-reranker-v1"
    ):
        raise ValueError("Meta model is not a serialized top-k state reranker")
    if list(meta_model.get("featureNames") or []) != list(META_FEATURE_NAMES):
        raise ValueError("Meta model feature contract does not match this scorer")
    family = str(meta_model.get("family") or "ridge-reward")
    if family not in {
        "ridge-reward",
        "pairwise-logistic",
        "hist-gradient-reward",
    }:
        raise ValueError(
            "Only JSON-serializable ridge, pairwise, and histogram meta models can be scored"
        )
    if family == "hist-gradient-reward" and (
        not isinstance(meta_model.get("trees"), list)
        or not meta_model.get("trees")
    ):
        raise ValueError("Histogram meta model is missing serialized runtime trees")
    maximum_states = max(4, min(256, int(base_model.get("maximumStates", 64))))
    top_k = max(2, min(maximum_states, int(meta_model.get("topK", 5))))
    margin = max(0.0, float(meta_model.get("minimumOverrideMargin", 0.0)))
    return maximum_states, top_k, margin


def cardinality_probabilities(
    cell: dict[str, Any], model: dict[str, Any], *, top_k: int
) -> np.ndarray:
    """Run a JSON-serialized multinomial cardinality model safely."""

    if model.get("schema") != "polymath-pitch-class-cardinality-reranker-v1":
        raise ValueError("Cardinality model has an unsupported schema")
    if model.get("family") != "multinomial-cardinality":
        raise ValueError("Only multinomial cardinality models are supported")
    if int(model.get("topK", 0)) != top_k:
        raise ValueError("Cardinality model top-k does not match the meta model")
    if list(model.get("featureNames") or []) != list(
        cardinality_feature_names(top_k)
    ):
        raise ValueError("Cardinality model feature contract does not match")
    values = cardinality_features(cell, top_k=top_k)
    means = np.asarray(model.get("means") or [], dtype=float)
    scales = np.asarray(model.get("scales") or [], dtype=float)
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    intercepts = np.asarray(model.get("intercepts") or [], dtype=float)
    classes = np.asarray(model.get("classes") or [], dtype=int)
    if (
        values.shape != means.shape
        or values.shape != scales.shape
        or coefficients.ndim != 2
        or coefficients.shape[1] != len(values)
        or coefficients.shape[0] != len(intercepts)
        or len(classes) != len(intercepts)
        or np.any(np.abs(scales) < 1e-9)
    ):
        raise ValueError("Cardinality model contains invalid coefficient arrays")
    logits = coefficients @ ((values - means) / scales) + intercepts
    logits -= float(np.max(logits))
    learned = np.exp(logits)
    learned /= max(1e-12, float(np.sum(learned)))
    output = np.full(6, 1e-6, dtype=float)
    for target_class, probability in zip(classes, learned, strict=True):
        output[min(6, max(1, int(target_class))) - 1] = float(probability)
    output /= float(np.sum(output))
    return output


def score_song(
    raw_song: dict[str, Any],
    base_model: dict[str, Any],
    meta_model: dict[str, Any],
    cardinality_model: dict[str, Any] | None = None,
    *,
    cardinality_strength: float = 0.0,
    minimum_override_margin: float | None = None,
    minimum_candidate_override_margin: float | None = None,
) -> dict[str, Any]:
    maximum_states, top_k, model_override_margin = validate_models(
        base_model, meta_model
    )
    active_override_margin = (
        model_override_margin
        if minimum_override_margin is None
        else max(0.0, float(minimum_override_margin))
    )
    active_candidate_override_margin = (
        max(0.0, float(meta_model.get("minimumCandidateOverrideMargin", 0.0)))
        if minimum_candidate_override_margin is None
        else max(0.0, float(minimum_candidate_override_margin))
    )
    copied = dict(raw_song)
    copied["groupId"] = canonical_song_id(raw_song.get("id"))
    copied["weight"] = 1.0
    prepared = prepare_song(copied, maximum_states=maximum_states)
    prepared_by_index = {
        int(cell["index"]): cell for cell in prepared.get("cells") or []
    }

    score_rows: list[dict[str, Any]] = []
    changed = base_overrides = meta_overrides = 0
    for index, raw_cell in enumerate(raw_song.get("cells") or []):
        candidate = pitch_classes(raw_cell.get("candidatePitchClasses") or [])
        cell = prepared_by_index.get(index)
        if cell is None:
            selected = candidate
            base_selected = candidate
            base_rank = meta_rank = 1
            meta_margin = 0.0
            decision = "preserve-unanchored-cell"
        else:
            base_scores = predict(cell["features"], base_model)
            maximum_cardinality_delta = meta_model.get("maximumCardinalityDelta")
            include_candidate_state = bool(
                meta_model.get("includeCandidateState", False)
            )
            indices = ranked_state_indices(
                cell,
                base_scores,
                top_k=top_k,
                maximum_cardinality_delta=(
                    None
                    if maximum_cardinality_delta is None
                    else max(0, int(maximum_cardinality_delta))
                ),
                include_candidate_state=include_candidate_state,
            )
            base_index = indices[0]
            base_selected = cell["states"][base_index]
            meta_matrix = meta_feature_matrix(cell["features"], base_scores, indices)
            meta_scores = meta_predict(meta_matrix, meta_model)
            if cardinality_model is not None and cardinality_strength > 0.0:
                meta_cell = {
                    "features": meta_matrix,
                    "baseScores": np.asarray(
                        [float(base_scores[state_index]) for state_index in indices],
                        dtype=float,
                    ),
                    "states": [cell["states"][state_index] for state_index in indices],
                }
                size_probabilities = cardinality_probabilities(
                    meta_cell, cardinality_model, top_k=top_k
                )
                meta_scores = meta_scores + cardinality_strength * np.asarray(
                    [
                        np.log(
                            max(
                                1e-6,
                                float(
                                    size_probabilities[
                                        min(6, max(1, len(state))) - 1
                                    ]
                                ),
                            )
                        )
                        for state in meta_cell["states"]
                    ],
                    dtype=float,
                )
            incumbent_offset = (
                indices.index(
                    next(
                        state_index
                        for state_index, state in enumerate(cell["states"])
                        if state == cell["candidate"]
                    )
                )
                if include_candidate_state
                else 0
            )
            selected_offset = select_meta_index(
                meta_scores,
                active_override_margin,
                incumbent_index=incumbent_offset,
                minimum_candidate_override_margin=active_candidate_override_margin,
            )
            selected_index = indices[selected_offset]
            selected = cell["states"][selected_index]
            base_rank = 1
            meta_rank = selected_offset + 1
            runner_up = max(
                (
                    float(score)
                    for offset, score in enumerate(meta_scores)
                    if offset != selected_offset
                ),
                default=float(meta_scores[selected_offset]),
            )
            meta_margin = float(meta_scores[selected_offset]) - runner_up
            decision = "ranked-anchored-cell"
            base_overrides += int(base_selected != candidate)
            meta_overrides += int(selected != base_selected)

        changed += int(selected != candidate)
        probabilities = [
            0.99 if pitch_class in selected else 0.01
            for pitch_class in range(12)
        ]
        score_rows.append(
            {
                "cellIndex": index,
                "sourceTime": round(float(raw_cell.get("sourceTime", 0.0)), 6),
                "probabilities": probabilities,
                "candidatePitchClasses": sorted(candidate),
                "basePitchClasses": sorted(base_selected),
                "selectedPitchClasses": sorted(selected),
                "baseRank": base_rank,
                "selectedMetaRank": meta_rank,
                "metaWinningMargin": round(meta_margin, 9),
                "decision": decision,
            }
        )

    return {
        "schema": "polymath-pitch-class-state-score-override-v1",
        "songId": str(raw_song.get("id") or ""),
        "modelId": "frozen-state-topk-reranker",
        "modelSha256": str(meta_model.get("sha256") or ""),
        "settings": {
            "maximumStates": maximum_states,
            "topK": top_k,
            "maximumCardinalityDelta": meta_model.get(
                "maximumCardinalityDelta"
            ),
            "minimumOverrideMargin": active_override_margin,
            "includeCandidateState": bool(
                meta_model.get("includeCandidateState", False)
            ),
            "minimumCandidateOverrideMargin": active_candidate_override_margin,
            "cardinalityStrength": max(0.0, float(cardinality_strength)),
            "cardinalityModelEnabled": cardinality_model is not None,
            "unanchoredCellPolicy": "preserve-candidate",
        },
        "diagnostics": {
            "cells": len(score_rows),
            "anchoredCells": len(prepared_by_index),
            "unanchoredCells": len(score_rows) - len(prepared_by_index),
            "baseOverridesFromCandidate": base_overrides,
            "metaOverridesFromBase": meta_overrides,
            "finalOverridesFromCandidate": changed,
        },
        "cells": score_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--meta-model", required=True)
    parser.add_argument("--cardinality-model")
    parser.add_argument("--cardinality-strength", type=float, default=0.0)
    parser.add_argument("--minimum-override-margin", type=float)
    parser.add_argument("--minimum-candidate-override-margin", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audit_path = Path(args.audit).resolve()
    base_path = Path(args.base_model).resolve()
    meta_path = Path(args.meta_model).resolve()
    audit = load_json(audit_path)
    matches = [
        song
        for song in audit.get("songs") or []
        if str(song.get("id") or "") == str(args.song)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one audit song named {args.song!r}")
    cardinality_path = (
        Path(args.cardinality_model).resolve() if args.cardinality_model else None
    )
    payload = score_song(
        matches[0],
        load_json(base_path),
        load_json(meta_path),
        load_json(cardinality_path) if cardinality_path else None,
        cardinality_strength=max(0.0, float(args.cardinality_strength)),
        minimum_override_margin=args.minimum_override_margin,
        minimum_candidate_override_margin=(
            args.minimum_candidate_override_margin
        ),
    )
    payload["inputs"] = {
        "audit": str(audit_path),
        "auditSha256": sha256_file(audit_path),
        "baseModel": str(base_path),
        "baseModelSha256": sha256_file(base_path),
        "metaModel": str(meta_path),
        "metaModelSha256": sha256_file(meta_path),
        "cardinalityModel": str(cardinality_path) if cardinality_path else None,
        "cardinalityModelSha256": (
            sha256_file(cardinality_path) if cardinality_path else None
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
