"""Score one unseen inference audit with a frozen addition-ranker model.

The scorer never reads reference notes.  It converts candidate/source-only
audit cells into the same 237-feature contract used by the phase-55 model,
then writes a deterministic probability interchange file for the existing
pitch-class note operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .pianist_pitch_class_context import (
    FEATURE_NAMES as CONTEXT_FEATURE_NAMES,
    context_examples_from_song,
)
from .pianist_pitch_class_evidence_context import (
    FEATURE_NAMES as EVIDENCE_FEATURE_NAMES,
    evidence_examples_from_song,
)
from .pianist_pitch_class_decoder_origin_context import (
    FEATURE_NAMES as ORIGIN_FEATURE_NAMES,
    decoder_origin_examples_from_song,
)
from .pianist_pitch_class_sequence_context import (
    FEATURE_NAMES as SEQUENCE_FEATURE_NAMES,
    sequence_examples_from_song,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_payload(
    song_id: str,
    probabilities: np.ndarray,
    *,
    model_id: str,
    model_sha256: str,
    training_song_ids: list[str],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    if probabilities.ndim != 2 or probabilities.shape[1] != 12:
        raise ValueError("Expected one 12-class probability row per cell")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain a non-finite value")
    return {
        "schema": "polymath-pianist-pitch-class-scores-v1",
        "evidenceBoundary": (
            "unseen-transfer-inference-candidate-and-source-features-only"
        ),
        "songId": song_id,
        "modelId": model_id,
        "modelSha256": model_sha256,
        "trainingSongIds": training_song_ids,
        "featureContractSha256": hashlib.sha256(
            "\n".join(feature_names).encode("utf-8")
        ).hexdigest(),
        "cells": [
            {
                "cellIndex": index,
                "probabilities": [round(float(value), 10) for value in row],
            }
            for index, row in enumerate(probabilities)
        ],
    }


def inference_arrays(
    song: dict[str, Any], *, feature_contract: str
) -> dict[str, np.ndarray]:
    if feature_contract == "context237":
        examples = context_examples_from_song(song)
    elif feature_contract == "evidence259":
        examples = evidence_examples_from_song(song)
    elif feature_contract == "origin274":
        examples = decoder_origin_examples_from_song(song)
    elif feature_contract == f"sequence{len(SEQUENCE_FEATURE_NAMES)}":
        examples = sequence_examples_from_song(song)
    else:
        raise ValueError(f"Unknown feature contract: {feature_contract}")
    if not examples or len(examples) % 12:
        raise ValueError(f"Song {song.get('id')!r} has an invalid example grid")
    cells = len(examples) // 12
    return {
        "features": np.asarray(
            [row["features"] for row in examples], dtype=np.float32
        ),
        "candidate": np.asarray(
            [row["candidate"] for row in examples], dtype=np.int8
        ).reshape(cells, 12),
        "source": np.asarray(
            [row["source"] for row in examples], dtype=np.int8
        ).reshape(cells, 12),
        "generated": np.asarray(
            [bool(row.get("decoderGenerated", False)) for row in examples],
            dtype=bool,
        ).reshape(cells, 12),
    }


def inference_probability_grid(
    arrays: dict[str, np.ndarray], model: Any, *, decision_scope: str = "additions"
) -> np.ndarray:
    candidate = arrays["candidate"].reshape(-1)
    source = arrays["source"].reshape(-1)
    probabilities = np.zeros(candidate.shape[0], dtype=float)
    if decision_scope == "additions":
        eligible = (candidate <= 0) & (source > 0)
        probabilities[candidate > 0] = 1.0
    elif decision_scope == "membership":
        eligible = (candidate > 0) | (source > 0)
    elif decision_scope == "decoder-generated-removals-only":
        generated = arrays["generated"].reshape(-1)
        eligible = generated
        probabilities[(candidate > 0) & ~generated] = 1.0
    else:
        raise ValueError(f"Unknown decision scope: {decision_scope}")
    if np.any(eligible):
        probabilities[eligible] = model.predict_proba(
            arrays["features"][eligible]
        )[:, 1]
    return probabilities.reshape(arrays["candidate"].shape)


def select_song(audit: dict[str, Any], song_id: str) -> dict[str, Any]:
    matches = [
        song
        for song in audit.get("songs") or []
        if str(song.get("id") or "") == song_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one audit song named {song_id!r}")
    return matches[0]


def validate_inference_song(song: dict[str, Any]) -> None:
    cells = list(song.get("cells") or [])
    if not cells:
        raise ValueError("Inference audit contains no cells")
    leaked = [
        index
        for index, cell in enumerate(cells)
        if cell.get("referencePitchClasses")
    ]
    if leaked:
        raise ValueError(
            "Inference audit contains reference-derived pitch classes; "
            "refusing a non-blind score"
        )


def score_song(
    song: dict[str, Any],
    model: Any,
    *,
    model_id: str,
    model_sha256: str,
    training_song_ids: list[str],
    feature_contract: str = "context237",
    decision_scope: str = "additions",
) -> dict[str, Any]:
    validate_inference_song(song)
    feature_names = (
        CONTEXT_FEATURE_NAMES
        if feature_contract == "context237"
        else EVIDENCE_FEATURE_NAMES
        if feature_contract == "evidence259"
        else ORIGIN_FEATURE_NAMES
        if feature_contract == "origin274"
        else SEQUENCE_FEATURE_NAMES
        if feature_contract == f"sequence{len(SEQUENCE_FEATURE_NAMES)}"
        else ()
    )
    if not feature_names:
        raise ValueError(f"Unknown feature contract: {feature_contract}")
    expected_features = len(feature_names)
    actual_features = int(getattr(model, "n_features_in_", -1))
    if actual_features != expected_features:
        raise ValueError(
            f"Model expects {actual_features} features; context contract has "
            f"{expected_features}"
        )
    arrays = inference_arrays(song, feature_contract=feature_contract)
    probabilities = inference_probability_grid(
        arrays, model, decision_scope=decision_scope
    )
    payload = score_payload(
        str(song.get("id") or ""),
        probabilities,
        model_id=model_id,
        model_sha256=model_sha256,
        training_song_ids=training_song_ids,
        feature_names=feature_names,
    )
    payload["referenceFieldsRead"] = False
    payload["decisionScope"] = decision_scope
    return payload


def training_ids(
    report: dict[str, Any],
    model_sha256: str,
    *,
    held_out_song: str | None = None,
) -> list[str]:
    if held_out_song:
        matches = [
            row
            for row in report.get("folds") or []
            if str(row.get("heldOutSong") or "") == held_out_song
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one training fold for {held_out_song!r}"
            )
        fold = matches[0]
        recorded_hash = str((fold.get("model") or {}).get("sha256") or "")
        if not recorded_hash or recorded_hash != model_sha256:
            raise ValueError("Training fold and model SHA-256 do not match")
        values = fold.get("trainingSongIds") or []
        parsed = [str(value) for value in values if str(value)]
        if not parsed or held_out_song in parsed:
            raise ValueError("Training fold does not prove whole-song exclusion")
        return parsed
    final = report.get("finalModel") or {}
    recorded_hash = str(final.get("sha256") or "")
    if recorded_hash and recorded_hash != model_sha256:
        raise ValueError("Training report and model SHA-256 do not match")
    values = final.get("trainingSongIds") or report.get("songs") or []
    parsed = [str(value) for value in values if str(value)]
    if not parsed:
        raise ValueError("Training report contains no training song ids")
    return parsed


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-id", default="phase55-context-addition-ranker-opened-v001"
    )
    parser.add_argument(
        "--feature-contract",
        choices=(
            "context237",
            "evidence259",
            "origin274",
            f"sequence{len(SEQUENCE_FEATURE_NAMES)}",
        ),
        default="context237",
    )
    parser.add_argument(
        "--decision-scope",
        choices=(
            "additions",
            "membership",
            "decoder-generated-removals-only",
        ),
        default="additions",
    )
    parser.add_argument(
        "--held-out-song",
        default="",
        help=(
            "Use and verify the report's whole-song-excluded fold for this song "
            "instead of the final opened-development model."
        ),
    )
    args = parser.parse_args()

    audit_path = args.audit.resolve()
    model_path = args.model.resolve()
    report_path = args.training_report.resolve()
    model_hash = sha256_file(model_path)
    report = load_json(report_path)
    model = joblib.load(model_path)
    payload = score_song(
        select_song(load_json(audit_path), str(args.song)),
        model,
        model_id=str(args.model_id),
        model_sha256=model_hash,
        training_song_ids=training_ids(
            report,
            model_hash,
            held_out_song=str(args.held_out_song or "") or None,
        ),
        feature_contract=str(args.feature_contract),
        decision_scope=str(args.decision_scope),
    )
    payload["inputs"] = {
        "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "model": {"path": str(model_path), "sha256": model_hash},
        "trainingReport": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
    }
    output_path = args.output.resolve()
    atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "songId": payload["songId"],
                "cells": len(payload["cells"]),
                "modelSha256": model_hash,
                "featureContract": str(args.feature_contract),
                "decisionScope": str(args.decision_scope),
                "referenceFieldsRead": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
