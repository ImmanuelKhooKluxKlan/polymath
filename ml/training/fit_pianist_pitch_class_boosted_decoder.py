"""Fit and export a whole-song-held-out boosted pitch-class decoder.

This is an opened-development experiment.  Every score file for a named song
is produced by a model that was trained on the other songs only.  The final
all-song estimator is exported for a *future* untouched transfer test and must
not be evaluated on any song listed in this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .fit_pianist_pitch_class_decoder import FEATURE_NAMES, examples_from_song


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
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


def safe_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("Song id cannot be converted into an artifact name")
    return result


def estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=40,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        learning_rate=0.08,
        random_state=17,
    )


def song_arrays(song: dict[str, Any]) -> dict[str, Any]:
    examples = examples_from_song(song)
    if not examples or len(examples) % 12:
        raise ValueError(f"Song {song.get('id')!r} has an invalid example grid")
    cells = len(examples) // 12
    return {
        "examples": examples,
        "features": np.asarray(
            [row["features"] for row in examples], dtype=np.float32
        ),
        "labels": np.asarray(
            [row["label"] for row in examples], dtype=np.int8
        ).reshape(cells, 12),
        "candidate": np.asarray(
            [row["candidate"] for row in examples], dtype=np.int8
        ).reshape(cells, 12),
        "source": np.asarray(
            [row["source"] for row in examples], dtype=np.int8
        ).reshape(cells, 12),
    }


def set_metrics(
    arrays: dict[str, Any],
    probabilities: np.ndarray,
    *,
    add_threshold: float,
    remove_threshold: float,
    source_supported_additions_only: bool,
) -> dict[str, Any]:
    labels = arrays["labels"].astype(bool)
    candidate = arrays["candidate"].astype(bool)
    source = arrays["source"].astype(bool)
    if probabilities.shape != labels.shape:
        raise ValueError("Probability grid does not match the song cell grid")
    decoded = candidate.copy()
    additions = (~candidate) & (probabilities >= add_threshold)
    if source_supported_additions_only:
        additions &= source
    removals = candidate & (probabilities <= remove_threshold)
    decoded |= additions
    decoded &= ~removals
    true_positive = int(np.sum(decoded & labels))
    false_positive = int(np.sum(decoded & ~labels))
    false_negative = int(np.sum(~decoded & labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "cells": int(labels.shape[0]),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "exactSetRate": round(
            float(np.mean(np.all(decoded == labels, axis=1))), 6
        ),
        "additions": int(np.sum(additions)),
        "removals": int(np.sum(removals)),
    }


def score_payload(
    song_id: str,
    probabilities: np.ndarray,
    *,
    model_id: str,
    model_sha256: str,
    training_song_ids: list[str],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, Any]:
    if probabilities.ndim != 2 or probabilities.shape[1] != 12:
        raise ValueError("Expected one 12-class probability row per cell")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain a non-finite value")
    return {
        "schema": "polymath-pianist-pitch-class-scores-v1",
        "evidenceBoundary": "whole-song-out-of-fold-opened-development",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--add-threshold", type=float, default=0.4)
    parser.add_argument("--remove-threshold", type=float, default=0.225)
    parser.add_argument(
        "--allow-unsupported-additions",
        action="store_true",
        help="Allow generated pitch classes without local source support.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.remove_threshold <= args.add_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= remove <= add <= 1")

    songs: list[dict[str, Any]] = []
    seen: set[str] = set()
    audit_inputs: list[dict[str, str]] = []
    for unresolved in args.audit:
        path = unresolved.resolve()
        audit_inputs.append({"path": str(path), "sha256": sha256_file(path)})
        for song in load_json(path).get("songs") or []:
            song_id = str(song.get("id") or "")
            if not song_id:
                raise ValueError(f"Audit {path} contains a song without an id")
            if song_id in seen:
                raise ValueError(f"Duplicate song id across audits: {song_id}")
            seen.add(song_id)
            songs.append(song)
    if len(songs) < 3:
        raise ValueError("At least three whole songs are required")

    output_dir = args.output_dir.resolve()
    folds_dir = output_dir / "folds"
    scores_dir = output_dir / "scores"
    folds_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    arrays = {str(song["id"]): song_arrays(song) for song in songs}
    song_ids = list(arrays)
    source_only = not bool(args.allow_unsupported_additions)
    folds: list[dict[str, Any]] = []

    for held_out in song_ids:
        training_ids = [song_id for song_id in song_ids if song_id != held_out]
        train_features = np.concatenate(
            [arrays[song_id]["features"] for song_id in training_ids]
        )
        train_labels = np.concatenate(
            [arrays[song_id]["labels"].reshape(-1) for song_id in training_ids]
        )
        model = estimator()
        model.fit(train_features, train_labels)
        held = arrays[held_out]
        probabilities = model.predict_proba(held["features"])[:, 1].reshape(
            held["labels"].shape
        )

        stem = safe_name(held_out)
        model_path = folds_dir / stem / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(model_path.name + ".tmp")
        joblib.dump(model, temporary)
        temporary.replace(model_path)
        model_sha = sha256_file(model_path)
        scores_path = scores_dir / f"{stem}.json"
        atomic_json(
            scores_path,
            score_payload(
                held_out,
                probabilities,
                model_id=f"boosted-pitch-class-loso-{stem}",
                model_sha256=model_sha,
                training_song_ids=training_ids,
            ),
        )
        baseline = set_metrics(
            held,
            np.zeros_like(probabilities),
            add_threshold=2.0,
            remove_threshold=-1.0,
            source_supported_additions_only=True,
        )
        candidate = set_metrics(
            held,
            probabilities,
            add_threshold=float(args.add_threshold),
            remove_threshold=float(args.remove_threshold),
            source_supported_additions_only=source_only,
        )
        folds.append(
            {
                "heldOutSong": held_out,
                "trainingSongIds": training_ids,
                "model": {"path": str(model_path), "sha256": model_sha},
                "scores": {
                    "path": str(scores_path),
                    "sha256": sha256_file(scores_path),
                },
                "baseline": baseline,
                "candidate": candidate,
                "deltas": {
                    "f1": round(candidate["f1"] - baseline["f1"], 6),
                    "exactSetRate": round(
                        candidate["exactSetRate"] - baseline["exactSetRate"],
                        6,
                    ),
                },
            }
        )

    final_features = np.concatenate(
        [arrays[song_id]["features"] for song_id in song_ids]
    )
    final_labels = np.concatenate(
        [arrays[song_id]["labels"].reshape(-1) for song_id in song_ids]
    )
    final_model = estimator()
    final_model.fit(final_features, final_labels)
    final_path = output_dir / "final-opened-development-model.joblib"
    final_temporary = final_path.with_name(final_path.name + ".tmp")
    joblib.dump(final_model, final_temporary)
    final_temporary.replace(final_path)

    f1_deltas = [float(row["deltas"]["f1"]) for row in folds]
    exact_deltas = [float(row["deltas"]["exactSetRate"]) for row in folds]
    policy = {
        "schema": "polymath-pianist-boosted-pitch-class-policy-v1",
        "id": "boosted-pitch-class-hist40-l15-opened-development-v001",
        "production": False,
        "commercialUseAllowed": False,
        "addThreshold": float(args.add_threshold),
        "removeThreshold": float(args.remove_threshold),
        "sourceSupportedAdditionsOnly": source_only,
        "registerStrategy": "nearest-candidate",
        "additionTiming": "candidate",
        "modelSha256": sha256_file(final_path),
        "warning": (
            "Opened-development model. Use only on a new untouched transfer "
            "song before considering promotion."
        ),
    }
    policy_path = output_dir / "policy.json"
    atomic_json(policy_path, policy)
    report = {
        "schema": "polymath-pianist-boosted-pitch-class-training-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "evidenceBoundary": (
            "Whole-song leave-one-out scores across opened development songs. "
            "The final estimator has no promotion evidence."
        ),
        "audits": audit_inputs,
        "featureCount": len(FEATURE_NAMES),
        "songs": song_ids,
        "hyperparameters": {
            "maxIter": 40,
            "maxLeafNodes": 15,
            "l2Regularization": 1.0,
            "learningRate": 0.08,
            "randomState": 17,
        },
        "policy": policy,
        "folds": folds,
        "aggregate": {
            "averageF1Delta": round(float(np.mean(f1_deltas)), 6),
            "worstF1Delta": round(min(f1_deltas), 6),
            "averageExactSetRateDelta": round(
                float(np.mean(exact_deltas)), 6
            ),
            "worstExactSetRateDelta": round(min(exact_deltas), 6),
        },
        "gates": {
            "allSongsPitchClassF1NonRegression": all(
                value >= 0.0 for value in f1_deltas
            ),
            "allSongsExactSetRateNonRegression": all(
                value >= 0.0 for value in exact_deltas
            ),
            "averageExactSetRateNonRegression": float(np.mean(exact_deltas))
            >= 0.0,
            "untouchedTransferStillRequired": True,
        },
        "decision": "RESEARCH_ONLY_NOT_PROMOTION_SAFE",
        "finalModel": {
            "path": str(final_path),
            "sha256": sha256_file(final_path),
            "trainingSongIds": song_ids,
        },
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "policy": str(policy_path),
                "aggregate": report["aggregate"],
                "gates": report["gates"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
