"""Fit a whole-song-held-out forest to correct chord membership.

The phase-45 decoder can both omit useful pitch classes and add locally
supported classes that do not belong in the authored keyboard part.  This
model scores the union of existing candidate classes and local source classes,
allowing a later deterministic operator to add high-confidence omissions and
remove only very-low-confidence incumbent classes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .fit_pianist_pitch_class_addition_ranker import (
    estimator,
    evidence_song_arrays,
)
from .fit_pianist_pitch_class_boosted_decoder import (
    atomic_json,
    load_json,
    safe_name,
    score_payload,
    set_metrics,
    sha256_file,
)
from .pianist_pitch_class_evidence_context import FEATURE_NAMES
from .pianist_pitch_class_decisions import membership_decision_mask


def membership_probability_grid(arrays: dict[str, Any], model: Any) -> np.ndarray:
    mask = membership_decision_mask(arrays)
    probabilities = np.zeros(mask.shape[0], dtype=float)
    if np.any(mask):
        probabilities[mask] = model.predict_proba(arrays["features"][mask])[:, 1]
    return probabilities.reshape(arrays["labels"].shape)


def feature_importance(model: Any, limit: int = 30) -> list[dict[str, Any]]:
    values = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if values.shape != (len(FEATURE_NAMES),):
        return []
    indexes = np.argsort(values)[::-1][:limit]
    return [
        {"feature": FEATURE_NAMES[int(index)], "importance": round(float(values[index]), 8)}
        for index in indexes
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--add-threshold", type=float, default=0.75)
    parser.add_argument("--remove-threshold", type=float, default=0.20)
    parser.add_argument(
        "--estimator",
        choices=("extra-trees", "random-forest"),
        default="random-forest",
    )
    args = parser.parse_args()
    if not 0.0 <= args.remove_threshold < args.add_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= remove < add <= 1")

    songs: list[dict[str, Any]] = []
    seen: set[str] = set()
    audit_inputs: list[dict[str, str]] = []
    for unresolved in args.audit:
        path = unresolved.resolve()
        audit_inputs.append({"path": str(path), "sha256": sha256_file(path)})
        for song in load_json(path).get("songs") or []:
            song_id = str(song.get("id") or "")
            if not song_id or song_id in seen:
                raise ValueError(f"Missing or duplicate song id: {song_id!r}")
            seen.add(song_id)
            songs.append(song)
    if len(songs) < 3:
        raise ValueError("At least three whole songs are required")

    arrays = {str(song["id"]): evidence_song_arrays(song) for song in songs}
    song_ids = list(arrays)
    output_dir = args.output_dir.resolve()
    folds: list[dict[str, Any]] = []
    for held_out in song_ids:
        training_ids = [song_id for song_id in song_ids if song_id != held_out]
        train_features = np.concatenate(
            [
                arrays[song_id]["features"][membership_decision_mask(arrays[song_id])]
                for song_id in training_ids
            ]
        )
        train_labels = np.concatenate(
            [
                arrays[song_id]["labels"].reshape(-1)[membership_decision_mask(arrays[song_id])]
                for song_id in training_ids
            ]
        )
        model = estimator(str(args.estimator))
        model.fit(train_features, train_labels)
        probabilities = membership_probability_grid(arrays[held_out], model)

        stem = safe_name(held_out)
        model_path = output_dir / "folds" / stem / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(model_path.name + ".tmp")
        joblib.dump(model, temporary)
        temporary.replace(model_path)
        model_sha = sha256_file(model_path)
        scores_path = output_dir / "scores" / f"{stem}.json"
        scores = score_payload(
            held_out,
            probabilities,
            model_id=f"membership-forest-loso-{stem}",
            model_sha256=model_sha,
            training_song_ids=training_ids,
            feature_names=FEATURE_NAMES,
        )
        scores["decisionScope"] = "membership"
        atomic_json(scores_path, scores)
        baseline = set_metrics(
            arrays[held_out],
            probabilities,
            add_threshold=2.0,
            remove_threshold=-1.0,
            source_supported_additions_only=True,
        )
        candidate = set_metrics(
            arrays[held_out],
            probabilities,
            add_threshold=float(args.add_threshold),
            remove_threshold=float(args.remove_threshold),
            source_supported_additions_only=True,
        )
        folds.append(
            {
                "heldOutSong": held_out,
                "trainingSongIds": training_ids,
                "eligibleTrainingDecisions": int(train_labels.shape[0]),
                "positiveTrainingDecisions": int(np.sum(train_labels)),
                "model": {"path": str(model_path), "sha256": model_sha},
                "scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
                "baseline": baseline,
                "candidate": candidate,
                "deltas": {
                    "f1": round(candidate["f1"] - baseline["f1"], 6),
                    "exactSetRate": round(candidate["exactSetRate"] - baseline["exactSetRate"], 6),
                },
                "featureImportance": feature_importance(model),
            }
        )

    all_features = np.concatenate(
        [arrays[song_id]["features"][membership_decision_mask(arrays[song_id])] for song_id in song_ids]
    )
    all_labels = np.concatenate(
        [arrays[song_id]["labels"].reshape(-1)[membership_decision_mask(arrays[song_id])] for song_id in song_ids]
    )
    final_model = estimator(str(args.estimator))
    final_model.fit(all_features, all_labels)
    final_path = output_dir / "final-opened-development-model.joblib"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(final_path.name + ".tmp")
    joblib.dump(final_model, temporary)
    temporary.replace(final_path)
    model_sha = sha256_file(final_path)

    policy = {
        "schema": "polymath-pianist-membership-forest-policy-v1",
        "id": f"source-supported-membership-{args.estimator}-evidence259-opened-v001",
        "production": False,
        "commercialUseAllowed": False,
        "decisionScope": "membership",
        "featureContract": "evidence259",
        "addThreshold": float(args.add_threshold),
        "removeThreshold": float(args.remove_threshold),
        "sourceSupportedAdditionsOnly": True,
        "registerStrategy": "nearest-candidate",
        "additionTiming": "candidate",
        "modelSha256": model_sha,
        "warning": "Opened development only; a new untouched transfer is mandatory.",
    }
    atomic_json(output_dir / "policy.json", policy)
    f1_deltas = [float(row["deltas"]["f1"]) for row in folds]
    report = {
        "schema": "polymath-pianist-membership-forest-training-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "evidenceBoundary": "Whole-song leave-one-out opened development.",
        "audits": audit_inputs,
        "songs": song_ids,
        "estimator": str(args.estimator),
        "featureContract": "evidence259",
        "featureCount": len(FEATURE_NAMES),
        "eligibleDecisions": int(all_labels.shape[0]),
        "positiveDecisions": int(np.sum(all_labels)),
        "policy": policy,
        "folds": folds,
        "aggregate": {
            "averageF1Delta": round(float(np.mean(f1_deltas)), 6),
            "worstF1Delta": round(min(f1_deltas), 6),
            "totalAcceptedAdditions": sum(int(row["candidate"]["additions"]) for row in folds),
            "totalRemovals": sum(int(row["candidate"]["removals"]) for row in folds),
        },
        "gates": {
            "allSongsCellPitchClassF1NonRegression": all(value >= 0.0 for value in f1_deltas),
            "playableNoteGateStillRequired": True,
            "untouchedTransferStillRequired": True,
        },
        "featureImportance": feature_importance(final_model),
        "decision": "RESEARCH_ONLY_NOT_PROMOTION_SAFE",
        "finalModel": {
            "path": str(final_path),
            "sha256": model_sha,
            "trainingSongIds": song_ids,
        },
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "aggregate": report["aggregate"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
