"""Fit an octave router that must agree with source-audio evidence.

Training labels come from fixed, aligned pianist references.  Validation is
split by complete song, and all mixtures of one song share the same fold.  At
runtime the router sees only the candidate score and its source WAV.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_pianist_register_errors import analyze_song
from .audio_register_evidence import (
    AUDIO_FEATURE_NAMES,
    AudioRegisterEvidence,
    audio_feature_values,
    audio_gate,
)
from .fit_pianist_register_router import (
    FEATURE_NAMES as SYMBOLIC_FEATURE_NAMES,
    SHIFT_CLASSES,
    balanced_weights,
    load_json,
    model_probabilities,
    policy_metrics,
    record_features,
    rounded_shift,
    route_predictions,
)
from .train_piano_arranger_adapter import train_logistic_model


FEATURE_NAMES = (*SYMBOLIC_FEATURE_NAMES, *AUDIO_FEATURE_NAMES)
AGREEMENT_MODES = ("both", "odd", "fundamental", "either", "none")


def fit_models(
    records: list[dict[str, Any]],
    *,
    class_balance: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    sample_weights = balanced_weights(records, class_balance)
    training_mask = np.ones(len(records), dtype=bool)
    classes: dict[str, Any] = {}
    for index, shift in enumerate(SHIFT_CLASSES):
        binary = (labels == shift).astype(np.float64)
        weights, means, scales, history = train_logistic_model(
            features,
            binary,
            sample_weights,
            training_mask,
            seed=seed + index * 101,
            iterations=iterations,
            learning_rate=0.014,
            l2=0.028,
        )
        classes[str(shift)] = {
            "weights": [round(float(value), 10) for value in weights],
            "means": [round(float(value), 10) for value in means],
            "scales": [round(float(value), 10) for value in scales],
            "finalTrainingLoss": history[-1]["weightedBinaryCrossEntropy"],
        }
    return {
        "type": "one-v-rest-audio-symbolic-register-router-v1",
        "featureNames": list(FEATURE_NAMES),
        "shiftClasses": list(SHIFT_CLASSES),
        "classBalance": round(float(class_balance), 4),
        "classes": classes,
    }


def load_records(
    rows: Iterable[dict[str, Any]], tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    descriptions: list[dict[str, Any]] = []
    for row in rows:
        audio_path = Path(str(row.get("audio") or "")).resolve()
        if not audio_path.is_file():
            raise ValueError(f"Audio evidence is missing: {audio_path}")
        analysis = analyze_song(row, tolerance)
        evidence_reader = AudioRegisterEvidence.from_wav(audio_path)
        descriptions.append(
            {
                "id": analysis["id"],
                "variant": row.get("variant"),
                "audio": str(audio_path),
                **analysis["counts"],
                "requiredShiftBands": analysis["requiredShiftBands"],
            }
        )
        for item in analysis["pairs"]:
            evidence = evidence_reader.score(
                float(item["time"]), int(item["candidateMidi"])
            )
            records.append(
                {
                    **item,
                    "songId": analysis["id"],
                    "variant": row.get("variant"),
                    "label": rounded_shift(item["requiredShiftSemitones"]),
                    "audioEvidence": evidence,
                    "features": [
                        *record_features(item),
                        *audio_feature_values(evidence),
                    ],
                }
            )
    return records, descriptions


def gated_predictions(
    records: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    threshold: float,
    margin: float,
    agreement: str,
    minimum_gain: float,
) -> np.ndarray:
    proposed = route_predictions(
        probabilities, threshold=threshold, margin=margin
    )
    return np.asarray(
        [
            int(prediction)
            if audio_gate(
                int(prediction),
                record["audioEvidence"],
                agreement=agreement,
                minimum_gain=minimum_gain,
            )
            else 0
            for record, prediction in zip(records, proposed)
        ],
        dtype=np.int64,
    )


def select_policy(
    folds: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for threshold in np.linspace(0.35, 0.90, 12):
        for margin in np.linspace(0.0, 0.40, 9):
            for agreement in AGREEMENT_MODES:
                for minimum_gain in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
                    fold_metrics: dict[str, Any] = {}
                    combined_records: list[dict[str, Any]] = []
                    combined_predictions: list[int] = []
                    for song, fold in folds.items():
                        predictions = gated_predictions(
                            fold["records"],
                            fold["probabilities"],
                            threshold=float(threshold),
                            margin=float(margin),
                            agreement=agreement,
                            minimum_gain=float(minimum_gain),
                        )
                        fold_metrics[song] = policy_metrics(
                            fold["records"], predictions
                        )
                        combined_records.extend(fold["records"])
                        combined_predictions.extend(int(value) for value in predictions)
                    aggregate = policy_metrics(
                        combined_records,
                        np.asarray(combined_predictions, dtype=np.int64),
                    )
                    worst_delta = min(
                        float(value["exactRateDelta"])
                        for value in fold_metrics.values()
                    )
                    trial = {
                        "threshold": round(float(threshold), 4),
                        "margin": round(float(margin), 4),
                        "audioAgreement": agreement,
                        "minimumAudioLogGain": round(float(minimum_gain), 4),
                        "worstSongExactRateDelta": round(worst_delta, 6),
                        "aggregate": aggregate,
                        "folds": fold_metrics,
                    }
                    trial["safe"] = bool(
                        worst_delta >= 0
                        and int(aggregate["changes"]) >= 5
                        and float(aggregate["changePrecision"]) >= 0.70
                    )
                    trials.append(trial)
    safe = [trial for trial in trials if trial["safe"]]
    ranked = sorted(
        safe or trials,
        key=lambda trial: (
            bool(trial["safe"]),
            float(trial["aggregate"]["exactRateDelta"]),
            float(trial["aggregate"]["changePrecision"]),
            float(trial["worstSongExactRateDelta"]),
            -int(trial["aggregate"]["changes"]),
        ),
        reverse=True,
    )
    return ranked[0], ranked


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0x41554449)
    parser.add_argument("--class-balance", type=float, default=0.55)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    rows = manifest.get("songs") or []
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("At least three complete-song rows are required")
    records, descriptions = load_records(
        rows, max(0.01, float(args.tolerance_seconds))
    )
    song_ids = sorted({str(record["songId"]) for record in records})
    if len(song_ids) < 3:
        raise ValueError("At least three unique song ids are required")

    folds: dict[str, dict[str, Any]] = {}
    for fold_index, held_out in enumerate(song_ids):
        training = [record for record in records if record["songId"] != held_out]
        validation = [record for record in records if record["songId"] == held_out]
        model = fit_models(
            training,
            class_balance=args.class_balance,
            iterations=max(200, args.iterations),
            seed=args.seed + fold_index * 1009,
        )
        features = np.asarray(
            [record["features"] for record in validation], dtype=np.float64
        )
        folds[held_out] = {
            "records": validation,
            "probabilities": model_probabilities(model, features),
            "trainingSongs": [song for song in song_ids if song != held_out],
        }

    best, trials = select_policy(folds)
    final_model = fit_models(
        records,
        class_balance=args.class_balance,
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    profile = {
        "schema": "polymath-audio-pianist-register-router-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(best["safe"]),
        "model": final_model,
        "policy": {
            "minimumConfidence": best["threshold"],
            "minimumMargin": best["margin"],
            "audioAgreement": best["audioAgreement"],
            "minimumAudioLogGain": best["minimumAudioLogGain"],
            "minimumMidi": 33,
            "maximumMidi": 108,
            "collisionPolicy": "abstain",
        },
        "training": {
            "manifest": str(manifest_path),
            "method": "complete-song leave-one-out audio-plus-symbolic octave routing",
            "commercialUseAllowed": False,
            "decision": (
                "CANDIDATE_FOR_APPLICATION_TEST"
                if best["safe"]
                else "REJECT_OR_RESEARCH_ONLY"
            ),
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_path = Path(args.output_profile).resolve()
    atomic_json(output_path, profile)
    report = {
        "schema": "polymath-audio-pianist-register-router-training-v1",
        "evidenceBoundary": (
            "Private research only; complete-song folds; source WAV evidence and "
            "candidate-side structure only at inference."
        ),
        "manifest": str(manifest_path),
        "dataset": descriptions,
        "shiftCounts": dict(Counter(str(record["label"]) for record in records)),
        "bestPolicy": best,
        "safePolicyCount": sum(int(trial["safe"]) for trial in trials),
        "topTrials": trials[:25],
        "outputProfile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "decision": profile["training"]["decision"],
    }
    report_path = Path(args.report).resolve()
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "outputProfile": str(output_path),
                "report": str(report_path),
                "bestPolicy": best,
                "safePolicyCount": report["safePolicyCount"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
