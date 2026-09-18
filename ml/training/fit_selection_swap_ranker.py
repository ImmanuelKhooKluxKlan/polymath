"""Fit a quota-preserving source-note swap ranker.

Training examples come only from same-onset, same-family pairs in the selection
swap audit.  Each positive vector says that the alignment-supported source
note should outrank the source note that occupied the slot.  The reversed
vector is added as a negative example, so the fitted decision boundary has no
song-specific intercept.  Complete songs are held out during validation.

The output is an ordinary arranger ``selectionModel`` and can therefore be
installed behind ``conditionalSelectionBlend``.  Runtime use must preserve the
base onset counts: this model changes *which pitch occupies a slot*, never how
many notes or attacks are played.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
)


FEATURE_GROUPS = {
    "compact": (
        "bias",
        "midi_centered",
        "midi_squared",
        "role_melody",
        "role_bass",
        "role_harmony",
        "cross_family_pitch_class_support",
        "same_instrument_pitch_class_recurrence",
        "is_local_onset_lowest",
        "is_local_onset_highest",
        "local_onset_pitch_percentile",
        "previous_same_pitch_gap_log",
        "next_same_pitch_gap_log",
        "wide_pitch_class_support",
        "wide_pitch_class_rank",
        "wide_pitch_class_dominance",
        "estimated_chord_member_root",
        "estimated_chord_member_third",
        "estimated_chord_member_fifth",
        "estimated_chord_nonmember",
        "estimated_bass_root_match",
    ),
    "position-only": (
        "bias",
        "midi_centered",
        "midi_squared",
        "role_melody",
        "role_bass",
        "role_harmony",
        "same_instrument_pitch_class_recurrence",
        "is_local_onset_lowest",
        "is_local_onset_highest",
        "local_onset_pitch_percentile",
        "previous_same_pitch_gap_log",
        "next_same_pitch_gap_log",
    ),
    "harmony-only": (
        "bias",
        "wide_pitch_class_support",
        "wide_pitch_class_rank",
        "wide_pitch_class_dominance",
        "estimated_chord_member_root",
        "estimated_chord_member_third",
        "estimated_chord_member_fifth",
        "estimated_chord_nonmember",
        "estimated_bass_root_match",
    ),
    "full": HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def scales_for_deltas(matrix: np.ndarray) -> np.ndarray:
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    if matrix.shape[1]:
        scales[0] = 1.0
    return scales


def fit_symmetric_logistic(
    positive_deltas: np.ndarray,
    *,
    ridge: float,
    scales: np.ndarray,
    iterations: int = 80,
) -> np.ndarray:
    standardized = positive_deltas / scales
    matrix = np.concatenate((standardized, -standardized), axis=0)
    labels = np.concatenate(
        (np.ones(len(standardized)), np.zeros(len(standardized))), axis=0
    )
    weights = np.zeros(matrix.shape[1], dtype=float)
    penalty = np.eye(matrix.shape[1], dtype=float) * ridge
    if matrix.shape[1]:
        penalty[0, 0] = 0.0
    for _iteration in range(iterations):
        probabilities = sigmoid(matrix @ weights)
        gradient = matrix.T @ (probabilities - labels) + penalty @ weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = matrix.T @ (matrix * curvature[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return weights


def pairwise_metrics(deltas: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    margins = (deltas / scales) @ weights
    wins = margins > 1e-12
    ties = np.abs(margins) <= 1e-12
    return {
        "pairs": len(deltas),
        "desiredWins": int(np.sum(wins)),
        "ties": int(np.sum(ties)),
        "desiredWinRate": round(float(np.mean(wins)) if len(wins) else 0.0, 6),
        "meanMargin": round(float(np.mean(margins)) if len(margins) else 0.0, 6),
        "medianMargin": round(float(np.median(margins)) if len(margins) else 0.0, 6),
    }


def examples_from_audit(
    audit: dict[str, Any],
    feature_names: tuple[str, ...],
    *,
    family: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for song in audit.get("songs") or []:
        song_id = str(song.get("id") or "")
        for pair in song.get("pairs") or []:
            if not pair.get("sameOnset") or not pair.get("sameFamily"):
                continue
            desired_family = str((pair.get("desired") or {}).get("family") or "")
            if family and desired_family != family:
                continue
            deltas = pair.get("featureDeltaDesiredMinusSelected") or {}
            if not all(name in deltas for name in feature_names):
                raise ValueError(f"{song_id}: swap audit lacks an inference feature")
            examples.append(
                {
                    "songId": song_id,
                    "delta": [float(deltas[name]) for name in feature_names],
                }
            )
    return examples


def parameter_grid() -> Iterable[tuple[str, float]]:
    for group in FEATURE_GROUPS:
        for ridge in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            yield group, ridge


def cross_validate(audit: dict[str, Any], *, family: str) -> dict[str, Any]:
    songs = sorted(str(song.get("id") or "") for song in audit.get("songs") or [])
    if len(songs) < 2:
        raise ValueError("Selection swap fitting requires at least two songs")
    trials: list[dict[str, Any]] = []
    for group, ridge in parameter_grid():
        names = tuple(FEATURE_GROUPS[group])
        examples = examples_from_audit(audit, names, family=family)
        folds = []
        for heldout in songs:
            training = [row for row in examples if row["songId"] != heldout]
            validation = [row for row in examples if row["songId"] == heldout]
            if not training or not validation:
                folds.append({"heldOutSong": heldout, "pairs": len(validation), "desiredWinRate": None})
                continue
            train_matrix = np.asarray([row["delta"] for row in training], dtype=float)
            validation_matrix = np.asarray([row["delta"] for row in validation], dtype=float)
            scales = scales_for_deltas(train_matrix)
            weights = fit_symmetric_logistic(train_matrix, ridge=ridge, scales=scales)
            folds.append(
                {
                    "heldOutSong": heldout,
                    **pairwise_metrics(validation_matrix, weights, scales),
                }
            )
        active = [fold for fold in folds if fold.get("desiredWinRate") is not None]
        total_pairs = sum(int(fold["pairs"]) for fold in active)
        aggregate_rate = sum(
            int(fold["desiredWins"]) for fold in active
        ) / max(1, total_pairs)
        worst_rate = min(float(fold["desiredWinRate"]) for fold in active)
        # A model used for music must transfer to every song; the weakest
        # complete-song fold matters more than a deceptively strong average.
        selection_score = 0.70 * worst_rate + 0.30 * aggregate_rate
        trials.append(
            {
                "featureGroup": group,
                "ridge": ridge,
                "selectionScore": round(selection_score, 6),
                "aggregateDesiredWinRate": round(aggregate_rate, 6),
                "worstSongDesiredWinRate": round(worst_rate, 6),
                "pairs": total_pairs,
                "folds": folds,
            }
        )
    trials.sort(
        key=lambda row: (
            -float(row["selectionScore"]),
            -float(row["worstSongDesiredWinRate"]),
            -float(row["aggregateDesiredWinRate"]),
            len(FEATURE_GROUPS[str(row["featureGroup"])]),
            float(row["ridge"]),
        )
    )
    return {"songs": songs, "best": trials[0], "topTrials": trials[:20]}


def sha256_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--source-family", default="guitar")
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    audit = load_json(audit_path)
    validation = cross_validate(audit, family=args.source_family)
    best = validation["best"]
    feature_names = tuple(FEATURE_GROUPS[str(best["featureGroup"])])
    examples = examples_from_audit(audit, feature_names, family=args.source_family)
    matrix = np.asarray([row["delta"] for row in examples], dtype=float)
    scales = scales_for_deltas(matrix)
    weights = fit_symmetric_logistic(
        matrix, ridge=float(best["ridge"]), scales=scales
    )
    fitted_weight_by_name = dict(zip(feature_names, weights))
    fitted_scale_by_name = dict(zip(feature_names, scales))
    runtime_feature_names = tuple(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
    model = {
        "schema": "polymath-piano-arranger-profile-v1",
        "version": 1,
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "selectionModel": {
            "type": "standardized-logistic-note-swap-ranker-v1",
            # Runtime feature contracts are exact/versioned tuples. Lift the
            # selected compact feature set into the complete harmonic contract
            # with zero coefficients instead of changing its learned logits.
            "featureNames": list(runtime_feature_names),
            "weights": [
                round(float(fitted_weight_by_name.get(name, 0.0)), 10)
                for name in runtime_feature_names
            ],
            # Pairwise rankings are translation invariant. Zero means make the
            # absolute logits easy to inspect while preserving every delta.
            "means": [0.0] * len(runtime_feature_names),
            "scales": [
                round(float(fitted_scale_by_name.get(name, 1.0)), 10)
                for name in runtime_feature_names
            ],
            "threshold": 0.5,
        },
        "training": {
            "schema": "polymath-local-note-swap-training-v1",
            "audit": str(audit_path),
            "sourceFamily": args.source_family,
            "examples": len(examples),
            "fittedFeatureGroup": str(best["featureGroup"]),
            "fittedFeatureNames": list(feature_names),
            "completeSongLeaveOneOut": True,
            "preservesOnsetAndNoteQuotas": True,
            "usesTargetAtInference": False,
            "rights": "private research only",
            "decision": "RESEARCH_ONLY",
        },
    }
    model["profileSha256"] = sha256_payload(model)
    output_profile = Path(args.output_profile).resolve()
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    importance = sorted(
        (
            {
                "feature": name,
                "weight": round(float(weight), 6),
                "absoluteWeight": round(abs(float(weight)), 6),
            }
            for name, weight in zip(feature_names, weights)
        ),
        key=lambda row: -float(row["absoluteWeight"]),
    )
    report = {
        "schema": "polymath-selection-swap-training-report-v1",
        "evidenceBoundary": (
            "Complete-song leave-one-out validation on same-onset, same-family "
            "source-note swaps. Kiss Me at/after 02:30 is excluded. Pairwise "
            "accuracy alone does not authorize promotion; rendered held-out "
            "scores and listening must still pass."
        ),
        "audit": str(audit_path),
        "sourceFamily": args.source_family,
        "examples": len(examples),
        "crossValidation": validation,
        "finalTrainingMetrics": pairwise_metrics(matrix, weights, scales),
        "featureImportance": importance,
        "profile": str(output_profile),
        "profileSha256": model["profileSha256"],
        "decision": "RESEARCH_ONLY",
    }
    output_report = Path(args.output_report).resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": str(output_profile),
                "report": str(output_report),
                "examples": len(examples),
                "best": best,
                "topFeatures": importance[:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
