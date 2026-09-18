"""Fit a same-gesture pairwise ranker for accompaniment swap proposals.

The classification model learns an absolute yes/no boundary across songs.
This alternative asks the decoder's real question: within one already-frozen
gesture, should proposal A rank above proposal B?  Training pairs never cross
song or gesture boundaries, and every positive/negative difference is mirrored
so the linear latent score has a meaningful zero-centered comparison space.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import (
    atomic_json,
    build_song_examples,
)
from .raw_support_swap_benefit import (
    SWAP_BENEFIT_FEATURE_NAMES,
    model_probabilities_numpy,
)
from .train_piano_arranger_adapter import classification_metrics


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pairwise_song(song: dict[str, Any], maximum_negatives: int) -> dict[str, Any]:
    by_group: dict[int, list[int]] = {}
    for index, proposal in enumerate(song["proposals"]):
        by_group.setdefault(int(proposal["groupNumber"]), []).append(index)
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    times: list[float] = []
    positive_pairs = 0
    positive_proposals_used: set[int] = set()
    for group_indices in by_group.values():
        positives = [index for index in group_indices if song["labels"][index] >= 0.5]
        negatives = [index for index in group_indices if song["labels"][index] < 0.5]
        if not positives or not negatives:
            continue
        for positive_index in positives:
            positive = song["proposals"][positive_index]

            def hardness(negative_index: int) -> tuple[float, ...]:
                negative = song["proposals"][negative_index]
                return (
                    float(negative["incumbentIndex"] == positive["incumbentIndex"]),
                    float(negative["sourceUsableIndex"] == positive["sourceUsableIndex"]),
                    -abs(float(negative["sourceProbability"]) - float(positive["sourceProbability"])),
                    -abs(float(negative["probabilityGain"]) - float(positive["probabilityGain"])),
                    -abs(int(negative["targetMidi"]) - int(positive["targetMidi"])),
                )

            ranked_negatives = sorted(negatives, key=hardness, reverse=True)[:maximum_negatives]
            positive_features = song["features"][positive_index]
            pair_weight = 1.0 + 0.10 * min(8.0, max(0.0, float(song["utilities"][positive_index])))
            for negative_index in ranked_negatives:
                difference = positive_features - song["features"][negative_index]
                feature_rows.extend((difference, -difference))
                labels.extend((1.0, 0.0))
                weights.extend((pair_weight, pair_weight))
                times.extend((float(positive["groupTime"]), float(positive["groupTime"])))
                positive_pairs += 1
                positive_proposals_used.add(positive_index)
    if positive_pairs < 10:
        raise ValueError(f"{song['id']} has too few same-gesture ranking pairs")
    features = np.asarray(feature_rows, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    # Equalize each song's total contribution regardless of proposal density.
    weights_array *= max(1.0, 1000.0 / float(weights_array.sum()))
    times_array = np.asarray(times, dtype=np.float64)
    return {
        "id": song["id"],
        "features": features,
        "labels": labels_array,
        "weights": weights_array,
        "times": times_array,
        "temporalTraining": (times_array // 20.0).astype(np.int64) % 5 != 4,
        "positivePairs": positive_pairs,
        "positiveProposalsUsed": len(positive_proposals_used),
    }


def combine_pair_songs(songs: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
        np.vstack([song["features"] for song in songs]),
        np.concatenate([song["labels"] for song in songs]),
        np.concatenate([song["weights"] for song in songs]),
        np.concatenate([song["temporalTraining"] for song in songs]),
    )


def train_pairwise_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    training_mask: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    means = features[training_mask].mean(axis=0)
    scales = features[training_mask].std(axis=0)
    scales = np.where(scales < 1e-7, 1.0, scales)
    x = np.clip((features - means) / scales, -8.0, 8.0)
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    first_w = np.zeros_like(coefficients)
    second_w = np.zeros_like(coefficients)
    first_b = second_b = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_coefficients = coefficients.copy()
    best_bias = bias

    def probability(mask: np.ndarray) -> np.ndarray:
        logits = x[mask] @ coefficients + bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def loss(mask: np.ndarray) -> float:
        predicted = np.clip(probability(mask), 1e-7, 1.0 - 1e-7)
        kept_labels = labels[mask]
        kept_weights = weights[mask]
        values = -(kept_labels * np.log(predicted) + (1.0 - kept_labels) * np.log(1.0 - predicted))
        return float(np.sum(kept_weights * values) / max(1e-9, float(kept_weights.sum())))

    training_indices = np.flatnonzero(training_mask)
    batch_size = min(1024, len(training_indices))
    rng = np.random.default_rng(0x50414952)
    for iteration in range(1, iterations + 1):
        batch = rng.choice(training_indices, size=batch_size, replace=False)
        logits = x[batch] @ coefficients + bias
        predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        kept_weights = weights[batch]
        error = (predicted - labels[batch]) * kept_weights / max(1e-9, float(kept_weights.sum()))
        gradient_w = x[batch].T @ error + l2 * coefficients
        gradient_b = float(error.sum())
        first_w = beta1 * first_w + (1.0 - beta1) * gradient_w
        second_w = beta2 * second_w + (1.0 - beta2) * (gradient_w * gradient_w)
        first_b = beta1 * first_b + (1.0 - beta1) * gradient_b
        second_b = beta2 * second_b + (1.0 - beta2) * gradient_b * gradient_b
        coefficients -= learning_rate * (first_w / (1.0 - beta1**iteration)) / (
            np.sqrt(second_w / (1.0 - beta2**iteration)) + epsilon
        )
        bias -= learning_rate * (first_b / (1.0 - beta1**iteration)) / (
            math.sqrt(second_b / (1.0 - beta2**iteration)) + epsilon
        )
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            training_loss = loss(training_mask)
            validation_loss = loss(~training_mask)
            history.append(
                {
                    "iteration": float(iteration),
                    "trainingWeightedBinaryCrossEntropy": round(training_loss, 8),
                    "validationWeightedBinaryCrossEntropy": round(validation_loss, 8),
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_coefficients = coefficients.copy()
                best_bias = bias
    model = {
        "type": "standardized-logistic-swap-benefit-v1",
        "objective": "same-gesture-symmetric-pairwise-v1",
        "featureNames": list(SWAP_BENEFIT_FEATURE_NAMES),
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "weights": [round(float(value), 10) for value in best_coefficients],
        "bias": round(float(best_bias), 10),
        "threshold": 0.5,
    }
    return model, history


def fit(
    songs: list[dict[str, Any]], *, iterations: int, learning_rate: float, l2: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    features, labels, weights, training = combine_pair_songs(songs)
    if training.all() or (~training).all():
        raise ValueError("Pairwise temporal training and validation blocks are required")
    model, history = train_pairwise_logistic(
        features,
        labels,
        weights,
        training,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
    )
    probabilities = model_probabilities_numpy(features, model)
    return model, {
        "pairs": len(labels) // 2,
        "temporalTraining": classification_metrics(labels[training], probabilities[training], weights[training], 0.5),
        "temporalValidation": classification_metrics(labels[~training], probabilities[~training], weights[~training], 0.5),
        "history": history,
    }


def evaluate_pairs(song: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    probabilities = model_probabilities_numpy(song["features"], model)
    metrics = classification_metrics(song["labels"], probabilities, song["weights"], 0.5)
    metrics["pairs"] = len(song["labels"]) // 2
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fold-profiles-root", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--maximum-negatives-per-positive", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=2600)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=0.003)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    selector_root = args.selector_profiles_root.resolve()
    manifest = load_json(manifest_path)
    proposal_songs = [
        build_song_examples(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            source_radius=0.18,
            minimum_source_midi=0,
            maximum_source_midi=59,
            minimum_output_midi=33,
            maximum_output_midi=71,
            register_shifts=(0, 12),
        )
        for row in manifest.get("songs") or []
    ]
    pair_songs = [
        pairwise_song(song, max(1, int(args.maximum_negatives_per_positive)))
        for song in proposal_songs
    ]
    fold_root = args.fold_profiles_root.resolve()
    folds: dict[str, Any] = {}
    for held_out in pair_songs:
        training = [song for song in pair_songs if song is not held_out]
        model, internal = fit(
            training,
            iterations=max(200, int(args.iterations)),
            learning_rate=float(args.learning_rate),
            l2=float(args.l2),
        )
        profile = {
            "schema": "polymath-raw-support-swap-benefit-profile-v1",
            "id": f"{args.profile_id}-without-{held_out['id']}",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "training": {
                "manifest": str(manifest_path),
                "heldOutSong": held_out["id"],
                "trainingSongs": [song["id"] for song in training],
                "objective": "same-gesture-symmetric-pairwise-v1",
                "commercialUseAllowed": False,
                "purpose": "private research evaluation",
            },
        }
        profile["profileSha256"] = profile_hash(profile)
        profile_path = fold_root / held_out["id"] / "profile.json"
        atomic_json(profile_path, profile)
        folds[held_out["id"]] = {
            "trainingSongs": [song["id"] for song in training],
            "internalTemporalValidation": internal["temporalValidation"],
            "heldOutPairs": evaluate_pairs(held_out, model),
            "profile": str(profile_path),
        }

    final_model, internal = fit(
        pair_songs,
        iterations=max(200, int(args.iterations)),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )
    profile = {
        "schema": "polymath-raw-support-swap-benefit-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": final_model,
        "training": {
            "manifest": str(manifest_path),
            "songs": [song["id"] for song in pair_songs],
            "objective": "same-gesture-symmetric-pairwise-v1",
            "commercialUseAllowed": False,
            "purpose": "private research evaluation",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    atomic_json(args.output_profile.resolve(), profile)
    heldout = [value["heldOutPairs"] for value in folds.values()]
    report = {
        "schema": "polymath-raw-support-swap-benefit-pairwise-training-v1",
        "evidenceBoundary": "Each held-out song uses a ranker and upstream selector trained without that complete song.",
        "manifest": str(manifest_path),
        "songs": [
            {
                "id": song["id"],
                "positivePairs": song["positivePairs"],
                "positiveProposalsUsed": song["positiveProposalsUsed"],
            }
            for song in pair_songs
        ],
        "folds": folds,
        "heldOutSummary": {
            "meanPairAccuracy": round(sum(float(row["accuracy"]) for row in heldout) / len(heldout), 6),
            "minimumPairAccuracy": round(min(float(row["accuracy"]) for row in heldout), 6),
        },
        "finalInternal": internal,
        "profile": str(args.output_profile.resolve()),
    }
    atomic_json(args.report.resolve(), report)
    print(json.dumps({"profile": str(args.output_profile.resolve()), "report": str(args.report.resolve()), "heldOutSummary": report["heldOutSummary"]}, indent=2))


if __name__ == "__main__":
    main()
