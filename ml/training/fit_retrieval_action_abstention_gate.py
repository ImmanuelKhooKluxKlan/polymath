"""Learn when the variable-size chord retriever should abstain.

The base retriever proposes musically plausible, source-supported chord
actions.  Some proposals still hurt a held-out song.  This second-stage gate
learns only from inference-safe proposal/context features and predicts whether
to accept the proposal or keep the incumbent gesture.  Validation again holds
out a complete song.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .fit_chord_gesture_library import membership_metrics
from .fit_pianist_texture_decoder import fit_logistic, sigmoid, standardize
from .fit_retrieval_action_chord_mapper import (
    build_rankings,
    decode_retrieval_action,
    load_json,
    source_supported_intervals,
)
from .fit_paired_chord_gesture_mapper import set_f1


BASE_FEATURE_NAMES = (
    "bias",
    "retrieval_gain",
    "incumbent_expected_f1",
    "proposal_expected_f1",
    "action_count",
    "nearest_distance",
    "mean_top5_distance",
    "incumbent_size",
    "proposal_size",
    "absolute_size_change",
    "added_intervals",
    "removed_intervals",
    "incumbent_proposal_f1",
    "source_supported_count",
)


def proposal_feature_names(context_size: int) -> list[str]:
    return list(BASE_FEATURE_NAMES) + [f"context_{index:03d}" for index in range(context_size)]


def proposal_features(
    sample: dict[str, Any],
    ranked: list[tuple[float, dict[str, Any]]],
    incumbent: set[int],
    proposal: set[int],
    gain: float,
    diagnostics: dict[str, Any],
) -> list[float]:
    distances = [float(distance) for distance, _sample in ranked[:5]]
    context = [float(value) for value in sample.get("context") or []]
    return [
        1.0,
        float(gain),
        float(diagnostics.get("incumbentExpectedF1", 0.0)),
        float(diagnostics.get("bestExpectedF1", 0.0)),
        min(20.0, float(diagnostics.get("actions", 1))) / 20.0,
        distances[0] if distances else 0.0,
        sum(distances) / max(1, len(distances)),
        len(incumbent) / 3.0,
        len(proposal) / 3.0,
        abs(len(proposal) - len(incumbent)),
        len(proposal - incumbent) / 3.0,
        len(incumbent - proposal) / 3.0,
        set_f1(incumbent, proposal),
        len(source_supported_intervals(sample)) / 12.0,
        *context,
    ]


def collect_proposals(
    by_song: dict[str, list[dict[str, Any]]],
    rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]],
    policy: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for song_id, samples in by_song.items():
        for sample_index, (sample, ranked) in enumerate(zip(samples, rankings[song_id])):
            incumbent = {int(value) % 12 for value in sample["incumbentIntervals"]}
            target = {int(value) % 12 for value in sample["targetIntervals"]}
            proposal, gain, diagnostics = decode_retrieval_action(
                ranked,
                incumbent,
                source_supported_intervals(sample),
                neighbors=int(policy["neighbors"]),
                temperature=float(policy["temperature"]),
                incumbent_prior=float(policy["incumbentPrior"]),
                minimum_gain=float(policy["minimumGain"]),
                maximum_size_change=int(policy["maximumSizeChange"]),
            )
            if proposal == incumbent:
                continue
            delta = set_f1(proposal, target) - set_f1(incumbent, target)
            proposals[song_id].append(
                {
                    "sampleIndex": sample_index,
                    "incumbent": incumbent,
                    "proposal": proposal,
                    "target": target,
                    "label": float(delta > 1e-9),
                    "delta": delta,
                    "features": proposal_features(
                        sample, ranked, incumbent, proposal, gain, diagnostics
                    ),
                }
            )
    return proposals


def evaluate_threshold(
    by_song: dict[str, list[dict[str, Any]]],
    proposals: dict[str, list[dict[str, Any]]],
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    all_predictions: list[set[int]] = []
    all_incumbents: list[set[int]] = []
    all_targets: list[set[int]] = []
    folds: dict[str, Any] = {}
    accepted = accepted_improved = accepted_worsened = 0
    for song_id, samples in by_song.items():
        predicted = [
            {int(value) % 12 for value in sample["incumbentIntervals"]}
            for sample in samples
        ]
        incumbents = [set(value) for value in predicted]
        targets = [
            {int(value) % 12 for value in sample["targetIntervals"]}
            for sample in samples
        ]
        fold_accepted = fold_improved = fold_worsened = 0
        for proposal, probability in zip(proposals.get(song_id, []), probabilities.get(song_id, [])):
            if float(probability) < threshold:
                continue
            index = int(proposal["sampleIndex"])
            predicted[index] = set(proposal["proposal"])
            fold_accepted += 1
            fold_improved += int(float(proposal["delta"]) > 1e-9)
            fold_worsened += int(float(proposal["delta"]) < -1e-9)
        baseline_metrics = membership_metrics(incumbents, targets)
        predicted_metrics = membership_metrics(predicted, targets)
        folds[song_id] = {
            "samples": len(samples),
            "proposals": len(proposals.get(song_id, [])),
            "accepted": fold_accepted,
            "acceptedImproved": fold_improved,
            "acceptedWorsened": fold_worsened,
            "baseline": baseline_metrics,
            "predicted": predicted_metrics,
            "membershipF1Delta": round(
                predicted_metrics["membershipF1"] - baseline_metrics["membershipF1"], 6
            ),
        }
        accepted += fold_accepted
        accepted_improved += fold_improved
        accepted_worsened += fold_worsened
        all_predictions.extend(predicted)
        all_incumbents.extend(incumbents)
        all_targets.extend(targets)
    baseline = membership_metrics(all_incumbents, all_targets)
    predicted = membership_metrics(all_predictions, all_targets)
    minimum_fold_delta = min(float(item["membershipF1Delta"]) for item in folds.values())
    return {
        "threshold": round(float(threshold), 6),
        "aggregate": predicted,
        "baseline": baseline,
        "membershipF1Delta": round(predicted["membershipF1"] - baseline["membershipF1"], 6),
        "minimumFoldDelta": round(minimum_fold_delta, 6),
        "accepted": accepted,
        "acceptedImproved": accepted_improved,
        "acceptedWorsened": accepted_worsened,
        "folds": folds,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mapper", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-id", required=True)
    args = parser.parse_args()

    base = load_json(args.base_mapper.resolve())
    retrieval_report = load_json(args.retrieval_report.resolve())
    policy = dict(retrieval_report["winner"])
    by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in base.get("samples") or []:
        if isinstance(sample, dict):
            by_song[str(sample.get("songId") or "")].append(sample)
    rankings = build_rankings(dict(by_song))
    proposals = collect_proposals(dict(by_song), rankings, policy)
    all_rows = [row for rows in proposals.values() for row in rows]
    if len(all_rows) < 20 or len({row["label"] for row in all_rows}) < 2:
        raise ValueError("Need at least 20 proposals with both outcome classes")

    feature_count = len(all_rows[0]["features"])
    trials: list[dict[str, Any]] = []
    thresholds = np.arange(0.35, 0.951, 0.025)
    for ridge in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        for positive_weight in (0.75, 1.0, 1.5, 2.0):
            heldout_probabilities: dict[str, np.ndarray] = {}
            for heldout_song in by_song:
                training = [row for song, rows in proposals.items() if song != heldout_song for row in rows]
                validation = proposals.get(heldout_song, [])
                if not validation or len({row["label"] for row in training}) < 2:
                    heldout_probabilities[heldout_song] = np.zeros(len(validation), dtype=float)
                    continue
                matrix = np.asarray([row["features"] for row in training], dtype=float)
                labels = np.asarray([row["label"] for row in training], dtype=float)
                standardized, means, scales = standardize(matrix)
                weights = fit_logistic(
                    standardized,
                    labels,
                    ridge=ridge,
                    positive_weight=positive_weight,
                )
                validation_matrix = np.asarray([row["features"] for row in validation], dtype=float)
                heldout_probabilities[heldout_song] = sigmoid(
                    ((validation_matrix - means) / scales) @ weights
                )
            for threshold in thresholds:
                evaluation = evaluate_threshold(
                    dict(by_song), proposals, heldout_probabilities, float(threshold)
                )
                evaluation.update({"ridge": ridge, "positiveWeight": positive_weight})
                trials.append(evaluation)

    safe = [
        row for row in trials
        if float(row["minimumFoldDelta"]) >= -0.002
        and int(row["acceptedImproved"]) >= int(row["acceptedWorsened"])
    ]
    candidates = safe or trials
    winner = max(
        candidates,
        key=lambda row: (
            float(row["membershipF1Delta"]),
            float(row["aggregate"]["exactSetShare"]),
            int(row["acceptedImproved"]) - int(row["acceptedWorsened"]),
            float(row["minimumFoldDelta"]),
            -int(row["accepted"]),
        ),
    )

    full_matrix = np.asarray([row["features"] for row in all_rows], dtype=float)
    full_labels = np.asarray([row["label"] for row in all_rows], dtype=float)
    standardized, means, scales = standardize(full_matrix)
    weights = fit_logistic(
        standardized,
        full_labels,
        ridge=float(winner["ridge"]),
        positive_weight=float(winner["positiveWeight"]),
    )
    names = proposal_feature_names(feature_count - len(BASE_FEATURE_NAMES))
    gate = {
        "id": args.gate_id,
        "type": "retrieval-action-abstention-logistic-v1",
        "featureNames": names,
        "weights": [round(float(value), 10) for value in weights],
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "threshold": winner["threshold"],
        "ridge": winner["ridge"],
        "positiveWeight": winner["positiveWeight"],
        "retrievalPolicy": {
            key: policy[key]
            for key in ("neighbors", "temperature", "incumbentPrior", "minimumGain", "maximumSizeChange")
        },
        "training": {
            "schema": "polymath-retrieval-action-abstention-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "policy": "whole-song leave-one-out proposal acceptance",
            "commercialUseAllowed": False,
        },
    }
    canonical = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    gate["gateSha256"] = hashlib.sha256(canonical).hexdigest()
    report = {
        "schema": "polymath-retrieval-action-abstention-report-v1",
        "proposalCounts": {song: len(rows) for song, rows in sorted(proposals.items())},
        "positiveProposals": int(sum(row["label"] for row in all_rows)),
        "negativeOrTiedProposals": int(sum(1.0 - row["label"] for row in all_rows)),
        "featureCount": feature_count,
        "safePolicyCount": len(safe),
        "winner": winner,
        "leaderboard": sorted(
            trials,
            key=lambda row: (
                float(row["membershipF1Delta"]),
                float(row["aggregate"]["exactSetShare"]),
                float(row["minimumFoldDelta"]),
            ),
            reverse=True,
        )[:30],
        "decision": "RESEARCH_ONLY",
    }
    atomic_json(args.output_gate.resolve(), gate)
    atomic_json(args.report.resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "leaderboard"}, indent=2))


if __name__ == "__main__":
    main()
