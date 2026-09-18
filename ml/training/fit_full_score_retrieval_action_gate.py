"""Fit a retrieval-action gate on every inference gesture.

The first abstention gate was trained only on candidate gestures that already
matched an authored target gesture.  At inference time, however, the mapper
also sees extra candidate gestures.  That mismatch made a conservative
held-out policy accept far more full-score edits than validation predicted.

This fitter reproduces the real inference path for every candidate gesture.
It labels each proposed replacement by its change to whole-song pitch-class
membership F1 after alignment, including unmatched candidate and target
gestures.  Model selection holds out a complete song, and the exported model
can additionally exclude one named song for an honest listening challenger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import normalize_source_notes  # noqa: E402

from .fit_chord_gesture_library import (  # noqa: E402
    aligned_target_left_notes,
    group_nearby_notes,
    membership_metrics,
    squared_distance,
)
from .fit_onset_gesture_ranker import trusted_windows  # noqa: E402
from .fit_paired_chord_gesture_mapper import (  # noqa: E402
    candidate_left_notes,
    canonical_manifest_pairs,
    match_group_sequences,
    paired_context,
    pitch_class_intervals,
)
from .fit_pianist_texture_decoder import fit_logistic, sigmoid, standardize  # noqa: E402
from .fit_retrieval_action_abstention_gate import (  # noqa: E402
    BASE_FEATURE_NAMES,
    proposal_feature_names,
    proposal_features,
)
from .fit_retrieval_action_chord_mapper import (  # noqa: E402
    decode_retrieval_action,
    load_json,
    source_supported_intervals,
)


def replacement_f1_delta(
    predictions: list[set[int]],
    targets: list[set[int]],
    index: int,
    proposal: set[int],
) -> float:
    """Return exact global membership-F1 change for one replacement."""

    incumbent = predictions[index]
    true_positive = sum(
        len(predicted & target)
        for predicted, target in zip(predictions, targets)
    )
    predicted_total = sum(len(predicted) for predicted in predictions)
    target_total = sum(len(target) for target in targets)
    baseline = 2.0 * true_positive / max(1, predicted_total + target_total)
    changed_true_positive = (
        true_positive
        - len(incumbent & targets[index])
        + len(proposal & targets[index])
    )
    changed_predicted_total = predicted_total - len(incumbent) + len(proposal)
    changed = 2.0 * changed_true_positive / max(
        1, changed_predicted_total + target_total
    )
    return changed - baseline


def build_full_song_data(
    pair: dict[str, Any],
    mapper: dict[str, Any],
    policy: dict[str, Any],
    *,
    reference_hand_split: int,
    match_tolerance: float,
    context_radius: float,
) -> dict[str, Any]:
    song_id = str(pair["id"])
    source = load_json(Path(pair["source"]).resolve())
    target = load_json(Path(pair["target"]).resolve())
    candidate = load_json(Path(pair["candidate"]).resolve())
    alignment = load_json(Path(pair["alignmentReport"]).resolve())
    candidate_end = (
        float(pair["candidateEndSeconds"])
        if pair.get("candidateEndSeconds") is not None
        else None
    )
    windows = trusted_windows(alignment)
    source_notes = normalize_source_notes(source.get("notes", []))
    source_times = [float(note["time"]) for note in source_notes]
    candidate_groups = group_nearby_notes(
        candidate_left_notes(candidate, windows, candidate_end)
    )
    target_groups = group_nearby_notes(
        [
            note
            for note in aligned_target_left_notes(
                target, alignment, reference_hand_split
            )
            if candidate_end is None or float(note["time"]) < candidate_end
        ]
    )
    matched_pairs = match_group_sequences(
        candidate_groups, target_groups, match_tolerance
    )
    target_by_candidate = {
        candidate_index: target_index
        for candidate_index, target_index in matched_pairs
    }
    matched_targets = {target_index for _candidate_index, target_index in matched_pairs}

    contexts: list[list[float]] = []
    anchors: list[int] = []
    predictions: list[set[int]] = []
    targets: list[set[int]] = []
    for candidate_index, _group in enumerate(candidate_groups):
        context, anchor, incumbent = paired_context(
            source_notes,
            source_times,
            candidate_groups,
            candidate_index,
            context_radius,
        )
        target_index = target_by_candidate.get(candidate_index)
        target_intervals = (
            pitch_class_intervals(target_groups[target_index], anchor)
            if target_index is not None
            else set()
        )
        contexts.append(context)
        anchors.append(anchor)
        predictions.append(set(incumbent))
        targets.append(target_intervals)

    # Missing authored gestures contribute false negatives to full-song F1.
    # Their interval origin is irrelevant because their prediction is empty.
    for target_index, target_group in enumerate(target_groups):
        if target_index in matched_targets:
            continue
        predictions.append(set())
        targets.append({int(note["midi"]) % 12 for note in target_group})

    training_samples = [
        sample
        for sample in mapper.get("samples") or []
        if isinstance(sample, dict) and str(sample.get("songId") or "") != song_id
    ]
    if not training_samples:
        raise ValueError(f"No cross-song retrieval samples remain for {song_id}")

    proposals: list[dict[str, Any]] = []
    for candidate_index, (context, incumbent) in enumerate(
        zip(contexts, predictions[: len(candidate_groups)])
    ):
        ranked = sorted(
            (
                (squared_distance(context, sample["context"]), sample)
                for sample in training_samples
            ),
            key=lambda item: item[0],
        )[:40]
        inference_sample = {
            "context": context,
            "incumbentIntervals": sorted(incumbent),
        }
        proposal, gain, diagnostics = decode_retrieval_action(
            ranked,
            incumbent,
            source_supported_intervals(inference_sample),
            neighbors=int(policy["neighbors"]),
            temperature=float(policy["temperature"]),
            incumbent_prior=float(policy["incumbentPrior"]),
            minimum_gain=float(policy["minimumGain"]),
            maximum_size_change=int(policy["maximumSizeChange"]),
            maximum_size=int(mapper.get("maximumChordSize", 3)),
        )
        if proposal == incumbent:
            continue
        delta = replacement_f1_delta(
            predictions, targets, candidate_index, proposal
        )
        proposals.append(
            {
                "candidateIndex": candidate_index,
                "time": round(float(candidate_groups[candidate_index][0]["time"]), 6),
                "matchedTarget": candidate_index in target_by_candidate,
                "incumbent": sorted(incumbent),
                "proposal": sorted(proposal),
                "label": float(delta > 1e-12),
                "delta": delta,
                "features": proposal_features(
                    inference_sample,
                    ranked,
                    incumbent,
                    proposal,
                    gain,
                    diagnostics,
                ),
            }
        )
    return {
        "songId": song_id,
        "predictions": predictions,
        "targets": targets,
        "candidateGestureCount": len(candidate_groups),
        "targetGestureCount": len(target_groups),
        "matchedGestureCount": len(matched_pairs),
        "proposals": proposals,
        "baseline": membership_metrics(predictions, targets),
    }


def evaluate_threshold(
    songs: dict[str, dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    all_predictions: list[set[int]] = []
    all_targets: list[set[int]] = []
    all_baselines: list[set[int]] = []
    folds: dict[str, Any] = {}
    accepted = improved = worsened = tied = 0
    for song_id, song in songs.items():
        baseline_predictions = [set(values) for values in song["predictions"]]
        predicted = [set(values) for values in baseline_predictions]
        fold_accepted = fold_improved = fold_worsened = fold_tied = 0
        for row, probability in zip(
            song["proposals"], probabilities.get(song_id, np.asarray([]))
        ):
            if float(probability) < threshold:
                continue
            predicted[int(row["candidateIndex"])] = set(row["proposal"])
            fold_accepted += 1
            fold_improved += int(float(row["delta"]) > 1e-12)
            fold_worsened += int(float(row["delta"]) < -1e-12)
            fold_tied += int(abs(float(row["delta"])) <= 1e-12)
        baseline_metrics = membership_metrics(
            baseline_predictions, song["targets"]
        )
        predicted_metrics = membership_metrics(predicted, song["targets"])
        folds[song_id] = {
            "candidateGestures": song["candidateGestureCount"],
            "targetGestures": song["targetGestureCount"],
            "matchedGestures": song["matchedGestureCount"],
            "proposals": len(song["proposals"]),
            "accepted": fold_accepted,
            "acceptedImproved": fold_improved,
            "acceptedWorsened": fold_worsened,
            "acceptedTied": fold_tied,
            "baseline": baseline_metrics,
            "predicted": predicted_metrics,
            "membershipF1Delta": round(
                predicted_metrics["membershipF1"]
                - baseline_metrics["membershipF1"],
                6,
            ),
        }
        accepted += fold_accepted
        improved += fold_improved
        worsened += fold_worsened
        tied += fold_tied
        all_predictions.extend(predicted)
        all_baselines.extend(baseline_predictions)
        all_targets.extend(song["targets"])
    baseline = membership_metrics(all_baselines, all_targets)
    predicted_metrics = membership_metrics(all_predictions, all_targets)
    return {
        "threshold": round(float(threshold), 6),
        "aggregate": predicted_metrics,
        "baseline": baseline,
        "membershipF1Delta": round(
            predicted_metrics["membershipF1"] - baseline["membershipF1"], 6
        ),
        "minimumFoldDelta": round(
            min(float(fold["membershipF1Delta"]) for fold in folds.values()), 6
        ),
        "accepted": accepted,
        "acceptedImproved": improved,
        "acceptedWorsened": worsened,
        "acceptedTied": tied,
        "folds": folds,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--exclude-final-song", default="")
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--match-tolerance", type=float, default=0.14)
    parser.add_argument("--context-radius", type=float, default=0.35)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    mapper_path = args.mapper.resolve()
    mapper = load_json(mapper_path)
    retrieval_report = load_json(args.retrieval_report.resolve())
    policy = {
        key: retrieval_report["winner"][key]
        for key in (
            "neighbors",
            "temperature",
            "incumbentPrior",
            "minimumGain",
            "maximumSizeChange",
        )
    }
    pairs = canonical_manifest_pairs(load_json(manifest_path))
    songs = {
        str(pair["id"]): build_full_song_data(
            pair,
            mapper,
            policy,
            reference_hand_split=args.reference_hand_split,
            match_tolerance=args.match_tolerance,
            context_radius=args.context_radius,
        )
        for pair in pairs
    }
    all_rows = [row for song in songs.values() for row in song["proposals"]]
    if len(songs) < 3 or len(all_rows) < 20:
        raise ValueError("Need at least three songs and twenty full-score proposals")

    trials: list[dict[str, Any]] = []
    for ridge in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        for positive_weight in (0.5, 0.75, 1.0, 1.5, 2.0):
            heldout_probabilities: dict[str, np.ndarray] = {}
            for heldout_song, validation_song in songs.items():
                training = [
                    row
                    for song_id, song in songs.items()
                    if song_id != heldout_song
                    for row in song["proposals"]
                ]
                validation = validation_song["proposals"]
                if (
                    not validation
                    or len({float(row["label"]) for row in training}) < 2
                ):
                    heldout_probabilities[heldout_song] = np.zeros(
                        len(validation), dtype=float
                    )
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
                validation_matrix = np.asarray(
                    [row["features"] for row in validation], dtype=float
                )
                heldout_probabilities[heldout_song] = sigmoid(
                    ((validation_matrix - means) / scales) @ weights
                )
            for threshold in np.arange(0.40, 0.976, 0.025):
                evaluation = evaluate_threshold(
                    songs, heldout_probabilities, float(threshold)
                )
                evaluation.update(
                    {"ridge": ridge, "positiveWeight": positive_weight}
                )
                trials.append(evaluation)

    safe = [
        trial
        for trial in trials
        if float(trial["minimumFoldDelta"]) >= 0.0
        and int(trial["acceptedImproved"]) >= int(trial["acceptedWorsened"])
    ]
    winner = max(
        safe or trials,
        key=lambda trial: (
            bool(trial in safe),
            float(trial["membershipF1Delta"]),
            float(trial["aggregate"]["exactSetShare"]),
            int(trial["acceptedImproved"]) - int(trial["acceptedWorsened"]),
            float(trial["minimumFoldDelta"]),
            -int(trial["accepted"]),
        ),
    )

    final_rows = [
        row
        for song_id, song in songs.items()
        if not args.exclude_final_song or song_id != args.exclude_final_song
        for row in song["proposals"]
    ]
    if len({float(row["label"]) for row in final_rows}) < 2:
        raise ValueError("Final gate training requires both outcome classes")
    matrix = np.asarray([row["features"] for row in final_rows], dtype=float)
    labels = np.asarray([row["label"] for row in final_rows], dtype=float)
    standardized, means, scales = standardize(matrix)
    weights = fit_logistic(
        standardized,
        labels,
        ridge=float(winner["ridge"]),
        positive_weight=float(winner["positiveWeight"]),
    )
    names = proposal_feature_names(len(final_rows[0]["features"]) - len(BASE_FEATURE_NAMES))
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
        "retrievalPolicy": policy,
        "training": {
            "schema": "polymath-full-score-retrieval-action-gate-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "mapper": str(mapper_path),
            "policy": "all candidate gestures; whole-song leave-one-out",
            "finalTrainingExcludedSongId": args.exclude_final_song or None,
            "commercialUseAllowed": False,
        },
    }
    canonical = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    gate["gateSha256"] = hashlib.sha256(canonical).hexdigest()
    report = {
        "schema": "polymath-full-score-retrieval-action-gate-report-v1",
        "songCounts": {
            song_id: {
                "candidateGestures": song["candidateGestureCount"],
                "targetGestures": song["targetGestureCount"],
                "matchedGestures": song["matchedGestureCount"],
                "proposals": len(song["proposals"]),
                "positiveProposals": int(
                    sum(float(row["label"]) for row in song["proposals"])
                ),
            }
            for song_id, song in sorted(songs.items())
        },
        "safePolicyCount": len(safe),
        "finalTrainingExamples": len(final_rows),
        "finalTrainingExcludedSongId": args.exclude_final_song or None,
        "winner": winner,
        "leaderboard": sorted(
            trials,
            key=lambda trial: (
                float(trial["membershipF1Delta"]),
                float(trial["aggregate"]["exactSetShare"]),
                float(trial["minimumFoldDelta"]),
            ),
            reverse=True,
        )[:30],
        "decision": "RESEARCH_ONLY",
    }
    atomic_json(args.output_gate.resolve(), gate)
    atomic_json(args.report.resolve(), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "leaderboard"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
