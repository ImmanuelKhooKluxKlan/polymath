"""Fit a conservative variable-size pianist chord action retriever.

The first paired chord mapper froze chord size.  That is safe, but it cannot
reproduce an authored one-note versus two-note accompaniment decision.  This
research fitter treats complete pianist interval sets from nearby training
contexts as candidate *actions*.  It scores those actions by their expected
set F1 among cross-song neighbours, requires local source support, and permits
at most one note of size change.

Every model-selection prediction is leave-one-song-out.  Absolute song time,
song identity, and held-out reference answers are absent at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .fit_chord_gesture_library import membership_metrics
from .fit_paired_chord_gesture_mapper import set_f1


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def source_supported_intervals(sample: dict[str, Any]) -> set[int]:
    context = list(sample.get("context") or [])
    supported = {
        interval for interval, value in enumerate(context[:12]) if float(value) > 1e-9
    }
    supported.update(int(value) % 12 for value in sample.get("incumbentIntervals") or [])
    return supported


def decode_retrieval_action(
    ranked: list[tuple[float, dict[str, Any]]],
    incumbent: set[int],
    available: set[int],
    *,
    neighbors: int,
    temperature: float,
    incumbent_prior: float,
    minimum_gain: float,
    maximum_size_change: int,
    maximum_size: int = 3,
) -> tuple[set[int], float, dict[str, Any]]:
    """Choose the source-supported action with best neighbour-expected F1."""

    if not incumbent or not ranked:
        return incumbent, 0.0, {"actions": 1, "reason": "no-evidence"}
    selected = ranked[: max(1, int(neighbors))]
    weighted: list[tuple[float, set[int]]] = []
    interval_votes = [0.0] * 12
    total_weight = 0.0
    for distance, sample in selected:
        weight = max(0.0, float(sample.get("weight", 1.0))) * math.exp(
            -float(distance) / max(1e-6, float(temperature))
        )
        target = {int(value) % 12 for value in sample.get("targetIntervals") or []}
        if not target or weight <= 0:
            continue
        weighted.append((weight, target))
        total_weight += weight
        for interval in target:
            interval_votes[interval] += weight
    if total_weight <= 0:
        return incumbent, 0.0, {"actions": 1, "reason": "zero-weight"}

    minimum_size = max(1, len(incumbent) - max(0, int(maximum_size_change)))
    maximum_allowed_size = min(
        max(1, int(maximum_size)),
        len(available),
        len(incumbent) + max(0, int(maximum_size_change)),
    )
    actions: set[frozenset[int]] = {frozenset(incumbent)}
    for _weight, target in weighted:
        if (
            minimum_size <= len(target) <= maximum_allowed_size
            and target.issubset(available)
        ):
            actions.add(frozenset(target))
        supported_target = target & available
        if minimum_size <= len(supported_target) <= maximum_allowed_size:
            actions.add(frozenset(supported_target))
    ranked_intervals = sorted(
        available,
        key=lambda interval: (interval_votes[interval], interval in incumbent, -interval),
        reverse=True,
    )
    for size in range(minimum_size, maximum_allowed_size + 1):
        action = frozenset(ranked_intervals[:size])
        if len(action) == size:
            actions.add(action)

    def expected_f1(action: set[int] | frozenset[int]) -> float:
        return sum(weight * set_f1(set(action), target) for weight, target in weighted) / total_weight

    incumbent_score = expected_f1(incumbent) + max(0.0, float(incumbent_prior))
    scored = sorted(
        ((expected_f1(action), action) for action in actions),
        key=lambda item: (
            item[0],
            item[1] == frozenset(incumbent),
            -abs(len(item[1]) - len(incumbent)),
            -len(item[1]),
            tuple(-value for value in sorted(item[1])),
        ),
        reverse=True,
    )
    best_score, best = scored[0]
    gain = best_score - incumbent_score
    if best == frozenset(incumbent) or gain < max(0.0, float(minimum_gain)):
        return incumbent, gain, {
            "actions": len(actions),
            "incumbentExpectedF1": round(incumbent_score, 6),
            "bestExpectedF1": round(best_score, 6),
            "reason": "incumbent-or-insufficient-gain",
        }
    return set(best), gain, {
        "actions": len(actions),
        "incumbentExpectedF1": round(incumbent_score, 6),
        "bestExpectedF1": round(best_score, 6),
        "reason": "accepted",
    }


def build_rankings(
    by_song: dict[str, list[dict[str, Any]]],
) -> dict[str, list[list[tuple[float, dict[str, Any]]]]]:
    rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]] = {}
    for song_id, held_out in by_song.items():
        training = [
            sample
            for other_id, samples in by_song.items()
            if other_id != song_id
            for sample in samples
        ]
        train_contexts = np.asarray([sample["context"] for sample in training], dtype=np.float32)
        held_contexts = np.asarray([sample["context"] for sample in held_out], dtype=np.float32)
        distances = np.sum(
            (held_contexts[:, None, :] - train_contexts[None, :, :]) ** 2,
            axis=2,
        )
        orders = np.argsort(distances, axis=1)
        rankings[song_id] = [
            [
                (float(distances[row, index]), training[int(index)])
                for index in order[:40]
            ]
            for row, order in enumerate(orders)
        ]
    return rankings


def evaluate_policy(
    by_song: dict[str, list[dict[str, Any]]],
    rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]],
    *,
    neighbors: int,
    temperature: float,
    incumbent_prior: float,
    minimum_gain: float,
    maximum_size_change: int,
) -> dict[str, Any]:
    all_predictions: list[set[int]] = []
    all_incumbents: list[set[int]] = []
    all_targets: list[set[int]] = []
    folds: dict[str, Any] = {}
    total_changes = 0
    total_improved = 0
    total_worsened = 0
    size_changes = 0
    for song_id, samples in by_song.items():
        predictions: list[set[int]] = []
        incumbents: list[set[int]] = []
        targets: list[set[int]] = []
        changes = improved = worsened = fold_size_changes = 0
        gains: list[float] = []
        for sample, ranked in zip(samples, rankings[song_id]):
            incumbent = {int(value) % 12 for value in sample["incumbentIntervals"]}
            target = {int(value) % 12 for value in sample["targetIntervals"]}
            prediction, gain, _diagnostics = decode_retrieval_action(
                ranked,
                incumbent,
                source_supported_intervals(sample),
                neighbors=neighbors,
                temperature=temperature,
                incumbent_prior=incumbent_prior,
                minimum_gain=minimum_gain,
                maximum_size_change=maximum_size_change,
            )
            if prediction != incumbent:
                changes += 1
                gains.append(gain)
                fold_size_changes += int(len(prediction) != len(incumbent))
                delta = set_f1(prediction, target) - set_f1(incumbent, target)
                improved += int(delta > 1e-9)
                worsened += int(delta < -1e-9)
            predictions.append(prediction)
            incumbents.append(incumbent)
            targets.append(target)
        predicted_metrics = membership_metrics(predictions, targets)
        baseline_metrics = membership_metrics(incumbents, targets)
        delta = predicted_metrics["membershipF1"] - baseline_metrics["membershipF1"]
        folds[song_id] = {
            "samples": len(samples),
            "changes": changes,
            "sizeChanges": fold_size_changes,
            "improvedChanges": improved,
            "worsenedChanges": worsened,
            "meanAcceptedGain": round(sum(gains) / max(1, len(gains)), 6),
            "baseline": baseline_metrics,
            "predicted": predicted_metrics,
            "membershipF1Delta": round(delta, 6),
        }
        total_changes += changes
        size_changes += fold_size_changes
        total_improved += improved
        total_worsened += worsened
        all_predictions.extend(predictions)
        all_incumbents.extend(incumbents)
        all_targets.extend(targets)
    aggregate = membership_metrics(all_predictions, all_targets)
    baseline = membership_metrics(all_incumbents, all_targets)
    minimum_fold_delta = min(
        float(fold["membershipF1Delta"]) for fold in folds.values()
    )
    return {
        "neighbors": neighbors,
        "temperature": temperature,
        "incumbentPrior": incumbent_prior,
        "minimumGain": minimum_gain,
        "maximumSizeChange": maximum_size_change,
        "aggregate": aggregate,
        "baseline": baseline,
        "membershipF1Delta": round(
            aggregate["membershipF1"] - baseline["membershipF1"], 6
        ),
        "minimumFoldDelta": round(minimum_fold_delta, 6),
        "changes": total_changes,
        "sizeChanges": size_changes,
        "improvedChanges": total_improved,
        "worsenedChanges": total_worsened,
        "folds": folds,
    }


def atomic_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        + ("" if compact else "\n"),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mapper", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()

    base_path = args.base_mapper.resolve()
    base = load_json(base_path)
    samples = [sample for sample in base.get("samples") or [] if isinstance(sample, dict)]
    by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_song[str(sample.get("songId") or "")].append(sample)
    if len(by_song) < 3 or any(not values for values in by_song.values()):
        raise ValueError("At least three non-empty songs are required")
    rankings = build_rankings(dict(by_song))

    grid: list[dict[str, Any]] = []
    for neighbors in (3, 5, 9, 15, 25, 40):
        for temperature in (0.03, 0.06, 0.12, 0.24, 0.48):
            for incumbent_prior in (0.0, 0.01, 0.03, 0.06, 0.10):
                for minimum_gain in (0.0, 0.01, 0.025, 0.05, 0.08):
                    for maximum_size_change in (0, 1):
                        grid.append(
                            evaluate_policy(
                                dict(by_song),
                                rankings,
                                neighbors=neighbors,
                                temperature=temperature,
                                incumbent_prior=incumbent_prior,
                                minimum_gain=minimum_gain,
                                maximum_size_change=maximum_size_change,
                            )
                        )
    safe = [
        row
        for row in grid
        if float(row["minimumFoldDelta"]) >= -0.002
        and int(row["improvedChanges"]) >= int(row["worsenedChanges"])
    ]
    candidates = safe or grid
    winner = max(
        candidates,
        key=lambda row: (
            float(row["membershipF1Delta"]),
            float(row["aggregate"]["exactSetShare"]),
            int(row["improvedChanges"]) - int(row["worsenedChanges"]),
            float(row["minimumFoldDelta"]),
            -int(row["changes"]),
        ),
    )
    profile = {
        "id": args.profile_id,
        "type": "paired-retrieval-action-chord-mapper-v2",
        "contextRadiusSeconds": float(base.get("contextRadiusSeconds", 0.35)),
        "neighbors": winner["neighbors"],
        "temperature": winner["temperature"],
        "incumbentPrior": winner["incumbentPrior"],
        "minimumGain": winner["minimumGain"],
        "maximumSizeChange": winner["maximumSizeChange"],
        "maximumChordSize": 3,
        "sourceSupportRequired": True,
        "onsetsFrozen": True,
        "samples": samples,
        "training": {
            "schema": "polymath-retrieval-action-chord-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "baseMapper": str(base_path),
            "selection": "whole-song leave-one-out; no held-out reference at inference",
            "commercialUseAllowed": False,
            "warning": "Private research profile; requires a sealed listening win before promotion.",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    leaderboard = sorted(
        grid,
        key=lambda row: (
            float(row["membershipF1Delta"]),
            float(row["aggregate"]["exactSetShare"]),
            float(row["minimumFoldDelta"]),
        ),
        reverse=True,
    )[:30]
    report = {
        "schema": "polymath-retrieval-action-chord-report-v1",
        "baseMapper": str(base_path),
        "profile": str(args.output_profile.resolve()),
        "sampleCounts": {song: len(values) for song, values in sorted(by_song.items())},
        "safePolicyCount": len(safe),
        "winner": winner,
        "leaderboard": leaderboard,
        "decision": "RESEARCH_ONLY",
    }
    atomic_json(args.output_profile.resolve(), profile, compact=True)
    atomic_json(args.report.resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "leaderboard"}, indent=2))


if __name__ == "__main__":
    main()
