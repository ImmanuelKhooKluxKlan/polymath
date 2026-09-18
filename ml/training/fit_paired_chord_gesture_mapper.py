"""Learn conservative pianist chord corrections from paired arranger outputs.

The earlier chord library tried to infer an authored left-hand shape from the
raw mixture alone.  This fitter uses the information that is actually
available in production after 3B has run:

* local non-vocal harmonic evidence from the uploaded song;
* the pitch classes, size and confidence of 3B's incumbent gesture;
* neighbouring incumbent gestures and their inter-onset timing.

Targets are approved pianist gestures mapped to the source timeline.  Every
reported validation fold leaves a complete song out.  The default decoder
keeps chord size and onset time frozen and falls back to the incumbent unless
cross-song neighbours show a sufficient vote advantage.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import instrument_family, normalize_source_notes  # noqa: E402

try:
    from .fit_chord_gesture_library import (
        aligned_target_left_notes,
        group_nearby_notes,
        harmonic_context,
        load_json,
        membership_metrics,
    )
    from .fit_onset_gesture_ranker import in_source_windows, trusted_windows
except ImportError:  # pragma: no cover - direct CLI execution
    from fit_chord_gesture_library import (  # type: ignore
        aligned_target_left_notes,
        group_nearby_notes,
        harmonic_context,
        load_json,
        membership_metrics,
    )
    from fit_onset_gesture_ranker import in_source_windows, trusted_windows  # type: ignore


def candidate_left_notes(
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    candidate_end: float | None,
) -> list[dict[str, Any]]:
    payload_items = [item for item in payload.get("notes", []) if isinstance(item, dict)]
    has_explicit_hands = any(
        str(item.get("hand") or "").strip().lower() in {"left", "right"}
        for item in payload_items
    )
    result: list[dict[str, Any]] = []
    for item in payload_items:
        try:
            midi = int(round(float(item["midi"])))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
        except (KeyError, TypeError, ValueError):
            continue
        if candidate_end is not None and time >= candidate_end:
            continue
        if not in_source_windows(time, windows):
            continue
        if (
            has_explicit_hands
            and str(item.get("hand") or "").strip().lower() != "left"
        ):
            continue
        if not has_explicit_hands and midi >= 72:
            continue
        if str(item.get("arrangementRole") or "") == "melody":
            continue
        if instrument_family(str(item.get("sourceInstrument") or "")) == "voice":
            continue
        result.append({**item, "midi": midi, "time": time})
    return sorted(result, key=lambda note: (float(note["time"]), int(note["midi"])))


def match_group_sequences(
    candidate_groups: list[list[dict[str, Any]]],
    target_groups: list[list[dict[str, Any]]],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Greedily form a one-to-one time-only match between gesture sequences."""

    possibilities: list[tuple[float, int, int]] = []
    target_times = [float(group[0]["time"]) for group in target_groups]
    for candidate_index, group in enumerate(candidate_groups):
        time = float(group[0]["time"])
        position = bisect.bisect_left(target_times, time)
        for target_index in range(
            max(0, position - 4), min(len(target_groups), position + 5)
        ):
            distance = abs(target_times[target_index] - time)
            if distance <= tolerance:
                possibilities.append((distance, candidate_index, target_index))
    used_candidate: set[int] = set()
    used_target: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _distance, candidate_index, target_index in sorted(possibilities):
        if candidate_index in used_candidate or target_index in used_target:
            continue
        used_candidate.add(candidate_index)
        used_target.add(target_index)
        matches.append((candidate_index, target_index))
    return sorted(matches)


def pitch_class_intervals(group: list[dict[str, Any]], anchor: int) -> set[int]:
    return {(int(note["midi"]) % 12 - anchor) % 12 for note in group}


def binary_intervals(intervals: set[int], weight: float = 1.0) -> list[float]:
    return [weight if interval in intervals else 0.0 for interval in range(12)]


def paired_context(
    source_notes: list[dict[str, Any]],
    source_times: list[float],
    groups: list[list[dict[str, Any]]],
    group_index: int,
    radius: float,
) -> tuple[list[float], int, set[int]]:
    """Describe a 3B gesture and its neighbours using inference-safe inputs."""

    group = groups[group_index]
    onset = float(group[0]["time"])
    local, anchor, _support = harmonic_context(
        source_notes, source_times, onset, radius
    )
    incumbent = pitch_class_intervals(group, anchor)

    previous = groups[group_index - 1] if group_index > 0 else None
    following = groups[group_index + 1] if group_index + 1 < len(groups) else None
    previous_intervals = (
        pitch_class_intervals(previous, anchor) if previous is not None else set()
    )
    following_intervals = (
        pitch_class_intervals(following, anchor) if following is not None else set()
    )
    previous_gap = (
        onset - float(previous[0]["time"]) if previous is not None else 0.0
    )
    next_gap = (
        float(following[0]["time"]) - onset if following is not None else 0.0
    )
    gap_total = max(1e-6, previous_gap + next_gap)

    velocities = [float(note.get("velocity", 0.7)) for note in group]
    durations = [float(note.get("duration", 0.2)) for note in group]
    probabilities = [float(note.get("selectionProbability", 0.5)) for note in group]
    bass_share = sum(
        str(note.get("arrangementRole") or "") == "bass" for note in group
    ) / max(1, len(group))
    gesture_stats = [
        min(1.0, len(incumbent) / 3.0),
        min(1.0, max(0.0, median(velocities))),
        min(1.0, max(0.0, median(durations)) / 2.0),
        min(1.0, max(0.0, median(probabilities))),
        bass_share,
    ]
    rhythm = [
        0.5 * min(1.0, max(0.0, previous_gap) / 1.5),
        0.5 * min(1.0, max(0.0, next_gap) / 1.5),
        0.5 * (previous_gap / gap_total if previous is not None else 0.0),
        0.5 * (next_gap / gap_total if following is not None else 0.0),
        0.5 if previous is None else 0.0,
        0.5 if following is None else 0.0,
    ]
    features = (
        local
        + binary_intervals(incumbent, 0.9)
        + gesture_stats
        + binary_intervals(previous_intervals, 0.45)
        + binary_intervals(following_intervals, 0.45)
        + rhythm
    )
    return features, anchor, incumbent


def build_song_samples(
    pair: dict[str, Any],
    reference_hand_split: int,
    match_tolerance: float,
    context_radius: float,
) -> list[dict[str, Any]]:
    source = load_json(Path(pair["source"]).resolve())
    target = load_json(Path(pair["target"]).resolve())
    candidate = load_json(Path(pair["candidate"]).resolve())
    report = load_json(Path(pair["alignmentReport"]).resolve())
    windows = trusted_windows(report)
    candidate_end = (
        float(pair["candidateEndSeconds"])
        if pair.get("candidateEndSeconds") is not None
        else None
    )
    source_notes = normalize_source_notes(source.get("notes", []))
    source_times = [float(note["time"]) for note in source_notes]
    candidate_groups = group_nearby_notes(
        candidate_left_notes(candidate, windows, candidate_end)
    )
    target_groups = group_nearby_notes(
        [
            note
            for note in aligned_target_left_notes(
                target, report, reference_hand_split
            )
            if candidate_end is None or float(note["time"]) < candidate_end
        ]
    )
    matches = match_group_sequences(candidate_groups, target_groups, match_tolerance)
    samples: list[dict[str, Any]] = []
    for candidate_index, target_index in matches:
        group = candidate_groups[candidate_index]
        target_group = target_groups[target_index]
        context, anchor, incumbent = paired_context(
            source_notes,
            source_times,
            candidate_groups,
            candidate_index,
            context_radius,
        )
        target_intervals = pitch_class_intervals(target_group, anchor)
        if not incumbent or not target_intervals:
            continue
        samples.append(
            {
                "songId": str(pair["id"]),
                "time": round(float(group[0]["time"]), 6),
                "targetTime": round(float(target_group[0]["time"]), 6),
                "context": [round(value, 7) for value in context],
                "incumbentIntervals": sorted(incumbent),
                "targetIntervals": sorted(target_intervals),
                "size": len(incumbent),
                "weight": max(0.01, float(pair.get("weight", 1.0))),
            }
        )
    return samples


def canonical_manifest_pairs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both training-pair and structural-audit manifests.

    The structural audit deliberately calls the approved score ``reference``
    and the time map ``alignment``.  Earlier fitters used ``target`` and
    ``alignmentReport``.  Normalising the two schemas here prevents researchers
    from hand-copying paths into a second manifest and, more importantly,
    keeps every chord experiment attached to the exact frozen candidate that
    was audited.
    """

    rows = manifest.get("pairs")
    if not isinstance(rows, list):
        rows = manifest.get("songs")
    if not isinstance(rows, list):
        return []
    canonical: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item.setdefault("target", item.get("reference"))
        item.setdefault("alignmentReport", item.get("alignment"))
        if not item.get("candidate") and item.get("baseline"):
            item["candidate"] = item["baseline"]
        required = ("id", "source", "target", "candidate", "alignmentReport")
        if any(not item.get(field) for field in required):
            continue
        canonical.append(item)
    return canonical


def ranked_vote_probabilities(
    ranked: list[tuple[float, dict[str, Any]]],
    neighbors: int,
    temperature: float,
) -> list[float]:
    votes = [0.0] * 12
    total = 0.0
    for distance, sample in ranked[:neighbors]:
        weight = float(sample.get("weight", 1.0)) * math.exp(
            -distance / max(1e-6, temperature)
        )
        total += weight
        for interval in sample["targetIntervals"]:
            votes[int(interval) % 12] += weight
    return [vote / max(1e-9, total) for vote in votes]


def decode_conservative_correction(
    votes: list[float],
    incumbent: set[int],
    incumbent_prior: float,
    minimum_gain: float,
) -> tuple[set[int], float]:
    size = max(1, min(3, len(incumbent)))
    adjusted = [
        vote + (incumbent_prior if interval in incumbent else 0.0)
        for interval, vote in enumerate(votes)
    ]
    predicted = set(
        sorted(
            range(12),
            key=lambda interval: (adjusted[interval], -interval),
            reverse=True,
        )[:size]
    )
    incumbent_score = sum(votes[interval] for interval in incumbent) / len(incumbent)
    predicted_score = sum(votes[interval] for interval in predicted) / len(predicted)
    gain = predicted_score - incumbent_score
    if predicted == incumbent or gain < minimum_gain:
        return incumbent, gain
    return predicted, gain


def set_f1(first: set[int], second: set[int]) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return 2.0 * len(first & second) / (len(first) + len(second))


def evaluate_policy(
    by_song: dict[str, list[dict[str, Any]]],
    rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]],
    neighbors: int,
    temperature: float,
    incumbent_prior: float,
    minimum_gain: float,
) -> dict[str, Any]:
    all_predictions: list[set[int]] = []
    all_incumbents: list[set[int]] = []
    all_targets: list[set[int]] = []
    folds: dict[str, Any] = {}
    total_changes = 0
    total_improved = 0
    total_worsened = 0
    for song_id, samples in by_song.items():
        predictions: list[set[int]] = []
        incumbents: list[set[int]] = []
        targets: list[set[int]] = []
        improved = 0
        worsened = 0
        changes = 0
        gains: list[float] = []
        for sample, ranked in zip(samples, rankings[song_id]):
            incumbent = set(sample["incumbentIntervals"])
            target = set(sample["targetIntervals"])
            votes = ranked_vote_probabilities(ranked, neighbors, temperature)
            predicted, gain = decode_conservative_correction(
                votes, incumbent, incumbent_prior, minimum_gain
            )
            if predicted != incumbent:
                changes += 1
                gains.append(gain)
                delta = set_f1(predicted, target) - set_f1(incumbent, target)
                improved += delta > 1e-9
                worsened += delta < -1e-9
            predictions.append(predicted)
            incumbents.append(incumbent)
            targets.append(target)
        predicted_metrics = membership_metrics(predictions, targets)
        baseline_metrics = membership_metrics(incumbents, targets)
        folds[song_id] = {
            "samples": len(samples),
            "changes": changes,
            "improvedChanges": improved,
            "worsenedChanges": worsened,
            "meanAcceptedVoteGain": round(sum(gains) / max(1, len(gains)), 6),
            "baseline": baseline_metrics,
            "predicted": predicted_metrics,
            "membershipF1Delta": round(
                predicted_metrics["membershipF1"] - baseline_metrics["membershipF1"],
                6,
            ),
        }
        total_changes += changes
        total_improved += improved
        total_worsened += worsened
        all_predictions.extend(predictions)
        all_incumbents.extend(incumbents)
        all_targets.extend(targets)
    aggregate = membership_metrics(all_predictions, all_targets)
    baseline = membership_metrics(all_incumbents, all_targets)
    return {
        "neighbors": neighbors,
        "temperature": temperature,
        "incumbentPrior": incumbent_prior,
        "minimumGain": minimum_gain,
        "aggregate": aggregate,
        "baseline": baseline,
        "membershipF1Delta": round(
            aggregate["membershipF1"] - baseline["membershipF1"], 6
        ),
        "changes": total_changes,
        "improvedChanges": total_improved,
        "worsenedChanges": total_worsened,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--match-tolerance", type=float, default=0.14)
    parser.add_argument("--context-radius", type=float, default=0.35)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    pairs = canonical_manifest_pairs(manifest)
    by_song = {
        str(pair["id"]): build_song_samples(
            pair,
            args.reference_hand_split,
            args.match_tolerance,
            args.context_radius,
        )
        for pair in pairs
    }
    if len(by_song) < 3 or any(not samples for samples in by_song.values()):
        raise ValueError("At least three non-empty songs are required.")

    rankings: dict[str, list[list[tuple[float, dict[str, Any]]]]] = {}
    for song_id, held_out in by_song.items():
        training = [
            sample
            for other_id, samples in by_song.items()
            if other_id != song_id
            for sample in samples
        ]
        train_contexts = np.asarray(
            [sample["context"] for sample in training], dtype=np.float32
        )
        held_contexts = np.asarray(
            [sample["context"] for sample in held_out], dtype=np.float32
        )
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

    grid: list[dict[str, Any]] = []
    for neighbors in (3, 5, 9, 15, 25, 40):
        for temperature in (0.03, 0.06, 0.12, 0.24, 0.48):
            for incumbent_prior in (0.0, 0.05, 0.1, 0.2, 0.35):
                for minimum_gain in (0.0, 0.02, 0.05, 0.08, 0.12, 0.18):
                    grid.append(
                        evaluate_policy(
                            by_song,
                            rankings,
                            neighbors,
                            temperature,
                            incumbent_prior,
                            minimum_gain,
                        )
                    )
    winner = max(
        grid,
        key=lambda row: (
            row["membershipF1Delta"],
            row["aggregate"]["exactSetShare"],
            row["improvedChanges"] - row["worsenedChanges"],
            -row["changes"],
            -row["neighbors"],
        ),
    )
    samples = [sample for values in by_song.values() for sample in values]
    profile = {
        "id": args.profile_id,
        "type": "paired-incumbent-sequence-chord-mapper-v1",
        "contextRadiusSeconds": args.context_radius,
        "matchToleranceSeconds": args.match_tolerance,
        "neighbors": winner["neighbors"],
        "temperature": winner["temperature"],
        "incumbentPrior": winner["incumbentPrior"],
        "minimumGain": winner["minimumGain"],
        "chordSizeFrozen": True,
        "onsetsFrozen": True,
        "samples": samples,
        "training": {
            "schema": "polymath-paired-chord-gesture-training-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "policy": "whole-song leave-one-out; paired incumbent and pianist gestures; source-only inference features",
            "commercialUseAllowed": False,
            "warning": "Private research profile. Do not deploy without rights clearance and a sealed listening win.",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    leaderboard = sorted(
        grid,
        key=lambda row: (
            row["membershipF1Delta"],
            row["aggregate"]["exactSetShare"],
            row["improvedChanges"] - row["worsenedChanges"],
        ),
        reverse=True,
    )[:20]
    report = {
        "schema": "polymath-paired-chord-gesture-report-v1",
        "manifest": str(manifest_path),
        "profile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "sampleCounts": {song: len(values) for song, values in by_song.items()},
        "featureCount": len(samples[0]["context"]),
        "winner": winner,
        "leaderboard": leaderboard,
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "leaderboard"}, indent=2))


if __name__ == "__main__":
    main()
