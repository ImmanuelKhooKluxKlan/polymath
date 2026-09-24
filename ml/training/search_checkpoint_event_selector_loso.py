"""Learn an inference-safe checkpoint event selector with whole-song LOSO.

Each event proposal is described only by information available at inference:
checkpoint identity, cross-checkpoint agreement, local note density, register,
duration, velocity, chord size, and repeated-note spacing.  Reference MIDI is
used to label proposals on training songs and to score the held-out song only.

The lightweight logistic model is implemented with NumPy so the research tool
does not add a production dependency on scikit-learn.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_piano_arranger import greedy_matches, load_json
from .search_checkpoint_consensus_loso import (
    aggregate_metrics,
    compact_metrics,
    load_song,
    one_to_one_matches,
)
from .transcription_accuracy_scorecard import atomic_json


VARIANTS = ("incumbent", "checkpointA", "checkpointB")


def cluster_variants(
    variants: dict[str, list[dict[str, Any]]], radius_seconds: float
) -> list[dict[str, Any]]:
    """Cluster same-pitch proposals without merging notes from one checkpoint."""

    incumbent = variants["incumbent"]
    checkpoint_a = variants["checkpointA"]
    checkpoint_b = variants["checkpointB"]
    clusters = [{"members": {"incumbent": note}} for note in incumbent]

    incumbent_a = one_to_one_matches(incumbent, checkpoint_a, radius_seconds)
    matched_a: set[int] = set()
    a_cluster: dict[int, int] = {}
    for incumbent_index, a_index in incumbent_a:
        clusters[incumbent_index]["members"]["checkpointA"] = checkpoint_a[a_index]
        matched_a.add(a_index)
        a_cluster[a_index] = incumbent_index
    for a_index, note in enumerate(checkpoint_a):
        if a_index in matched_a:
            continue
        a_cluster[a_index] = len(clusters)
        clusters.append({"members": {"checkpointA": note}})

    incumbent_b = one_to_one_matches(incumbent, checkpoint_b, radius_seconds)
    matched_b: set[int] = set()
    for incumbent_index, b_index in incumbent_b:
        clusters[incumbent_index]["members"]["checkpointB"] = checkpoint_b[b_index]
        matched_b.add(b_index)

    remaining_a_indices = [index for index in range(len(checkpoint_a)) if index not in matched_a]
    remaining_b_indices = [index for index in range(len(checkpoint_b)) if index not in matched_b]
    remaining_a = [checkpoint_a[index] for index in remaining_a_indices]
    remaining_b = [checkpoint_b[index] for index in remaining_b_indices]
    paired_remaining_b: set[int] = set()
    for local_a, local_b in one_to_one_matches(remaining_a, remaining_b, radius_seconds):
        a_index = remaining_a_indices[local_a]
        b_index = remaining_b_indices[local_b]
        clusters[a_cluster[a_index]]["members"]["checkpointB"] = checkpoint_b[b_index]
        paired_remaining_b.add(b_index)
    for b_index, note in enumerate(checkpoint_b):
        if b_index in matched_b or b_index in paired_remaining_b:
            continue
        clusters.append({"members": {"checkpointB": note}})

    for cluster in clusters:
        times = [float(note["time"]) for note in cluster["members"].values()]
        cluster["centerTime"] = float(np.median(times))
        cluster["midi"] = int(next(iter(cluster["members"].values()))["midi"])
    return sorted(clusters, key=lambda item: (item["centerTime"], item["midi"]))


def proposal_labels(
    reference: list[dict[str, Any]], variants: dict[str, list[dict[str, Any]]]
) -> dict[str, set[int]]:
    """Label each variant's one-to-one strict-100ms true-positive proposals."""

    return {
        name: {
            id(candidate)
            for _target, candidate in greedy_matches(
                reference, notes, 0.1, octave_equivalent=False
            )
        }
        for name, notes in variants.items()
    }


def variant_context(notes: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(note["time"]) for note in notes]
    by_pitch: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        by_pitch.setdefault(int(note["midi"]), []).append(note)
    repeated: dict[int, tuple[float, float]] = {}
    for pitch_notes in by_pitch.values():
        for index, note in enumerate(pitch_notes):
            previous = (
                float(note["time"]) - float(pitch_notes[index - 1]["time"])
                if index > 0
                else 2.0
            )
            following = (
                float(pitch_notes[index + 1]["time"]) - float(note["time"])
                if index + 1 < len(pitch_notes)
                else 2.0
            )
            repeated[id(note)] = (min(2.0, previous), min(2.0, following))
    duration = max(1.0, max(times, default=0.0))
    return {
        "times": times,
        "repeated": repeated,
        "duration": duration,
        "density": len(notes) / duration,
    }


def agreement_share(
    left: list[dict[str, Any]], right: list[dict[str, Any]], radius_seconds: float
) -> float:
    return len(one_to_one_matches(left, right, radius_seconds)) / max(1, len(left))


FEATURE_NAMES = [
    "model_incumbent",
    "model_checkpoint_a",
    "model_checkpoint_b",
    "presence_incumbent",
    "presence_checkpoint_a",
    "presence_checkpoint_b",
    "support_share",
    "midi_centered",
    "pitch_class_sin",
    "pitch_class_cos",
    "song_time_fraction",
    "duration",
    "log_duration",
    "velocity",
    "proposal_minus_median_100ms",
    "cluster_spread_100ms",
    "nearest_other_distance_100ms",
    "mean_other_distance_100ms",
    "agreement_30ms_share",
    "agreement_60ms_share",
    "agreement_100ms_share",
    "local_density_250ms",
    "local_density_500ms",
    "onset_chord_size",
    "previous_same_pitch_gap",
    "next_same_pitch_gap",
    "song_density_incumbent",
    "song_density_checkpoint_a",
    "song_density_checkpoint_b",
    "song_agreement_incumbent_a",
    "song_agreement_incumbent_b",
    "song_agreement_a_b",
]


def feature_rows(
    variants: dict[str, list[dict[str, Any]]], clusters: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    contexts = {name: variant_context(notes) for name, notes in variants.items()}
    song_duration = max(context["duration"] for context in contexts.values())
    agreement = {
        "incumbentA": agreement_share(variants["incumbent"], variants["checkpointA"], 0.1),
        "incumbentB": agreement_share(variants["incumbent"], variants["checkpointB"], 0.1),
        "aB": agreement_share(variants["checkpointA"], variants["checkpointB"], 0.1),
    }
    rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters):
        members = cluster["members"]
        all_times = [float(note["time"]) for note in members.values()]
        center = float(np.median(all_times))
        spread = max(all_times) - min(all_times)
        for variant_name, note in members.items():
            proposal_time = float(note["time"])
            other_distances = [
                abs(proposal_time - float(other["time"]))
                for name, other in members.items()
                if name != variant_name
            ]
            context = contexts[variant_name]
            left_250 = bisect.bisect_left(context["times"], proposal_time - 0.25)
            right_250 = bisect.bisect_right(context["times"], proposal_time + 0.25)
            left_500 = bisect.bisect_left(context["times"], proposal_time - 0.5)
            right_500 = bisect.bisect_right(context["times"], proposal_time + 0.5)
            left_chord = bisect.bisect_left(context["times"], proposal_time - 0.03)
            right_chord = bisect.bisect_right(context["times"], proposal_time + 0.03)
            previous_gap, next_gap = context["repeated"][id(note)]
            midi = int(note["midi"])
            pitch_angle = 2.0 * math.pi * (midi % 12) / 12.0
            vector = [
                float(variant_name == "incumbent"),
                float(variant_name == "checkpointA"),
                float(variant_name == "checkpointB"),
                float("incumbent" in members),
                float("checkpointA" in members),
                float("checkpointB" in members),
                len(members) / 3.0,
                (midi - 60.0) / 24.0,
                math.sin(pitch_angle),
                math.cos(pitch_angle),
                proposal_time / song_duration,
                min(4.0, float(note["duration"])),
                math.log1p(max(0.0, float(note["duration"]))),
                float(note["velocity"]),
                (proposal_time - center) / 0.1,
                spread / 0.1,
                min(other_distances, default=0.3) / 0.1,
                (sum(other_distances) / len(other_distances) if other_distances else 0.3) / 0.1,
                sum(distance <= 0.03 for distance in other_distances) / 2.0,
                sum(distance <= 0.06 for distance in other_distances) / 2.0,
                sum(distance <= 0.1 for distance in other_distances) / 2.0,
                (right_250 - left_250) / 10.0,
                (right_500 - left_500) / 20.0,
                (right_chord - left_chord) / 8.0,
                previous_gap / 2.0,
                next_gap / 2.0,
                contexts["incumbent"]["density"] / 20.0,
                contexts["checkpointA"]["density"] / 20.0,
                contexts["checkpointB"]["density"] / 20.0,
                agreement["incumbentA"],
                agreement["incumbentB"],
                agreement["aB"],
            ]
            rows.append(
                {
                    "clusterIndex": cluster_index,
                    "variant": variant_name,
                    "note": note,
                    "features": vector,
                }
            )
    return rows


class LogisticModel:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        self.mean = mean
        self.scale = scale
        self.weights = weights

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        logits = np.clip(design @ self.weights, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float) -> LogisticModel:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * float(l2)
    penalty[0, 0] = 0.0
    for _iteration in range(40):
        logits = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(1e-6, probabilities * (1.0 - probabilities))
        gradient = design.T @ (probabilities - labels) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return LogisticModel(mean, scale, weights)


def training_matrix(song_data: list[dict[str, Any]], radius_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    all_features: list[list[float]] = []
    all_labels: list[float] = []
    for song in song_data:
        clusters = cluster_variants(song["variants"], radius_seconds)
        rows = feature_rows(song["variants"], clusters)
        labels = proposal_labels(song["reference"], song["variants"])
        for row in rows:
            all_features.append(row["features"])
            all_labels.append(float(id(row["note"]) in labels[row["variant"]]))
    return np.asarray(all_features, dtype=float), np.asarray(all_labels, dtype=float)


def prepare_selector_inputs(
    variants: dict[str, list[dict[str, Any]]],
    radius_seconds: float,
) -> dict[str, Any]:
    clusters = cluster_variants(variants, radius_seconds)
    rows = feature_rows(variants, clusters)
    return {
        "clusters": clusters,
        "rows": rows,
        "features": np.asarray([row["features"] for row in rows], dtype=float),
    }


def select_prepared(
    prepared: dict[str, Any],
    probabilities: np.ndarray,
    *,
    switch_margin: float,
    addition_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[tuple[int, str], ...]]:
    clusters = prepared["clusters"]
    rows = prepared["rows"]
    proposals_by_cluster: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for row, probability in zip(rows, probabilities):
        proposals_by_cluster.setdefault(row["clusterIndex"], []).append((float(probability), row))
    output: list[dict[str, Any]] = []
    switched = 0
    additions = 0
    rejected_non_incumbent = 0
    signature: list[tuple[int, str]] = []
    for cluster_index, cluster in enumerate(clusters):
        proposals = proposals_by_cluster[cluster_index]
        incumbent = next((item for item in proposals if item[1]["variant"] == "incumbent"), None)
        best = max(
            proposals,
            key=lambda item: (
                item[0],
                item[1]["variant"] == "incumbent",
                -VARIANTS.index(item[1]["variant"]),
            ),
        )
        chosen = best
        if incumbent is not None:
            if best[1]["variant"] != "incumbent" and best[0] < incumbent[0] + switch_margin:
                chosen = incumbent
            if chosen[1]["variant"] != "incumbent":
                switched += 1
        elif best[0] < addition_threshold:
            rejected_non_incumbent += 1
            continue
        else:
            additions += 1
        output.append(deepcopy(chosen[1]["note"]))
        signature.append((cluster_index, chosen[1]["variant"]))
    output.sort(key=lambda note: (float(note["time"]), int(note["midi"])))
    return output, {
        "clusters": len(clusters),
        "outputNotes": len(output),
        "switchedIncumbentOnsets": switched,
        "addedNonIncumbentEvents": additions,
        "rejectedNonIncumbentClusters": rejected_non_incumbent,
        "meanSelectedProbability": round(
            float(np.mean([max(item[0] for item in proposals_by_cluster[index]) for index in proposals_by_cluster])),
            6,
        ),
    }, tuple(signature)


def select_events(
    variants: dict[str, list[dict[str, Any]]],
    *,
    radius_seconds: float,
    model: LogisticModel,
    switch_margin: float,
    addition_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepare_selector_inputs(variants, radius_seconds)
    probabilities = model.probabilities(prepared["features"])
    output, diagnostics, _signature = select_prepared(
        prepared,
        probabilities,
        switch_margin=switch_margin,
        addition_threshold=addition_threshold,
    )
    return output, diagnostics


def selection_key(metrics: dict[str, Any], mutation_cost: float) -> tuple[float, ...]:
    return (
        float(metrics["exactPitchOnset100ms"]["f1"]),
        float(metrics["exactPitchOnset100ms"]["recall"]),
        float(metrics["exactPitchOnset250ms"]["f1"]),
        float(metrics["pitchClassOnset250ms"]["f1"]),
        -mutation_cost,
    )


def run(manifest: dict[str, Any]) -> dict[str, Any]:
    songs = [load_song(row) for row in manifest["songs"]]
    baseline_by_song = {
        song["id"]: compact_metrics(song["reference"], song["variants"]["incumbent"])
        for song in songs
    }
    search = manifest["eventSelectorSearch"]
    radii = [float(value) for value in search["clusterRadiusSeconds"]]
    prepared_cache: dict[tuple[str, float], dict[str, Any]] = {}
    for song in songs:
        labels = proposal_labels(song["reference"], song["variants"])
        for radius in radii:
            prepared = prepare_selector_inputs(song["variants"], radius)
            prepared["labels"] = np.asarray(
                [
                    float(id(row["note"]) in labels[row["variant"]])
                    for row in prepared["rows"]
                ],
                dtype=float,
            )
            prepared_cache[(song["id"], radius)] = prepared
    folds: list[dict[str, Any]] = []
    heldout_metrics: list[dict[str, Any]] = []
    for holdout in songs:
        training = [song for song in songs if song["id"] != holdout["id"]]
        ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        fitted: dict[tuple[float, float], LogisticModel] = {}
        metric_cache: dict[tuple[str, float, tuple[tuple[int, str], ...]], dict[str, Any]] = {}
        diagnostics_cache: dict[tuple[str, float, tuple[tuple[int, str], ...]], dict[str, Any]] = {}
        for radius, l2 in itertools.product(radii, search["l2"]):
            training_prepared = [prepared_cache[(song["id"], radius)] for song in training]
            features = np.concatenate([item["features"] for item in training_prepared], axis=0)
            labels = np.concatenate([item["labels"] for item in training_prepared], axis=0)
            model = fit_logistic(features, labels, float(l2))
            fitted[(radius, float(l2))] = model
            probabilities_by_song = {
                song["id"]: model.probabilities(prepared_cache[(song["id"], radius)]["features"])
                for song in songs
            }
            for margin, threshold in itertools.product(
                search["switchMargin"], search["additionThreshold"]
            ):
                rows: list[dict[str, Any]] = []
                mutations = 0
                for song in training:
                    prepared = prepared_cache[(song["id"], radius)]
                    notes, diagnostics, signature = select_prepared(
                        prepared,
                        probabilities_by_song[song["id"]],
                        switch_margin=float(margin),
                        addition_threshold=float(threshold),
                    )
                    cache_key = (song["id"], radius, signature)
                    if cache_key not in metric_cache:
                        metric_cache[cache_key] = compact_metrics(song["reference"], notes)
                        diagnostics_cache[cache_key] = diagnostics
                    rows.append(metric_cache[cache_key])
                    mutations += int(diagnostics["switchedIncumbentOnsets"])
                    mutations += int(diagnostics["addedNonIncumbentEvents"])
                aggregate = aggregate_metrics(rows)
                parameters = {
                    "clusterRadiusSeconds": radius,
                    "l2": float(l2),
                    "switchMargin": float(margin),
                    "additionThreshold": float(threshold),
                }
                ranked.append(
                    (
                        selection_key(aggregate, mutations / max(1, aggregate["observedNotes"])),
                        {"parameters": parameters, "trainingAggregate": aggregate},
                    )
                )
        _rank, selected = max(ranked, key=lambda item: item[0])
        parameters = selected["parameters"]
        model = fitted[(parameters["clusterRadiusSeconds"], parameters["l2"])]
        heldout_prepared = prepared_cache[(holdout["id"], parameters["clusterRadiusSeconds"])]
        heldout_probabilities = model.probabilities(heldout_prepared["features"])
        notes, diagnostics, _signature = select_prepared(
            heldout_prepared,
            heldout_probabilities,
            switch_margin=parameters["switchMargin"],
            addition_threshold=parameters["additionThreshold"],
        )
        metrics = compact_metrics(holdout["reference"], notes)
        heldout_metrics.append(metrics)
        weight_rows = sorted(
            zip(FEATURE_NAMES, model.weights[1:]), key=lambda item: abs(float(item[1])), reverse=True
        )[:10]
        folds.append(
            {
                "heldOutSong": holdout["id"],
                "trainingSongs": [song["id"] for song in training],
                "selectedParameters": parameters,
                "trainingAggregate": selected["trainingAggregate"],
                "heldOutBaseline": baseline_by_song[holdout["id"]],
                "heldOutCandidate": metrics,
                "diagnostics": diagnostics,
                "topStandardizedFeatureWeights": [
                    {"feature": name, "weight": round(float(weight), 6)}
                    for name, weight in weight_rows
                ],
            }
        )
    baseline = aggregate_metrics(list(baseline_by_song.values()))
    loso = aggregate_metrics(heldout_metrics)
    delta = round(
        float(loso["exactPitchOnset100ms"]["f1"])
        - float(baseline["exactPitchOnset100ms"]["f1"]),
        6,
    )
    no_regressions = all(
        float(fold["heldOutCandidate"]["exactPitchOnset100ms"]["f1"]) + 1e-12
        >= float(fold["heldOutBaseline"]["exactPitchOnset100ms"]["f1"])
        for fold in folds
    )
    return {
        "schema": "polymath-checkpoint-event-selector-loso-report-v1",
        "manifestId": manifest.get("id"),
        "candidateGenerationUsesTarget": False,
        "model": "standardized-logistic-regression-newton-irls",
        "featureNames": FEATURE_NAMES,
        "warning": "Opened development evidence only; a new sealed song remains mandatory.",
        "baselineAggregate": baseline,
        "losoAggregate": loso,
        "strictExact100F1Delta": delta,
        "noHeldOutSongRegressed": no_regressions,
        "folds": folds,
        "decision": (
            "CANDIDATE_FOR_NEW_SEALED_HOLDOUT"
            if delta > 0.001 and no_regressions
            else "REJECT_NO_SAFE_CROSS_SONG_GAIN"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_json(Path(args.manifest))
    report = run(manifest)
    atomic_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "baselineExact100F1": report["baselineAggregate"]["exactPitchOnset100ms"]["f1"],
                "losoExact100F1": report["losoAggregate"]["exactPitchOnset100ms"]["f1"],
                "delta": report["strictExact100F1Delta"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
