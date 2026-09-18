"""Fit a post-arranger ranker for exact accompaniment-note replacements.

Targets are intentionally conservative.  A proposal is positive only when it
can recover an unmatched exact reference note without sacrificing an existing
exact or pitch-class match.  The reference is used to create labels offline;
all runtime features come from the arranged candidate, separated source, and
the frozen phase-47 source-support selector.

Every reported song is held out as a complete song.  Selector probabilities
for a song also come from a selector trained without that song, so the stacked
ranker never receives an in-sample selector score for its held-out report.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .apply_raw_support_recovery import load_json
    from .evaluate_piano_arranger import (
        greedy_matches,
        prepare_reference_notes,
        trusted_source_ranges,
    )
    from .raw_support_swap_benefit import (
        SWAP_BENEFIT_FEATURE_NAMES,
        generate_swap_proposals,
        model_probabilities_numpy,
    )
    from .train_piano_arranger_adapter import classification_metrics
except ImportError:  # pragma: no cover
    from apply_raw_support_recovery import load_json  # type: ignore
    from evaluate_piano_arranger import (  # type: ignore
        greedy_matches,
        prepare_reference_notes,
        trusted_source_ranges,
    )
    from raw_support_swap_benefit import (  # type: ignore
        SWAP_BENEFIT_FEATURE_NAMES,
        generate_swap_proposals,
        model_probabilities_numpy,
    )
    from train_piano_arranger_adapter import classification_metrics  # type: ignore


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inside_ranges(time: float, ranges: list[tuple[float, float]] | None) -> bool:
    return ranges is None or any(start <= time < end for start, end in ranges)


def normalized_candidate_notes(
    payload: dict[str, Any], row: dict[str, Any], alignment: dict[str, Any]
) -> list[dict[str, Any]]:
    ranges = trusted_source_ranges(alignment)
    hard_end = (
        float(row["candidateEndSeconds"])
        if row.get("candidateEndSeconds") is not None
        else math.inf
    )
    notes: list[dict[str, Any]] = []
    for candidate_index, item in enumerate(payload.get("notes") or []):
        try:
            midi = int(round(float(item.get("midi", item.get("pitch")))))
            time = float(item.get("time", item.get("startTime", item.get("start"))))
            duration = max(0.01, float(item.get("duration", 0.2)))
        except (TypeError, ValueError):
            continue
        if (
            not 0 <= midi <= 127
            or not math.isfinite(time)
            or not math.isfinite(duration)
            or time < 0
            or time >= hard_end
            or not inside_ranges(time, ranges)
        ):
            continue
        notes.append(
            {
                "candidateIndex": candidate_index,
                "midi": midi,
                "time": time,
                "duration": duration,
                "velocity": max(0.01, min(1.0, float(item.get("velocity", 0.72)))),
            }
        )
    return sorted(notes, key=lambda note: (float(note["time"]), int(note["midi"])))


def match_state(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], tolerance: float, *, octave_equivalent: bool
) -> tuple[set[int], set[int]]:
    matches = greedy_matches(
        reference,
        candidate,
        tolerance,
        octave_equivalent=octave_equivalent,
    )
    return (
        {int(observed["candidateIndex"]) for _target, observed in matches},
        {int(target["referenceIndex"]) for target, _observed in matches},
    )


def reference_index(
    reference: list[dict[str, Any]], *, octave_equivalent: bool
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[float]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for note in reference:
        key = int(note["midi"]) % 12 if octave_equivalent else int(note["midi"])
        grouped.setdefault(key, []).append(note)
    for values in grouped.values():
        values.sort(key=lambda note: float(note["time"]))
    return grouped, {
        key: [float(note["time"]) for note in values]
        for key, values in grouped.items()
    }


def unmatched_target_available(
    midi: int,
    time: float,
    tolerance: float,
    grouped: dict[int, list[dict[str, Any]]],
    times: dict[int, list[float]],
    used_reference_indices: set[int],
    *,
    octave_equivalent: bool,
) -> tuple[int, float]:
    key = midi % 12 if octave_equivalent else midi
    pool = grouped.get(key, [])
    if not pool:
        return 0, math.inf
    values = times[key]
    position = bisect.bisect_left(values, time)
    candidates: list[tuple[float, dict[str, Any]]] = []
    left = position - 1
    while left >= 0 and time - values[left] <= tolerance:
        if int(pool[left]["referenceIndex"]) not in used_reference_indices:
            candidates.append((abs(values[left] - time), pool[left]))
        left -= 1
    right = position
    while right < len(pool) and values[right] - time <= tolerance:
        if int(pool[right]["referenceIndex"]) not in used_reference_indices:
            candidates.append((abs(values[right] - time), pool[right]))
        right += 1
    if not candidates:
        return 0, math.inf
    distance, _note = min(
        candidates,
        key=lambda item: (item[0], abs(int(item[1]["midi"]) - midi)),
    )
    return 1, distance


def label_proposals(
    proposals: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    candidate_notes: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    exact100_candidates, exact100_refs = match_state(
        reference, candidate_notes, 0.10, octave_equivalent=False
    )
    exact250_candidates, exact250_refs = match_state(
        reference, candidate_notes, 0.25, octave_equivalent=False
    )
    pc250_candidates, pc250_refs = match_state(
        reference, candidate_notes, 0.25, octave_equivalent=True
    )
    exact_index, exact_times = reference_index(reference, octave_equivalent=False)
    pc_index, pc_times = reference_index(reference, octave_equivalent=True)
    labels: list[float] = []
    utilities: list[float] = []
    details: list[dict[str, Any]] = []
    for proposal in proposals:
        incumbent_index = int(proposal["incumbentIndex"])
        target_midi = int(proposal["targetMidi"])
        time = float(proposal["groupTime"])
        proposed_exact100, distance100 = unmatched_target_available(
            target_midi,
            time,
            0.10,
            exact_index,
            exact_times,
            exact100_refs,
            octave_equivalent=False,
        )
        proposed_exact250, distance250 = unmatched_target_available(
            target_midi,
            time,
            0.25,
            exact_index,
            exact_times,
            exact250_refs,
            octave_equivalent=False,
        )
        proposed_pc250, distance_pc = unmatched_target_available(
            target_midi,
            time,
            0.25,
            pc_index,
            pc_times,
            pc250_refs,
            octave_equivalent=True,
        )
        delta_exact100 = proposed_exact100 - int(incumbent_index in exact100_candidates)
        delta_exact250 = proposed_exact250 - int(incumbent_index in exact250_candidates)
        delta_pc250 = proposed_pc250 - int(incumbent_index in pc250_candidates)
        exact_gain = delta_exact100 > 0 or delta_exact250 > 0
        safe = delta_exact100 >= 0 and delta_exact250 >= 0 and delta_pc250 >= 0
        label = float(exact_gain and safe)
        utility = 4.0 * delta_exact100 + 3.0 * delta_exact250 + 1.0 * delta_pc250
        labels.append(label)
        utilities.append(utility)
        details.append(
            {
                "label": int(label),
                "utility": utility,
                "deltaExact100": delta_exact100,
                "deltaExact250": delta_exact250,
                "deltaPitchClass250": delta_pc250,
                "nearestUnmatchedExact100Seconds": None if not math.isfinite(distance100) else round(distance100, 6),
                "nearestUnmatchedExact250Seconds": None if not math.isfinite(distance250) else round(distance250, 6),
                "nearestUnmatchedPitchClass250Seconds": None if not math.isfinite(distance_pc) else round(distance_pc, 6),
            }
        )
    return (
        np.asarray(labels, dtype=np.float64),
        np.asarray(utilities, dtype=np.float64),
        details,
    )


def build_song_examples(
    row: dict[str, Any],
    selector_profile: dict[str, Any],
    *,
    source_radius: float,
    minimum_source_midi: int,
    maximum_source_midi: int,
    minimum_output_midi: int,
    maximum_output_midi: int,
    register_shifts: tuple[int, ...],
) -> dict[str, Any]:
    candidate = load_json(Path(str(row["candidate"])).resolve())
    source = load_json(Path(str(row["source"])).resolve())
    alignment = load_json(Path(str(row["alignment"])).resolve())
    reference_payload = load_json(Path(str(row["reference"])).resolve())
    reference = prepare_reference_notes(row, reference_payload, alignment)
    reference = [
        {**note, "referenceIndex": index}
        for index, note in enumerate(reference)
    ]
    candidate_notes = normalized_candidate_notes(candidate, row, alignment)
    eligible_indices = {int(note["candidateIndex"]) for note in candidate_notes}
    proposals, diagnostics = generate_swap_proposals(
        candidate,
        source,
        selector_profile,
        source_radius=source_radius,
        minimum_source_midi=minimum_source_midi,
        maximum_source_midi=maximum_source_midi,
        minimum_output_midi=minimum_output_midi,
        maximum_output_midi=maximum_output_midi,
        register_shifts=register_shifts,
    )
    proposals = [
        proposal
        for proposal in proposals
        if int(proposal["incumbentIndex"]) in eligible_indices
    ]
    labels, utilities, label_details = label_proposals(
        proposals, reference, candidate_notes
    )
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives < 10 or negatives < 10:
        raise ValueError(
            f"{row['id']} lacks both swap-benefit classes: {positives} positive, {negatives} negative"
        )
    song_weight = max(0.01, float(row.get("weight", 1.0)))
    weights = np.asarray(
        [
            song_weight * len(labels) * (0.5 / positives if label else 0.5 / negatives)
            * (1.0 + 0.12 * min(4.0, abs(float(utility))))
            for label, utility in zip(labels, utilities)
        ],
        dtype=np.float64,
    )
    times = np.asarray([float(item["groupTime"]) for item in proposals], dtype=np.float64)
    temporal_training = (times // 20.0).astype(np.int64) % 5 != 4
    return {
        "id": str(row["id"]),
        "features": np.asarray([item["features"] for item in proposals], dtype=np.float64),
        "labels": labels,
        "utilities": utilities,
        "weights": weights,
        "times": times,
        "temporalTraining": temporal_training,
        "proposals": proposals,
        "labelDetails": label_details,
        "diagnostics": diagnostics,
        "referenceNotes": len(reference),
        "candidateNotes": len(candidate_notes),
        "examples": len(labels),
        "positives": positives,
        "positiveShare": positives / len(labels),
    }


def combine_songs(songs: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    return (
        np.vstack([song["features"] for song in songs]),
        np.concatenate([song["labels"] for song in songs]),
        np.concatenate([song["weights"] for song in songs]),
        np.concatenate([song["temporalTraining"] for song in songs]),
    )


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    training_mask: np.ndarray,
    *,
    seed: int,
    iterations: int,
    hidden_size: int = 18,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, float]]]:
    if not training_mask.any() or training_mask.all():
        raise ValueError("Training and temporal-validation examples are both required.")
    means = features[training_mask].mean(axis=0)
    scales = features[training_mask].std(axis=0)
    scales = np.where(scales < 1e-7, 1.0, scales)
    standardized = np.clip((features - means) / scales, -8.0, 8.0)
    train_indices = np.flatnonzero(training_mask)
    validation_mask = ~training_mask
    rng = np.random.default_rng(seed)
    input_size = features.shape[1]
    hidden_weights = rng.normal(
        0.0,
        math.sqrt(2.0 / max(1, input_size + hidden_size)),
        size=(input_size, hidden_size),
    )
    hidden_biases = np.zeros(hidden_size, dtype=np.float64)
    output_weights = rng.normal(0.0, 0.10, size=hidden_size)
    output_bias = np.zeros(1, dtype=np.float64)
    parameters = [hidden_weights, hidden_biases, output_weights, output_bias]
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    learning_rate, l2 = 0.003, 0.0035
    batch_size = min(512, len(train_indices))
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_parameters = [parameter.copy() for parameter in parameters]

    def probabilities(indices: np.ndarray) -> np.ndarray:
        hidden = np.tanh(standardized[indices] @ hidden_weights + hidden_biases)
        logits = hidden @ output_weights + float(output_bias[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def weighted_loss(mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        predicted = np.clip(probabilities(indices), 1e-7, 1.0 - 1e-7)
        kept_weights = weights[indices]
        losses = -(labels[indices] * np.log(predicted) + (1.0 - labels[indices]) * np.log(1.0 - predicted))
        return float(np.sum(kept_weights * losses) / max(1e-9, np.sum(kept_weights)))

    for iteration in range(1, iterations + 1):
        batch = rng.choice(train_indices, size=batch_size, replace=False)
        x = standardized[batch]
        y = labels[batch]
        kept_weights = weights[batch]
        hidden = np.tanh(x @ hidden_weights + hidden_biases)
        logits = hidden @ output_weights + float(output_bias[0])
        predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (predicted - y) * kept_weights / max(1e-9, float(kept_weights.sum()))
        gradients = [
            x.T @ (error[:, None] * output_weights[None, :] * (1.0 - hidden * hidden)) + l2 * hidden_weights,
            (error[:, None] * output_weights[None, :] * (1.0 - hidden * hidden)).sum(axis=0),
            hidden.T @ error + l2 * output_weights,
            np.asarray([error.sum()], dtype=np.float64),
        ]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
            second[index] = beta2 * second[index] + (1.0 - beta2) * (gradient * gradient)
            first_hat = first[index] / (1.0 - beta1 ** iteration)
            second_hat = second[index] / (1.0 - beta2 ** iteration)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            training_loss = weighted_loss(training_mask)
            validation_loss = weighted_loss(validation_mask)
            history.append(
                {
                    "iteration": float(iteration),
                    "trainingWeightedBinaryCrossEntropy": round(training_loss, 8),
                    "validationWeightedBinaryCrossEntropy": round(validation_loss, 8),
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_parameters = [parameter.copy() for parameter in parameters]

    hidden_weights[:], hidden_biases[:], output_weights[:], output_bias[:] = best_parameters
    all_probabilities = probabilities(np.arange(len(features)))
    model = {
        "type": "standardized-mlp-swap-benefit-v1",
        "featureNames": list(SWAP_BENEFIT_FEATURE_NAMES),
        "means": [round(float(value), 10) for value in means],
        "scales": [round(float(value), 10) for value in scales],
        "hiddenWeights": [[round(float(value), 10) for value in row] for row in hidden_weights],
        "hiddenBiases": [round(float(value), 10) for value in hidden_biases],
        "outputWeights": [round(float(value), 10) for value in output_weights],
        "outputBias": round(float(output_bias[0]), 10),
        "hiddenActivation": "tanh",
        "hiddenSize": hidden_size,
    }
    return model, all_probabilities, history


def threshold_search(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    trials = [
        classification_metrics(labels[mask], probabilities[mask], weights[mask], float(threshold))
        for threshold in np.linspace(0.20, 0.98, 157)
    ]
    # A wrong replacement is more costly than abstention.  Prefer a precision
    # floor, then F1; end-to-end policy search remains the final authority.
    return max(
        trials,
        key=lambda item: (
            float(item["precision"] >= 0.72),
            float(item["f1"]),
            float(item["precision"]),
            float(item["threshold"]),
        ),
    )


def fit(songs: list[dict[str, Any]], *, iterations: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    features, labels, weights, training = combine_songs(songs)
    model, probabilities, history = train_mlp(
        features,
        labels,
        weights,
        training,
        seed=seed,
        iterations=iterations,
    )
    validation = threshold_search(labels, probabilities, weights, ~training)
    model["threshold"] = validation["threshold"]
    return model, {
        "examples": len(labels),
        "positives": int(labels.sum()),
        "temporalTraining": classification_metrics(
            labels[training], probabilities[training], weights[training], float(model["threshold"])
        ),
        "temporalValidation": validation,
        "history": history,
    }


def evaluate_song(song: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    probabilities = model_probabilities_numpy(song["features"], model)
    metrics = classification_metrics(
        song["labels"],
        probabilities,
        song["weights"],
        float(model.get("threshold", 0.5)),
    )
    metrics["examples"] = song["examples"]
    metrics["positives"] = song["positives"]
    metrics["positiveShare"] = round(float(song["positiveShare"]), 6)
    metrics["probabilityQuantiles"] = {
        str(fraction): round(float(np.quantile(probabilities, fraction)), 6)
        for fraction in (0.1, 0.5, 0.9, 0.99)
    }
    positive_probabilities = probabilities[song["labels"] >= 0.5]
    negative_probabilities = probabilities[song["labels"] < 0.5]
    metrics["positiveProbabilityMedian"] = round(float(np.median(positive_probabilities)), 6)
    metrics["negativeProbabilityMedian"] = round(float(np.median(negative_probabilities)), 6)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fold-profiles-root", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--minimum-source-midi", type=int, default=0)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    parser.add_argument("--register-shifts", default="0,12")
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0x42454E45)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    selector_root = args.selector_profiles_root.resolve()
    manifest = load_json(manifest_path)
    rows = manifest.get("songs") or []
    if len(rows) < 4:
        raise ValueError("At least four complete songs are required.")
    register_shifts = tuple(
        dict.fromkeys(int(value.strip()) for value in args.register_shifts.split(",") if value.strip())
    )
    if not register_shifts:
        raise ValueError("At least one register shift is required.")
    songs: list[dict[str, Any]] = []
    for row in rows:
        song_id = str(row["id"])
        selector_path = selector_root / song_id / "profile.json"
        if not selector_path.is_file():
            raise FileNotFoundError(f"Missing selector fold for {song_id}: {selector_path}")
        songs.append(
            build_song_examples(
                row,
                load_json(selector_path),
                source_radius=max(0.01, float(args.source_radius_seconds)),
                minimum_source_midi=int(args.minimum_source_midi),
                maximum_source_midi=int(args.maximum_source_midi),
                minimum_output_midi=int(args.minimum_output_midi),
                maximum_output_midi=int(args.maximum_output_midi),
                register_shifts=register_shifts,
            )
        )

    folds: dict[str, Any] = {}
    fold_root = args.fold_profiles_root.resolve()
    for index, held_out in enumerate(songs):
        training_songs = [song for song in songs if song is not held_out]
        model, internal = fit(
            training_songs,
            iterations=max(200, int(args.iterations)),
            seed=int(args.seed) + index * 1009,
        )
        profile = {
            "schema": "polymath-raw-support-swap-benefit-profile-v1",
            "id": f"{args.profile_id}-without-{held_out['id']}",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "training": {
                "manifest": str(manifest_path),
                "heldOutSong": held_out["id"],
                "trainingSongs": [song["id"] for song in training_songs],
                "labelContract": "exact-gain-with-no-exact-or-pitch-class-regression-v1",
                "selectorEvidence": "per-song out-of-fold phase-48 raw-support selector",
                "commercialUseAllowed": False,
                "purpose": "private research evaluation",
            },
        }
        profile["profileSha256"] = profile_hash(profile)
        profile_path = fold_root / held_out["id"] / "profile.json"
        atomic_json(profile_path, profile)
        folds[held_out["id"]] = {
            "trainingSongs": [song["id"] for song in training_songs],
            "internalTemporalValidation": internal["temporalValidation"],
            "heldOutSong": evaluate_song(held_out, model),
            "profile": str(profile_path),
        }

    final_model, internal = fit(
        songs,
        iterations=max(200, int(args.iterations)),
        seed=int(args.seed),
    )
    final_profile = {
        "schema": "polymath-raw-support-swap-benefit-profile-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": final_model,
        "proposalContract": {
            "sourceRadiusSeconds": max(0.01, float(args.source_radius_seconds)),
            "sourceMidiRange": [int(args.minimum_source_midi), int(args.maximum_source_midi)],
            "outputMidiRange": [int(args.minimum_output_midi), int(args.maximum_output_midi)],
            "registerShifts": list(register_shifts),
        },
        "training": {
            "manifest": str(manifest_path),
            "songs": [song["id"] for song in songs],
            "labelContract": "exact-gain-with-no-exact-or-pitch-class-regression-v1",
            "selectorEvidence": "per-song out-of-fold phase-48 raw-support selector",
            "commercialUseAllowed": False,
            "purpose": "private research evaluation",
        },
    }
    final_profile["profileSha256"] = profile_hash(final_profile)
    atomic_json(args.output_profile.resolve(), final_profile)
    heldout_f1 = [float(value["heldOutSong"]["f1"]) for value in folds.values()]
    heldout_precision = [float(value["heldOutSong"]["precision"]) for value in folds.values()]
    report = {
        "schema": "polymath-raw-support-swap-benefit-training-v1",
        "manifest": str(manifest_path),
        "evidenceBoundary": (
            "Each held-out song uses a benefit model trained without that complete song and "
            "selector probabilities from a selector also trained without that song. Reference "
            "fields create labels only and are absent from the runtime feature contract."
        ),
        "featureNames": list(SWAP_BENEFIT_FEATURE_NAMES),
        "songs": [
            {
                "id": song["id"],
                "examples": song["examples"],
                "positives": song["positives"],
                "positiveShare": round(float(song["positiveShare"]), 6),
                "referenceNotes": song["referenceNotes"],
                "candidateNotes": song["candidateNotes"],
                "proposalGeneration": song["diagnostics"],
            }
            for song in songs
        ],
        "folds": folds,
        "heldOutSummary": {
            "meanF1": round(sum(heldout_f1) / len(heldout_f1), 6),
            "minimumF1": round(min(heldout_f1), 6),
            "meanPrecision": round(sum(heldout_precision) / len(heldout_precision), 6),
            "minimumPrecision": round(min(heldout_precision), 6),
        },
        "finalInternal": internal,
        "profile": str(args.output_profile.resolve()),
    }
    atomic_json(args.report.resolve(), report)
    print(
        json.dumps(
            {
                "profile": str(args.output_profile.resolve()),
                "report": str(args.report.resolve()),
                "songs": len(songs),
                "examples": sum(song["examples"] for song in songs),
                "positives": sum(song["positives"] for song in songs),
                "heldOutSummary": report["heldOutSummary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
