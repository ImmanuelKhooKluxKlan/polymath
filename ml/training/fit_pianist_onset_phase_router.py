"""Fit an inference-safe router for one-pulse pianist onset corrections.

Answer keys label whether an arranged gesture is one source pulse early, on
time, or one source pulse late.  Runtime features contain only candidate and
source evidence.  Complete songs are held out from every fold so the router
cannot memorize a song's answer-key coordinates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from .analyze_pianist_gesture_patterns import group_onsets
from .analyze_pianist_timing_residuals import attach_candidate_roles
from .evaluate_piano_arranger import greedy_matches, load_json, prepare_reference_notes
from .search_default_piano_pipeline import clip_candidate
from .train_pianist_intro_motif import (
    estimate_source_pulse,
    is_voice,
    source_groups,
    source_instrument,
)
from .train_piano_arranger_adapter import train_logistic_model


CLASSES = (-1, 0, 1)
LAGS = (-2, -1, 0, 1, 2)
BASE_FEATURE_NAMES = (
    "bias",
    "song_progress",
    "song_progress_squared",
    "previous_gap_pulses",
    "next_gap_pulses",
    "gesture_size",
    "gesture_pitch_span",
    "gesture_mean_midi",
    "gesture_velocity",
    "gesture_duration",
    "left_hand_share",
    "right_hand_share",
    "role_melody_share",
    "role_bass_share",
    "role_harmony_share",
    "family_voice_share",
    "family_guitar_share",
    "family_bass_share",
    "family_piano_share",
    "family_strings_share",
    "phase2_sin",
    "phase2_cos",
    "phase4_sin",
    "phase4_cos",
    "phase8_sin",
    "phase8_cos",
)
LAG_FEATURE_SUFFIXES = (
    "pitch_class_recall",
    "exact_pitch_recall",
    "source_note_density",
    "voice_share",
    "guitar_share",
    "bass_share",
    "piano_share",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    f"lag_{lag:+d}_{suffix}" for lag in LAGS for suffix in LAG_FEATURE_SUFFIXES
)


def family(value: Any) -> str:
    name = str(value or "").lower()
    if name == "voice" or "vocal" in name:
        return "voice"
    if "guitar" in name or "ukulele" in name:
        return "guitar"
    if "bass" in name:
        return "bass"
    if "piano" in name or "keyboard" in name:
        return "piano"
    if "string" in name or "violin" in name or "cello" in name:
        return "strings"
    return "other"


def share(values: Iterable[str], expected: str) -> float:
    items = list(values)
    return sum(value == expected for value in items) / max(1, len(items))


def local_source_features(
    candidate_group: list[dict[str, Any]],
    source: list[list[dict[str, Any]]],
    target_time: float,
    radius: float,
) -> list[float]:
    pool = [
        note
        for group in source
        if abs(float(group[0]["time"]) - target_time) <= radius
        for note in group
    ]
    candidate_midis = {int(note["midi"]) for note in candidate_group}
    candidate_pcs = {midi % 12 for midi in candidate_midis}
    source_midis = {int(note["midi"]) for note in pool}
    source_pcs = {midi % 12 for midi in source_midis}
    families = [family(source_instrument(note)) for note in pool]
    return [
        len(candidate_pcs & source_pcs) / max(1, len(candidate_pcs)),
        len(candidate_midis & source_midis) / max(1, len(candidate_midis)),
        min(16, len(pool)) / 16.0,
        share(families, "voice"),
        share(families, "guitar"),
        share(families, "bass"),
        share(families, "piano"),
    ]


def gesture_features(
    groups: list[list[dict[str, Any]]],
    index: int,
    source: list[list[dict[str, Any]]],
    *,
    pulse: float,
    origin: float,
    song_end: float,
) -> list[float]:
    group = groups[index]
    time = float(group[0]["time"])
    previous_gap = time - float(groups[index - 1][0]["time"]) if index else pulse
    next_gap = (
        float(groups[index + 1][0]["time"]) - time
        if index + 1 < len(groups)
        else pulse
    )
    midis = [int(note["midi"]) for note in group]
    roles = [str(note.get("_arrangementRole") or "").lower() for note in group]
    families = [
        family(note.get("_sourceInstrument") or note.get("sourceInstrument"))
        for note in group
    ]
    hands = [str(note.get("hand") or ("left" if int(note["midi"]) < 60 else "right")) for note in group]
    phase = (time - origin) / max(1e-6, pulse)
    progress = min(1.0, max(0.0, time / max(1.0, song_end)))
    features = [
        1.0,
        progress,
        progress * progress,
        min(4.0, max(0.0, previous_gap / pulse)) / 4.0,
        min(4.0, max(0.0, next_gap / pulse)) / 4.0,
        min(8, len(group)) / 8.0,
        min(48, max(midis) - min(midis)) / 48.0,
        (sum(midis) / len(midis) - 69.0) / 36.0,
        median(float(note.get("velocity", 0.72)) for note in group),
        min(3.0, median(float(note.get("duration", 0.2)) for note in group)) / 3.0,
        share(hands, "left"),
        share(hands, "right"),
        share(roles, "melody"),
        share(roles, "bass"),
        share(roles, "harmony"),
        share(families, "voice"),
        share(families, "guitar"),
        share(families, "bass"),
        share(families, "piano"),
        share(families, "strings"),
    ]
    for period in (2, 4, 8):
        angle = 2.0 * math.pi * phase / period
        features.extend((math.sin(angle), math.cos(angle)))
    radius = max(0.055, min(0.10, pulse * 0.42))
    for lag in LAGS:
        features.extend(
            local_source_features(group, source, time + lag * pulse, radius)
        )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"Feature length {len(features)} != {len(FEATURE_NAMES)}")
    return features


def load_song_records(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alignment = load_json(Path(str(row["alignment"])).resolve())
    reference = prepare_reference_notes(
        row, load_json(Path(str(row["reference"])).resolve()), alignment
    )
    if row.get("candidateEndSeconds") is not None:
        end = float(row["candidateEndSeconds"])
        reference = [note for note in reference if float(note["time"]) < end]
    candidate_payload = load_json(Path(str(row["candidate"])).resolve())
    candidate = clip_candidate(candidate_payload, row, alignment)
    attach_candidate_roles(candidate, candidate_payload)
    groups = group_onsets(candidate, 0.035)
    source_payload = load_json(Path(str(row["source"])).resolve())
    source = source_groups(source_payload)
    harmonic = [group for group in source if not all(is_voice(note) for note in group)]
    song_end = max((float(note["time"]) for note in candidate), default=1.0) + 1.0
    pulse = float(estimate_source_pulse(harmonic, song_end))
    origin = min(float(group[0]["time"]) for group in harmonic)

    matches = greedy_matches(reference, candidate, 0.25, octave_equivalent=True)
    targets_by_observed = {id(observed): target for target, observed in matches}
    records: list[dict[str, Any]] = []
    excluded_ambiguous = 0
    for index, group in enumerate(groups):
        labels = []
        for note in group:
            target = targets_by_observed.get(id(note))
            if target is None:
                continue
            desired = (float(target["time"]) - float(note["time"])) / pulse
            labels.append(max(-1, min(1, int(round(desired)))))
        if not labels:
            continue
        counts = Counter(labels)
        label, votes = max(
            counts.items(), key=lambda item: (item[1], item[0] == 0, -abs(item[0]))
        )
        if votes / len(labels) < 0.60:
            excluded_ambiguous += 1
            continue
        records.append(
            {
                "songId": str(row["id"]),
                "time": round(float(group[0]["time"]), 6),
                "pulseSeconds": round(pulse, 6),
                "label": int(label),
                "matchedNotes": len(labels),
                "labelAgreement": round(votes / len(labels), 6),
                "features": gesture_features(
                    groups,
                    index,
                    source,
                    pulse=pulse,
                    origin=origin,
                    song_end=song_end,
                ),
            }
        )
    return records, {
        "id": str(row["id"]),
        "pulseSeconds": round(pulse, 6),
        "candidateGestures": len(groups),
        "labelledGestures": len(records),
        "excludedAmbiguousGestures": excluded_ambiguous,
        "labels": dict(sorted(Counter(record["label"] for record in records).items())),
    }


def balanced_weights(records: list[dict[str, Any]]) -> np.ndarray:
    song_counts = Counter(str(record["songId"]) for record in records)
    song_label_counts = Counter(
        (str(record["songId"]), int(record["label"])) for record in records
    )
    labels_by_song: dict[str, set[int]] = {}
    for song, label in song_label_counts:
        labels_by_song.setdefault(song, set()).add(label)
    values = []
    for record in records:
        song = str(record["songId"])
        label = int(record["label"])
        song_equal = 1.0 / max(1, song_counts[song])
        class_equal = 1.0 / (
            max(1, len(labels_by_song[song])) * max(1, song_label_counts[(song, label)])
        )
        values.append(0.40 * song_equal + 0.60 * class_equal)
    weights = np.asarray(values, dtype=np.float64)
    return weights * len(weights) / max(1e-12, float(weights.sum()))


def fit_model(records: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    sample_weights = balanced_weights(records)
    mask = np.ones(len(records), dtype=bool)
    models = {}
    for index, label in enumerate(CLASSES):
        binary = (labels == label).astype(np.float64)
        weights, means, scales, history = train_logistic_model(
            features,
            binary,
            sample_weights,
            mask,
            seed=seed + index * 1009,
            iterations=iterations,
            learning_rate=0.012,
            l2=0.035,
        )
        models[str(label)] = {
            "weights": [round(float(value), 10) for value in weights],
            "means": [round(float(value), 10) for value in means],
            "scales": [round(float(value), 10) for value in scales],
            "finalTrainingLoss": history[-1]["weightedBinaryCrossEntropy"],
        }
    return {
        "type": "one-v-rest-standardized-logistic-onset-phase-router-v1",
        "classes": list(CLASSES),
        "featureNames": list(FEATURE_NAMES),
        "models": models,
    }


def probabilities(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    columns = []
    for label in CLASSES:
        item = model["models"][str(label)]
        weights = np.asarray(item["weights"], dtype=np.float64)
        means = np.asarray(item["means"], dtype=np.float64)
        scales = np.asarray(item["scales"], dtype=np.float64)
        logits = np.clip(((features - means) / scales) @ weights, -30.0, 30.0)
        columns.append(1.0 / (1.0 + np.exp(-logits)))
    return np.column_stack(columns)


def route(values: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    winners = np.argmax(values, axis=1)
    ordered = np.sort(values, axis=1)
    confidence = ordered[:, -1]
    separation = ordered[:, -1] - ordered[:, -2]
    labels = np.asarray(CLASSES, dtype=np.int64)[winners]
    accepted = (labels != 0) & (confidence >= threshold) & (separation >= margin)
    return np.where(accepted, labels, 0)


def policy_metrics(records: list[dict[str, Any]], predicted: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    baseline = labels == 0
    correct = predicted == labels
    changed = predicted != 0
    correct_changes = changed & correct
    harmful = changed & baseline
    return {
        "examples": len(records),
        "baselineExactRate": round(float(baseline.mean()), 6),
        "routedExactRate": round(float(correct.mean()), 6),
        "exactRateDelta": round(float(correct.mean() - baseline.mean()), 6),
        "changes": int(changed.sum()),
        "correctChanges": int(correct_changes.sum()),
        "harmfulChanges": int(harmful.sum()),
        "changePrecision": round(
            float(correct_changes.sum()) / max(1, int(changed.sum())), 6
        ),
    }


def select_policy(
    folds: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    for threshold in np.linspace(0.40, 0.90, 11):
        for margin in np.linspace(0.0, 0.40, 9):
            fold_metrics = {}
            all_records = []
            all_predictions = []
            for song, fold in folds.items():
                predicted = route(fold["probabilities"], float(threshold), float(margin))
                fold_metrics[song] = policy_metrics(fold["records"], predicted)
                all_records.extend(fold["records"])
                all_predictions.extend(int(value) for value in predicted)
            aggregate = policy_metrics(
                all_records, np.asarray(all_predictions, dtype=np.int64)
            )
            worst = min(float(value["exactRateDelta"]) for value in fold_metrics.values())
            trial = {
                "threshold": round(float(threshold), 4),
                "margin": round(float(margin), 4),
                "worstSongExactRateDelta": round(worst, 6),
                "aggregate": aggregate,
                "folds": fold_metrics,
            }
            trial["safe"] = bool(
                worst >= 0
                and int(aggregate["changes"]) >= 10
                and float(aggregate["changePrecision"]) >= 0.70
            )
            trials.append(trial)
    ranked = sorted(
        trials,
        key=lambda item: (
            bool(item["safe"]),
            float(item["aggregate"]["exactRateDelta"]),
            float(item["aggregate"]["changePrecision"]),
            float(item["worstSongExactRateDelta"]),
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0x50484153)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    records = []
    descriptions = []
    for row in manifest.get("songs") or []:
        song_records, description = load_song_records(row)
        records.extend(song_records)
        descriptions.append(description)
    song_ids = sorted({str(record["songId"]) for record in records})
    if len(song_ids) < 3:
        raise ValueError("At least three complete songs are required")
    folds = {}
    for fold_index, held_out in enumerate(song_ids):
        training = [record for record in records if record["songId"] != held_out]
        validation = [record for record in records if record["songId"] == held_out]
        model = fit_model(training, max(200, args.iterations), args.seed + fold_index * 4099)
        features = np.asarray(
            [record["features"] for record in validation], dtype=np.float64
        )
        folds[held_out] = {
            "records": validation,
            "probabilities": probabilities(model, features),
            "trainingSongs": [song for song in song_ids if song != held_out],
        }
    best, trials = select_policy(folds)
    final_model = fit_model(records, max(200, args.iterations), args.seed)
    profile = {
        "schema": "polymath-pianist-onset-phase-router-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(best["safe"]),
        "model": final_model,
        "policy": {
            "minimumConfidence": best["threshold"],
            "minimumMargin": best["margin"],
            "maximumPulseShift": 1,
            "collisionPolicy": "abstain",
        },
        "training": {
            "manifest": str(manifest_path),
            "method": "complete-song leave-one-out one-pulse onset routing",
            "commercialUseAllowed": False,
            "decision": (
                "CANDIDATE_FOR_APPLICATION_TEST"
                if best["safe"]
                else "REJECT_OR_RESEARCH_ONLY"
            ),
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    report = {
        "schema": "polymath-pianist-onset-phase-router-training-v1",
        "evidenceBoundary": (
            "Answer keys label offline examples only. Every probability is from a "
            "model that excluded the complete validation song."
        ),
        "songs": descriptions,
        "bestPolicy": best,
        "topTrials": trials[:12],
        "decision": profile["training"]["decision"],
    }
    atomic_json(args.output_profile.resolve(), profile)
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
