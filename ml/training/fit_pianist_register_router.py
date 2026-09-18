"""Fit a conservative, song-independent octave router for arranged piano notes.

The piano arranger often finds the correct pitch class but places it in the
wrong octave.  This trainer learns only that final octave decision from
candidate-side information (hand, role, source family, register and position
inside the gesture).  It never receives reference coordinates at inference.

Every policy is selected with complete-song leave-one-out validation.  A
non-zero octave move is permitted only when the winning class is sufficiently
confident and separated from the runner-up; otherwise the approved candidate
is left unchanged.
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

try:  # Package and direct-script execution.
    from .analyze_pianist_register_errors import analyze_song
    from .train_piano_arranger_adapter import train_logistic_model
except ImportError:  # pragma: no cover
    from analyze_pianist_register_errors import analyze_song  # type: ignore
    from train_piano_arranger_adapter import train_logistic_model  # type: ignore


SHIFT_CLASSES = (-24, -12, 0, 12, 24)
FEATURE_NAMES = (
    "bias",
    "candidate_midi_centered",
    "candidate_midi_squared",
    "source_midi_centered",
    "candidate_source_interval",
    "gesture_pitch_rank",
    "gesture_size",
    "same_pitch_class_layers",
    "is_singleton",
    "is_lowest",
    "is_highest",
    "hand_left",
    "hand_right",
    "role_bass",
    "role_harmony",
    "role_melody",
    "family_voice",
    "family_guitar",
    "family_bass",
    "family_piano",
    "family_strings",
    "band_bass",
    "band_lower",
    "band_middle",
    "band_upper",
    "band_high",
    *(f"pitch_class_{value:02d}" for value in range(12)),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def rounded_shift(value: Any) -> int:
    semitones = int(round(float(value) / 12.0)) * 12
    return max(SHIFT_CLASSES[0], min(SHIFT_CLASSES[-1], semitones))


def record_features(record: dict[str, Any]) -> list[float]:
    midi = int(record["candidateMidi"])
    source_midi = int(record.get("sourceMidi", midi))
    centered = (midi - 69.0) / 24.0
    source_centered = (source_midi - 69.0) / 24.0
    hand = str(record.get("candidateHand") or "unknown")
    role = str(record.get("candidateRole") or "unknown")
    family = str(record.get("sourceFamily") or "other")
    band = str(record.get("candidatePitchBand") or "")
    pitch_class = midi % 12
    return [
        1.0,
        centered,
        centered * centered,
        source_centered,
        (midi - source_midi) / 24.0,
        float(record.get("pitchRank", 0.0)),
        min(8.0, float(record.get("gestureSize", 1))) / 8.0,
        min(4.0, float(record.get("samePitchClassLayers", 1))) / 4.0,
        float(int(record.get("gestureSize", 1)) == 1),
        float(bool(record.get("isLowest"))),
        float(bool(record.get("isHighest"))),
        float(hand == "left"),
        float(hand == "right"),
        float(role == "bass"),
        float(role == "harmony"),
        float(role == "melody"),
        float(family == "voice"),
        float(family == "guitar"),
        float(family == "bass"),
        float(family == "piano"),
        float(family == "strings"),
        float(band == "bass-<C3"),
        float(band == "lower-C3-B3"),
        float(band == "middle-C4-B4"),
        float(band == "upper-C5-B5"),
        float(band == "high->=C6"),
        *(float(pitch_class == value) for value in range(12)),
    ]


def balanced_weights(
    records: list[dict[str, Any]], class_balance: float
) -> np.ndarray:
    class_balance = max(0.0, min(1.0, float(class_balance)))
    song_counts = Counter(str(record["songId"]) for record in records)
    label_counts = Counter(
        (str(record["songId"]), int(record["label"])) for record in records
    )
    song_labels: dict[str, set[int]] = {}
    for song, label in label_counts:
        song_labels.setdefault(song, set()).add(label)
    song_total = max(1, len(song_counts))
    values: list[float] = []
    for record in records:
        song = str(record["songId"])
        label = int(record["label"])
        song_equal = 1.0 / (song_total * song_counts[song])
        class_equal = 1.0 / (
            song_total
            * max(1, len(song_labels[song]))
            * label_counts[(song, label)]
        )
        values.append((1.0 - class_balance) * song_equal + class_balance * class_equal)
    weights = np.asarray(values, dtype=np.float64)
    return weights * len(weights) / max(1e-12, float(weights.sum()))


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
        "type": "one-v-rest-standardized-logistic-register-router-v1",
        "featureNames": list(FEATURE_NAMES),
        "shiftClasses": list(SHIFT_CLASSES),
        "classBalance": round(class_balance, 4),
        "classes": classes,
    }


def model_probabilities(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    columns: list[np.ndarray] = []
    for shift in SHIFT_CLASSES:
        payload = model["classes"][str(shift)]
        weights = np.asarray(payload["weights"], dtype=np.float64)
        means = np.asarray(payload["means"], dtype=np.float64)
        scales = np.asarray(payload["scales"], dtype=np.float64)
        logits = np.clip(((features - means) / scales) @ weights, -30.0, 30.0)
        columns.append(1.0 / (1.0 + np.exp(-logits)))
    return np.column_stack(columns)


def route_predictions(
    probabilities: np.ndarray,
    *,
    threshold: float,
    margin: float,
) -> np.ndarray:
    winners = np.argmax(probabilities, axis=1)
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    separation = ordered[:, -1] - ordered[:, -2]
    output = np.zeros(len(probabilities), dtype=np.int64)
    accepted = (confidence >= threshold) & (separation >= margin)
    output[accepted] = np.asarray(SHIFT_CLASSES, dtype=np.int64)[winners[accepted]]
    return output


def policy_metrics(
    records: list[dict[str, Any]], predictions: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    baseline_correct = labels == 0
    predicted_correct = predictions == labels
    changed = predictions != 0
    correct_changes = changed & predicted_correct
    harmful_changes = changed & ~predicted_correct & baseline_correct
    return {
        "examples": len(records),
        "baselineExactRate": round(float(baseline_correct.mean()), 6),
        "routedExactRate": round(float(predicted_correct.mean()), 6),
        "exactRateDelta": round(
            float(predicted_correct.mean() - baseline_correct.mean()), 6
        ),
        "changes": int(changed.sum()),
        "correctChanges": int(correct_changes.sum()),
        "harmfulChanges": int(harmful_changes.sum()),
        "changePrecision": round(
            float(correct_changes.sum()) / max(1, int(changed.sum())), 6
        ),
    }


def load_records(
    rows: Iterable[dict[str, Any]], tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    descriptions: list[dict[str, Any]] = []
    for row in rows:
        analysis = analyze_song(row, tolerance)
        descriptions.append(
            {
                "id": analysis["id"],
                **analysis["counts"],
                "requiredShiftBands": analysis["requiredShiftBands"],
            }
        )
        for item in analysis["pairs"]:
            records.append(
                {
                    **item,
                    "songId": analysis["id"],
                    "label": rounded_shift(item["requiredShiftSemitones"]),
                    "features": record_features(item),
                }
            )
    return records, descriptions


def select_policy(
    folds: dict[str, dict[str, Any]],
    thresholds: Iterable[float],
    margins: Iterable[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for threshold in thresholds:
        for margin in margins:
            fold_metrics: dict[str, Any] = {}
            combined_records: list[dict[str, Any]] = []
            combined_predictions: list[int] = []
            for song, fold in folds.items():
                predictions = route_predictions(
                    fold["probabilities"], threshold=threshold, margin=margin
                )
                fold_metrics[song] = policy_metrics(fold["records"], predictions)
                combined_records.extend(fold["records"])
                combined_predictions.extend(int(value) for value in predictions)
            aggregate = policy_metrics(
                combined_records, np.asarray(combined_predictions, dtype=np.int64)
            )
            worst_delta = min(
                float(value["exactRateDelta"]) for value in fold_metrics.values()
            )
            trial = {
                "threshold": round(float(threshold), 4),
                "margin": round(float(margin), 4),
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
    return ranked[0], sorted(
        trials,
        key=lambda trial: (
            bool(trial["safe"]),
            float(trial["aggregate"]["exactRateDelta"]),
            float(trial["aggregate"]["changePrecision"]),
        ),
        reverse=True,
    )


def profile_hash(profile: dict[str, Any]) -> str:
    clone = copy.deepcopy(profile)
    clone.pop("profileSha256", None)
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0x52454749)
    parser.add_argument("--class-balance", type=float, default=0.55)
    parser.add_argument("--exclude-song", action="append", default=[])
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    excluded = {str(value) for value in args.exclude_song if str(value)}
    rows = [
        row
        for row in manifest.get("songs") or []
        if str(row.get("id")) not in excluded
    ]
    if len(rows) < 3:
        raise ValueError("At least three complete songs are required.")
    records, descriptions = load_records(rows, max(0.01, args.tolerance_seconds))
    song_ids = sorted({str(record["songId"]) for record in records})
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

    best, trials = select_policy(
        folds,
        thresholds=np.linspace(0.35, 0.90, 12),
        margins=np.linspace(0.0, 0.40, 9),
    )
    final_model = fit_models(
        records,
        class_balance=args.class_balance,
        iterations=max(200, args.iterations),
        seed=args.seed,
    )
    profile = {
        "schema": "polymath-pianist-register-router-v1",
        "id": args.profile_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(best["safe"]),
        "model": final_model,
        "policy": {
            "minimumConfidence": best["threshold"],
            "minimumMargin": best["margin"],
            "minimumMidi": 33,
            "maximumMidi": 108,
            "collisionPolicy": "abstain",
        },
        "training": {
            "manifest": str(manifest_path),
            "excludedSongs": sorted(excluded),
            "method": "complete-song leave-one-out conservative octave routing",
            "commercialUseAllowed": False,
            "decision": "CANDIDATE_FOR_APPLICATION_TEST" if best["safe"] else "REJECT_OR_RESEARCH_ONLY",
        },
    }
    profile["profileSha256"] = profile_hash(profile)
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "polymath-pianist-register-router-training-v1",
        "evidenceBoundary": (
            "Private research only; complete-song folds; reference coordinates are "
            "used only for offline labels and never by the runtime router."
        ),
        "manifest": str(manifest_path),
        "excludedSongs": sorted(excluded),
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
