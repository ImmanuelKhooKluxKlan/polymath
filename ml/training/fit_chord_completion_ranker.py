"""Fit a chord-completion ranker at the exact onsets retained by a baseline.

The ordinary note classifier can score well while choosing a worse top-two
chord.  This trainer instead creates positive-vs-negative pairs inside each
retained accompaniment onset, which matches the decoder's real decision.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    normalize_source_notes,
    selection_feature_rows,
)
try:
    from .analyze_left_hand_accompaniment import (
        candidate_left_notes,
        onset_groups,
        reference_left_notes,
    )
    from .evaluate_piano_arranger import load_json, trusted_source_ranges
    from .fit_left_hand_accompaniment import (
        classification_metrics,
        model_payload,
        train_logistic_model,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from analyze_left_hand_accompaniment import (  # type: ignore
        candidate_left_notes,
        onset_groups,
        reference_left_notes,
    )
    from evaluate_piano_arranger import load_json, trusted_source_ranges  # type: ignore
    from fit_left_hand_accompaniment import (  # type: ignore
        classification_metrics,
        model_payload,
        train_logistic_model,
    )


VOICE_INSTRUMENTS = {"voice", "vocal", "vocals", "singing_voice"}


def match_onset_groups(
    reference_groups: list[list[dict[str, Any]]],
    candidate_groups: list[list[dict[str, Any]]],
    tolerance: float,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    available = set(range(len(reference_groups)))
    matches: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for candidate_group in candidate_groups:
        time = float(candidate_group[0]["time"])
        possible = [
            index
            for index in available
            if abs(float(reference_groups[index][0]["time"]) - time) <= tolerance
        ]
        if not possible:
            continue
        reference_index = min(
            possible,
            key=lambda index: abs(float(reference_groups[index][0]["time"]) - time),
        )
        available.remove(reference_index)
        matches.append((candidate_group, reference_groups[reference_index]))
    return matches


def representative_by_pitch_class(
    raw_notes: list[dict[str, Any]],
    rows_by_source_index: dict[int, np.ndarray],
) -> dict[int, tuple[dict[str, Any], np.ndarray]]:
    feature_indices = {
        name: index
        for index, name in enumerate(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
    }
    representatives: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    for note in raw_notes:
        if str(note.get("instrument") or "").lower() in VOICE_INSTRUMENTS:
            continue
        row = rows_by_source_index[int(note["sourceIndex"])]
        pitch_class = int(note["midi"]) % 12
        salience = (
            float(row[feature_indices["wide_pitch_class_support"]])
            + 0.30 * float(row[feature_indices["cross_family_pitch_class_support"]])
            + 0.10 * float(row[feature_indices["local_duration_percentile"]])
        )
        previous = representatives.get(pitch_class)
        if previous is None:
            representatives[pitch_class] = (note, row)
            continue
        previous_row = previous[1]
        previous_salience = (
            float(previous_row[feature_indices["wide_pitch_class_support"]])
            + 0.30 * float(previous_row[feature_indices["cross_family_pitch_class_support"]])
            + 0.10 * float(previous_row[feature_indices["local_duration_percentile"]])
        )
        if salience > previous_salience:
            representatives[pitch_class] = (note, row)
    return representatives


def top_k_audit(
    matched: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    raw: list[dict[str, Any]],
    raw_times: list[float],
    rows_by_source_index: dict[int, np.ndarray],
    learned_weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    radius: float,
    *,
    held_out: bool,
) -> dict[str, Any]:
    ideal_memberships = 0
    current_hits = 0
    model_hits = 0
    improved = 0
    worsened = 0
    unchanged = 0
    onsets = 0
    for candidate_group, reference_group in matched:
        time = float(candidate_group[0]["time"])
        if (int(time // 20.0) % 5 == 4) != held_out:
            continue
        left = bisect.bisect_left(raw_times, time - radius)
        right = bisect.bisect_right(raw_times, time + radius)
        representatives = representative_by_pitch_class(
            raw[left:right], rows_by_source_index
        )
        target = {int(note["midi"]) % 12 for note in reference_group}
        current = {int(note["midi"]) % 12 for note in candidate_group}
        scored = []
        for pitch_class, (_note, row) in representatives.items():
            standardized = (row - means) / scales
            score = float(standardized @ learned_weights)
            scored.append((score, pitch_class))
        selected = {
            pitch_class
            for _score, pitch_class in sorted(scored, reverse=True)[: max(1, len(current))]
        }
        current_count = len(current & target)
        model_count = len(selected & target)
        ideal_memberships += len(target)
        current_hits += current_count
        model_hits += model_count
        onsets += 1
        if model_count > current_count:
            improved += 1
        elif model_count < current_count:
            worsened += 1
        else:
            unchanged += 1
    return {
        "onsets": onsets,
        "idealPitchClassMemberships": ideal_memberships,
        "baselineMembershipRecall": round(current_hits / max(1, ideal_memberships), 6),
        "modelTopKMembershipRecall": round(model_hits / max(1, ideal_memberships), 6),
        "improvedOnsets": improved,
        "worsenedOnsets": worsened,
        "unchangedOnsets": unchanged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--aligned-target", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--onset-template", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--match-tolerance", type=float, default=0.25)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--candidate-hand-split", type=int, default=72)
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=0x43484F52)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    target_path = Path(args.aligned_target).resolve()
    alignment_path = Path(args.alignment).resolve()
    onset_path = Path(args.onset_template).resolve()
    source_payload = load_json(source_path)
    ranges = trusted_source_ranges(load_json(alignment_path))
    raw = normalize_source_notes(source_payload.get("notes", []))
    feature_names = tuple(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
    feature_rows = selection_feature_rows(raw, feature_names)
    rows_by_source_index = {
        int(note["sourceIndex"]): np.asarray(row, dtype=np.float64)
        for note, row in zip(raw, feature_rows)
    }
    raw_times = [float(note["time"]) for note in raw]
    reference = reference_left_notes(
        load_json(target_path), args.reference_hand_split, 12
    )
    candidate = candidate_left_notes(
        load_json(onset_path), ranges, args.candidate_hand_split
    )
    matched = match_onset_groups(
        onset_groups(reference), onset_groups(candidate), args.match_tolerance
    )

    pair_rows: list[np.ndarray] = []
    pair_labels: list[float] = []
    pair_weights: list[float] = []
    training_flags: list[bool] = []
    used_onsets = 0
    skipped_without_both_classes = 0
    for candidate_group, reference_group in matched:
        time = float(candidate_group[0]["time"])
        left = bisect.bisect_left(raw_times, time - args.radius)
        right = bisect.bisect_right(raw_times, time + args.radius)
        representatives = representative_by_pitch_class(
            raw[left:right], rows_by_source_index
        )
        target_pitch_classes = {
            int(note["midi"]) % 12 for note in reference_group
        }
        positives = [
            row
            for pitch_class, (_note, row) in representatives.items()
            if pitch_class in target_pitch_classes
        ]
        negatives = [
            row
            for pitch_class, (_note, row) in representatives.items()
            if pitch_class not in target_pitch_classes
        ]
        if not positives or not negatives:
            skipped_without_both_classes += 1
            continue
        used_onsets += 1
        is_training = int(time // 20.0) % 5 != 4
        for positive in positives:
            for negative in negatives:
                difference = positive - negative
                pair_rows.extend((difference, -difference))
                pair_labels.extend((1.0, 0.0))
                pair_weights.extend((1.0, 1.0))
                training_flags.extend((is_training, is_training))

    features = np.asarray(pair_rows, dtype=np.float64)
    labels = np.asarray(pair_labels, dtype=np.float64)
    weights = np.asarray(pair_weights, dtype=np.float64)
    training_mask = np.asarray(training_flags, dtype=bool)
    validation_mask = ~training_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("Structured ranker requires both training and held-out onset blocks.")
    learned_weights, means, scales, history = train_logistic_model(
        features,
        labels,
        weights,
        training_mask,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=0.014,
        l2=0.01,
    )
    probabilities = 1.0 / (
        1.0
        + np.exp(
            -np.clip(((features - means) / scales) @ learned_weights, -30.0, 30.0)
        )
    )
    model = model_payload(feature_names, learned_weights, means, scales, 0.5)
    model["type"] = "standardized-logistic-structured-chord-ranker-v1"
    profile = {
        "id": args.profile_id,
        "selectionModel": model,
        "training": {
            "schema": "polymath-structured-chord-completion-training-v1",
            "source": str(source_path),
            "alignedTarget": str(target_path),
            "alignment": str(alignment_path),
            "onsetTemplate": str(onset_path),
            "radiusSeconds": args.radius,
            "matchToleranceSeconds": args.match_tolerance,
            "warning": "Kiss Me calibration data; cross-song transfer remains unproven.",
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    output_path = Path(args.output_profile).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema": "polymath-structured-chord-completion-report-v1",
        "profile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "matchedOnsets": len(matched),
        "usedOnsets": used_onsets,
        "skippedWithoutBothClasses": skipped_without_both_classes,
        "pairExamples": len(features),
        "trainingPairs": int(training_mask.sum()),
        "heldOutPairs": int(validation_mask.sum()),
        "pairwiseTrainingMetrics": classification_metrics(
            labels[training_mask], probabilities[training_mask], weights[training_mask], 0.5
        ),
        "pairwiseHeldOutMetrics": classification_metrics(
            labels[validation_mask], probabilities[validation_mask], weights[validation_mask], 0.5
        ),
        "trainingTopK": top_k_audit(
            matched, raw, raw_times, rows_by_source_index,
            learned_weights, means, scales, args.radius, held_out=False
        ),
        "heldOutTopK": top_k_audit(
            matched, raw, raw_times, rows_by_source_index,
            learned_weights, means, scales, args.radius, held_out=True
        ),
        "optimizationHistory": history,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
