"""Fit an audio-aware pitch-class membership decoder with chord-set output.

Each pitch class is scored relative to the current candidate, raw source,
neighboring gestures, and target-blind full-mix chroma.  The twelve
probabilities are not thresholded independently: a compact whole-gesture
lattice is scored under the joint Bernoulli likelihood, preserving coherent
chord output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .analyze_pitch_class_state_lattice import micro_metrics, set_f1
from .fit_pitch_class_state_ranker import (
    canonical_song_id,
    finite,
    normalize_sample_weights,
    prepare_song,
)


CONTEXT_FIELDS = (
    (0, "candidatePitchClasses", "candidate_now"),
    (0, "sourcePitchClasses", "source_now"),
    (-1, "candidatePitchClasses", "candidate_previous"),
    (-1, "sourcePitchClasses", "source_previous"),
    (1, "candidatePitchClasses", "candidate_next"),
    (1, "sourcePitchClasses", "source_next"),
)
AUDIO_VECTOR_FIELDS = (
    ("meanChroma", "audio_mean"),
    ("peakChroma", "audio_peak"),
    ("attackChroma", "audio_attack"),
    ("bassChroma", "audio_bass"),
    ("trebleChroma", "audio_treble"),
)
SPECIFIC_OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
MULTIVIEW_NAMES = ("fullMix", "vocals", "backing")
MULTIVIEW_CONTEXT_OFFSETS = (-2, -1, 0, 1, 2)
MULTIVIEW_RELATIVE_NAMES = (
    "view_agreement",
    "attack_agreement",
    "fullmix_presence",
    "vocal_presence",
    "backing_presence",
)
MULTIVIEW_FEATURE_NAMES = tuple(
    f"multiview_{name}_relative_{interval:+d}"
    for name in MULTIVIEW_RELATIVE_NAMES
    for interval in range(12)
) + tuple(
    f"multiview_offset_{offset:+d}_{name}"
    for offset in MULTIVIEW_CONTEXT_OFFSETS
    for name in MULTIVIEW_RELATIVE_NAMES
) + (
    "multiview_available",
    "multiview_current_agreement",
    "multiview_current_attack_agreement",
) + tuple(
    f"multiview_current_{view}_{field}"
    for view in MULTIVIEW_NAMES
    for field in ("near_count", "attack_count", "onset_closeness", "velocity")
)
FEATURE_NAMES = tuple(
    f"{name}_relative_{interval:+d}"
    for _offset, _field, name in CONTEXT_FIELDS
    for interval in range(12)
) + tuple(
    f"{name}_relative_{interval:+d}"
    for _field, name in AUDIO_VECTOR_FIELDS
    for interval in range(12)
) + tuple(
    f"offset_{offset:+d}_{kind}"
    for offset in SPECIFIC_OFFSETS
    for kind in (
        "candidate",
        "source",
        "audio_mean",
        "audio_attack",
        "audio_treble",
    )
) + (
    "source_note_count",
    "source_attack_count",
    "source_sustain_only_count",
    "source_onset_offset_signed",
    "source_onset_closeness",
    "source_duration",
    "source_velocity",
    "source_register_closeness",
    "source_exact_candidate_midi",
    "source_voice_share",
    "source_piano_share",
    "source_guitar_share",
    "source_bass_share",
    "source_strings_share",
    "source_other_share",
    "candidate_size",
    "source_size",
    "candidate_left_share",
    "candidate_melody_share",
    "candidate_bass_share",
    "source_voice_cell_share",
    "source_guitar_cell_share",
    "source_bass_cell_share",
    "previous_gap_log",
    "next_gap_log",
    "local_onset_density",
    "candidate_duration",
    "candidate_velocity",
    "role_has_melody",
    "role_has_harmony",
    "role_has_bass",
) + MULTIVIEW_FEATURE_NAMES


def pitch_set(values: Iterable[Any]) -> set[int]:
    return {
        int(value) % 12
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def relative_bits(values: Iterable[Any], pitch_class: int) -> list[float]:
    selected = pitch_set(values)
    return [float((pitch_class + interval) % 12 in selected) for interval in range(12)]


def source_evidence(cell: dict[str, Any], pitch_class: int) -> dict[str, Any]:
    return next(
        (
            row
            for row in cell.get("sourcePitchClassEvidence") or []
            if int(row.get("pitchClass", -1)) % 12 == pitch_class
        ),
        {},
    )


def audio_evidence(cell: dict[str, Any], pitch_class: int) -> dict[str, Any]:
    return next(
        (
            row
            for row in cell.get("audioPitchClassEvidence") or []
            if int(row.get("pitchClass", -1)) % 12 == pitch_class
        ),
        {},
    )


def multiview_evidence(cell: dict[str, Any], pitch_class: int) -> dict[str, Any]:
    return next(
        (
            row
            for row in cell.get("multiViewPitchClassEvidence") or []
            if int(row.get("pitchClass", -1)) % 12 == pitch_class
        ),
        {},
    )


def multiview_relative_value(
    cell: dict[str, Any], pitch_class: int, name: str
) -> float:
    evidence = multiview_evidence(cell, pitch_class)
    if name == "view_agreement":
        return min(1.0, max(0.0, finite(evidence.get("viewAgreementCount"))) / 3.0)
    if name == "attack_agreement":
        return min(1.0, max(0.0, finite(evidence.get("attackAgreementCount"))) / 3.0)
    view_name = {
        "fullmix_presence": "fullMix",
        "vocal_presence": "vocals",
        "backing_presence": "backing",
    }[name]
    view = (evidence.get("views") or {}).get(view_name) or {}
    return float(bool(view.get("available")) and finite(view.get("nearCount")) > 0)


def audio_vector(cell: dict[str, Any], field: str) -> list[float]:
    evidence = {
        int(row.get("pitchClass", -1)) % 12: row
        for row in cell.get("audioPitchClassEvidence") or []
        if isinstance(row, dict) and isinstance(row.get("pitchClass"), (int, float))
    }
    return [max(0.0, min(1.0, finite(evidence.get(pc, {}).get(field)))) for pc in range(12)]


def relative_audio(cell: dict[str, Any], pitch_class: int, field: str) -> list[float]:
    values = audio_vector(cell, field)
    return [values[(pitch_class + interval) % 12] for interval in range(12)]


def cell_at(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return rows[min(len(rows) - 1, max(0, index))]


def feature_vector(
    rows: list[dict[str, Any]], index: int, pitch_class: int
) -> list[float]:
    cell = rows[index]
    values: list[float] = []
    for offset, field, _name in CONTEXT_FIELDS:
        values.extend(relative_bits(cell_at(rows, index + offset).get(field) or [], pitch_class))
    for field, _name in AUDIO_VECTOR_FIELDS:
        values.extend(relative_audio(cell, pitch_class, field))
    for offset in SPECIFIC_OFFSETS:
        neighbor = cell_at(rows, index + offset)
        candidate = pitch_set(neighbor.get("candidatePitchClasses") or [])
        source = pitch_set(neighbor.get("sourcePitchClasses") or [])
        audio = audio_evidence(neighbor, pitch_class)
        values.extend(
            [
                float(pitch_class in candidate),
                float(pitch_class in source),
                max(0.0, min(1.0, finite(audio.get("meanChroma")))),
                max(0.0, min(1.0, finite(audio.get("attackChroma")))),
                max(0.0, min(1.0, finite(audio.get("trebleChroma")))),
            ]
        )
    evidence = source_evidence(cell, pitch_class)
    families = evidence.get("familyShares") or {}
    onset_offset = finite(evidence.get("nearestOnsetOffsetSeconds"))
    role = str(cell.get("candidateRoleSignature") or "")
    values.extend(
        [
            min(1.0, max(0.0, finite(evidence.get("noteCount"))) / 3.0),
            min(1.0, max(0.0, finite(evidence.get("attackCount"))) / 2.0),
            min(1.0, max(0.0, finite(evidence.get("sustainOnlyCount"))) / 3.0),
            max(-1.0, min(1.0, onset_offset / 0.14)),
            math.exp(-abs(onset_offset) / 0.09),
            min(1.0, max(0.0, finite(evidence.get("medianDurationSeconds"))) / 2.0),
            min(1.0, max(0.0, finite(evidence.get("maximumVelocity")))),
            math.exp(
                -max(0.0, finite(evidence.get("minimumRegisterDistanceSemitones")))
                / 12.0
            ),
            float(bool(evidence.get("exactCandidateMidiMatch"))),
            finite(families.get("voice")),
            finite(families.get("piano")),
            finite(families.get("guitar")),
            finite(families.get("bass")),
            finite(families.get("strings")),
            finite(families.get("other")),
            min(1.0, len(pitch_set(cell.get("candidatePitchClasses") or [])) / 6.0),
            min(1.0, len(pitch_set(cell.get("sourcePitchClasses") or [])) / 12.0),
            min(1.0, max(0.0, finite(cell.get("candidateLeftShare")))),
            min(1.0, max(0.0, finite(cell.get("candidateMelodyShare")))),
            min(1.0, max(0.0, finite(cell.get("candidateBassShare")))),
            min(1.0, max(0.0, finite(cell.get("sourceVoiceShare")))),
            min(1.0, max(0.0, finite(cell.get("sourceGuitarShare")))),
            min(1.0, max(0.0, finite(cell.get("sourceBassShare")))),
            math.log1p(max(0.03, finite(cell.get("previousGapSeconds"), 0.3))),
            math.log1p(max(0.03, finite(cell.get("nextGapSeconds"), 0.3))),
            min(1.0, max(0.0, finite(cell.get("localCandidateOnsetsPerSecond"))) / 8.0),
            min(1.0, max(0.03, finite(cell.get("candidateDuration"), 0.3)) / 2.0),
            min(1.0, max(0.01, finite(cell.get("candidateVelocity"), 0.7))),
            float("melody" in role),
            float("harmony" in role),
            float("bass" in role),
        ]
    )
    for name in MULTIVIEW_RELATIVE_NAMES:
        values.extend(
            multiview_relative_value(cell, (pitch_class + interval) % 12, name)
            for interval in range(12)
        )
    for offset in MULTIVIEW_CONTEXT_OFFSETS:
        neighbor = cell_at(rows, index + offset)
        values.extend(
            multiview_relative_value(neighbor, pitch_class, name)
            for name in MULTIVIEW_RELATIVE_NAMES
        )
    current_multiview = multiview_evidence(cell, pitch_class)
    current_views = current_multiview.get("views") or {}
    values.extend(
        [
            float(bool(cell.get("multiViewAvailable"))),
            min(
                1.0,
                max(0.0, finite(current_multiview.get("viewAgreementCount"))) / 3.0,
            ),
            min(
                1.0,
                max(0.0, finite(current_multiview.get("attackAgreementCount"))) / 3.0,
            ),
        ]
    )
    for view_name in MULTIVIEW_NAMES:
        view = current_views.get(view_name) or {}
        available = bool(view.get("available"))
        near_count = max(0.0, finite(view.get("nearCount"))) if available else 0.0
        attack_count = max(0.0, finite(view.get("attackCount"))) if available else 0.0
        distance = max(
            0.0, finite(view.get("nearestOnsetDistanceSeconds"), 0.25)
        )
        values.extend(
            [
                min(1.0, near_count / 3.0),
                min(1.0, attack_count / 2.0),
                math.exp(-distance / 0.10) if near_count > 0 else 0.0,
                min(1.0, max(0.0, finite(view.get("maximumVelocity"))))
                if available
                else 0.0,
            ]
        )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(
            f"Pitch membership feature drift: {len(values)} != {len(FEATURE_NAMES)}"
        )
    return values


def prepare_membership_song(
    raw_song: dict[str, Any], *, maximum_states: int
) -> dict[str, Any]:
    prepared = prepare_song(raw_song, maximum_states=maximum_states)
    raw_cells = list(raw_song.get("cells") or [])
    for cell in prepared["cells"]:
        index = int(cell["index"])
        cell["membershipFeatures"] = np.asarray(
            [feature_vector(raw_cells, index, pc) for pc in range(12)],
            dtype=np.float32,
        )
        cell["membershipLabels"] = np.asarray(
            [float(pc in cell["target"]) for pc in range(12)], dtype=np.int8
        )
    return prepared


def training_arrays(
    songs: list[dict[str, Any]], *, positive_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_totals: dict[str, int] = {}
    for song in songs:
        group_totals[song["groupId"]] = group_totals.get(song["groupId"], 0) + 1
    for song in songs:
        cell_weight = (
            max(1e-9, song["weight"])
            / max(1, group_totals[song["groupId"]])
            / max(1, len(song["cells"]))
        )
        for cell in song["cells"]:
            cell_labels = cell["membershipLabels"]
            cell_weights = np.full(12, cell_weight / 12.0, dtype=float)
            cell_weights *= np.where(cell_labels > 0, max(0.1, positive_weight), 1.0)
            matrices.append(cell["membershipFeatures"])
            labels.append(cell_labels)
            weights.append(cell_weights)
    return np.vstack(matrices), np.concatenate(labels), np.concatenate(weights)


def fit_model(
    songs: list[dict[str, Any]],
    *,
    positive_weight: float,
    maximum_leaf_nodes: int,
    maximum_iterations: int,
) -> HistGradientBoostingClassifier:
    matrix, labels, weights = training_arrays(songs, positive_weight=positive_weight)
    weights = normalize_sample_weights(weights)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=max(1, int(maximum_iterations)),
        max_leaf_nodes=maximum_leaf_nodes,
        min_samples_leaf=180,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=6701,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def lattice_state_score(state: frozenset[int], probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    selected = np.zeros(12, dtype=bool)
    selected[list(state)] = True
    return float(np.sum(np.where(selected, np.log(clipped), np.log1p(-clipped))))


def included_logit_score(state: frozenset[int], probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped) - np.log1p(-clipped)
    return float(np.sum(logits[list(state)]))


def score_song_cells(
    song: dict[str, Any], model: HistGradientBoostingClassifier
) -> list[dict[str, Any]]:
    matrix = np.vstack([cell["membershipFeatures"] for cell in song["cells"]])
    probabilities = model.predict_proba(matrix)[:, 1].reshape(len(song["cells"]), 12)
    return [
        {"cell": cell, "probabilities": row}
        for cell, row in zip(song["cells"], probabilities)
    ]


def evaluate_scored(
    song: dict[str, Any],
    scored: list[dict[str, Any]],
    *,
    candidate_bonus: float,
    decode_mode: str = "joint",
    size_penalty: float = 0.0,
) -> dict[str, Any]:
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    baseline_counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    exact = baseline_exact = changes = improvements = regressions = 0
    reward = baseline_reward = 0.0
    for item in scored:
        cell = item["cell"]
        probabilities = item["probabilities"]
        candidate = cell["candidate"]
        scores: list[float] = []
        for state in cell["states"]:
            size_difference = abs(len(state) - len(candidate))
            if decode_mode == "same-size" and size_difference:
                score = -1e9
            elif decode_mode in {"same-size", "within-one"}:
                if decode_mode == "within-one" and size_difference > 1:
                    score = -1e9
                else:
                    score = included_logit_score(state, probabilities)
                    score -= max(0.0, size_penalty) * size_difference
            elif decode_mode == "joint":
                score = lattice_state_score(state, probabilities)
            else:
                raise ValueError(f"Unknown decode mode: {decode_mode}")
            if state == candidate:
                score += max(0.0, candidate_bonus)
            scores.append(score)
        selected = cell["states"][int(np.argmax(scores))]
        target = cell["target"]
        for bucket, state in ((counts, selected), (baseline_counts, candidate)):
            bucket["truePositive"] += len(target & state)
            bucket["falsePositive"] += len(state - target)
            bucket["falseNegative"] += len(target - state)
        selected_reward = set_f1(target, selected)
        current_reward = set_f1(target, candidate)
        reward += selected_reward
        baseline_reward += current_reward
        exact += int(selected == target)
        baseline_exact += int(candidate == target)
        changes += int(selected != candidate)
        improvements += int(selected_reward > current_reward + 1e-9)
        regressions += int(selected_reward + 1e-9 < current_reward)
    cells = len(song["cells"])
    return {
        "id": song["id"],
        "groupId": song["groupId"],
        "cells": cells,
        "baseline": {
            **micro_metrics(baseline_counts),
            "macroF1": round(baseline_reward / max(1, cells), 6),
            "exactSetRate": round(baseline_exact / max(1, cells), 6),
        },
        "decoder": {
            **micro_metrics(counts),
            "macroF1": round(reward / max(1, cells), 6),
            "exactSetRate": round(exact / max(1, cells), 6),
        },
        "changes": changes,
        "improvements": improvements,
        "regressions": regressions,
    }


def weighted(rows: list[dict[str, Any]], weights: dict[str, float], field: str) -> float:
    total = sum(max(0.0, weights.get(row["id"], 1.0)) for row in rows)
    return sum(
        max(0.0, weights.get(row["id"], 1.0)) * float(row["decoder"][field])
        for row in rows
    ) / max(1e-12, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, nargs="+")
    parser.add_argument("--evaluation-song-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--extra-audit-weight", type=float, default=0.5)
    parser.add_argument("--positive-weight-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--leaf-grid", type=int, nargs="+", default=[15])
    parser.add_argument("--maximum-iterations", type=int, default=150)
    parser.add_argument(
        "--candidate-bonus-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--decode-mode-grid",
        nargs="+",
        choices=["joint", "same-size", "within-one"],
        default=["joint", "same-size", "within-one"],
    )
    parser.add_argument(
        "--size-penalty-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0]
    )
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.audit]
    raw_songs: list[dict[str, Any]] = []
    for audit_index, path in enumerate(paths):
        audit = json.loads(path.read_text(encoding="utf-8-sig"))
        multiplier = 1.0 if audit_index == 0 else max(0.0, args.extra_audit_weight)
        for song in audit.get("songs") or []:
            copied = dict(song)
            copied["weight"] = max(0.0, finite(song.get("weight"), 1.0)) * multiplier
            copied["groupId"] = canonical_song_id(song.get("id"))
            raw_songs.append(copied)
    maximum_states = max(4, min(256, int(args.maximum_states)))
    songs = [
        prepare_membership_song(song, maximum_states=maximum_states)
        for song in raw_songs
    ]
    wanted = set(args.evaluation_song_ids)
    evaluation = [song for song in songs if song["id"] in wanted]
    missing = wanted - {song["id"] for song in evaluation}
    if missing:
        raise ValueError(f"Unknown evaluation songs: {sorted(missing)}")
    song_weights = {song["id"]: song["weight"] for song in evaluation}

    trials: list[dict[str, Any]] = []
    fold_cache: dict[tuple[float, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    for positive_weight in args.positive_weight_grid:
        for leaves in args.leaf_grid:
            scored_folds: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for held_out in evaluation:
                training = [
                    song for song in songs if song["groupId"] != held_out["groupId"]
                ]
                model = fit_model(
                    training,
                    positive_weight=max(0.1, float(positive_weight)),
                    maximum_leaf_nodes=max(3, int(leaves)),
                    maximum_iterations=max(1, args.maximum_iterations),
                )
                scored_folds.append((held_out, score_song_cells(held_out, model)))
            fold_cache[(float(positive_weight), int(leaves))] = scored_folds
            for decode_mode in args.decode_mode_grid:
                penalties = args.size_penalty_grid if decode_mode == "within-one" else [0.0]
                for size_penalty in penalties:
                    for bonus in args.candidate_bonus_grid:
                        folds = [
                            evaluate_scored(
                                song,
                                scored,
                                candidate_bonus=float(bonus),
                                decode_mode=decode_mode,
                                size_penalty=float(size_penalty),
                            )
                            for song, scored in scored_folds
                        ]
                        trials.append(
                            {
                                "positiveWeight": float(positive_weight),
                                "maximumLeafNodes": int(leaves),
                                "decodeMode": decode_mode,
                                "sizePenalty": float(size_penalty),
                                "candidateBonus": float(bonus),
                                "weightedPitchClassF1": round(weighted(folds, song_weights, "f1"), 6),
                                "worstSongPitchClassF1": round(
                                    min(float(row["decoder"]["f1"]) for row in folds), 6
                                ),
                                "weightedExactSetRate": round(
                                    weighted(folds, song_weights, "exactSetRate"), 6
                                ),
                                "folds": folds,
                            }
                        )
    best = max(
        trials,
        key=lambda row: (
            float(row["weightedPitchClassF1"]),
            float(row["worstSongPitchClassF1"]),
            float(row["weightedExactSetRate"]),
        ),
    )
    final = fit_model(
        songs,
        positive_weight=float(best["positiveWeight"]),
        maximum_leaf_nodes=int(best["maximumLeafNodes"]),
        maximum_iterations=max(1, args.maximum_iterations),
    )
    model_path = Path(args.model_output).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "polymath-audio-pitch-class-membership-decoder-v1",
            "featureNames": FEATURE_NAMES,
            "maximumStates": maximum_states,
            "positiveWeight": best["positiveWeight"],
            "maximumLeafNodes": best["maximumLeafNodes"],
            "candidateBonus": best["candidateBonus"],
            "decodeMode": best["decodeMode"],
            "sizePenalty": best["sizePenalty"],
            "model": final,
        },
        model_path,
    )
    report = {
        "schema": "polymath-audio-pitch-class-membership-training-v1",
        "status": "opened-development-only",
        "evidenceBoundary": (
            "Candidate/raw/source-family/full-mix-audio/neighbor evidence only. "
            "Synchronized full-mix/vocal/backing evidence is target-blind. Targets "
            "create labels and are excluded from every held-out composition group."
        ),
        "audits": [str(path) for path in paths],
        "evaluationSongIds": args.evaluation_song_ids,
        "featureCount": len(FEATURE_NAMES),
        "best": best,
        "trials": trials,
        "model": str(model_path),
        "decision": {
            "promotable": False,
            "reason": "Opened development only; exact-note and unopened-song gates remain.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "model": str(model_path),
                "featureCount": len(FEATURE_NAMES),
                "positiveWeight": best["positiveWeight"],
                "decodeMode": best["decodeMode"],
                "sizePenalty": best["sizePenalty"],
                "candidateBonus": best["candidateBonus"],
                "weightedPitchClassF1": best["weightedPitchClassF1"],
                "worstSongPitchClassF1": best["worstSongPitchClassF1"],
                "weightedExactSetRate": best["weightedExactSetRate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
