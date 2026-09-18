"""Audit local source-note swaps that separate an arranger from a pianist.

The ordinary source-decision audit compares ``sourceIndex`` values.  That is
useful, but separated stems can contain duplicate events at the same pitch and
time.  Treating those duplicates as different musical choices overstates the
error.  This audit first reconciles event-equivalent selections, then pairs a
missed alignment-supported note with an unsupported selected note from the
same local musical moment.

The resulting pairs answer the actionable question: *which source note should
have occupied an already-budgeted slot?*  No target field is intended for
runtime inference.  A later pairwise ranker may learn only the source-side
feature delta and must preserve the frozen onset/note quotas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import (  # noqa: E402
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    instrument_family,
    normalize_source_notes,
    selection_feature_rows,
    selection_scores,
)

try:  # Support package and direct-script execution.
    from .analyze_aligned_source_decisions import (
        inside_ranges,
        strongest_alignment_matches,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover - direct CLI compatibility path.
    from analyze_aligned_source_decisions import (
        inside_ranges,
        strongest_alignment_matches,
        trusted_source_ranges,
    )


CORE_FEATURES = (
    "midi_centered",
    "midi_squared",
    "velocity",
    "duration_log",
    "onset_cluster",
    "local_density",
    "role_melody",
    "role_bass",
    "role_harmony",
    "voice_within_060ms",
    "voice_within_180ms",
    "cross_family_onset_support",
    "cross_family_pitch_class_support",
    "local_duration_percentile",
    "local_velocity_percentile",
    "same_instrument_pitch_class_recurrence",
    "is_local_onset_lowest",
    "is_local_onset_highest",
    "local_onset_pitch_percentile",
    "local_onset_unique_pitch_count",
    "previous_same_pitch_gap_log",
    "next_same_pitch_gap_log",
    "wide_pitch_class_support",
    "wide_pitch_class_rank",
    "wide_pitch_class_dominance",
    "estimated_chord_member_root",
    "estimated_chord_member_third",
    "estimated_chord_member_fifth",
    "estimated_chord_nonmember",
    "estimated_chord_confidence",
    "estimated_bass_root_match",
)


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def candidate_source_indices(payload: dict[str, Any]) -> set[int]:
    selected: set[int] = set()
    for note in payload.get("notes") or []:
        if not isinstance(note, dict):
            continue
        try:
            selected.add(int(note["sourceIndex"]))
        except (KeyError, TypeError, ValueError):
            continue
    return selected


def pitch_band(midi: int) -> str:
    if midi < 48:
        return "bass-<C3"
    if midi < 60:
        return "lower-C3-B3"
    if midi < 72:
        return "middle-C4-B4"
    if midi < 84:
        return "upper-C5-B5"
    return "high->=C6"


def reconcile_equivalent_events(
    desired: set[int],
    selected: set[int],
    notes_by_index: dict[int, dict[str, Any]],
    *,
    radius_seconds: float,
) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    """Remove musically equivalent source events before counting errors."""

    exact = desired & selected
    remaining_desired = set(desired - exact)
    remaining_selected = set(selected - exact)
    matches: list[dict[str, Any]] = [
        {"desiredSourceIndex": index, "selectedSourceIndex": index, "kind": "same-index"}
        for index in sorted(exact)
    ]
    edges: list[tuple[tuple[float, ...], int, int, str]] = []
    for desired_index in remaining_desired:
        target = notes_by_index[desired_index]
        target_family = instrument_family(str(target.get("instrument") or ""))
        for selected_index in remaining_selected:
            observed = notes_by_index[selected_index]
            distance = abs(float(target["time"]) - float(observed["time"]))
            if distance > radius_seconds:
                continue
            exact_pitch = int(target["midi"]) == int(observed["midi"])
            same_pitch_class = int(target["midi"]) % 12 == int(observed["midi"]) % 12
            if not same_pitch_class:
                continue
            observed_family = instrument_family(str(observed.get("instrument") or ""))
            same_family = target_family == observed_family
            kind = "same-pitch" if exact_pitch else "same-pitch-class"
            edges.append(
                (
                    (
                        0.0 if exact_pitch else 1.0,
                        0.0 if same_family else 1.0,
                        distance,
                        abs(int(target["midi"]) - int(observed["midi"])),
                        float(desired_index),
                        float(selected_index),
                    ),
                    desired_index,
                    selected_index,
                    kind,
                )
            )
    for _cost, desired_index, selected_index, kind in sorted(edges):
        if desired_index not in remaining_desired or selected_index not in remaining_selected:
            continue
        remaining_desired.remove(desired_index)
        remaining_selected.remove(selected_index)
        left = notes_by_index[desired_index]
        right = notes_by_index[selected_index]
        matches.append(
            {
                "desiredSourceIndex": desired_index,
                "selectedSourceIndex": selected_index,
                "kind": kind,
                "timeDistanceSeconds": round(abs(float(left["time"]) - float(right["time"])), 6),
                "sameFamily": instrument_family(str(left.get("instrument") or ""))
                == instrument_family(str(right.get("instrument") or "")),
            }
        )
    return remaining_desired, remaining_selected, matches


def pair_local_swaps(
    missed: set[int],
    unsupported: set[int],
    notes_by_index: dict[int, dict[str, Any]],
    *,
    radius_seconds: float,
    onset_radius_seconds: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedily pair missed/unsupported notes from one local musical moment."""

    remaining_missed = set(missed)
    remaining_unsupported = set(unsupported)
    edges: list[tuple[tuple[float, ...], int, int]] = []
    for desired_index in remaining_missed:
        target = notes_by_index[desired_index]
        target_family = instrument_family(str(target.get("instrument") or ""))
        for selected_index in remaining_unsupported:
            observed = notes_by_index[selected_index]
            distance = abs(float(target["time"]) - float(observed["time"]))
            if distance > radius_seconds:
                continue
            observed_family = instrument_family(str(observed.get("instrument") or ""))
            circular_distance = min(
                (int(target["midi"]) - int(observed["midi"])) % 12,
                (int(observed["midi"]) - int(target["midi"])) % 12,
            )
            edges.append(
                (
                    (
                        0.0 if distance <= onset_radius_seconds else 1.0,
                        0.0 if target_family == observed_family else 1.0,
                        distance,
                        float(circular_distance),
                        float(abs(int(target["midi"]) - int(observed["midi"]))),
                        float(desired_index),
                        float(selected_index),
                    ),
                    desired_index,
                    selected_index,
                )
            )
    pairs: list[tuple[int, int]] = []
    for _cost, desired_index, selected_index in sorted(edges):
        if desired_index not in remaining_missed or selected_index not in remaining_unsupported:
            continue
        remaining_missed.remove(desired_index)
        remaining_unsupported.remove(selected_index)
        pairs.append((desired_index, selected_index))
    return pairs, remaining_missed, remaining_unsupported


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    if not kept:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": len(kept),
        "mean": round(mean(kept), 6),
        "minimum": round(min(kept), 6),
        "maximum": round(max(kept), 6),
    }


def model_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decoder = profile.get("decoder") or {}
    left = decoder.get("leftHandAccompaniment") or {}
    conditional = decoder.get("conditionalSelectionBlend") or {}
    models = {
        "base": profile.get("selectionModel"),
        "conditional": conditional.get("selectionModel"),
        "leftHand": left.get("selectionModel"),
        "leftHandOnset": left.get("onsetSelectionModel"),
        "leftHandSupplemental": left.get("supplementalSelectionModel"),
        "chordCompletion": left.get("chordCompletionModel"),
    }
    return {name: value for name, value in models.items() if isinstance(value, dict)}


def analyze_song(
    row: dict[str, Any],
    *,
    profile: dict[str, Any] | None,
    equivalence_radius_seconds: float,
    swap_radius_seconds: float,
    onset_radius_seconds: float,
) -> dict[str, Any]:
    source_path = Path(row["source"]).resolve()
    alignment_path = Path(row["alignment"]).resolve()
    candidate_path = Path(row.get("baseline") or row["candidate"]).resolve()
    source_payload = load_json(source_path)
    alignment = load_json(alignment_path)
    candidate = load_json(candidate_path)
    all_notes = normalize_source_notes(source_payload.get("notes") or [])
    ranges = trusted_source_ranges(alignment)
    candidate_end = finite(row.get("candidateEndSeconds"), float("inf"))
    eligible = [
        note
        for note in all_notes
        if float(note["time"]) < candidate_end
        and inside_ranges(float(note["time"]), ranges)
    ]
    notes_by_index = {int(note["sourceIndex"]): note for note in eligible}
    desired = set(strongest_alignment_matches(alignment)) & set(notes_by_index)
    selected = candidate_source_indices(candidate) & set(notes_by_index)
    missed, unsupported, equivalent = reconcile_equivalent_events(
        desired,
        selected,
        notes_by_index,
        radius_seconds=equivalence_radius_seconds,
    )
    swap_pairs, unpaired_missed, unpaired_unsupported = pair_local_swaps(
        missed,
        unsupported,
        notes_by_index,
        radius_seconds=swap_radius_seconds,
        onset_radius_seconds=onset_radius_seconds,
    )

    feature_rows = selection_feature_rows(
        all_notes, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
    )
    feature_by_index = {
        int(note["sourceIndex"]): dict(
            zip(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES, features)
        )
        for note, features in zip(all_notes, feature_rows)
        if int(note["sourceIndex"]) in notes_by_index
    }
    model_scores: dict[str, dict[int, float]] = {}
    if profile:
        for name, model in model_map(profile).items():
            scores = selection_scores(all_notes, {"selectionModel": model})
            model_scores[name] = {
                int(note["sourceIndex"]): float(score)
                for note, score in zip(all_notes, scores)
                if int(note["sourceIndex"]) in notes_by_index
            }

    pair_rows: list[dict[str, Any]] = []
    for desired_index, selected_index in swap_pairs:
        target = notes_by_index[desired_index]
        observed = notes_by_index[selected_index]
        target_features = feature_by_index[desired_index]
        observed_features = feature_by_index[selected_index]
        distance = abs(float(target["time"]) - float(observed["time"]))
        target_family = instrument_family(str(target.get("instrument") or ""))
        observed_family = instrument_family(str(observed.get("instrument") or ""))
        scores = {
            name: {
                "desired": round(values[desired_index], 6),
                "selected": round(values[selected_index], 6),
                "desiredMinusSelected": round(
                    values[desired_index] - values[selected_index], 6
                ),
                "desiredWins": values[desired_index] > values[selected_index],
            }
            for name, values in model_scores.items()
        }
        pair_rows.append(
            {
                "desiredSourceIndex": desired_index,
                "selectedSourceIndex": selected_index,
                "timeSeconds": round(float(target["time"]), 6),
                "timeDistanceSeconds": round(distance, 6),
                "sameOnset": distance <= onset_radius_seconds,
                "sameFamily": target_family == observed_family,
                "desired": {
                    "midi": int(target["midi"]),
                    "pitchBand": pitch_band(int(target["midi"])),
                    "family": target_family,
                    "instrument": str(target.get("instrument") or "other"),
                    "velocity": round(float(target["velocity"]), 6),
                    "duration": round(float(target["duration"]), 6),
                },
                "selected": {
                    "midi": int(observed["midi"]),
                    "pitchBand": pitch_band(int(observed["midi"])),
                    "family": observed_family,
                    "instrument": str(observed.get("instrument") or "other"),
                    "velocity": round(float(observed["velocity"]), 6),
                    "duration": round(float(observed["duration"]), 6),
                },
                "featureDeltaDesiredMinusSelected": {
                    name: round(float(target_features[name] - observed_features[name]), 6)
                    for name in HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
                },
                "modelScores": scores,
            }
        )

    high_confidence = [
        item for item in pair_rows if item["sameOnset"] and item["sameFamily"]
    ]
    feature_summary = {
        name: distribution(
            item["featureDeltaDesiredMinusSelected"][name]
            for item in high_confidence
        )
        for name in CORE_FEATURES
    }
    model_summary = {}
    for name in model_scores:
        deltas = [
            float(item["modelScores"][name]["desiredMinusSelected"])
            for item in high_confidence
        ]
        model_summary[name] = {
            "pairs": len(deltas),
            "desiredWinRate": round(
                sum(value > 0 for value in deltas) / max(1, len(deltas)), 6
            ),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "scoreDelta": distribution(deltas),
        }

    transition_counts = Counter(
        f'{item["selected"]["pitchBand"]} -> {item["desired"]["pitchBand"]}'
        for item in high_confidence
    )
    family_counts = Counter(
        f'{item["selected"]["family"]} -> {item["desired"]["family"]}'
        for item in pair_rows
    )
    exact_index_overlap = len(desired & selected)
    reconciled_count = len(equivalent)
    return {
        "id": str(row.get("id") or source_path.stem),
        "inputs": {
            "source": str(source_path),
            "alignment": str(alignment_path),
            "candidate": str(candidate_path),
        },
        "counts": {
            "eligibleSourceNotes": len(eligible),
            "desiredSourceNotes": len(desired),
            "selectedSourceNotes": len(selected),
            "exactIndexOverlap": exact_index_overlap,
            "equivalentOverlap": reconciled_count,
            "equivalentDesiredRetention": round(reconciled_count / max(1, len(desired)), 6),
            "equivalentSelectedPrecision": round(reconciled_count / max(1, len(selected)), 6),
            "missedAfterEquivalence": len(missed),
            "unsupportedAfterEquivalence": len(unsupported),
            "localSwapPairs": len(pair_rows),
            "highConfidenceSameOnsetSameFamilyPairs": len(high_confidence),
            "unpairedMissed": len(unpaired_missed),
            "unpairedUnsupported": len(unpaired_unsupported),
        },
        "equivalenceKinds": dict(sorted(Counter(item["kind"] for item in equivalent).items())),
        "highConfidenceFeatureDelta": feature_summary,
        "highConfidenceModelRanking": model_summary,
        "highConfidencePitchBandSwaps": [
            {"transition": key, "pairs": value}
            for key, value in transition_counts.most_common()
        ],
        "allPairFamilySwaps": [
            {"transition": key, "pairs": value}
            for key, value in family_counts.most_common()
        ],
        "pairs": pair_rows,
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [item for report in reports for item in report["pairs"]]
    high_confidence = [item for item in pairs if item["sameOnset"] and item["sameFamily"]]
    model_names = sorted(
        {
            name
            for item in high_confidence
            for name in item.get("modelScores", {})
        }
    )
    return {
        "songs": len(reports),
        "localSwapPairs": len(pairs),
        "highConfidenceSameOnsetSameFamilyPairs": len(high_confidence),
        "featureDeltaDesiredMinusSelected": {
            name: distribution(
                item["featureDeltaDesiredMinusSelected"][name]
                for item in high_confidence
            )
            for name in CORE_FEATURES
        },
        "modelRanking": {
            name: {
                "pairs": len(high_confidence),
                "desiredWinRate": round(
                    sum(
                        bool(item["modelScores"][name]["desiredWins"])
                        for item in high_confidence
                        if name in item.get("modelScores", {})
                    )
                    / max(
                        1,
                        sum(name in item.get("modelScores", {}) for item in high_confidence),
                    ),
                    6,
                ),
            }
            for name in model_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--equivalence-radius-seconds", type=float, default=0.035)
    parser.add_argument("--swap-radius-seconds", type=float, default=0.08)
    parser.add_argument("--onset-radius-seconds", type=float, default=0.035)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    profile = load_json(Path(args.profile).resolve()) if args.profile else None
    reports = [
        analyze_song(
            row,
            profile=profile,
            equivalence_radius_seconds=max(0.001, args.equivalence_radius_seconds),
            swap_radius_seconds=max(0.001, args.swap_radius_seconds),
            onset_radius_seconds=max(0.001, args.onset_radius_seconds),
        )
        for row in manifest.get("songs") or []
    ]
    payload = {
        "schema": "polymath-selection-swap-audit-v1",
        "evidenceBoundary": (
            "Private development references only. Alignment-supported source indices are "
            "proxies, duplicate-equivalent events are reconciled, and target fields are "
            "never runtime features. Kiss Me at/after 02:30 remains excluded."
        ),
        "manifest": str(manifest_path),
        "profile": str(Path(args.profile).resolve()) if args.profile else None,
        "parameters": {
            "equivalenceRadiusSeconds": args.equivalence_radius_seconds,
            "swapRadiusSeconds": args.swap_radius_seconds,
            "onsetRadiusSeconds": args.onset_radius_seconds,
        },
        "aggregate": aggregate(reports),
        "songs": reports,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "aggregate": payload["aggregate"],
                "songs": [
                    {"id": report["id"], **report["counts"]} for report in reports
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
