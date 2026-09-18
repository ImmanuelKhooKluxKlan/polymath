"""Measure a piano reduction's accompaniment against an aligned ideal sheet.

Unlike the full-mix diagnostic, this report deliberately excludes the frozen
melody/right-hand layer.  It answers whether left-hand pitch choices, chord
shape, attack density, holds, and retriggers moved closer to the teacher sheet.
Only pre-approved trusted alignment windows are scored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger_adapter import normalize_source_notes, selection_scores  # noqa: E402

try:
    from .analyze_full_mix_arranger import (
        filter_metadata_notes,
        greedy_match_indices,
        normalize_notes_with_metadata,
    )
    from .evaluate_piano_arranger import f1_metrics, load_json, trusted_source_ranges
except ImportError:  # pragma: no cover - direct CLI execution
    from analyze_full_mix_arranger import (
        filter_metadata_notes,
        greedy_match_indices,
        normalize_notes_with_metadata,
    )
    from evaluate_piano_arranger import f1_metrics, load_json, trusted_source_ranges


VOICE_INSTRUMENTS = {"voice", "vocals", "vocal", "singing_voice"}


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    progress = position - lower
    return ordered[lower] * (1.0 - progress) + ordered[upper] * progress


def onset_groups(
    notes: list[dict[str, Any]], tolerance: float = 0.035
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for note in sorted(notes, key=lambda item: (float(item["time"]), int(item["midi"]))):
        if not groups or float(note["time"]) - float(groups[-1][0]["time"]) > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def fast_retriggers(notes: list[dict[str, Any]]) -> int:
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in notes:
        by_pitch[int(note["midi"])].append(float(note["time"]))
    return sum(
        0.065 <= current - previous < 0.18
        for times in by_pitch.values()
        for previous, current in zip(sorted(times), sorted(times)[1:])
    )


def interval_histogram(notes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for group in onset_groups(notes):
        bass = min(int(note["midi"]) for note in group)
        for note in group:
            counts[(int(note["midi"]) - bass) % 12] += 1
    return {str(interval): counts[interval] for interval in range(12)}


def interval_translation_table(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float = 0.25,
) -> dict[str, dict[str, int]]:
    """Describe how candidate chord intervals differ at nearby ideal onsets."""

    reference_groups = onset_groups(reference)
    candidate_groups = onset_groups(candidate)
    available = set(range(len(reference_groups)))
    table: dict[int, Counter[int]] = defaultdict(Counter)
    for candidate_group in candidate_groups:
        if len(candidate_group) < 2:
            continue
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
        reference_group = reference_groups[reference_index]
        if len(reference_group) < 2:
            continue
        candidate_bass = min(int(note["midi"]) for note in candidate_group)
        reference_bass = min(int(note["midi"]) for note in reference_group)
        source_intervals = [
            (int(note["midi"]) - candidate_bass) % 12
            for note in candidate_group
            if int(note["midi"]) != candidate_bass
        ]
        target_intervals = [
            (int(note["midi"]) - reference_bass) % 12
            for note in reference_group
            if int(note["midi"]) != reference_bass
        ]
        for source_interval in source_intervals:
            if not target_intervals:
                continue
            target_interval = min(
                target_intervals,
                key=lambda target: (
                    min((target - source_interval) % 12, (source_interval - target) % 12),
                    target,
                ),
            )
            table[source_interval][target_interval] += 1
    return {
        str(source): {
            str(target): count
            for target, count in sorted(targets.items())
        }
        for source, targets in sorted(table.items())
    }


def selected_onset_support_ceiling(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    *,
    match_tolerance: float = 0.25,
    support_radius: float = 0.18,
) -> dict[str, Any]:
    """Measure whether raw stems contain the ideal chord at retained onsets."""

    reference_groups = onset_groups(reference)
    candidate_groups = onset_groups(candidate)
    raw = sorted(raw, key=lambda note: float(note["time"]))
    available = set(range(len(reference_groups)))
    matched_onsets = 0
    target_memberships = 0
    candidate_memberships = 0
    raw_memberships = 0
    for candidate_group in candidate_groups:
        time = float(candidate_group[0]["time"])
        possible = [
            index
            for index in available
            if abs(float(reference_groups[index][0]["time"]) - time) <= match_tolerance
        ]
        if not possible:
            continue
        reference_index = min(
            possible,
            key=lambda index: abs(float(reference_groups[index][0]["time"]) - time),
        )
        available.remove(reference_index)
        matched_onsets += 1
        target_pitch_classes = {
            int(note["midi"]) % 12 for note in reference_groups[reference_index]
        }
        candidate_pitch_classes = {
            int(note["midi"]) % 12 for note in candidate_group
        }
        raw_pitch_classes = {
            int(note["midi"]) % 12
            for note in raw
            if abs(float(note["time"]) - time) <= support_radius
            and str(note.get("instrument") or note.get("sourceInstrument") or "").lower()
            not in VOICE_INSTRUMENTS | {"drums", "percussion", "timpani"}
        }
        target_memberships += len(target_pitch_classes)
        candidate_memberships += len(target_pitch_classes & candidate_pitch_classes)
        raw_memberships += len(target_pitch_classes & raw_pitch_classes)
    return {
        "matchedCandidateOnsets": matched_onsets,
        "idealPitchClassMemberships": target_memberships,
        "candidateMembershipRecall": round(
            candidate_memberships / max(1, target_memberships), 6
        ),
        "rawSupportRecallCeiling": round(
            raw_memberships / max(1, target_memberships), 6
        ),
        "supportRadiusSeconds": support_radius,
    }


def completion_model_diagnostic(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    raw_payload: dict[str, Any],
    model: dict[str, Any],
    *,
    radius: float = 0.08,
) -> dict[str, Any]:
    """Score top-k pitch-class choice at the chord/onset level."""

    raw = normalize_source_notes(raw_payload.get("notes", []))
    probabilities = selection_scores(raw, {"selectionModel": model})
    score_by_index = {
        int(note["sourceIndex"]): float(score)
        for note, score in zip(raw, probabilities)
    }
    reference_groups = onset_groups(reference)
    candidate_groups = onset_groups(candidate)
    available = set(range(len(reference_groups)))
    current_hits = 0
    model_hits = 0
    ideal_memberships = 0
    improved_onsets = 0
    worsened_onsets = 0
    unchanged_onsets = 0
    matched_onsets = 0
    for candidate_group in candidate_groups:
        time = float(candidate_group[0]["time"])
        possible = [
            index
            for index in available
            if abs(float(reference_groups[index][0]["time"]) - time) <= 0.25
        ]
        if not possible:
            continue
        reference_index = min(
            possible,
            key=lambda index: abs(float(reference_groups[index][0]["time"]) - time),
        )
        available.remove(reference_index)
        target = {int(note["midi"]) % 12 for note in reference_groups[reference_index]}
        current = {int(note["midi"]) % 12 for note in candidate_group}
        options: dict[int, float] = {}
        for raw_note in raw:
            if abs(float(raw_note["time"]) - time) > radius:
                continue
            if str(raw_note.get("instrument") or "").lower() in VOICE_INSTRUMENTS:
                continue
            pitch_class = int(raw_note["midi"]) % 12
            options[pitch_class] = max(
                options.get(pitch_class, 0.0),
                score_by_index[int(raw_note["sourceIndex"])],
            )
        selected = {
            pitch_class
            for pitch_class, _score in sorted(
                options.items(), key=lambda item: (-item[1], item[0])
            )[: max(1, len(current))]
        }
        current_count = len(current & target)
        selected_count = len(selected & target)
        current_hits += current_count
        model_hits += selected_count
        ideal_memberships += len(target)
        matched_onsets += 1
        if selected_count > current_count:
            improved_onsets += 1
        elif selected_count < current_count:
            worsened_onsets += 1
        else:
            unchanged_onsets += 1
    return {
        "matchedOnsets": matched_onsets,
        "idealPitchClassMemberships": ideal_memberships,
        "currentMembershipRecall": round(current_hits / max(1, ideal_memberships), 6),
        "modelTopKMembershipRecall": round(model_hits / max(1, ideal_memberships), 6),
        "improvedOnsets": improved_onsets,
        "worsenedOnsets": worsened_onsets,
        "unchangedOnsets": unchanged_onsets,
        "radiusSeconds": radius,
    }


def accompaniment_summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = onset_groups(notes)
    durations = [float(note.get("duration", 0.2)) for note in notes]
    sizes = [len(group) for group in groups]
    return {
        "notes": len(notes),
        "onsets": len(groups),
        "notesPerOnset": round(len(notes) / max(1, len(groups)), 6),
        "multiNoteOnsetShare": round(
            sum(size >= 2 for size in sizes) / max(1, len(sizes)), 6
        ),
        "maximumNotesPerOnset": max(sizes, default=0),
        "onsetSizeHistogram": {
            str(size): count for size, count in sorted(Counter(sizes).items())
        },
        "durationSeconds": {
            "p10": round(quantile(durations, 0.10), 6),
            "median": round(median(durations), 6) if durations else 0.0,
            "p90": round(quantile(durations, 0.90), 6),
        },
        "fastSamePitchRetriggers65To180ms": fast_retriggers(notes),
        "intervalAboveLocalBassPitchClass": interval_histogram(notes),
    }


def match_metrics(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for milliseconds in (50, 100, 250):
        tolerance = milliseconds / 1000.0
        exact = greedy_match_indices(reference, candidate, tolerance)
        pitch_class = greedy_match_indices(
            reference, candidate, tolerance, octave_equivalent=True
        )
        output[f"exactPitchOnset{milliseconds}ms"] = f1_metrics(
            len(reference), len(candidate), len(exact)
        )
        output[f"pitchClassOnset{milliseconds}ms"] = f1_metrics(
            len(reference), len(candidate), len(pitch_class)
        )
    return output


def onset_only_metrics(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_times = [float(group[0]["time"]) for group in onset_groups(reference)]
    candidate_times = [float(group[0]["time"]) for group in onset_groups(candidate)]
    output: dict[str, Any] = {}
    for milliseconds in (50, 100, 250):
        tolerance = milliseconds / 1000.0
        used: set[int] = set()
        matches = 0
        for reference_time in reference_times:
            possible = [
                (abs(candidate_time - reference_time), index)
                for index, candidate_time in enumerate(candidate_times)
                if index not in used and abs(candidate_time - reference_time) <= tolerance
            ]
            if not possible:
                continue
            _distance, index = min(possible)
            used.add(index)
            matches += 1
        output[f"onsetOnly{milliseconds}ms"] = f1_metrics(
            len(reference_times), len(candidate_times), matches
        )
    return output


def candidate_left_notes(
    payload: dict[str, Any], ranges: list[tuple[float, float]], hand_split: int
) -> list[dict[str, Any]]:
    notes = filter_metadata_notes(normalize_notes_with_metadata(payload), ranges)
    return [
        note
        for note in notes
        if int(note["midi"]) < hand_split
        and str(note.get("arrangementRole") or "") != "melody"
        and str(note.get("sourceInstrument") or "").lower() not in VOICE_INSTRUMENTS
    ]


def reference_left_notes(
    payload: dict[str, Any], reference_split: int, transpose: int
) -> list[dict[str, Any]]:
    notes = normalize_notes_with_metadata(payload)
    return [
        {**note, "midi": int(note["midi"]) + transpose}
        for note in notes
        if bool(note.get("trainingEligible"))
        and int(note["midi"]) < reference_split
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-target", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--raw")
    parser.add_argument("--completion-profile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-hand-split", type=int, default=60)
    parser.add_argument("--candidate-hand-split", type=int, default=72)
    parser.add_argument("--reference-transpose-semitones", type=int, default=12)
    args = parser.parse_args()

    alignment = load_json(Path(args.alignment).resolve())
    ranges = trusted_source_ranges(alignment)
    if not ranges:
        raise ValueError("Alignment contains no trusted source-time ranges.")
    reference = reference_left_notes(
        load_json(Path(args.aligned_target).resolve()),
        args.reference_hand_split,
        args.reference_transpose_semitones,
    )
    candidate = candidate_left_notes(
        load_json(Path(args.candidate).resolve()), ranges, args.candidate_hand_split
    )
    report: dict[str, Any] = {
        "schema": "polymath-left-hand-accompaniment-analysis-v1",
        "evidencePolicy": "Aligned ideal notes and candidate notes from trusted windows only; melody/voice excluded.",
        "reference": accompaniment_summary(reference),
        "candidate": {
            **accompaniment_summary(candidate),
            "matches": {
                **match_metrics(reference, candidate),
                **onset_only_metrics(reference, candidate),
            },
            "candidateToIdealChordIntervalTable250ms": interval_translation_table(
                reference, candidate
            ),
        },
    }
    if args.raw:
        raw_payload = load_json(Path(args.raw).resolve())
        raw = filter_metadata_notes(normalize_notes_with_metadata(raw_payload), ranges)
        report["candidate"]["selectedOnsetSupport"] = {
            str(radius): selected_onset_support_ceiling(
                reference, candidate, raw, support_radius=radius
            )
            for radius in (0.08, 0.18, 0.35)
        }
        if args.completion_profile:
            completion_profile = load_json(Path(args.completion_profile).resolve())
            completion_model = completion_profile.get("selectionModel") or (
                completion_profile.get("decoder", {})
                .get("leftHandAccompaniment", {})
                .get("selectionModel")
            )
            if not completion_model:
                raise ValueError("Completion profile contains no selection model.")
            report["candidate"]["completionModelDiagnostic"] = (
                completion_model_diagnostic(
                    reference, candidate, raw_payload, completion_model
                )
            )
    if args.baseline:
        baseline = candidate_left_notes(
            load_json(Path(args.baseline).resolve()), ranges, args.candidate_hand_split
        )
        baseline_matches = {
            **match_metrics(reference, baseline),
            **onset_only_metrics(reference, baseline),
        }
        candidate_matches = report["candidate"]["matches"]
        report["baseline"] = {
            **accompaniment_summary(baseline),
            "matches": baseline_matches,
        }
        report["candidateMinusBaseline"] = {
            metric: round(
                float(candidate_matches[metric]["f1"])
                - float(baseline_matches[metric]["f1"]),
                6,
            )
            for metric in candidate_matches
        }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
