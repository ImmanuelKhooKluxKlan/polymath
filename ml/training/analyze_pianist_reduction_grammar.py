"""Forensically compare a full-mix reduction with an approved pianist target.

The ordinary note score hides several musically different failure modes.  This
audit keeps them separate:

* onset reduction (did the decoder create too many physical strikes?);
* hand occupancy (left only, right only, or both hands at a strike);
* chord texture (notes and distinct pitch classes per strike);
* source-note selection (which detected events were retained or ignored); and
* touch transfer (whether source loudness or the arranger better predicts the
  pianist's hammer velocity).

References are admitted only through explicit alignment quality windows.  The
tool is evaluation-only and never modifies a model or candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:  # Support module and direct-script execution.
    from .analyze_pianist_gesture_patterns import (
        PERCUSSION,
        distribution,
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        pitch_set,
        set_f1,
        trusted_source_ranges,
    )
    from .analyze_pianist_sequence_alignment import align_sequences
except ImportError:  # pragma: no cover - CLI compatibility path.
    from analyze_pianist_gesture_patterns import (
        PERCUSSION,
        distribution,
        finite,
        group_onsets,
        inside_ranges,
        map_reference,
        monotonic_anchors,
        normalize_notes,
        pitch_set,
        set_f1,
        trusted_source_ranges,
    )
    from analyze_pianist_sequence_alignment import align_sequences


OCCUPANCIES = ("left-only", "right-only", "both")


def rounded(value: float) -> float:
    return round(float(value), 6)


def explicit_hand(note: dict[str, Any]) -> str:
    value = str(note.get("hand") or "").strip().lower()
    if value in {"left", "right"}:
        return value
    return "left" if int(note["midi"]) < 60 else "right"


def occupancy(group: list[dict[str, Any]]) -> str:
    hands = {explicit_hand(note) for note in group}
    if hands == {"left", "right"}:
        return "both"
    return "left-only" if "left" in hands else "right-only"


def occupancy_counts(groups: Iterable[list[dict[str, Any]]]) -> dict[str, int]:
    counts = Counter(occupancy(group) for group in groups)
    return {name: int(counts[name]) for name in OCCUPANCIES}


def chord_size_counts(
    groups: Iterable[list[dict[str, Any]]], *, pitch_class: bool
) -> dict[str, int]:
    counts = Counter(len(pitch_set(group, pitch_class)) for group in groups)
    return {str(size): int(count) for size, count in sorted(counts.items())}


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return rounded(numerator / denominator) if denominator > 1e-12 else None


def error_summary(predicted: list[float], target: list[float]) -> dict[str, Any]:
    if len(predicted) != len(target) or not predicted:
        return {"count": 0, "bias": None, "mae": None, "correlation": None}
    errors = [left - right for left, right in zip(predicted, target)]
    return {
        "count": len(errors),
        "bias": rounded(sum(errors) / len(errors)),
        "mae": rounded(sum(abs(value) for value in errors) / len(errors)),
        "correlation": pearson(predicted, target),
    }


def velocity_band(value: float) -> str:
    if value < 0.55:
        return "quiet-<0.55"
    if value < 0.65:
        return "soft-0.55-0.64"
    if value < 0.75:
        return "medium-0.65-0.74"
    if value < 0.85:
        return "strong-0.75-0.84"
    return "accent->=0.85"


def duration_band(value: float) -> str:
    if value < 0.18:
        return "short-<0.18s"
    if value < 0.55:
        return "medium-0.18-0.54s"
    return "long->=0.55s"


def family(value: Any) -> str:
    name = str(value or "unknown").lower()
    if "voice" in name or "vocal" in name or "choir" in name:
        return "voice"
    if "guitar" in name:
        return "guitar"
    if "bass" in name or "contrabass" in name:
        return "bass"
    if "piano" in name or "keyboard" in name:
        return "piano"
    if "string" in name or "violin" in name or "cello" in name:
        return "strings"
    return "other"


def source_rank_labels(group: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Return interpretable within-attack ranks without inventing tie order."""

    velocities = [float(note["velocity"]) for note in group]
    durations = [float(note["duration"]) for note in group]
    midis = [int(note["midi"]) for note in group]
    maximum_velocity = max(velocities)
    minimum_velocity = min(velocities)
    maximum_duration = max(durations)
    minimum_duration = min(durations)
    output: dict[int, dict[str, str]] = {}
    for note in group:
        index = int(note["sourceIndex"])
        velocity = float(note["velocity"])
        duration = float(note["duration"])
        midi = int(note["midi"])
        if math.isclose(maximum_velocity, minimum_velocity, abs_tol=0.005):
            velocity_rank = "all-velocity-tied"
        elif math.isclose(velocity, maximum_velocity, abs_tol=0.005):
            velocity_rank = "loudest"
        elif math.isclose(velocity, minimum_velocity, abs_tol=0.005):
            velocity_rank = "quietest"
        else:
            velocity_rank = "middle-velocity"
        if math.isclose(maximum_duration, minimum_duration, abs_tol=0.012):
            duration_rank = "all-duration-tied"
        elif math.isclose(duration, maximum_duration, abs_tol=0.012):
            duration_rank = "longest"
        elif math.isclose(duration, minimum_duration, abs_tol=0.012):
            duration_rank = "shortest"
        else:
            duration_rank = "middle-duration"
        if len(set(midis)) == 1:
            pitch_rank = "only-pitch"
        elif midi == min(midis):
            pitch_rank = "lowest"
        elif midi == max(midis):
            pitch_rank = "highest"
        else:
            pitch_rank = "inner"
        output[index] = {
            "velocityRank": velocity_rank,
            "durationRank": duration_rank,
            "pitchRank": pitch_rank,
        }
    return output


def strongest_alignment_matches(
    report: dict[str, Any], eligible_reference_indices: set[int]
) -> dict[int, dict[str, Any]]:
    strongest: dict[int, tuple[tuple[int, float], dict[str, Any]]] = {}
    for item in report.get("matches") or []:
        if not isinstance(item, dict):
            continue
        observed = item.get("observed") or {}
        reference = item.get("reference") or {}
        try:
            observed_index = int(observed["sourceIndex"])
            reference_index = int(reference["sourceIndex"])
            residual = abs(finite(item.get("coarseResidual"), 9e9))
        except (KeyError, TypeError, ValueError):
            continue
        if reference_index not in eligible_reference_indices:
            continue
        quality = (int(bool(item.get("exactPitch"))), -residual)
        if observed_index not in strongest or quality > strongest[observed_index][0]:
            strongest[observed_index] = (quality, item)
    return {index: value[1] for index, value in strongest.items()}


def selected_candidate_notes(
    notes: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        try:
            output[int(note["sourceIndex"])].append(note)
        except (KeyError, TypeError, ValueError):
            continue
    return dict(output)


def decision_breakdown(
    source: list[dict[str, Any]],
    desired: set[int],
    selected: set[int],
    labels: dict[int, dict[str, str]],
    key: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for note in source:
        index = int(note["sourceIndex"])
        if key == "instrumentFamily":
            name = family(note.get("instrument"))
        elif key == "velocityBand":
            name = velocity_band(float(note["velocity"]))
        elif key == "durationBand":
            name = duration_band(float(note["duration"]))
        else:
            name = labels[index][key]
        row = buckets[name]
        row["source"] += 1
        row["desired"] += int(index in desired)
        row["selected"] += int(index in selected)
        row["desiredSelected"] += int(index in desired and index in selected)
    output: list[dict[str, Any]] = []
    for name, row in buckets.items():
        output.append(
            {
                "name": name,
                **dict(row),
                "desiredRate": rounded(row["desired"] / max(1, row["source"])),
                "candidateSelectionRate": rounded(
                    row["selected"] / max(1, row["source"])
                ),
                "desiredRetention": rounded(
                    row["desiredSelected"] / max(1, row["desired"])
                ),
                "candidatePrecisionProxy": rounded(
                    row["desiredSelected"] / max(1, row["selected"])
                ),
            }
        )
    return sorted(output, key=lambda item: (-int(item["source"]), str(item["name"])))


def hand_feature_summary(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    matches: list[tuple[int, int]],
) -> dict[str, Any]:
    cross = Counter(
        (occupancy(reference[left]), occupancy(candidate[right]))
        for left, right in matches
    )
    transitions = Counter()
    candidate_ngrams = Counter()
    reference_ngrams = Counter()
    for groups, output in (
        (reference, reference_ngrams),
        (candidate, candidate_ngrams),
    ):
        states = [occupancy(group)[0].upper() if occupancy(group) != "both" else "B" for group in groups]
        for index in range(max(0, len(states) - 3)):
            output["".join(states[index : index + 4])] += 1
    for left, right in zip(matches, matches[1:]):
        if left[0] + 1 == right[0] and left[1] + 1 == right[1]:
            transitions[
                f"{occupancy(reference[left[0]])}->{occupancy(reference[right[0]])}"
            ] += 1
    return {
        "reference": occupancy_counts(reference),
        "candidate": occupancy_counts(candidate),
        "pairedCrossTable": {
            f"{target}->{observed}": int(count)
            for (target, observed), count in cross.most_common()
        },
        "referenceTopFourGesturePatterns": [
            {"pattern": name, "count": int(count)}
            for name, count in reference_ngrams.most_common(15)
        ],
        "candidateTopFourGesturePatterns": [
            {"pattern": name, "count": int(count)}
            for name, count in candidate_ngrams.most_common(15)
        ],
        "consecutiveReferenceTransitions": dict(transitions.most_common()),
    }


def _eligible_reference_indices(notes: list[dict[str, Any]]) -> set[int]:
    output: set[int] = set()
    for note in notes:
        try:
            output.add(int(note.get("sourceIndex", note["_payloadIndex"])))
        except (KeyError, TypeError, ValueError):
            output.add(int(note["_payloadIndex"]))
    return output


def analyze_song(
    row: dict[str, Any],
    *,
    onset_window: float,
    maximum_time_distance: float,
    gap_cost: float,
) -> dict[str, Any]:
    song_id = str(row.get("id") or "").strip()
    if not song_id:
        raise ValueError("Every manifest row requires an id")
    reference_path = Path(str(row["reference"])).resolve()
    source_path = Path(str(row["source"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = Path(str(row["candidate"])).resolve()
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    source_payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    transpose = int(row.get("referenceTransposeSemitones") or 0)
    reference_end_value = finite(row.get("referenceEndSeconds"), math.nan)
    reference_end = reference_end_value if math.isfinite(reference_end_value) else None
    candidate_end_value = finite(row.get("candidateEndSeconds"), math.nan)
    candidate_end = candidate_end_value if math.isfinite(candidate_end_value) else None
    ranges = trusted_source_ranges(alignment)
    reference_original = normalize_notes(
        reference_payload,
        transpose=transpose,
        hard_end=reference_end,
    )
    if bool(row.get("referenceAlreadyAligned")):
        reference_mapped = reference_original
    else:
        reference_mapped = map_reference(reference_original, monotonic_anchors(alignment))
    reference = [
        note
        for note in reference_mapped
        if note.get("trainingEligible") is not False
        and inside_ranges(float(note["time"]), ranges)
        and (candidate_end is None or float(note["time"]) < candidate_end)
    ]
    source = [
        note
        for note in normalize_notes(source_payload, hard_end=candidate_end, source_indices=True)
        if inside_ranges(float(note["time"]), ranges)
        and str(note.get("instrument") or "").lower() not in PERCUSSION
    ]
    candidate = [
        note
        for note in normalize_notes(candidate_payload, hard_end=candidate_end)
        if inside_ranges(float(note["time"]), ranges)
    ]
    reference_groups = group_onsets(reference, onset_window)
    source_groups = group_onsets(source, onset_window)
    candidate_groups = group_onsets(candidate, onset_window)
    matches, missing, extra, sequence_score = align_sequences(
        reference_groups,
        candidate_groups,
        maximum_time_distance=maximum_time_distance,
        gap_cost=gap_cost,
    )

    pc_f1_values = [
        set_f1(
            pitch_set(reference_groups[left], True),
            pitch_set(candidate_groups[right], True),
        )
        for left, right in matches
    ]
    source_labels: dict[int, dict[str, str]] = {}
    for group in source_groups:
        source_labels.update(source_rank_labels(group))
    supported_matches = strongest_alignment_matches(
        alignment, _eligible_reference_indices(reference_original)
    )
    source_lookup = {int(note["sourceIndex"]): note for note in source}
    supported_matches = {
        index: item for index, item in supported_matches.items() if index in source_lookup
    }
    candidate_by_source = selected_candidate_notes(candidate)
    selected = set(candidate_by_source) & set(source_lookup)
    desired = set(supported_matches)

    source_velocities: list[float] = []
    candidate_velocities: list[float] = []
    target_velocities: list[float] = []
    for source_index in sorted(desired & selected):
        match = supported_matches[source_index]
        target = match.get("reference") or {}
        target_midi = int(round(finite(target.get("midi"), -999))) + transpose
        outputs = candidate_by_source[source_index]
        best = min(
            outputs,
            key=lambda note: (
                int(int(note["midi"]) % 12 != target_midi % 12),
                abs(int(note["midi"]) - target_midi),
            ),
        )
        source_velocities.append(float(source_lookup[source_index]["velocity"]))
        candidate_velocities.append(float(best["velocity"]))
        target_velocities.append(
            max(0.01, min(1.0, finite(target.get("velocity"), 0.72)))
        )

    reference_gesture_velocity = [
        median(float(note["velocity"]) for note in group)
        for group in reference_groups
    ]
    candidate_gesture_velocity = [
        median(float(note["velocity"]) for note in group)
        for group in candidate_groups
    ]
    shared_reference = [
        max(float(note["velocity"]) for note in group)
        - min(float(note["velocity"]) for note in group)
        <= 1e-9
        for group in reference_groups
        if len(group) > 1
    ]
    shared_candidate = [
        max(float(note["velocity"]) for note in group)
        - min(float(note["velocity"]) for note in group)
        <= 1e-9
        for group in candidate_groups
        if len(group) > 1
    ]
    return {
        "id": song_id,
        "inputs": {
            "reference": str(reference_path),
            "source": str(source_path),
            "alignment": str(alignment_path),
            "candidate": str(candidate_path),
        },
        "boundary": {
            "referenceAlreadyAligned": bool(row.get("referenceAlreadyAligned")),
            "referenceTransposeSemitones": transpose,
            "referenceEndSecondsExclusive": reference_end,
            "candidateEndSecondsExclusive": candidate_end,
            "trustedSourceRanges": ranges,
        },
        "timeline": {
            "sourceNotes": len(source),
            "referenceNotes": len(reference),
            "candidateNotes": len(candidate),
            "sourceAttackGroups": len(source_groups),
            "referenceGestures": len(reference_groups),
            "candidateGestures": len(candidate_groups),
            "referenceReductionFromSourceAttacks": rounded(
                1.0 - len(reference_groups) / max(1, len(source_groups))
            ),
            "candidateReductionFromSourceAttacks": rounded(
                1.0 - len(candidate_groups) / max(1, len(source_groups))
            ),
        },
        "sequenceAlignment": {
            "score": rounded(sequence_score),
            "matchedGestures": len(matches),
            "missingReferenceGestures": len(missing),
            "extraCandidateGestures": len(extra),
            "matchRateAgainstReference": rounded(
                len(matches) / max(1, len(reference_groups))
            ),
            "meanPitchClassF1": rounded(
                sum(pc_f1_values) / max(1, len(pc_f1_values))
            ),
        },
        "handOccupancy": hand_feature_summary(
            reference_groups, candidate_groups, matches
        ),
        "chordTexture": {
            "referenceExactKeyCount": chord_size_counts(
                reference_groups, pitch_class=False
            ),
            "candidateExactKeyCount": chord_size_counts(
                candidate_groups, pitch_class=False
            ),
            "referencePitchClassCount": chord_size_counts(
                reference_groups, pitch_class=True
            ),
            "candidatePitchClassCount": chord_size_counts(
                candidate_groups, pitch_class=True
            ),
        },
        "sourceSelection": {
            "eligibleSourceNotes": len(source),
            "alignmentSupportedSourceNotes": len(desired),
            "selectedSourceNotes": len(selected),
            "supportedAndSelected": len(desired & selected),
            "supportedButIgnored": len(desired - selected),
            "selectedWithoutAlignmentSupport": len(selected - desired),
            "supportedRetention": rounded(len(desired & selected) / max(1, len(desired))),
            "candidatePrecisionProxy": rounded(
                len(desired & selected) / max(1, len(selected))
            ),
            "byInstrumentFamily": decision_breakdown(
                source, desired, selected, source_labels, "instrumentFamily"
            ),
            "byVelocityBand": decision_breakdown(
                source, desired, selected, source_labels, "velocityBand"
            ),
            "byDurationBand": decision_breakdown(
                source, desired, selected, source_labels, "durationBand"
            ),
            "byWithinAttackVelocityRank": decision_breakdown(
                source, desired, selected, source_labels, "velocityRank"
            ),
            "byWithinAttackDurationRank": decision_breakdown(
                source, desired, selected, source_labels, "durationRank"
            ),
            "byWithinAttackPitchRank": decision_breakdown(
                source, desired, selected, source_labels, "pitchRank"
            ),
        },
        "touch": {
            "referenceGestureVelocity": distribution(reference_gesture_velocity),
            "candidateGestureVelocity": distribution(candidate_gesture_velocity),
            "referenceSharedVelocityMultiNoteShare": rounded(
                sum(shared_reference) / max(1, len(shared_reference))
            ),
            "candidateSharedVelocityMultiNoteShare": rounded(
                sum(shared_candidate) / max(1, len(shared_candidate))
            ),
            "onSameSupportedSourceEvents": {
                "sourceToTarget": error_summary(source_velocities, target_velocities),
                "candidateToTarget": error_summary(
                    candidate_velocities, target_velocities
                ),
            },
        },
    }


def aggregate_songs(songs: list[dict[str, Any]]) -> dict[str, Any]:
    if not songs:
        return {}
    timeline_fields = (
        "sourceNotes",
        "referenceNotes",
        "candidateNotes",
        "sourceAttackGroups",
        "referenceGestures",
        "candidateGestures",
    )
    return {
        "songs": len(songs),
        "timeline": {
            name: sum(int(song["timeline"][name]) for song in songs)
            for name in timeline_fields
        },
        "sequence": {
            "matchedGestures": sum(
                int(song["sequenceAlignment"]["matchedGestures"]) for song in songs
            ),
            "missingReferenceGestures": sum(
                int(song["sequenceAlignment"]["missingReferenceGestures"])
                for song in songs
            ),
            "extraCandidateGestures": sum(
                int(song["sequenceAlignment"]["extraCandidateGestures"])
                for song in songs
            ),
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Pianist reduction grammar audit",
        "",
        "This report separates timing, hand choreography, chord texture, source selection, and touch.",
        "",
    ]
    for song in report["songs"]:
        timeline = song["timeline"]
        sequence = song["sequenceAlignment"]
        hands = song["handOccupancy"]
        selection = song["sourceSelection"]
        touch = song["touch"]["onSameSupportedSourceEvents"]
        lines.extend(
            [
                f"## {song['id']}",
                "",
                f"- Physical strikes: reference {timeline['referenceGestures']}, candidate {timeline['candidateGestures']}.",
                f"- Sequence matches: {sequence['matchedGestures']} ({sequence['matchRateAgainstReference']:.1%}); mean pitch-class F1 {sequence['meanPitchClassF1']:.3f}.",
                f"- Reference hands: {hands['reference']}; candidate hands: {hands['candidate']}.",
                f"- Supported source retention: {selection['supportedRetention']:.1%}; precision proxy {selection['candidatePrecisionProxy']:.1%}.",
                f"- Source-to-target touch: {touch['sourceToTarget']}; arranged-to-target touch: {touch['candidateToTarget']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence boundary",
            "",
            "These are fixed-song private research diagnostics. Alignment support is incomplete ground truth, and results do not prove unseen-song generalization or commercial rights.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--maximum-time-distance-seconds", type=float, default=0.55)
    parser.add_argument("--gap-cost", type=float, default=0.85)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    songs = [
        analyze_song(
            row,
            onset_window=max(0.005, float(args.onset_window_seconds)),
            maximum_time_distance=max(
                0.05, float(args.maximum_time_distance_seconds)
            ),
            gap_cost=max(0.05, float(args.gap_cost)),
        )
        for row in manifest.get("songs") or []
    ]
    report = {
        "schema": "polymath-pianist-reduction-grammar-v1",
        "evidenceBoundary": (
            "Evaluation only; quality-window gated; Kiss Me material at and "
            "after 02:30 must remain excluded."
        ),
        "manifest": str(manifest_path),
        "aggregate": aggregate_songs(songs),
        "songs": songs,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        markdown_path = Path(args.markdown).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "songs": [
                    {
                        "id": song["id"],
                        **song["timeline"],
                        **song["sequenceAlignment"],
                    }
                    for song in songs
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
