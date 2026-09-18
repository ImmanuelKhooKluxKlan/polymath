"""Apply an experimental authored-gesture library to a piano candidate.

This utility is intentionally offline and profile-gated.  It never changes
onset count, chord size, the melody/right hand, or any pitch without local raw
stem support.  It is used to prove a chord-retrieval policy before equivalent
logic is considered for the live arranger.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .fit_chord_gesture_library import (
        gesture_context,
        harmonic_context,
        load_json,
        predict_size_from_ranked_neighbors,
        squared_distance,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from fit_chord_gesture_library import (  # type: ignore
        gesture_context,
        harmonic_context,
        load_json,
        predict_size_from_ranked_neighbors,
        squared_distance,
    )

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger import collapse_exact_left_hand_duplicates  # noqa: E402
from piano_arranger_adapter import instrument_family, normalize_source_notes  # noqa: E402


def gesture_votes(
    context: list[float],
    prototypes: list[dict[str, Any]],
    neighbors: int,
    temperature: float,
) -> tuple[list[float], float]:
    nearest = sorted(
        prototypes,
        key=lambda prototype: squared_distance(context, prototype["context"]),
    )[:neighbors]
    votes = [0.0] * 12
    total = 0.0
    for prototype in nearest:
        distance = squared_distance(context, prototype["context"])
        weight = float(prototype.get("weight", 1.0)) * math.exp(
            -distance / max(1e-6, temperature)
        )
        total += weight
        for interval in prototype["intervals"]:
            votes[int(interval) % 12] += weight
    return (
        [vote / max(1e-9, total) for vote in votes],
        squared_distance(context, nearest[0]["context"]) if nearest else math.inf,
    )


def gesture_size_prediction(
    context: list[float],
    prototypes: list[dict[str, Any]],
    neighbors: int,
    temperature: float,
) -> tuple[int, float, float]:
    ranked = sorted(
        (
            (squared_distance(context, prototype["context"]), prototype)
            for prototype in prototypes
        ),
        key=lambda item: item[0],
    )
    return predict_size_from_ranked_neighbors(
        ranked, neighbors=neighbors, temperature=temperature
    )


def best_supported_intervals(
    votes: list[float],
    available: set[int],
    count: int,
) -> set[int]:
    if not available:
        return set()
    return set(
        sorted(available, key=lambda interval: (votes[interval], -interval), reverse=True)[
            : min(count, len(available))
        ]
    )


def closest_joint_voicing(
    original: list[int],
    pitch_classes: set[int],
    minimum: int = 46,
    maximum: int = 70,
) -> list[int] | None:
    if len(original) != len(pitch_classes):
        return None
    options = {
        pitch_class: [
            midi for midi in range(minimum, maximum + 1) if midi % 12 == pitch_class
        ]
        for pitch_class in pitch_classes
    }
    best: tuple[float, list[int]] | None = None
    for order in itertools.permutations(sorted(pitch_classes)):
        for values in itertools.product(*(options[pitch_class] for pitch_class in order)):
            if len(set(values)) != len(values):
                continue
            voiced = sorted(values)
            movement = sum(abs(left - right) for left, right in zip(sorted(original), voiced))
            span = max(voiced) - min(voiced)
            cost = movement + 0.35 * max(0, span - 12)
            if best is None or cost < best[0]:
                best = (cost, voiced)
    return best[1] if best else None


def closest_variable_voicing(
    original: list[int],
    pitch_classes: set[int],
    minimum: int = 46,
    maximum: int = 70,
) -> list[int] | None:
    """Voice a one-to-three-note prediction near the incumbent hand shape."""

    if not original or not pitch_classes:
        return None
    options = {
        pitch_class: [
            midi for midi in range(minimum, maximum + 1) if midi % 12 == pitch_class
        ]
        for pitch_class in pitch_classes
    }
    best: tuple[float, list[int]] | None = None
    for values in itertools.product(*(options[pitch_class] for pitch_class in sorted(pitch_classes))):
        voiced = sorted(values)
        if len(set(voiced)) != len(voiced):
            continue
        forward = sum(min(abs(value - old) for old in original) for value in voiced)
        backward = sum(min(abs(old - value) for value in voiced) for old in original)
        span = max(voiced) - min(voiced) if len(voiced) > 1 else 0
        center_shift = abs(
            sum(voiced) / len(voiced) - sum(original) / len(original)
        )
        cost = forward + 0.55 * backward + 0.35 * max(0, span - 12) + 0.2 * center_shift
        if best is None or cost < best[0]:
            best = (cost, voiced)
    return best[1] if best else None


def minimum_note_assignment(
    original: list[int], voiced: list[int]
) -> tuple[dict[int, int], list[int]]:
    """Match existing note objects to new pitches and return added pitches."""

    if not original or not voiced:
        return {}, list(voiced)
    best: tuple[float, dict[int, int], list[int]] | None = None
    if len(original) >= len(voiced):
        for indices in itertools.combinations(range(len(original)), len(voiced)):
            for pitches in itertools.permutations(voiced):
                mapping = dict(zip(indices, pitches))
                cost = sum(abs(original[index] - pitch) for index, pitch in mapping.items())
                candidate = (cost, mapping, [])
                if best is None or candidate[0] < best[0]:
                    best = candidate
    else:
        for pitch_indices in itertools.combinations(range(len(voiced)), len(original)):
            selected = [voiced[index] for index in pitch_indices]
            for pitches in itertools.permutations(selected):
                mapping = dict(enumerate(pitches))
                cost = sum(abs(original[index] - pitch) for index, pitch in mapping.items())
                additions = [
                    pitch for index, pitch in enumerate(voiced) if index not in pitch_indices
                ]
                candidate = (cost, mapping, additions)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    return (best[1], best[2]) if best else ({}, list(voiced))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-song-id", default="")
    parser.add_argument("--minimum-gain", type=float, default=0.08)
    parser.add_argument("--support-radius", type=float, default=0.12)
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--replace-octave-gestures",
        action="store_true",
        help="Allow a high-confidence retrieved two-note gesture to replace an incumbent octave pair.",
    )
    parser.add_argument(
        "--replace-singleton-gestures",
        action="store_true",
        help="Allow a high-confidence retrieved authored pitch to replace a one-note accompaniment gesture.",
    )
    parser.add_argument("--maximum-neighbor-distance", type=float, default=math.inf)
    parser.add_argument(
        "--allow-chord-size-changes",
        action="store_true",
        help=(
            "Use cross-song neighbour votes to change a left-hand gesture by "
            "a bounded number of notes while preserving its onset."
        ),
    )
    parser.add_argument("--maximum-chord-size-change", type=int, default=1)
    parser.add_argument("--minimum-size-confidence", type=float, default=0.5)
    parser.add_argument("--minimum-size-margin", type=float, default=0.0)
    args = parser.parse_args()

    candidate = load_json(Path(args.candidate).resolve())
    source = load_json(Path(args.source).resolve())
    library = load_json(Path(args.library).resolve())
    prototypes = [
        prototype
        for prototype in library.get("prototypes", [])
        if not args.exclude_song_id or prototype.get("songId") != args.exclude_song_id
    ]
    if not prototypes:
        raise ValueError("The chord gesture library has no eligible prototypes.")
    raw = normalize_source_notes(source.get("notes", []))
    raw_times = [float(note["time"]) for note in raw]
    neighbors = max(
        1,
        int(args.neighbors if args.neighbors is not None else library.get("neighbors", 40)),
    )
    temperature = max(
        1e-6,
        float(
            args.temperature
            if args.temperature is not None
            else library.get("temperature", 0.24)
        ),
    )

    result = copy.deepcopy(candidate)
    left_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in result.get("notes", []):
        if (
            int(note.get("midi", 999)) < 72
            and note.get("arrangementRole") != "melody"
            and str(note.get("sourceInstrument") or "").lower() != "voice"
        ):
            left_groups[int(round(float(note["time"]) * 1000.0))].append(note)
    selected_times = sorted(key / 1000.0 for key in left_groups)
    context_mode = str(library.get("contextMode") or "local-v1")

    examined = 0
    changed_groups = 0
    changed_notes = 0
    added_notes: list[dict[str, Any]] = []
    removed_note_ids: set[int] = set()
    enlarged_groups = 0
    reduced_groups = 0
    size_predictions_accepted = 0
    skipped_octave_groups = 0
    accepted_neighbor_distances: list[float] = []
    for key, group in left_groups.items():
        original_midis = sorted(int(note["midi"]) for note in group)
        original_pitch_classes = {midi % 12 for midi in original_midis}
        # Retain A/v073's blind-approved octave gestures.  Retrieval is used
        # only where the incumbent already intended distinct chord members.
        if (
            len(group) > 1
            and len(original_pitch_classes) == 1
            and not args.replace_octave_gestures
        ):
            skipped_octave_groups += 1
            continue
        distinct_count = (
            min(3, len(group))
            if args.replace_octave_gestures and len(group) > 1
            else len(original_pitch_classes)
        )
        if distinct_count < 2 and not args.replace_singleton_gestures:
            continue
        onset = key / 1000.0
        context, anchor, _support = gesture_context(
            raw,
            raw_times,
            onset,
            float(library.get("contextRadiusSeconds", 0.35)),
            selected_times,
            context_mode,
        )
        votes, nearest_distance = gesture_votes(
            context, prototypes, neighbors, temperature
        )
        if nearest_distance > args.maximum_neighbor_distance:
            continue
        left = bisect.bisect_left(raw_times, onset - args.support_radius)
        right = bisect.bisect_right(raw_times, onset + args.support_radius)
        available = {
            (int(note["midi"]) % 12 - anchor) % 12
            for note in raw[left:right]
            if instrument_family(str(note.get("instrument") or "")) != "voice"
        }
        existing_intervals = {
            (pitch_class - anchor) % 12 for pitch_class in original_pitch_classes
        }
        predicted_size = distinct_count
        size_confidence = 0.0
        size_margin = 0.0
        if args.allow_chord_size_changes:
            predicted_size, size_confidence, size_margin = gesture_size_prediction(
                context, prototypes, neighbors, temperature
            )
            if (
                size_confidence >= max(0.0, min(1.0, args.minimum_size_confidence))
                and size_margin >= max(0.0, min(1.0, args.minimum_size_margin))
            ):
                maximum_size_change = max(0, args.maximum_chord_size_change)
                predicted_size = max(
                    distinct_count - maximum_size_change,
                    min(distinct_count + maximum_size_change, predicted_size),
                )
                predicted_size = max(1, min(3, len(available), predicted_size))
                if predicted_size != distinct_count:
                    size_predictions_accepted += 1
                    distinct_count = predicted_size
        available.update(existing_intervals)
        predicted = best_supported_intervals(votes, available, distinct_count)
        if any(votes[interval] <= 0.0 for interval in predicted):
            continue
        if len(predicted) != distinct_count or predicted == existing_intervals:
            continue
        if args.allow_chord_size_changes and len(predicted) != len(existing_intervals):
            existing_score = sum(votes[interval] for interval in existing_intervals) / max(
                1, len(existing_intervals)
            )
            predicted_score = sum(votes[interval] for interval in predicted) / max(
                1, len(predicted)
            )
        else:
            existing_score = sum(votes[interval] for interval in existing_intervals)
            predicted_score = sum(votes[interval] for interval in predicted)
        examined += 1
        if predicted_score - existing_score < args.minimum_gain:
            continue
        predicted_pitch_classes = {(anchor + interval) % 12 for interval in predicted}
        voiced = (
            closest_variable_voicing(original_midis, predicted_pitch_classes)
            if args.allow_chord_size_changes
            else closest_joint_voicing(original_midis, predicted_pitch_classes)
        )
        if voiced is None:
            continue
        changed_groups += 1
        accepted_neighbor_distances.append(nearest_distance)
        ordered_group = sorted(group, key=lambda item: int(item["midi"]))
        mapping, additions = minimum_note_assignment(original_midis, voiced)
        for index, note in enumerate(ordered_group):
            if index not in mapping:
                removed_note_ids.add(id(note))
                changed_notes += 1
                continue
            midi = mapping[index]
            previous = int(note["midi"])
            if previous == midi:
                continue
            note["chordGestureOriginalMidi"] = previous
            note["chordGestureRetrievalGain"] = round(predicted_score - existing_score, 6)
            note["chordGestureNearestDistance"] = round(nearest_distance, 7)
            note["midi"] = midi
            note["note"] = f"{('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')[midi % 12]}{midi // 12 - 1}"
            note["voice"] = f"{note.get('arrangementRole', 'harmony')}-{midi}"
            changed_notes += 1

        for midi in additions:
            template = min(
                ordered_group,
                key=lambda note: abs(int(note["midi"]) - midi),
            )
            note = copy.deepcopy(template)
            previous = int(note["midi"])
            note["chordGestureOriginalMidi"] = previous
            note["chordGestureRetrievalGain"] = round(
                predicted_score - existing_score, 6
            )
            note["chordGestureNearestDistance"] = round(nearest_distance, 7)
            note["chordGestureAddedNote"] = True
            note["generatedBy"] = "cross-song-chord-size-retrieval-v1"
            note["midi"] = midi
            note["note"] = f"{('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')[midi % 12]}{midi // 12 - 1}"
            note["voice"] = f"{note.get('arrangementRole', 'harmony')}-{midi}"
            added_notes.append(note)
            changed_notes += 1

        if len(voiced) < len(original_midis):
            reduced_groups += 1
        elif len(voiced) > len(original_midis):
            enlarged_groups += 1

    if removed_note_ids or added_notes:
        result["notes"] = [
            note for note in result.get("notes", []) if id(note) not in removed_note_ids
        ] + added_notes
    result["notes"], collapsed_duplicates = collapse_exact_left_hand_duplicates(
        result.get("notes", [])
    )
    arrangement = result.setdefault("pianoArrangement", {})
    arrangement["chordGestureRetrieval"] = {
        "profile": library.get("id"),
        "profileSha256": library.get("profileSha256"),
        "neighbors": neighbors,
        "temperature": temperature,
        "contextMode": context_mode,
        "minimumGain": args.minimum_gain,
        "supportRadiusSeconds": args.support_radius,
        "excludedSongId": args.exclude_song_id or None,
        "examinedGroups": examined,
        "changedGroups": changed_groups,
        "changedNotes": changed_notes,
        "allowChordSizeChanges": args.allow_chord_size_changes,
        "maximumChordSizeChange": max(0, args.maximum_chord_size_change),
        "minimumSizeConfidence": max(0.0, min(1.0, args.minimum_size_confidence)),
        "minimumSizeMargin": max(0.0, min(1.0, args.minimum_size_margin)),
        "acceptedSizePredictions": size_predictions_accepted,
        "reducedGroups": reduced_groups,
        "enlargedGroups": enlarged_groups,
        "removedNotesForSize": len(removed_note_ids),
        "addedNotesForSize": len(added_notes),
        "preservedOctaveGroups": skipped_octave_groups,
        "octaveGestureReplacementEnabled": args.replace_octave_gestures,
        "singletonGestureReplacementEnabled": args.replace_singleton_gestures,
        "maximumNeighborDistance": (
            None
            if not math.isfinite(args.maximum_neighbor_distance)
            else args.maximum_neighbor_distance
        ),
        "maximumAcceptedNeighborDistance": (
            round(max(accepted_neighbor_distances), 7)
            if accepted_neighbor_distances
            else None
        ),
        "rightHandFrozen": True,
        "onsetsFrozen": True,
        "chordSizesFrozen": not args.allow_chord_size_changes,
        "collapsedSimultaneousLeftHandStrikes": collapsed_duplicates,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(arrangement["chordGestureRetrieval"], indent=2))


if __name__ == "__main__":
    main()
