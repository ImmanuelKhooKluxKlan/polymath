"""Apply a whole-song-validated paired chord mapper offline.

The mapper is trained by :mod:`fit_paired_chord_gesture_mapper`.  This applier
uses the exact same 83-value inference context as training, leaves onset count,
chord size, melody and right hand frozen, and accepts a replacement only when
every proposed pitch class is present in the nearby non-vocal transcription.
It is a research utility; it never edits the deployed arranger profile.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from piano_arranger import collapse_exact_left_hand_duplicates  # noqa: E402
from piano_arranger_adapter import instrument_family, normalize_source_notes  # noqa: E402

try:
    from .apply_chord_gesture_library import (
        closest_joint_voicing,
        minimum_note_assignment,
    )
    from .fit_chord_gesture_library import squared_distance
    from .fit_paired_chord_gesture_mapper import (
        decode_conservative_correction,
        paired_context,
        ranked_vote_probabilities,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from apply_chord_gesture_library import (  # type: ignore
        closest_joint_voicing,
        minimum_note_assignment,
    )
    from fit_chord_gesture_library import squared_distance  # type: ignore
    from fit_paired_chord_gesture_mapper import (  # type: ignore
        decode_conservative_correction,
        paired_context,
        ranked_vote_probabilities,
    )


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def tight_left_groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Return mutable non-melody left-hand groups in chronological order."""

    by_onset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in payload.get("notes", []):
        if not isinstance(note, dict):
            continue
        hand = str(note.get("hand") or "").strip().lower()
        midi = int(round(float(note.get("midi", 999))))
        if hand:
            is_left = hand == "left"
        else:
            is_left = midi < 60
        if (
            not is_left
            or str(note.get("arrangementRole") or "") == "melody"
            or instrument_family(str(note.get("sourceInstrument") or "")) == "voice"
        ):
            continue
        by_onset[int(round(float(note["time"]) * 1000.0))].append(note)
    return [
        sorted(by_onset[key], key=lambda note: int(note["midi"]))
        for key in sorted(by_onset)
    ]


def apply_mapper(
    candidate: dict[str, Any],
    source: dict[str, Any],
    mapper: dict[str, Any],
    *,
    exclude_song_id: str = "",
    support_radius_seconds: float = 0.12,
    minimum_midi: int = 33,
    maximum_midi: int = 59,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mapper.get("type") != "paired-incumbent-sequence-chord-mapper-v1":
        raise ValueError("Unsupported paired chord mapper profile.")
    if not 0.02 <= support_radius_seconds <= 0.5:
        raise ValueError("support_radius_seconds must be between 0.02 and 0.5")
    if minimum_midi >= maximum_midi:
        raise ValueError("minimum_midi must be below maximum_midi")

    samples = [
        sample
        for sample in mapper.get("samples", [])
        if isinstance(sample, dict)
        and (not exclude_song_id or sample.get("songId") != exclude_song_id)
    ]
    if not samples:
        raise ValueError("The paired chord mapper has no eligible samples.")

    result = copy.deepcopy(candidate)
    groups = tight_left_groups(result)
    source_notes = normalize_source_notes(source.get("notes", []))
    source_times = [float(note["time"]) for note in source_notes]
    neighbors = max(1, int(mapper.get("neighbors", 25)))
    temperature = max(1e-6, float(mapper.get("temperature", 0.48)))
    incumbent_prior = max(0.0, float(mapper.get("incumbentPrior", 0.35)))
    minimum_gain = max(0.0, float(mapper.get("minimumGain", 0.18)))
    context_radius = max(0.05, float(mapper.get("contextRadiusSeconds", 0.35)))

    examined = 0
    changed_groups = 0
    changed_notes = 0
    unsupported_predictions = 0
    unchanged_predictions = 0
    unvoiceable_predictions = 0
    accepted_gains: list[float] = []

    for group_index, group in enumerate(groups):
        original_midis = [int(note["midi"]) for note in group]
        original_pitch_classes = {midi % 12 for midi in original_midis}
        # The frozen mapper predicts a set, not doubled octaves.  Preserve
        # octave gestures until a dedicated register model is validated.
        if len(original_pitch_classes) != len(original_midis):
            continue
        context, anchor, incumbent = paired_context(
            source_notes,
            source_times,
            groups,
            group_index,
            context_radius,
        )
        if len(context) != len(samples[0].get("context", [])):
            raise ValueError("Paired chord mapper context does not match training.")
        ranked = sorted(
            (
                (squared_distance(context, sample["context"]), sample)
                for sample in samples
            ),
            key=lambda item: item[0],
        )
        votes = ranked_vote_probabilities(ranked, neighbors, temperature)
        predicted, gain = decode_conservative_correction(
            votes,
            incumbent,
            incumbent_prior,
            minimum_gain,
        )
        examined += 1
        if predicted == incumbent:
            unchanged_predictions += 1
            continue

        onset = float(group[0]["time"])
        left = bisect.bisect_left(source_times, onset - support_radius_seconds)
        right = bisect.bisect_right(source_times, onset + support_radius_seconds)
        available = {
            (int(note["midi"]) % 12 - anchor) % 12
            for note in source_notes[left:right]
            if instrument_family(str(note.get("instrument") or "")) != "voice"
        }
        # Incumbent notes have already passed the production source decoder;
        # retaining one remains safe even if its raw attack sits just outside
        # the narrow support window.
        available.update(incumbent)
        if not predicted.issubset(available):
            unsupported_predictions += 1
            continue

        predicted_pitch_classes = {(anchor + interval) % 12 for interval in predicted}
        voiced = closest_joint_voicing(
            original_midis,
            predicted_pitch_classes,
            minimum=minimum_midi,
            maximum=maximum_midi,
        )
        if voiced is None:
            unvoiceable_predictions += 1
            continue
        mapping, additions = minimum_note_assignment(original_midis, voiced)
        if additions or len(mapping) != len(group):
            unvoiceable_predictions += 1
            continue

        changed_groups += 1
        accepted_gains.append(gain)
        for index, note in enumerate(group):
            midi = int(mapping[index])
            previous = int(note["midi"])
            if midi == previous:
                continue
            note["pairedChordOriginalMidi"] = previous
            note["pairedChordVoteGain"] = round(gain, 6)
            note["pairedChordMapperProfile"] = mapper.get("id")
            note["midi"] = midi
            note["note"] = midi_name(midi)
            note["voice"] = f"{note.get('arrangementRole', 'harmony')}-{midi}"
            changed_notes += 1

    result["notes"], collapsed = collapse_exact_left_hand_duplicates(
        result.get("notes", [])
    )
    diagnostics = {
        "profile": mapper.get("id"),
        "profileSha256": mapper.get("profileSha256"),
        "excludedSongId": exclude_song_id or None,
        "trainingContextFeatureCount": len(samples[0].get("context", [])),
        "neighbors": neighbors,
        "temperature": temperature,
        "incumbentPrior": incumbent_prior,
        "minimumGain": minimum_gain,
        "supportRadiusSeconds": support_radius_seconds,
        "eligibleTrainingSamples": len(samples),
        "examinedGroups": examined,
        "changedGroups": changed_groups,
        "changedNotes": changed_notes,
        "unchangedPredictions": unchanged_predictions,
        "unsupportedPredictions": unsupported_predictions,
        "unvoiceablePredictions": unvoiceable_predictions,
        "meanAcceptedVoteGain": round(
            sum(accepted_gains) / max(1, len(accepted_gains)), 6
        ),
        "collapsedExactLeftHandDuplicates": collapsed,
        "onsetsFrozen": True,
        "chordSizesFrozen": True,
        "rightHandFrozen": True,
        "sourceSupportRequired": True,
        "researchOnly": True,
    }
    result.setdefault("pianoArrangement", {})["pairedChordGestureMapper"] = diagnostics
    return result, diagnostics


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-song-id", default="")
    parser.add_argument("--support-radius", type=float, default=0.12)
    parser.add_argument("--minimum-midi", type=int, default=33)
    parser.add_argument("--maximum-midi", type=int, default=59)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.resolve().read_text(encoding="utf-8"))
    source = json.loads(args.source.resolve().read_text(encoding="utf-8"))
    mapper = json.loads(args.mapper.resolve().read_text(encoding="utf-8"))
    result, diagnostics = apply_mapper(
        candidate,
        source,
        mapper,
        exclude_song_id=args.exclude_song_id,
        support_radius_seconds=args.support_radius,
        minimum_midi=args.minimum_midi,
        maximum_midi=args.maximum_midi,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
