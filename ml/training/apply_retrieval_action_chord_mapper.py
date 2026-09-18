"""Apply the gated variable-size chord action retriever offline.

Only non-melody left-hand groups are eligible.  Proposed pitch classes must be
present in local source evidence (or already in the incumbent), onsets and the
right hand remain frozen, and the independently cross-validated abstention
gate can reject any proposal.
"""

from __future__ import annotations

import argparse
import copy
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

from piano_arranger import collapse_exact_left_hand_duplicates  # noqa: E402
from piano_arranger_adapter import normalize_source_notes  # noqa: E402

from .apply_chord_gesture_library import (  # noqa: E402
    closest_variable_voicing,
    minimum_note_assignment,
)
from .apply_paired_chord_gesture_mapper import tight_left_groups  # noqa: E402
from .fit_chord_gesture_library import squared_distance  # noqa: E402
from .fit_paired_chord_gesture_mapper import paired_context  # noqa: E402
from .fit_retrieval_action_abstention_gate import (  # noqa: E402
    proposal_feature_names,
    proposal_features,
)
from .fit_retrieval_action_chord_mapper import (  # noqa: E402
    decode_retrieval_action,
    load_json,
    source_supported_intervals,
)


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def gate_probability(features: list[float], gate: dict[str, Any]) -> float:
    names = list(gate.get("featureNames") or [])
    if names != proposal_feature_names(len(features) - 14):
        raise ValueError("Abstention gate feature contract does not match")
    weights = np.asarray(gate.get("weights") or [], dtype=float)
    means = np.asarray(gate.get("means") or [], dtype=float)
    scales = np.asarray(gate.get("scales") or [], dtype=float)
    vector = np.asarray(features, dtype=float)
    if not (
        len(vector) == len(weights) == len(means) == len(scales)
        and np.all(np.isfinite(vector))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(np.abs(scales) > 1e-9)
    ):
        raise ValueError("Abstention gate contains invalid coefficient arrays")
    logit = float(((vector - means) / scales) @ weights)
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))


def apply_mapper(
    candidate: dict[str, Any],
    source: dict[str, Any],
    mapper: dict[str, Any],
    gate: dict[str, Any],
    *,
    exclude_song_id: str = "",
    minimum_midi: int = 33,
    maximum_midi: int = 59,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mapper.get("type") != "paired-retrieval-action-chord-mapper-v2":
        raise ValueError("Unsupported retrieval-action chord mapper")
    if gate.get("type") != "retrieval-action-abstention-logistic-v1":
        raise ValueError("Unsupported retrieval-action abstention gate")
    if minimum_midi >= maximum_midi:
        raise ValueError("minimum_midi must be below maximum_midi")
    samples = [
        sample for sample in mapper.get("samples") or []
        if isinstance(sample, dict)
        and (not exclude_song_id or str(sample.get("songId")) != exclude_song_id)
    ]
    if not samples:
        raise ValueError("No eligible cross-song chord samples remain")

    result = copy.deepcopy(candidate)
    groups = tight_left_groups(result)
    source_notes = normalize_source_notes(source.get("notes", []))
    source_times = [float(note["time"]) for note in source_notes]
    policy = dict(gate.get("retrievalPolicy") or {})
    context_radius = float(mapper.get("contextRadiusSeconds", 0.35))
    threshold = float(gate.get("threshold", 0.5))
    removed_note_ids: set[int] = set()
    additions: list[dict[str, Any]] = []
    proposed = accepted = changed_notes = enlarged = reduced = 0
    skipped_octaves = unsupported = unvoiceable = 0
    probabilities: list[float] = []

    for group_index, group in enumerate(groups):
        original_midis = [int(note["midi"]) for note in group]
        incumbent_pitch_classes = {midi % 12 for midi in original_midis}
        if len(original_midis) > 1 and len(incumbent_pitch_classes) == 1:
            skipped_octaves += 1
            continue
        context, anchor, incumbent = paired_context(
            source_notes, source_times, groups, group_index, context_radius
        )
        ranked = sorted(
            (
                (squared_distance(context, sample["context"]), sample)
                for sample in samples
            ),
            key=lambda item: item[0],
        )[:40]
        inference_sample = {
            "context": context,
            "incumbentIntervals": sorted(incumbent),
        }
        available = source_supported_intervals(inference_sample)
        proposal, gain, retrieval_diagnostics = decode_retrieval_action(
            ranked,
            incumbent,
            available,
            neighbors=int(policy.get("neighbors", mapper.get("neighbors", 40))),
            temperature=float(policy.get("temperature", mapper.get("temperature", 0.48))),
            incumbent_prior=float(policy.get("incumbentPrior", mapper.get("incumbentPrior", 0.1))),
            minimum_gain=float(policy.get("minimumGain", mapper.get("minimumGain", 0.08))),
            maximum_size_change=int(policy.get("maximumSizeChange", mapper.get("maximumSizeChange", 1))),
            maximum_size=int(mapper.get("maximumChordSize", 3)),
        )
        if proposal == incumbent:
            continue
        proposed += 1
        features = proposal_features(
            inference_sample,
            ranked,
            incumbent,
            proposal,
            gain,
            retrieval_diagnostics,
        )
        probability = gate_probability(features, gate)
        if probability < threshold:
            continue
        predicted_pitch_classes = {(anchor + interval) % 12 for interval in proposal}
        if not proposal.issubset(available):
            unsupported += 1
            continue
        voiced = closest_variable_voicing(
            original_midis,
            predicted_pitch_classes,
            minimum=minimum_midi,
            maximum=maximum_midi,
        )
        if voiced is None:
            unvoiceable += 1
            continue
        mapping, added_midis = minimum_note_assignment(original_midis, voiced)
        ordered_group = sorted(group, key=lambda note: int(note["midi"]))
        accepted += 1
        probabilities.append(probability)
        reduced += int(len(voiced) < len(original_midis))
        enlarged += int(len(voiced) > len(original_midis))
        for index, note in enumerate(ordered_group):
            if index not in mapping:
                removed_note_ids.add(id(note))
                changed_notes += 1
                continue
            midi = int(mapping[index])
            previous = int(note["midi"])
            note["retrievalActionAcceptanceProbability"] = round(probability, 6)
            note["retrievalActionExpectedGain"] = round(gain, 6)
            if midi == previous:
                continue
            note["retrievalActionOriginalMidi"] = previous
            note["midi"] = midi
            note["note"] = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
            note["voice"] = f"{note.get('arrangementRole', 'harmony')}-{midi}"
            changed_notes += 1
        for midi in added_midis:
            template = min(ordered_group, key=lambda note: abs(int(note["midi"]) - midi))
            note = copy.deepcopy(template)
            note["retrievalActionOriginalMidi"] = int(note["midi"])
            note["retrievalActionAddedNote"] = True
            note["retrievalActionAcceptanceProbability"] = round(probability, 6)
            note["retrievalActionExpectedGain"] = round(gain, 6)
            note["generatedBy"] = "cross-song-retrieval-action-chord-v2"
            note["midi"] = int(midi)
            note["note"] = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
            note["voice"] = f"{note.get('arrangementRole', 'harmony')}-{midi}"
            additions.append(note)
            changed_notes += 1

    if removed_note_ids or additions:
        result["notes"] = [
            note for note in result.get("notes", []) if id(note) not in removed_note_ids
        ] + additions
    result["notes"], collapsed = collapse_exact_left_hand_duplicates(result.get("notes", []))
    diagnostics = {
        "profile": mapper.get("id"),
        "profileSha256": mapper.get("profileSha256"),
        "abstentionGate": gate.get("id"),
        "abstentionGateSha256": gate.get("gateSha256"),
        "excludedSongId": exclude_song_id or None,
        "eligibleTrainingSamples": len(samples),
        "proposedGroups": proposed,
        "acceptedGroups": accepted,
        "changedNotes": changed_notes,
        "reducedGroups": reduced,
        "enlargedGroups": enlarged,
        "preservedOctaveGroups": skipped_octaves,
        "unsupportedProposals": unsupported,
        "unvoiceableProposals": unvoiceable,
        "acceptanceThreshold": round(threshold, 6),
        "meanAcceptedProbability": round(sum(probabilities) / max(1, len(probabilities)), 6),
        "collapsedExactLeftHandDuplicates": collapsed,
        "rightHandFrozen": True,
        "onsetsFrozen": True,
        "sourceSupportRequired": True,
        "researchOnly": True,
    }
    result.setdefault("pianoArrangement", {})["retrievalActionChordMapper"] = diagnostics
    return result, diagnostics


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-song-id", default="")
    parser.add_argument("--minimum-midi", type=int, default=33)
    parser.add_argument("--maximum-midi", type=int, default=59)
    args = parser.parse_args()
    result, diagnostics = apply_mapper(
        load_json(args.candidate.resolve()),
        load_json(args.source.resolve()),
        load_json(args.mapper.resolve()),
        load_json(args.gate.resolve()),
        exclude_song_id=args.exclude_song_id,
        minimum_midi=args.minimum_midi,
        maximum_midi=args.maximum_midi,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
