"""Replace weak accompaniment tones with stronger raw support in-place.

Unlike note supplementation, this operator freezes every gesture onset, size,
duration and velocity.  It changes at most a small number of left-hand pitch
classes inside an existing attack, and only when a frozen selector gives the
new raw event a clear probability margin over the incumbent tone.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # Package and direct-script execution.
    from server.piano_arranger_adapter import (
        instrument_family,
        normalize_source_notes,
        selection_scores,
    )
except ImportError:  # pragma: no cover
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SERVER_ROOT = REPO_ROOT / "server"
    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))
    from piano_arranger_adapter import (  # type: ignore
        instrument_family,
        normalize_source_notes,
        selection_scores,
    )

try:
    from .apply_raw_support_recovery import fold_midi, load_json, selector_model
except ImportError:  # pragma: no cover
    from apply_raw_support_recovery import fold_midi, load_json, selector_model  # type: ignore


def note_time(note: dict[str, Any]) -> float:
    return float(note.get("time", note.get("startTime", note.get("start", 0.0))))


def note_midi(note: dict[str, Any]) -> int:
    return int(round(float(note.get("midi", note.get("pitch")))))


def onset_groups(notes: list[dict[str, Any]], window: float = 0.035) -> list[list[int]]:
    ordered = sorted(range(len(notes)), key=lambda index: (note_time(notes[index]), note_midi(notes[index])))
    groups: list[list[int]] = []
    for index in ordered:
        if not groups or note_time(notes[index]) - note_time(notes[groups[-1][0]]) > window:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def raw_window(
    notes: list[dict[str, Any]], times: list[float], time: float, radius: float
) -> list[int]:
    left = bisect.bisect_left(times, time - radius)
    right = bisect.bisect_right(times, time + radius)
    return list(range(left, right))


def best_by_pitch_class(
    indices: list[int], notes: list[dict[str, Any]], scores: list[float]
) -> dict[int, int]:
    best: dict[int, int] = {}
    for index in indices:
        pitch_class = int(notes[index]["midi"]) % 12
        previous = best.get(pitch_class)
        if previous is None or float(scores[index]) > float(scores[previous]):
            best[pitch_class] = index
    return best


def apply_swaps(
    candidate: dict[str, Any],
    source: dict[str, Any],
    profile: dict[str, Any],
    *,
    threshold: float,
    minimum_gain: float,
    source_radius: float,
    maximum_replacements_per_gesture: int,
    maximum_source_midi: int,
    register_shift: int,
    minimum_output_midi: int,
    maximum_output_midi: int,
    minimum_source_midi: int = 0,
    source_families: set[str] | None = None,
    replace_hands: set[str] | None = None,
    replace_roles: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(candidate)
    notes = output.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Candidate has no notes list.")
    raw_notes = normalize_source_notes(source.get("notes") or [])
    scores = selection_scores(
        raw_notes, {"selectionModel": selector_model(profile)}
    )
    allowed_source_families = (
        {str(value).strip().lower() for value in source_families}
        if source_families
        else None
    )
    allowed_hands = (
        {str(value).strip().lower() for value in replace_hands}
        if replace_hands
        else {"left"}
    )
    allowed_roles = (
        {str(value).strip().lower() for value in replace_roles}
        if replace_roles
        else {"bass", "harmony"}
    )
    usable_indices = [
        index
        for index, note in enumerate(raw_notes)
        if (
            instrument_family(str(note.get("instrument") or ""))
            in allowed_source_families
            if allowed_source_families is not None
            else instrument_family(str(note.get("instrument") or "")) != "voice"
        )
        and int(note["midi"]) >= minimum_source_midi
        and int(note["midi"]) <= maximum_source_midi
    ]
    usable_notes = [raw_notes[index] for index in usable_indices]
    usable_scores = [float(scores[index]) for index in usable_indices]
    usable_times = [float(note["time"]) for note in usable_notes]

    replacements = 0
    skipped_no_proposal = skipped_gain = skipped_collision = 0
    replaced_roles: Counter[str] = Counter()
    for group in onset_groups(notes):
        time = note_time(notes[group[0]])
        window_indices = raw_window(usable_notes, usable_times, time, source_radius)
        raw_by_pc = best_by_pitch_class(window_indices, usable_notes, usable_scores)
        current_pcs = {note_midi(notes[index]) % 12 for index in group}
        proposals = [
            index
            for pitch_class, index in raw_by_pc.items()
            if pitch_class not in current_pcs and usable_scores[index] >= threshold
        ]
        if not proposals:
            skipped_no_proposal += 1
            continue
        replaceable = [
            index
            for index in group
            if str(notes[index].get("hand") or "").lower() in allowed_hands
            and str(notes[index].get("arrangementRole") or "").lower()
            in allowed_roles
        ]
        if not replaceable:
            continue

        def incumbent_support(index: int) -> float:
            raw_index = raw_by_pc.get(note_midi(notes[index]) % 12)
            return usable_scores[raw_index] if raw_index is not None else 0.0

        available = sorted(replaceable, key=lambda index: (incumbent_support(index), note_midi(notes[index])))
        accepted_in_group = 0
        for proposal_index in sorted(
            proposals, key=lambda index: usable_scores[index], reverse=True
        ):
            if accepted_in_group >= maximum_replacements_per_gesture or not available:
                break
            incumbent_index = available.pop(0)
            gain = usable_scores[proposal_index] - incumbent_support(incumbent_index)
            if gain < minimum_gain:
                skipped_gain += 1
                continue
            proposal = usable_notes[proposal_index]
            target_midi = fold_midi(
                int(proposal["midi"]),
                minimum_output_midi,
                maximum_output_midi,
                register_shift,
            )
            occupied = {
                note_midi(notes[index])
                for index in group
                if index != incumbent_index
            }
            if target_midi in occupied or target_midi % 12 in {
                midi % 12 for midi in occupied
            }:
                skipped_collision += 1
                continue
            incumbent = notes[incumbent_index]
            original_midi = note_midi(incumbent)
            incumbent["rawSupportSwapOriginalMidi"] = original_midi
            incumbent["rawSupportSwapProbability"] = round(
                usable_scores[proposal_index], 6
            )
            incumbent["rawSupportSwapIncumbentProbability"] = round(
                usable_scores[proposal_index] - gain, 6
            )
            incumbent["rawSupportSwapProfile"] = profile.get("id")
            incumbent["rawSupportSwapSourceIndex"] = proposal.get("sourceIndex")
            incumbent["midi"] = target_midi
            if "pitch" in incumbent:
                incumbent["pitch"] = target_midi
            incumbent["sourceInstrument"] = proposal.get("instrument")
            incumbent["sourceMidiBeforeArrangement"] = int(proposal["midi"])
            replaced_roles[str(incumbent.get("arrangementRole") or "unknown")] += 1
            current_pcs.discard(original_midi % 12)
            current_pcs.add(target_midi % 12)
            replacements += 1
            accepted_in_group += 1
    diagnostics = {
        "profile": profile.get("id"),
        "threshold": threshold,
        "minimumProbabilityGain": minimum_gain,
        "sourceRadiusSeconds": source_radius,
        "maximumReplacementsPerGesture": maximum_replacements_per_gesture,
        "minimumSourceMidi": minimum_source_midi,
        "maximumSourceMidi": maximum_source_midi,
        "sourceFamilies": sorted(allowed_source_families or []),
        "replaceHands": sorted(allowed_hands),
        "replaceRoles": sorted(allowed_roles),
        "candidateNotes": len(notes),
        "gestureCount": len(onset_groups(notes)),
        "replacements": replacements,
        "replacedRoles": dict(replaced_roles),
        "skippedNoProposal": skipped_no_proposal,
        "skippedInsufficientGain": skipped_gain,
        "skippedCollision": skipped_collision,
        "frozenProperties": ["onset", "gesture-size", "duration", "velocity"],
        "warning": "Calibration/research operator; promotion requires song-level transfer and sealed listening evidence.",
    }
    output.setdefault("diagnostics", {})["rawSupportGestureSwap"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--minimum-probability-gain", type=float, default=0.20)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--maximum-replacements-per-gesture", type=int, default=1)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument("--minimum-source-midi", type=int, default=0)
    parser.add_argument(
        "--source-families",
        default="",
        help="Comma-separated normalized families; empty preserves non-voice behavior.",
    )
    parser.add_argument("--replace-hands", default="left")
    parser.add_argument("--replace-roles", default="bass,harmony")
    parser.add_argument("--register-shift-semitones", type=int, default=0)
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    args = parser.parse_args()
    output, diagnostics = apply_swaps(
        load_json(Path(args.candidate).resolve()),
        load_json(Path(args.source).resolve()),
        load_json(Path(args.profile).resolve()),
        threshold=max(0.0, min(1.0, float(args.threshold))),
        minimum_gain=max(0.0, float(args.minimum_probability_gain)),
        source_radius=max(0.01, float(args.source_radius_seconds)),
        maximum_replacements_per_gesture=max(
            1, int(args.maximum_replacements_per_gesture)
        ),
        maximum_source_midi=int(args.maximum_source_midi),
        register_shift=int(args.register_shift_semitones),
        minimum_output_midi=int(args.minimum_output_midi),
        maximum_output_midi=int(args.maximum_output_midi),
        minimum_source_midi=int(args.minimum_source_midi),
        source_families={
            value.strip().lower()
            for value in args.source_families.split(",")
            if value.strip()
        }
        or None,
        replace_hands={
            value.strip().lower()
            for value in args.replace_hands.split(",")
            if value.strip()
        },
        replace_roles={
            value.strip().lower()
            for value in args.replace_roles.split(",")
            if value.strip()
        },
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
