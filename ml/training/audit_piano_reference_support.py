"""Locate piano note losses without changing a model, scorer, or arrangement.

The maximum-cardinality figures are reference-aware diagnostic ceilings, not
deployable selection rules. Primary frozen nearest-match scores are reported
alongside them and remain unchanged by this audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from ml.training.evaluate_piano_arranger import (
    f1_metrics, greedy_matches, load_json, normalize_notes, notes_inside_ranges,
    prepare_reference_notes, trusted_source_ranges,
)
from ml.training.train_muscriptor_piano import sha256_file


def maximum_time_matches(reference: list[dict], observed: list[dict], tolerance: float, *, octave_equivalent: bool = False) -> list[tuple[int, int]]:
    """Maximum one-to-one matches for a fixed pitch key and time tolerance.

    Earliest feasible pairs are optimal on sorted one-dimensional timelines.
    Matching the *nearest* candidate instead can steal a later target's only
    match. This diagnostic never changes the pre-existing frozen gate scorer.
    """
    if tolerance < 0:
        raise ValueError("Onset tolerance cannot be negative")
    groups = defaultdict(lambda: ([], []))
    for side, notes in enumerate((reference, observed)):
        for index, note in enumerate(notes):
            pitch = int(note["midi"])
            groups[pitch % 12 if octave_equivalent else pitch][side].append((float(note["time"]), index))
    matches = []
    for targets, candidates in groups.values():
        targets.sort()
        candidates.sort()
        left = right = 0
        while left < len(targets) and right < len(candidates):
            target_time, target_index = targets[left]
            candidate_time, candidate_index = candidates[right]
            if target_time < candidate_time - tolerance - 1e-12:
                left += 1
            elif candidate_time < target_time - tolerance - 1e-12:
                right += 1
            else:
                matches.append((target_index, candidate_index))
                left += 1
                right += 1
    return sorted(matches)


def support_audit(reference: list[dict], stages: dict[str, list[dict]]) -> dict[str, Any]:
    reports = {}
    for name, observed in stages.items():
        scores = {}
        for milliseconds in (50, 100, 250):
            tolerance = milliseconds / 1000
            exact = maximum_time_matches(reference, observed, tolerance)
            pitch_class = maximum_time_matches(reference, observed, tolerance, octave_equivalent=True)
            frozen = greedy_matches(reference, observed, tolerance)
            scores[f"{milliseconds}ms"] = {
                "frozenExactPitchOnset": f1_metrics(len(reference), len(observed), len(frozen)),
                "maximumExactPitchOnset": f1_metrics(len(reference), len(observed), len(exact)),
                "oraclePitchClassOnsetSupport": f1_metrics(len(reference), len(observed), len(pitch_class)),
                "nearestMatchingUndercount": len(exact) - len(frozen),
            }
        reports[name] = {"notesInApprovedRanges": len(observed), "scores": scores}
    segments = []
    end = max((float(row["time"]) for row in reference), default=0)
    for start in range(0, int(end) + 1, 30):
        target = [row for row in reference if start <= float(row["time"]) < start + 30]
        if not target:
            continue
        row = {"startSeconds": start, "endSeconds": start + 30, "referenceNotes": len(target), "stages": {}}
        for name, observed in stages.items():
            candidates = [note for note in observed if start - .25 <= float(note["time"]) < start + 30.25]
            exact = len(maximum_time_matches(target, candidates, .1))
            flexible = len(maximum_time_matches(target, candidates, .25, octave_equivalent=True))
            row["stages"][name] = {"exact100msSupportedTargets": exact, "exact100msSupportRecall": round(exact / len(target), 6), "oraclePitchClass250msSupportedTargets": flexible, "oraclePitchClass250msSupportRecall": round(flexible / len(target), 6)}
        segments.append(row)
    registers = []
    for label, lower, upper in (("A1-B2", 33, 47), ("C3-B3", 48, 59), ("C4-B4", 60, 71), ("C5-B5", 72, 83), ("C6-C8", 84, 108)):
        target = [row for row in reference if lower <= int(row["midi"]) <= upper]
        registers.append({"register": label, "referenceNotes": len(target), "exact100msSupportRecall": {name: round(len(maximum_time_matches(target, observed, .1)) / max(1, len(target)), 6) for name, observed in stages.items()}})
    transitions = {}
    if "raw" in stages and "arranged" in stages:
        raw = {left for left, _ in maximum_time_matches(reference, stages["raw"], .1)}
        arranged = {left for left, _ in maximum_time_matches(reference, stages["arranged"], .1)}
        transitions = {"sameTargetExact100msSupportLostAfterProcessing": len(raw - arranged), "sameTargetExact100msSupportGainedAfterProcessing": len(arranged - raw), "exactSupportPreserved": len(raw & arranged), "unsupportedAtEitherStage": len(reference) - len(raw | arranged), "interpretation": "Reference-aware paired diagnostic; not a causal proof or an inference-time filtering rule."}
    velocities = Counter(round(float(row["velocity"]), 4) for row in reference)
    return {"schema": "polymath-piano-reference-support-audit-v1", "referenceNotes": len(reference), "stages": reports, "pairedStageSupport": transitions, "segments30Seconds": segments, "registers": registers, "referenceVelocity": {"distinctValues": len(velocities), "mostFrequent": velocities.most_common(8), "warning": "Visual or inferred velocity labels are not measured finger force. Do not infer a pianist's dynamics from constant/default labels."}, "interpretation": "Opened-development failure localization. Oracle pitch-class matching ignores octave errors and needs the answer key; it is not model accuracy or a deployable solution. The frozen primary scorer and production weights are unchanged."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--alignment-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--arranged", type=Path, required=True)
    parser.add_argument("--arranged-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Audit result already exists")
    integrity = {}
    for name in ("reference", "alignment", "raw", "arranged"):
        path = getattr(args, name)
        expected = getattr(args, f"{name}_sha256")
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen audit input changed: {path}")
        integrity[name] = {"path": str(path), "sha256": expected}
    alignment = load_json(args.alignment)
    reference = prepare_reference_notes({"referenceAlreadyAligned": True, "referenceTransposeSemitones": 0}, load_json(args.reference), alignment)
    stages = {name: notes_inside_ranges(normalize_notes(load_json(getattr(args, name))), trusted_source_ranges(alignment)) for name in ("raw", "arranged")}
    result = support_audit(reference, stages)
    result["integrity"] = integrity
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "referenceNotes": len(reference), "stages": result["stages"], "pairedStageSupport": result["pairedStageSupport"], "referenceVelocity": result["referenceVelocity"]}, indent=2))


if __name__ == "__main__":
    main()
