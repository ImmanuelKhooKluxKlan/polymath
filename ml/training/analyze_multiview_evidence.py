"""Measure whether synchronized audio views add missing transcription evidence.

This is an oracle audit, not an inference policy.  It asks whether a reference
event can be supported by the full mix, a separated vocal view, a separated
backing view, or their union under one frozen alignment.  The target is used
only for retrospective measurement and is never used to select runtime notes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .analyze_arranger_error_budget import (
        clip_observed_notes,
        pitched_payload,
    )
    from .evaluate_piano_arranger import (
        load_json,
        normalize_notes,
        prepare_reference_notes,
        trusted_source_ranges,
    )
except ImportError:  # pragma: no cover
    from analyze_arranger_error_budget import (  # type: ignore
        clip_observed_notes,
        pitched_payload,
    )
    from evaluate_piano_arranger import (  # type: ignore
        load_json,
        normalize_notes,
        prepare_reference_notes,
        trusted_source_ranges,
    )


def maximum_tolerance_matches(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    tolerance: float,
    pitch_mode: str,
) -> int:
    """Return a maximum-cardinality one-to-one match for sorted time points.

    Nearest-first greedy matching is not monotonic: adding an observation can
    steal a match from a later reference event.  Within a pitch (or pitch
    class) group, the standard two-pointer interval matcher is maximum-cardinality
    and guarantees that adding a view cannot reduce an oracle ceiling.
    """

    if pitch_mode not in {"onset", "pitch-class", "exact"}:
        raise ValueError(f"Unsupported pitch mode: {pitch_mode}")

    def key(note: dict[str, float | int]) -> int:
        if pitch_mode == "onset":
            return 0
        midi = int(note["midi"])
        return midi % 12 if pitch_mode == "pitch-class" else midi

    reference_groups: dict[int, list[float]] = defaultdict(list)
    observed_groups: dict[int, list[float]] = defaultdict(list)
    for note in reference:
        reference_groups[key(note)].append(float(note["time"]))
    for note in observed:
        observed_groups[key(note)].append(float(note["time"]))

    matches = 0
    for group, reference_times in reference_groups.items():
        observed_times = observed_groups.get(group, [])
        reference_times.sort()
        observed_times.sort()
        reference_index = 0
        observed_index = 0
        while (
            reference_index < len(reference_times)
            and observed_index < len(observed_times)
        ):
            reference_time = reference_times[reference_index]
            observed_time = observed_times[observed_index]
            if observed_time < reference_time - tolerance:
                observed_index += 1
            elif reference_time < observed_time - tolerance:
                reference_index += 1
            else:
                matches += 1
                reference_index += 1
                observed_index += 1
    return matches


def evidence_counts(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    tolerance: float,
) -> dict[str, int | float]:
    reference_count = len(reference)
    onset = maximum_tolerance_matches(reference, observed, tolerance, "onset")
    pitch_class = maximum_tolerance_matches(
        reference, observed, tolerance, "pitch-class"
    )
    exact = maximum_tolerance_matches(reference, observed, tolerance, "exact")
    return {
        "observedNotes": len(observed),
        "onsetMatches": onset,
        "pitchClassMatches": pitch_class,
        "exactPitchMatches": exact,
        "onsetRecallCeiling": round(onset / max(1, reference_count), 6),
        "pitchClassRecallCeiling": round(
            pitch_class / max(1, reference_count), 6
        ),
        "exactPitchRecallCeiling": round(exact / max(1, reference_count), 6),
    }


def primary_preserving_union(
    views: dict[str, list[dict[str, float | int]]],
    primary_name: str,
    duplicate_window_seconds: float = 0.04,
) -> list[dict[str, float | int]]:
    """Combine views without counting one detected event two or three times.

    The primary view remains byte-for-byte present.  A secondary event is added
    only when no already-kept event has the same MIDI pitch within the small
    cross-view window.  Same-view retriggers are never collapsed.
    """

    ordered_names = [primary_name] + [name for name in views if name != primary_name]
    union: list[dict[str, float | int]] = []
    by_midi: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for name in ordered_names:
        for source_note in views[name]:
            note = {**source_note, "_view": name}
            midi = int(note["midi"])
            time = float(note["time"])
            if name != primary_name and any(
                abs(float(existing["time"]) - time) <= duplicate_window_seconds
                for existing in by_midi[midi]
            ):
                continue
            union.append(note)
            by_midi[midi].append(note)
    return sorted(union, key=lambda note: (float(note["time"]), int(note["midi"])))


def analyze_pair(pair: dict[str, Any], tolerance: float) -> dict[str, Any]:
    alignment = load_json(Path(pair["alignmentReport"]))
    ranges = trusted_source_ranges(alignment)
    reference = prepare_reference_notes(
        pair,
        load_json(Path(pair["target"])),
        alignment,
    )
    views_config = pair.get("views")
    if not isinstance(views_config, dict) or len(views_config) < 2:
        raise ValueError("Each pair requires at least two named audio views.")

    views: dict[str, list[dict[str, float | int]]] = {}
    for name, filename in views_config.items():
        path = Path(str(filename))
        payload = pitched_payload(load_json(path))
        views[str(name)] = clip_observed_notes(
            normalize_notes(payload), pair, ranges
        )

    primary_name = str(pair.get("primaryView") or next(iter(views)))
    if primary_name not in views:
        raise ValueError(f"Unknown primaryView: {primary_name}")
    individual = {
        name: evidence_counts(reference, notes, tolerance)
        for name, notes in views.items()
    }
    union_notes = primary_preserving_union(views, primary_name)
    union = evidence_counts(reference, union_notes, tolerance)
    primary = individual[primary_name]
    return {
        "referenceNotes": len(reference),
        "toleranceSeconds": tolerance,
        "primaryView": primary_name,
        "individualViews": individual,
        "union": union,
        "unionMinusPrimary": {
            "onsetRecallCeiling": round(
                float(union["onsetRecallCeiling"])
                - float(primary["onsetRecallCeiling"]),
                6,
            ),
            "pitchClassRecallCeiling": round(
                float(union["pitchClassRecallCeiling"])
                - float(primary["pitchClassRecallCeiling"]),
                6,
            ),
            "exactPitchRecallCeiling": round(
                float(union["exactPitchRecallCeiling"])
                - float(primary["exactPitchRecallCeiling"]),
                6,
            ),
        },
        "deduplication": {
            "strategy": "preserve-primary-and-collapse-cross-view-exact-midi",
            "windowSeconds": 0.04,
            "rawViewNotes": sum(len(notes) for notes in views.values()),
            "unionNotes": len(union_notes),
            "collapsedCrossViewDuplicates": sum(len(notes) for notes in views.values())
            - len(union_notes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance-ms", type=float, default=250.0)
    args = parser.parse_args()
    tolerance = args.tolerance_ms / 1000.0
    if tolerance <= 0:
        raise ValueError("--tolerance-ms must be positive")
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    songs = {
        str(pair["id"]): analyze_pair(pair, tolerance)
        for pair in manifest.get("pairs", [])
    }
    report = {
        "schema": "polymath-multiview-evidence-audit-v1",
        "evidenceBoundary": (
            "Retrospective oracle only; references are never available to inference."
        ),
        "manifest": str(manifest_path),
        "songs": songs,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
