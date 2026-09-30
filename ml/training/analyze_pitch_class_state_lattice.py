"""Audit a compact, inference-safe lattice of whole pitch-class gestures.

Independent note classifiers can improve a few labels while making a chord
less coherent.  This tool asks a more useful question before fitting another
model: does a compact list of musically plausible *sets* contain a good answer
at each frozen arranger onset?

References are used only to score the lattice after it has been generated.
The generator itself sees candidate and raw-source evidence, neighboring
source cells, and fixed interval templates.  It never receives a target pitch
class, song title, or absolute timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:  # Package and direct-script execution.
    from .analyze_pianist_texture_patterns import analyze_song
except ImportError:  # pragma: no cover
    from analyze_pianist_texture_patterns import analyze_song  # type: ignore


TRIAD_TEMPLATES = (
    (0, 4, 7),
    (0, 3, 7),
    (0, 5, 7),
    (0, 4, 7, 10),
    (0, 3, 7, 10),
    (0, 4, 7, 11),
    (0, 3, 6),
)


def pitch_classes(values: Iterable[Any]) -> frozenset[int]:
    return frozenset(
        int(value) % 12
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )


def target_set(cell: dict[str, Any]) -> frozenset[int]:
    return pitch_classes(
        value
        for gesture in cell.get("referencePitchClasses") or []
        for value in gesture
    )


def set_f1(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    overlap = len(left & right)
    return 2.0 * overlap / max(1, len(left) + len(right))


def evidence_by_pitch_class(cell: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = cell.get("sourcePitchClassEvidence") or []
    return {
        int(row.get("pitchClass", -1)) % 12: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("pitchClass"), (int, float))
    }


def multiview_evidence_by_pitch_class(
    cell: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    rows = cell.get("multiViewPitchClassEvidence") or []
    return {
        int(row.get("pitchClass", -1)) % 12: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("pitchClass"), (int, float))
    }


def multiview_pitch_classes(cell: dict[str, Any]) -> frozenset[int]:
    return frozenset(
        pitch_class
        for pitch_class, row in multiview_evidence_by_pitch_class(cell).items()
        if float(row.get("viewAgreementCount") or 0.0) > 0.0
    )


def evidence_score(cell: dict[str, Any], pitch_class: int) -> float:
    """Rank source pitches using only evidence available during inference."""

    candidate = pitch_classes(cell.get("candidatePitchClasses") or [])
    evidence = evidence_by_pitch_class(cell).get(pitch_class, {})
    families = evidence.get("familyShares") or {}
    attack_count = max(0.0, float(evidence.get("attackCount") or 0.0))
    note_count = max(0.0, float(evidence.get("noteCount") or 0.0))
    velocity = max(0.0, min(1.0, float(evidence.get("maximumVelocity") or 0.0)))
    register_distance = max(
        0.0, float(evidence.get("minimumRegisterDistanceSemitones") or 0.0)
    )
    melody_family = float(families.get("voice") or 0.0) + 0.45 * float(
        families.get("piano") or 0.0
    )
    multiview = multiview_evidence_by_pitch_class(cell).get(pitch_class, {})
    view_agreement = max(0.0, float(multiview.get("viewAgreementCount") or 0.0))
    attack_agreement = max(
        0.0, float(multiview.get("attackAgreementCount") or 0.0)
    )
    return (
        2.2 * float(pitch_class in candidate)
        + 1.7 * min(1.0, attack_count)
        + 0.45 * min(2.0, note_count)
        + 0.75 * velocity
        + 0.55 * melody_family
        + 0.30 * min(3.0, view_agreement)
        + 0.45 * min(3.0, attack_agreement)
        - 0.018 * min(36.0, register_distance)
    )


def ranked_source(cell: dict[str, Any]) -> list[int]:
    source = pitch_classes(cell.get("sourcePitchClasses") or []) | multiview_pitch_classes(cell)
    return sorted(source, key=lambda pc: (-evidence_score(cell, pc), pc))


def state_prior_score(
    state: frozenset[int],
    cell: dict[str, Any],
    previous_source: frozenset[int],
    next_source: frozenset[int],
) -> float:
    candidate = pitch_classes(cell.get("candidatePitchClasses") or [])
    source = pitch_classes(cell.get("sourcePitchClasses") or []) | multiview_pitch_classes(cell)
    evidence = sum(evidence_score(cell, pc) for pc in state)
    unsupported = len(state - source - candidate)
    persistent = len(state & previous_source) + len(state & next_source)
    size = len(state)
    size_penalty = max(0, size - 5) ** 2 + 0.15 * abs(size - min(4, max(1, len(candidate))))
    return (
        evidence
        + 0.45 * persistent
        + 0.6 * len(state & candidate)
        - 2.5 * unsupported
        - size_penalty
    )


def generate_states(
    rows: list[dict[str, Any]],
    index: int,
    *,
    maximum_states: int = 64,
    allow_harmonic_completion: bool = False,
) -> list[frozenset[int]]:
    """Generate a bounded whole-gesture lattice without reference leakage."""

    cell = rows[index]
    candidate = pitch_classes(cell.get("candidatePitchClasses") or [])
    source = pitch_classes(cell.get("sourcePitchClasses") or []) | multiview_pitch_classes(cell)
    previous_source = (
        pitch_classes(rows[index - 1].get("sourcePitchClasses") or [])
        | multiview_pitch_classes(rows[index - 1])
        if index
        else frozenset()
    )
    next_source = (
        pitch_classes(rows[index + 1].get("sourcePitchClasses") or [])
        | multiview_pitch_classes(rows[index + 1])
        if index + 1 < len(rows)
        else frozenset()
    )
    ranked = ranked_source(cell)
    states: set[frozenset[int]] = set()

    def add(values: Iterable[int]) -> None:
        state = pitch_classes(values)
        if state:
            states.add(state)

    add(candidate)
    add(source)
    add(candidate & source)
    add(source & (previous_source | next_source))
    add(candidate | (source & previous_source & next_source))

    for size in range(1, min(6, len(ranked)) + 1):
        add(ranked[:size])

    for pc in ranked[:6]:
        add(candidate | {pc})
        add((candidate & source) | {pc})
    candidate_ranked = sorted(
        candidate, key=lambda pc: (evidence_score(cell, pc), pc)
    )
    for count in range(1, min(3, len(candidate_ranked)) + 1):
        add(candidate - set(candidate_ranked[:count]))
    for remove in candidate_ranked[:3]:
        for addition in ranked[:4]:
            add((candidate - {remove}) | {addition})

    local_support = source | previous_source | next_source | candidate
    roots = ranked[:5] or sorted(candidate)[:5]
    for root in roots:
        for template in TRIAD_TEMPLATES:
            chord = frozenset((root + interval) % 12 for interval in template)
            supported = chord & local_support
            if len(supported) >= 2:
                add(supported)
                # A pianist may complete an implied triad even when source
                # separation missed one chord tone.  Keep this experimental
                # expansion opt-in so the accepted lattice stays frozen.
                if allow_harmonic_completion:
                    add(chord)
            if chord <= local_support:
                add(chord)

    ordered = sorted(
        states,
        key=lambda state: (
            -state_prior_score(state, cell, previous_source, next_source),
            len(state),
            tuple(sorted(state)),
        ),
    )
    # The unchanged candidate is always retained, even when its prior is low.
    selected = ordered[: max(1, maximum_states)]
    if candidate and candidate not in selected:
        selected[-1] = candidate
    return selected


def micro_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["truePositive"]
    fp = counts["falsePositive"]
    fn = counts["falseNegative"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        **counts,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2.0 * precision * recall / max(1e-12, precision + recall), 6),
    }


def score_song(song: dict[str, Any], *, maximum_states: int) -> dict[str, Any]:
    rows = list(song.get("cells") or [])
    baseline = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    oracle = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    support = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0}
    exact = baseline_exact = anchored = 0
    state_counts: list[int] = []
    for index, cell in enumerate(rows):
        target = target_set(cell)
        if not target:
            continue
        anchored += 1
        candidate = pitch_classes(cell.get("candidatePitchClasses") or [])
        source = pitch_classes(cell.get("sourcePitchClasses") or []) | multiview_pitch_classes(cell)
        states = generate_states(rows, index, maximum_states=maximum_states)
        chosen = max(
            states,
            key=lambda state: (
                set_f1(target, state),
                -len(target ^ state),
                state_prior_score(
                    state,
                    cell,
                    pitch_classes(rows[index - 1].get("sourcePitchClasses") or [])
                    | multiview_pitch_classes(rows[index - 1])
                    if index
                    else frozenset(),
                    pitch_classes(rows[index + 1].get("sourcePitchClasses") or [])
                    | multiview_pitch_classes(rows[index + 1])
                    if index + 1 < len(rows)
                    else frozenset(),
                ),
            ),
        )
        supported_target = target & (source | candidate)
        for counts, predicted in (
            (baseline, candidate),
            (oracle, chosen),
            (support, supported_target),
        ):
            counts["truePositive"] += len(target & predicted)
            counts["falsePositive"] += len(predicted - target)
            counts["falseNegative"] += len(target - predicted)
        baseline_exact += int(candidate == target)
        exact += int(chosen == target)
        state_counts.append(len(states))
    return {
        "id": song.get("id"),
        "anchoredCells": anchored,
        "baseline": {
            **micro_metrics(baseline),
            "exactSetRate": round(baseline_exact / max(1, anchored), 6),
        },
        "compactLatticeOracle": {
            **micro_metrics(oracle),
            "exactSetRate": round(exact / max(1, anchored), 6),
        },
        "perfectSupportedSetOracle": micro_metrics(support),
        "lattice": {
            "maximumStates": maximum_states,
            "meanStates": round(sum(state_counts) / max(1, len(state_counts)), 3),
            "maximumObservedStates": max(state_counts, default=0),
        },
    }


def texture_audit_from_arranger_manifest(
    manifest_path: Path,
    candidate_directory: Path,
    *,
    onset_window: float,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    songs: list[dict[str, Any]] = []
    for pair in payload.get("pairs") or []:
        song_id = str(pair["id"])
        candidate_options = (
            candidate_directory / f"{song_id}-arranged.json",
            candidate_directory / f"{song_id}.json",
        )
        candidate = next((path for path in candidate_options if path.is_file()), None)
        if candidate is None:
            raise FileNotFoundError(
                f"No candidate for {song_id!r} in {candidate_directory}"
            )
        row = {
            "id": song_id,
            "reference": pair["target"],
            "source": pair["source"],
            "alignment": pair["alignmentReport"],
            "candidate": str(candidate),
            "referenceAlreadyAligned": bool(pair.get("referenceAlreadyAligned")),
            "referenceTransposeSemitones": int(
                pair.get("referenceTransposeSemitones") or 0
            ),
            "referenceEndSeconds": pair.get("referenceEndSeconds"),
            "candidateEndSeconds": pair.get("candidateEndSeconds"),
            "weight": float(pair.get("weight", 1.0)),
        }
        report = analyze_song(row, onset_window)
        report["weight"] = row["weight"]
        songs.append(report)
    return {
        "schema": "polymath-pitch-class-state-lattice-audit-v1",
        "sourceManifest": str(manifest_path),
        "candidateDirectory": str(candidate_directory),
        "songs": songs,
    }


def aggregate(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for section in ("baseline", "compactLatticeOracle", "perfectSupportedSetOracle"):
        total = sum(max(0.0, weights.get(str(row["id"]), 1.0)) for row in rows)
        metrics[section] = {
            name: round(
                sum(
                    max(0.0, weights.get(str(row["id"]), 1.0))
                    * float(row[section][name])
                    for row in rows
                )
                / max(1e-12, total),
                6,
            )
            for name in ("precision", "recall", "f1")
        }
    metrics["recoverableF1Points"] = round(
        metrics["compactLatticeOracle"]["f1"] - metrics["baseline"]["f1"], 6
    )
    metrics["generatorGapToSupportedOracle"] = round(
        metrics["perfectSupportedSetOracle"]["f1"]
        - metrics["compactLatticeOracle"]["f1"],
        6,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--texture-audit")
    source.add_argument("--arranger-manifest")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output")
    parser.add_argument("--maximum-states", type=int, default=64)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    args = parser.parse_args()

    if args.texture_audit:
        audit = json.loads(Path(args.texture_audit).read_text(encoding="utf-8-sig"))
    else:
        if not args.candidate_dir:
            parser.error("--candidate-dir is required with --arranger-manifest")
        audit = texture_audit_from_arranger_manifest(
            Path(args.arranger_manifest).resolve(),
            Path(args.candidate_dir).resolve(),
            onset_window=max(0.005, float(args.onset_window_seconds)),
        )
    if args.audit_output:
        audit_path = Path(args.audit_output).resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    maximum_states = max(4, min(256, int(args.maximum_states)))
    rows = [score_song(song, maximum_states=maximum_states) for song in audit["songs"]]
    weights = {
        str(song.get("id")): float(song.get("weight", 1.0))
        for song in audit["songs"]
    }
    report = {
        "schema": "polymath-pitch-class-state-lattice-report-v1",
        "evidenceBoundary": (
            "States use candidate/source/neighbor evidence and fixed interval templates only. "
            "References choose the retrospective oracle state but never generate a state."
        ),
        "settings": {
            "maximumStates": maximum_states,
            "onsetWindowSeconds": max(0.005, float(args.onset_window_seconds)),
        },
        "aggregate": aggregate(rows, weights),
        "songs": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
