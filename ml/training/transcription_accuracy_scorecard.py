"""Build an honest, route-aware transcription accuracy scorecard.

The direct-piano and full-mix piano-reduction routes solve different problems.
Combining them into one percentage hides the actual bottleneck, while repeatedly
opening the same target turns a holdout into development data.  This evaluator
therefore reports every route and evidence split separately, computes micro-F1
from event counts, and refuses to certify the 90% goal without a sealed holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .evaluate_piano_arranger import (
    greedy_matches,
    load_json,
    prepare_reference_notes,
)
from .search_default_piano_pipeline import clip_candidate, measure


SEALED_EVIDENCE_ROLES = {"sealed-holdout", "untouched-holdout"}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(float(ordered[position]), 6)


def onset_error_summary(
    reference: list[dict[str, float | int]],
    observed: list[dict[str, float | int]],
    *,
    tolerance_seconds: float,
    octave_equivalent: bool,
) -> dict[str, float | int | None]:
    matches = greedy_matches(
        reference,
        observed,
        tolerance_seconds,
        octave_equivalent=octave_equivalent,
    )
    errors = [
        abs(float(candidate["time"]) - float(target["time"]))
        for target, candidate in matches
    ]
    return {
        "matches": len(matches),
        "medianAbsoluteErrorSeconds": round(median(errors), 6) if errors else None,
        "p95AbsoluteErrorSeconds": percentile(errors, 0.95),
    }


def f1_from_counts(reference: int, observed: int, matches: int) -> dict[str, float | int]:
    precision = matches / max(1, observed)
    recall = matches / max(1, reference)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "matches": matches,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def bottleneck_breakdown(note_metrics: dict[str, Any]) -> dict[str, Any]:
    exact_100 = float(note_metrics["exactPitchOnset100ms"]["f1"])
    pitch_class_100 = float(note_metrics["pitchClassOnset100ms"]["f1"])
    pitch_class_250 = float(note_metrics["pitchClassOnset250ms"]["f1"])
    components = {
        "registerPlacementGap": max(0.0, pitch_class_100 - exact_100),
        "coarseTimingGap": max(0.0, pitch_class_250 - pitch_class_100),
        "missingOrExtraEventGap": max(0.0, 1.0 - pitch_class_250),
    }
    rounded = {key: round(value, 6) for key, value in components.items()}
    dominant = max(components, key=components.get)
    return {
        **rounded,
        "dominant": dominant,
        "explanation": (
            "The three non-negative gaps decompose the remaining distance from "
            "exact pitch+onset F1@100ms to 1.0. They are diagnostics, not an "
            "assumption that every error can be independently recovered."
        ),
    }


def evaluate_song(row: dict[str, Any]) -> dict[str, Any]:
    song_id = str(row.get("id") or "").strip()
    route = str(row.get("route") or "").strip()
    evidence_role = str(row.get("evidenceRole") or "").strip()
    if not song_id or not route or not evidence_role:
        raise ValueError("Every song requires id, route, and evidenceRole")

    reference_path = Path(str(row["reference"])).resolve()
    alignment_path = Path(str(row["alignment"])).resolve()
    candidate_path = Path(str(row["candidate"])).resolve()
    source_path = Path(str(row["source"])).resolve() if row.get("source") else None
    for path in (reference_path, alignment_path, candidate_path, source_path):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)

    alignment = load_json(alignment_path)
    reference = prepare_reference_notes(row, load_json(reference_path), alignment)
    observed = clip_candidate(load_json(candidate_path), row, alignment)
    metrics = measure(reference, observed)
    notes = metrics["notes"]
    physical = notes["physicalDuration"]
    velocity = notes["velocity"]
    source_notes = None
    source_coverage = None
    if source_path is not None:
        source_observed = clip_candidate(load_json(source_path), row, alignment)
        source_notes = measure(reference, source_observed)["notes"]
        source_coverage = {
            "observedNotes": source_notes["observedNotes"],
            "noteCountRatio": source_notes["noteCountRatio"],
            "exactPitchOnset100msRecall": source_notes[
                "exactPitchOnset100ms"
            ]["recall"],
            "exactPitchOnset250msRecall": source_notes[
                "exactPitchOnset250ms"
            ]["recall"],
            "pitchClassOnset100msRecall": source_notes[
                "pitchClassOnset100ms"
            ]["recall"],
            "pitchClassOnset250msRecall": source_notes[
                "pitchClassOnset250ms"
            ]["recall"],
            "upstreamPitchClassMissingAt100ms": round(
                max(0.0, 1.0 - float(source_notes["pitchClassOnset100ms"]["recall"])),
                6,
            ),
            "arrangerPitchClassRecallLossAt100ms": round(
                max(
                    0.0,
                    float(source_notes["pitchClassOnset100ms"]["recall"])
                    - float(notes["pitchClassOnset100ms"]["recall"]),
                ),
                6,
            ),
            "foundationHasEnoughPitchClassEvidenceFor90At250ms": (
                float(source_notes["pitchClassOnset250ms"]["recall"]) >= 0.9
            ),
        }

    return {
        "id": song_id,
        "route": route,
        "evidenceRole": evidence_role,
        "claimEligible": evidence_role in SEALED_EVIDENCE_ROLES,
        "referenceNotes": len(reference),
        "observedNotes": len(observed),
        "artifacts": {
            "reference": str(reference_path),
            "referenceSha256": sha256_file(reference_path),
            "alignment": str(alignment_path),
            "alignmentSha256": sha256_file(alignment_path),
            "candidate": str(candidate_path),
            "candidateSha256": sha256_file(candidate_path),
            **(
                {"source": str(source_path), "sourceSha256": sha256_file(source_path)}
                if source_path is not None
                else {}
            ),
        },
        "accuracy": {
            "exactPitchOnset50ms": notes["exactPitchOnset50ms"],
            "exactPitchOnset100ms": notes["exactPitchOnset100ms"],
            "exactPitchOnset250ms": notes["exactPitchOnset250ms"],
            "pitchClassOnset100ms": notes["pitchClassOnset100ms"],
            "pitchClassOnset250ms": notes["pitchClassOnset250ms"],
            "physicalOnsetAndOffset250ms": physical["onsetAndOffset250ms"],
        },
        "timing": {
            "exactPitchOnset100ms": onset_error_summary(
                reference,
                observed,
                tolerance_seconds=0.1,
                octave_equivalent=False,
            ),
            "pitchClassOnset250ms": onset_error_summary(
                reference,
                observed,
                tolerance_seconds=0.25,
                octave_equivalent=True,
            ),
        },
        "performance": {
            "physicalDurationMedianAbsoluteErrorSeconds": physical[
                "medianAbsoluteErrorSeconds"
            ],
            "physicalDurationMedianRelativeError": physical["medianRelativeError"],
            "physicalSevereCutoffs": physical["severeCutoffs"],
            "physicalSevereCutoffRate": physical["severeCutoffRate"],
            "velocityMeanAbsoluteError": velocity["meanAbsoluteError"],
            "velocityMeanBias": velocity["meanBias"],
            "rapidRetriggersUnder100ms": notes["rapidRetriggersUnder100ms"],
            "noteCountRatio": notes["noteCountRatio"],
            "chordSizeDistance": metrics["chordSizeDistance"],
            "handOccupancyDistance": metrics["handOccupancyDistance"],
        },
        "bottleneck": bottleneck_breakdown(notes),
        "sourceCoverage": source_coverage,
        "_aggregation": {
            "physicalMatchedNotes": physical["matchedNotes"],
            "physicalOffsetMatches": physical["onsetAndOffset250ms"]["matches"],
            "velocityMatchedNotes": velocity["matchedNotes"],
            "sourceObservedNotes": (
                int(source_notes["observedNotes"]) if source_notes is not None else None
            ),
            "sourceExact100Matches": (
                int(source_notes["exactPitchOnset100ms"]["matches"])
                if source_notes is not None
                else None
            ),
            "sourceExact250Matches": (
                int(source_notes["exactPitchOnset250ms"]["matches"])
                if source_notes is not None
                else None
            ),
            "sourcePitchClass100Matches": (
                int(source_notes["pitchClassOnset100ms"]["matches"])
                if source_notes is not None
                else None
            ),
            "sourcePitchClass250Matches": (
                int(source_notes["pitchClassOnset250ms"]["matches"])
                if source_notes is not None
                else None
            ),
        },
    }


def weighted_value(rows: list[dict[str, Any]], getter: Any, weight_getter: Any) -> float | None:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = getter(row)
        weight = float(weight_getter(row))
        if value is not None and weight > 0:
            values.append((float(value), weight))
    if not values:
        return None
    return round(sum(value * weight for value, weight in values) / sum(weight for _, weight in values), 6)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = sum(int(row["referenceNotes"]) for row in rows)
    observed = sum(int(row["observedNotes"]) for row in rows)

    def aggregate_metric(name: str) -> dict[str, float | int]:
        matches = sum(int(row["accuracy"][name]["matches"]) for row in rows)
        return f1_from_counts(reference, observed, matches)

    physical_matches = sum(
        int(row["_aggregation"]["physicalMatchedNotes"]) for row in rows
    )
    physical_cutoffs = sum(
        int(row["performance"]["physicalSevereCutoffs"]) for row in rows
    )
    physical_offset_matches = sum(
        int(row["_aggregation"]["physicalOffsetMatches"]) for row in rows
    )
    rapid_retriggers = sum(
        int(row["performance"]["rapidRetriggersUnder100ms"]) for row in rows
    )
    exact_100 = aggregate_metric("exactPitchOnset100ms")
    pitch_class_100 = aggregate_metric("pitchClassOnset100ms")
    pitch_class_250 = aggregate_metric("pitchClassOnset250ms")
    bottleneck = bottleneck_breakdown(
        {
            "exactPitchOnset100ms": exact_100,
            "pitchClassOnset100ms": pitch_class_100,
            "pitchClassOnset250ms": pitch_class_250,
        }
    )
    rows_with_source = [
        row
        for row in rows
        if row["_aggregation"].get("sourceObservedNotes") is not None
    ]
    source_coverage = None
    if rows_with_source:
        source_reference = sum(int(row["referenceNotes"]) for row in rows_with_source)
        source_observed = sum(
            int(row["_aggregation"]["sourceObservedNotes"])
            for row in rows_with_source
        )

        def source_recall(key: str) -> float:
            matches = sum(int(row["_aggregation"][key]) for row in rows_with_source)
            return round(matches / max(1, source_reference), 6)

        source_pc_100 = source_recall("sourcePitchClass100Matches")
        candidate_pc_100_matches = sum(
            int(row["accuracy"]["pitchClassOnset100ms"]["matches"])
            for row in rows_with_source
        )
        candidate_pc_100_recall = candidate_pc_100_matches / max(1, source_reference)
        source_pc_250 = source_recall("sourcePitchClass250Matches")
        source_coverage = {
            "songs": len(rows_with_source),
            "referenceNotes": source_reference,
            "observedNotes": source_observed,
            "noteCountRatio": round(source_observed / max(1, source_reference), 6),
            "exactPitchOnset100msRecall": source_recall("sourceExact100Matches"),
            "exactPitchOnset250msRecall": source_recall("sourceExact250Matches"),
            "pitchClassOnset100msRecall": source_pc_100,
            "pitchClassOnset250msRecall": source_pc_250,
            "upstreamPitchClassMissingAt100ms": round(max(0.0, 1.0 - source_pc_100), 6),
            "arrangerPitchClassRecallLossAt100ms": round(
                max(0.0, source_pc_100 - candidate_pc_100_recall), 6
            ),
            "foundationHasEnoughPitchClassEvidenceFor90At250ms": source_pc_250 >= 0.9,
        }
    return {
        "songs": len(rows),
        "referenceNotes": reference,
        "observedNotes": observed,
        "noteCountRatio": round(observed / max(1, reference), 6),
        "accuracy": {
            "exactPitchOnset50ms": aggregate_metric("exactPitchOnset50ms"),
            "exactPitchOnset100ms": exact_100,
            "exactPitchOnset250ms": aggregate_metric("exactPitchOnset250ms"),
            "pitchClassOnset100ms": pitch_class_100,
            "pitchClassOnset250ms": pitch_class_250,
            "physicalOnsetAndOffset250ms": f1_from_counts(
                reference, observed, physical_offset_matches
            ),
        },
        "performance": {
            "physicalDurationWeightedMedianAbsoluteErrorSeconds": weighted_value(
                rows,
                lambda row: row["performance"][
                    "physicalDurationMedianAbsoluteErrorSeconds"
                ],
                lambda row: row["_aggregation"]["physicalMatchedNotes"],
            ),
            "physicalDurationWeightedMedianRelativeError": weighted_value(
                rows,
                lambda row: row["performance"][
                    "physicalDurationMedianRelativeError"
                ],
                lambda row: row["_aggregation"]["physicalMatchedNotes"],
            ),
            "physicalSevereCutoffs": physical_cutoffs,
            "physicalSevereCutoffRate": round(
                physical_cutoffs / max(1, physical_matches), 6
            ),
            "velocityMeanAbsoluteError": weighted_value(
                rows,
                lambda row: row["performance"]["velocityMeanAbsoluteError"],
                lambda row: row["_aggregation"]["velocityMatchedNotes"],
            ),
            "rapidRetriggersUnder100ms": rapid_retriggers,
            "rapidRetriggersPer1000Notes": round(
                1000 * rapid_retriggers / max(1, observed), 6
            ),
        },
        "bottleneck": bottleneck,
        "sourceCoverage": source_coverage,
    }


def objective_result(
    route: str,
    rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    all_route_rows = [row for row in rows if row["route"] == route]
    sealed_rows = [row for row in all_route_rows if row["claimEligible"]]
    development = aggregate_rows(all_route_rows) if all_route_rows else None
    sealed = aggregate_rows(sealed_rows) if sealed_rows else None
    evaluation = sealed or development
    if evaluation is None:
        return {
            "status": "NO_EVIDENCE",
            "claimEligible": False,
            "target": target,
        }

    gates = {
        "exactPitchOnset100msF1": (
            float(evaluation["accuracy"]["exactPitchOnset100ms"]["f1"])
            >= float(target["exactPitchOnset100msF1"])
        ),
        "pitchClassOnset250msF1": (
            float(evaluation["accuracy"]["pitchClassOnset250ms"]["f1"])
            >= float(target["pitchClassOnset250msF1"])
        ),
        "physicalOnsetAndOffset250msF1": (
            float(evaluation["accuracy"]["physicalOnsetAndOffset250ms"]["f1"])
            >= float(target["physicalOnsetAndOffset250msF1"])
        ),
        "physicalSevereCutoffRate": (
            float(evaluation["performance"]["physicalSevereCutoffRate"])
            <= float(target["maximumPhysicalSevereCutoffRate"])
        ),
        "rapidRetriggersPer1000Notes": (
            float(evaluation["performance"]["rapidRetriggersPer1000Notes"])
            <= float(target["maximumRapidRetriggersPer1000Notes"])
        ),
    }
    claim_eligible = bool(sealed_rows)
    return {
        "status": (
            "TARGET_MET_ON_SEALED_HOLDOUT"
            if claim_eligible and all(gates.values())
            else "SEALED_HOLDOUT_BELOW_TARGET"
            if claim_eligible
            else "DEVELOPMENT_ONLY_NEEDS_NEW_SEALED_HOLDOUT"
        ),
        "claimEligible": claim_eligible,
        "target": target,
        "gates": gates,
        "measuredOn": "sealed-holdout" if sealed_rows else "development-regression",
        "current": evaluation,
        "gapToHeadline90": round(
            float(target["exactPitchOnset100msF1"])
            - float(evaluation["accuracy"]["exactPitchOnset100ms"]["f1"]),
            6,
        ),
    }


def build_scorecard(manifest: dict[str, Any]) -> dict[str, Any]:
    raw_songs = manifest.get("songs")
    if not isinstance(raw_songs, list) or not raw_songs:
        raise ValueError("manifest must contain a non-empty songs list")
    ids = [str(row.get("id") or "") for row in raw_songs]
    if len(ids) != len(set(ids)):
        raise ValueError("song ids must be unique")
    rows = [evaluate_song(row) for row in raw_songs]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["route"], row["evidenceRole"])].append(row)
    aggregates = {
        f"{route}/{role}": aggregate_rows(group_rows)
        for (route, role), group_rows in sorted(grouped.items())
    }
    objectives = {
        route: objective_result(route, rows, target)
        for route, target in (manifest.get("objectives") or {}).items()
    }
    public_rows = []
    for row in rows:
        clean = dict(row)
        clean.pop("_aggregation", None)
        public_rows.append(clean)
    return {
        "schema": "polymath-transcription-accuracy-scorecard-v1",
        "id": manifest.get("id"),
        "purpose": manifest.get("purpose"),
        "accuracyDefinition": {
            "headline": "micro exact-pitch + onset F1 at +/-100 ms",
            "whyMicro": "Every reference and predicted event contributes to the total; short easy songs cannot dominate the percentage.",
            "claimRule": "90% may be claimed only when every gate passes on a pre-registered sealed holdout that was not used for fitting or model selection.",
            "fullMixCaveat": "Matching a chosen solo-piano arrangement is a conditional arrangement task, not literal source transcription.",
        },
        "songs": public_rows,
        "aggregates": aggregates,
        "objectives": objectives,
        "overallClaim": (
            "NINETY_PERCENT_VERIFIED"
            if objectives and all(
                value["status"] == "TARGET_MET_ON_SEALED_HOLDOUT"
                for value in objectives.values()
            )
            else "NOT_YET_VERIFIED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_scorecard(load_json(args.manifest.resolve()))
    atomic_json(args.output.resolve(), result)
    concise = {
        route: {
            "status": value["status"],
            "exactPitchOnset100msF1": value.get("current", {})
            .get("accuracy", {})
            .get("exactPitchOnset100ms", {})
            .get("f1"),
            "gapToHeadline90": value.get("gapToHeadline90"),
            "dominantBottleneck": value.get("current", {})
            .get("bottleneck", {})
            .get("dominant"),
        }
        for route, value in result["objectives"].items()
    }
    print(json.dumps({"output": str(args.output.resolve()), "routes": concise}, indent=2))


if __name__ == "__main__":
    main()
