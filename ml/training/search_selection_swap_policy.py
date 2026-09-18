"""Search a conservative, quota-preserving local note-swap policy.

This is the rendered validation stage for ``fit_selection_swap_ranker.py``.
Every trial replaces only the conditional guitar selector, freezes the base
onset slot count, runs the complete piano arranger, and compares the result
with the frozen baseline through each song's approved alignment ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
TRAINING_ROOT = REPO_ROOT / "ml" / "training"
for path in (SERVER_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piano_arranger import arrange_payload  # noqa: E402
from apply_conditional_selection_profile import apply_conditional_selector  # noqa: E402
from evaluate_piano_arranger import (  # noqa: E402
    evaluate,
    map_reference_notes,
    monotonic_anchors,
    normalize_notes,
    notes_inside_ranges,
    reference_transpose_semitones,
    trusted_source_ranges,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def numeric_grid(value: str, *, integer: bool = False) -> list[float | int]:
    parsed: list[float | int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        number = float(token)
        if not math.isfinite(number):
            raise argparse.ArgumentTypeError("Grid values must be finite")
        parsed.append(int(round(number)) if integer else number)
    if not parsed:
        raise argparse.ArgumentTypeError("Grid needs at least one value")
    return list(dict.fromkeys(parsed))


def metric(metrics: dict[str, Any], family: str, tolerance: int, statistic: str) -> float:
    return float(metrics[f"{family}Onset{tolerance}ms"][statistic])


def average(rows: Iterable[tuple[float, float]]) -> float:
    values = list(rows)
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / max(1e-12, total)


def evaluate_trial(
    songs: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in songs:
        song_id = str(row["id"])
        weight = float(row.get("weight", 1.0))
        source = load_json(Path(row["source"]).resolve())
        candidate_payload = arrange_payload(source, "full", style_profile=profile)
        outputs[song_id] = candidate_payload
        alignment = load_json(Path(row["alignment"]).resolve())
        ranges = trusted_source_ranges(alignment)
        anchors = monotonic_anchors(alignment)
        target = normalize_notes(
            load_json(Path(row["reference"]).resolve()),
            transpose_semitones=reference_transpose_semitones(row),
        )
        target = notes_inside_ranges(map_reference_notes(target, anchors), ranges)
        baseline = notes_inside_ranges(
            normalize_notes(load_json(Path(row["baseline"]).resolve())), ranges
        )
        candidate = notes_inside_ranges(normalize_notes(candidate_payload), ranges)
        baseline_metrics = evaluate(target, baseline)
        candidate_metrics = evaluate(target, candidate)
        deltas = {
            "exactF1_100ms": round(
                metric(candidate_metrics, "exactPitch", 100, "f1")
                - metric(baseline_metrics, "exactPitch", 100, "f1"),
                6,
            ),
            "exactRecall_100ms": round(
                metric(candidate_metrics, "exactPitch", 100, "recall")
                - metric(baseline_metrics, "exactPitch", 100, "recall"),
                6,
            ),
            "pitchClassF1_250ms": round(
                metric(candidate_metrics, "pitchClass", 250, "f1")
                - metric(baseline_metrics, "pitchClass", 250, "f1"),
                6,
            ),
            "pitchClassRecall_250ms": round(
                metric(candidate_metrics, "pitchClass", 250, "recall")
                - metric(baseline_metrics, "pitchClass", 250, "recall"),
                6,
            ),
            "durationMedianAbsoluteErrorSeconds": round(
                float(candidate_metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0)
                - float(baseline_metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0),
                6,
            ),
            "severeCutoffRate": round(
                float(candidate_metrics["duration"]["severeCutoffRate"])
                - float(baseline_metrics["duration"]["severeCutoffRate"]),
                6,
            ),
            "rapidRetriggersUnder100ms": int(
                candidate_metrics["rapidRetriggersUnder100ms"]
                - baseline_metrics["rapidRetriggersUnder100ms"]
            ),
            "outputNotes": len(candidate) - len(baseline),
        }
        reports.append(
            {
                "id": song_id,
                "weight": weight,
                "referenceNotes": len(target),
                "baselineNotes": len(baseline),
                "candidateNotes": len(candidate),
                "deltas": deltas,
                "conditionalSelection": (
                    candidate_payload.get("pianoArrangement", {}).get(
                        "conditionalSelectionBlend"
                    )
                ),
            }
        )
        for name, value in deltas.items():
            aggregate[name].append((float(value), weight))

    aggregate_deltas = {
        name: round(average(values), 6) for name, values in aggregate.items()
    }
    minimum_pc_f1 = min(
        float(report["deltas"]["pitchClassF1_250ms"]) for report in reports
    )
    minimum_exact_f1 = min(
        float(report["deltas"]["exactF1_100ms"]) for report in reports
    )
    maximum_cutoff_regression = max(
        float(report["deltas"]["severeCutoffRate"]) for report in reports
    )
    gates = {
        "aggregateExactF1DoesNotRegress": aggregate_deltas["exactF1_100ms"] >= -0.0005,
        "aggregateExactRecallDoesNotRegress": aggregate_deltas["exactRecall_100ms"] >= -0.0005,
        "aggregatePitchClassF1Improves": aggregate_deltas["pitchClassF1_250ms"] > 0.0,
        "aggregatePitchClassRecallDoesNotRegress": aggregate_deltas["pitchClassRecall_250ms"] >= -0.0005,
        "noSongPitchClassF1RegressesOver003": minimum_pc_f1 >= -0.003,
        "noSongExactF1RegressesOver005": minimum_exact_f1 >= -0.005,
        "noSongSevereCutoffRateRegressesOver001": maximum_cutoff_regression <= 0.001,
    }
    score = (
        5.0 * aggregate_deltas["exactF1_100ms"]
        + 3.0 * aggregate_deltas["exactRecall_100ms"]
        + 4.0 * aggregate_deltas["pitchClassF1_250ms"]
        + 2.0 * aggregate_deltas["pitchClassRecall_250ms"]
        + 0.5 * minimum_pc_f1
        + 0.25 * minimum_exact_f1
        - 0.20 * max(0.0, aggregate_deltas["durationMedianAbsoluteErrorSeconds"])
        - 0.20 * max(0.0, aggregate_deltas["severeCutoffRate"])
        - 0.0001 * max(0.0, aggregate_deltas["rapidRetriggersUnder100ms"])
    )
    return (
        {
            "aggregateDeltas": aggregate_deltas,
            "minimumSongPitchClassF1Delta250ms": round(minimum_pc_f1, 6),
            "minimumSongExactF1Delta100ms": round(minimum_exact_f1, 6),
            "maximumSongSevereCutoffRateRegression": round(maximum_cutoff_regression, 6),
            "gates": gates,
            "passedAllGates": all(gates.values()),
            "selectionScore": round(score, 8),
            "songs": reports,
        },
        outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--alternative-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-midi-grid", default="48,52,55,57,60")
    parser.add_argument("--maximum-midi-grid", default="76")
    parser.add_argument("--alternative-share-grid", default="0.1,0.2,0.3,0.4,0.5,0.65,0.8,1.0")
    parser.add_argument("--onset-radius-grid", default="0.025,0.035,0.05")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    base_path = Path(args.base_profile).resolve()
    alternative_path = Path(args.alternative_profile).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    songs = list(manifest.get("songs") or [])
    base = load_json(base_path)
    alternative = load_json(alternative_path)
    minimum_grid = numeric_grid(args.minimum_midi_grid, integer=True)
    maximum_grid = numeric_grid(args.maximum_midi_grid, integer=True)
    share_grid = numeric_grid(args.alternative_share_grid)
    onset_grid = numeric_grid(args.onset_radius_grid)
    trials: list[dict[str, Any]] = []
    outputs_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for minimum in minimum_grid:
        for maximum in maximum_grid:
            if int(maximum) < int(minimum):
                continue
            for share in share_grid:
                for onset_radius in onset_grid:
                    trial_id = f"m{int(minimum):03d}-x{int(maximum):03d}-s{int(round(float(share)*100)):03d}-r{int(round(float(onset_radius)*1000)):03d}"
                    profile = apply_conditional_selector(
                        base,
                        alternative,
                        profile_id=f"pianella-local-swap-{trial_id}",
                        alternative_profile_path=str(alternative_path),
                        alternative_share=float(share),
                        source_families=["guitar"],
                        minimum_source_midi=int(minimum),
                        maximum_source_midi=int(maximum),
                        preserve_base_onset_counts=True,
                        onset_slot_window_seconds=float(onset_radius),
                        validation_report_path=None,
                    )
                    result, outputs = evaluate_trial(songs, profile)
                    trials.append(
                        {
                            "id": trial_id,
                            "minimumSourceMidi": int(minimum),
                            "maximumSourceMidi": int(maximum),
                            "alternativeShare": float(share),
                            "onsetSlotWindowSeconds": float(onset_radius),
                            **result,
                        }
                    )
                    outputs_by_id[trial_id] = outputs
    trials.sort(
        key=lambda row: (
            not bool(row["passedAllGates"]),
            -float(row["selectionScore"]),
            -float(row["aggregateDeltas"]["pitchClassF1_250ms"]),
            float(row["alternativeShare"]),
        )
    )
    best = trials[0]
    best_profile = apply_conditional_selector(
        base,
        alternative,
        profile_id=f"pianella-local-swap-{best['id']}",
        alternative_profile_path=str(alternative_path),
        alternative_share=float(best["alternativeShare"]),
        source_families=["guitar"],
        minimum_source_midi=int(best["minimumSourceMidi"]),
        maximum_source_midi=int(best["maximumSourceMidi"]),
        preserve_base_onset_counts=True,
        onset_slot_window_seconds=float(best["onsetSlotWindowSeconds"]),
        validation_report_path=str(output_dir / "SEARCH-REPORT.json"),
    )
    best_profile_path = output_dir / "best-profile.json"
    best_profile_path.write_text(json.dumps(best_profile, indent=2) + "\n", encoding="utf-8")
    best_output_dir = output_dir / "best-outputs"
    best_output_dir.mkdir(parents=True, exist_ok=True)
    for song_id, payload in outputs_by_id[best["id"]].items():
        (best_output_dir / f"{song_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema": "polymath-selection-swap-search-v1",
        "evidenceBoundary": (
            "Private aligned references only. All trials preserve base onset slots. "
            "A passing metric search is still experimental until listening review."
        ),
        "manifest": str(manifest_path),
        "baseProfile": str(base_path),
        "alternativeProfile": str(alternative_path),
        "trials": len(trials),
        "passingTrials": sum(bool(row["passedAllGates"]) for row in trials),
        "best": best,
        "topTrials": trials[:30],
        "decision": "LISTENING_REQUIRED" if best["passedAllGates"] else "REJECTED",
    }
    report_path = output_dir / "SEARCH-REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "bestProfile": str(best_profile_path),
                "bestOutputs": str(best_output_dir),
                "trials": len(trials),
                "passingTrials": report["passingTrials"],
                "best": best,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
