"""Select an audio-onset snapping policy with whole-song LOSO evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from .evaluate_piano_arranger import load_json, prepare_reference_notes
from .search_default_piano_pipeline import clip_candidate, compact_metrics, measure
from .snap_piano_onsets_to_audio import atomic_json, onset_envelope, snap_payload


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def micro_f1(rows: list[dict[str, Any]], metric: str) -> float:
    reference = sum(int(row["referenceNotes"]) for row in rows)
    observed = sum(int(row["observedNotes"]) for row in rows)
    matches = sum(int(row[metric]["matches"]) for row in rows)
    precision = matches / max(1, observed)
    recall = matches / max(1, reference)
    return 2 * precision * recall / max(1e-12, precision + recall)


def profile_id(parameters: tuple[float, float, float, float, float]) -> str:
    radius, strength, distance, offset, gain = parameters
    return (
        f"r{radius:.3f}-s{strength:.2f}-d{distance:.2f}-"
        f"o{offset:+.3f}-g{gain:.2f}"
    )


def select_trial(
    trials: dict[str, dict[str, Any]], training_ids: set[str]
) -> dict[str, Any]:
    candidates = []
    for trial in trials.values():
        songs = [trial["songs"][song_id] for song_id in sorted(training_ids)]
        baseline_exact_250 = micro_f1([row["baselineNotes"] for row in songs], "exactPitchOnset250ms")
        candidate_exact_250 = micro_f1([row["candidateNotes"] for row in songs], "exactPitchOnset250ms")
        baseline_pc_250 = micro_f1([row["baselineNotes"] for row in songs], "pitchClassOnset250ms")
        candidate_pc_250 = micro_f1([row["candidateNotes"] for row in songs], "pitchClassOnset250ms")
        baseline_onoff = micro_f1([row["baselineNotes"] for row in songs], "physicalOnsetAndOffset250ms")
        candidate_onoff = micro_f1([row["candidateNotes"] for row in songs], "physicalOnsetAndOffset250ms")
        exact_100 = micro_f1([row["candidateNotes"] for row in songs], "exactPitchOnset100ms")
        exact_50 = micro_f1([row["candidateNotes"] for row in songs], "exactPitchOnset50ms")
        passes = (
            candidate_exact_250 >= baseline_exact_250 - 0.001
            and candidate_pc_250 >= baseline_pc_250 - 0.001
            and candidate_onoff >= baseline_onoff - 0.005
        )
        candidates.append(
            {
                **trial,
                "training": {
                    "songs": sorted(training_ids),
                    "exactPitchOnset100msF1": round(exact_100, 6),
                    "exactPitchOnset50msF1": round(exact_50, 6),
                    "exactPitchOnset250msDelta": round(candidate_exact_250 - baseline_exact_250, 6),
                    "pitchClassOnset250msDelta": round(candidate_pc_250 - baseline_pc_250, 6),
                    "physicalOnsetAndOffset250msDelta": round(candidate_onoff - baseline_onoff, 6),
                    "passesSafetyGates": passes,
                },
            }
        )
    eligible = [trial for trial in candidates if trial["training"]["passesSafetyGates"]]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda trial: (
            trial["training"]["exactPitchOnset100msF1"],
            trial["training"]["exactPitchOnset50msF1"],
            trial["training"]["physicalOnsetAndOffset250msDelta"],
            -float(trial["parameters"]["radiusSeconds"]),
        ),
    )


def metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    notes = metrics["notes"]
    return {
        "referenceNotes": notes["referenceNotes"],
        "observedNotes": notes["observedNotes"],
        "exactPitchOnset50ms": notes["exactPitchOnset50ms"],
        "exactPitchOnset100ms": notes["exactPitchOnset100ms"],
        "exactPitchOnset250ms": notes["exactPitchOnset250ms"],
        "pitchClassOnset250ms": notes["pitchClassOnset250ms"],
        "physicalOnsetAndOffset250ms": notes["physicalDuration"]["onsetAndOffset250ms"],
    }


def delta(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    return round(float(candidate[metric]["f1"]) - float(baseline[metric]["f1"]), 6)


def promotion_decision(aggregate: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    gates = {
        "exactPitchOnset100msImproves": (
            float(aggregate["candidateExactPitchOnset100msF1"])
            >= float(aggregate["baselineExactPitchOnset100msF1"]) + 0.001
        ),
        "exactPitchOnset50msNotWorse": (
            float(aggregate["candidateExactPitchOnset50msF1"])
            >= float(aggregate["baselineExactPitchOnset50msF1"]) - 0.0005
        ),
        "exactPitchOnset250msNotWorse": (
            float(aggregate["candidateExactPitchOnset250msF1"])
            >= float(aggregate["baselineExactPitchOnset250msF1"]) - 0.0005
        ),
        "everyHeldoutSongExact100NotWorse": bool(
            aggregate["allHeldoutExact100NonRegressing"]
        ),
    }
    return (
        "CANDIDATE_FOR_NEW_SEALED_HOLDOUT"
        if all(gates.values())
        else "REJECT_NO_CROSS_SONG_TIMING_GAIN",
        gates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radii", type=parse_grid, default=(0.04, 0.06, 0.08, 0.10, 0.12))
    parser.add_argument("--strengths", type=parse_grid, default=(0.30, 0.45, 0.60, 0.75))
    parser.add_argument("--distance-weights", type=parse_grid, default=(0.15, 0.30, 0.50))
    parser.add_argument("--peak-offsets", type=parse_grid, default=(-0.04, -0.02, 0.0))
    parser.add_argument("--minimum-gains", type=parse_grid, default=(0.0, 0.05))
    args = parser.parse_args()

    manifest = load_json(args.manifest.resolve())
    rows = manifest.get("songs") or []
    if len(rows) < 3:
        raise ValueError("LOSO onset search requires at least three complete songs")
    song_ids = [str(row.get("id") or "") for row in rows]
    if any(not song_id for song_id in song_ids) or len(song_ids) != len(set(song_ids)):
        raise ValueError("song ids must be non-empty and unique")

    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        song_id = str(row["id"])
        alignment = load_json(Path(str(row["alignment"])).resolve())
        reference = prepare_reference_notes(
            row, load_json(Path(str(row["reference"])).resolve()), alignment
        )
        candidate_payload = load_json(Path(str(row["candidate"])).resolve())
        candidate = clip_candidate(candidate_payload, row, alignment)
        times, strengths, duration = onset_envelope(Path(str(row["audio"])).resolve())
        cache[song_id] = {
            "row": row,
            "alignment": alignment,
            "reference": reference,
            "candidatePayload": candidate_payload,
            "baseline": measure(reference, candidate),
            "featureTimes": times,
            "strengths": strengths,
            "audioDuration": duration,
        }

    trials: dict[str, dict[str, Any]] = {}
    candidate_cache: dict[tuple[str, str], dict[str, Any]] = {}
    grid = itertools.product(
        args.radii,
        args.strengths,
        args.distance_weights,
        args.peak_offsets,
        args.minimum_gains,
    )
    for parameters in grid:
        trial_id = profile_id(parameters)
        radius, strength, distance, offset, gain = parameters
        per_song: dict[str, Any] = {}
        for song_id in song_ids:
            item = cache[song_id]
            output, diagnostics = snap_payload(
                item["candidatePayload"],
                item["featureTimes"],
                item["strengths"],
                radius_seconds=radius,
                minimum_peak_strength=strength,
                distance_weight=distance,
                peak_time_offset_seconds=offset,
                minimum_strength_gain=gain,
                audio_duration_seconds=item["audioDuration"],
            )
            candidate = clip_candidate(output, item["row"], item["alignment"])
            metrics = measure(item["reference"], candidate)
            baseline_notes = metric_payload(item["baseline"])
            candidate_notes = metric_payload(metrics)
            per_song[song_id] = {
                "baselineNotes": baseline_notes,
                "candidateNotes": candidate_notes,
                "diagnostics": diagnostics,
            }
            candidate_cache[(trial_id, song_id)] = output
        trials[trial_id] = {
            "id": trial_id,
            "parameters": {
                "radiusSeconds": radius,
                "minimumPeakStrength": strength,
                "distanceWeight": distance,
                "peakTimeOffsetSeconds": offset,
                "minimumStrengthGain": gain,
                "groupWindowSeconds": 0.035,
                "hopSeconds": 0.01,
                "fftSeconds": 0.064,
            },
            "songs": per_song,
        }

    output_root = args.output_dir.resolve()
    folds: list[dict[str, Any]] = []
    for holdout_id in song_ids:
        winner = select_trial(trials, set(song_ids) - {holdout_id})
        held = winner["songs"][holdout_id]
        baseline = held["baselineNotes"]
        candidate = held["candidateNotes"]
        output = candidate_cache[(winner["id"], holdout_id)]
        atomic_json(output_root / "loso-candidates" / f"{holdout_id}.json", output)
        folds.append(
            {
                "holdoutSong": holdout_id,
                "trainingSongs": [song_id for song_id in song_ids if song_id != holdout_id],
                "selectedProfile": winner["id"],
                "parameters": winner["parameters"],
                "training": winner["training"],
                "heldout": {
                    "baseline": baseline,
                    "candidate": candidate,
                    "deltas": {
                        metric: delta(candidate, baseline, metric)
                        for metric in (
                            "exactPitchOnset50ms",
                            "exactPitchOnset100ms",
                            "exactPitchOnset250ms",
                            "pitchClassOnset250ms",
                            "physicalOnsetAndOffset250ms",
                        )
                    },
                    "diagnostics": held["diagnostics"],
                },
            }
        )

    development_winner = select_trial(trials, set(song_ids))
    atomic_json(output_root / "development-profile.json", {
        "schema": "polymath-audio-onset-snap-profile-v1",
        "id": development_winner["id"],
        **development_winner["parameters"],
        "evidenceRole": "opened-development-only",
        "promotionRequires": "one new pre-registered sealed direct-piano holdout",
    })
    for song_id in song_ids:
        atomic_json(
            output_root / "development-candidates" / f"{song_id}.json",
            candidate_cache[(development_winner["id"], song_id)],
        )

    heldout_baselines = [fold["heldout"]["baseline"] for fold in folds]
    heldout_candidates = [fold["heldout"]["candidate"] for fold in folds]
    aggregate = {
        "baselineExactPitchOnset100msF1": round(
            micro_f1(heldout_baselines, "exactPitchOnset100ms"), 6
        ),
        "candidateExactPitchOnset100msF1": round(
            micro_f1(heldout_candidates, "exactPitchOnset100ms"), 6
        ),
        "baselineExactPitchOnset50msF1": round(
            micro_f1(heldout_baselines, "exactPitchOnset50ms"), 6
        ),
        "candidateExactPitchOnset50msF1": round(
            micro_f1(heldout_candidates, "exactPitchOnset50ms"), 6
        ),
        "baselineExactPitchOnset250msF1": round(
            micro_f1(heldout_baselines, "exactPitchOnset250ms"), 6
        ),
        "candidateExactPitchOnset250msF1": round(
            micro_f1(heldout_candidates, "exactPitchOnset250ms"), 6
        ),
        "allHeldoutExact100NonRegressing": all(
            float(fold["heldout"]["deltas"]["exactPitchOnset100ms"]) >= -0.0005
            for fold in folds
        ),
    }
    decision, promotion_gates = promotion_decision(aggregate)
    report = {
        "schema": "polymath-audio-onset-snap-loso-v1",
        "evidenceBoundary": "Each fold's policy is selected without the complete held-out song. All songs were previously opened, so this is development evidence, not a final claim.",
        "songs": song_ids,
        "gridTrials": len(trials),
        "folds": folds,
        "aggregate": aggregate,
        "promotionGates": promotion_gates,
        "developmentProfile": development_winner["parameters"],
        "decision": decision,
    }
    atomic_json(output_root / "report.json", report)
    print(json.dumps({"output": str(output_root / "report.json"), "decision": decision, **report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
