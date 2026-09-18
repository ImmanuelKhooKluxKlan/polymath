"""Rendered search for the post-hand-split supporting-tone ranker."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
TRAINING_ROOT = REPO_ROOT / "ml" / "training"
for path in (SERVER_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from piano_arranger import arrange_payload  # noqa: E402
from apply_supporting_tone_profile import apply_supporting_tone_ranker  # noqa: E402
from evaluate_piano_arranger import (  # noqa: E402
    evaluate,
    map_reference_notes,
    monotonic_anchors,
    normalize_notes,
    notes_inside_ranges,
    reference_transpose_semitones,
    trusted_source_ranges,
)
from search_selection_swap_policy import average, metric, numeric_grid  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def onset_count(notes: list[dict[str, Any]]) -> int:
    return len({int(round(float(note["time"]) * 1000)) for note in notes})


def prepare_songs(
    rows: list[dict[str, Any]], base_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        source = load_json(Path(row["source"]).resolve())
        alignment = load_json(Path(row["alignment"]).resolve())
        ranges = trusted_source_ranges(alignment)
        target = normalize_notes(
            load_json(Path(row["reference"]).resolve()),
            transpose_semitones=reference_transpose_semitones(row),
        )
        target = notes_inside_ranges(
            map_reference_notes(target, monotonic_anchors(alignment)), ranges
        )
        control_payload = arrange_payload(source, "full", style_profile=base_profile)
        control = notes_inside_ranges(normalize_notes(control_payload), ranges)
        prepared.append(
            {
                "id": str(row["id"]),
                "weight": float(row.get("weight", 1.0)),
                "source": source,
                "ranges": ranges,
                "target": target,
                "controlPayload": control_payload,
                "control": control,
                "controlMetrics": evaluate(target, control),
            }
        )
    return prepared


def evaluate_trial(
    songs: list[dict[str, Any]], profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reports = []
    outputs: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in songs:
        payload = arrange_payload(row["source"], "full", style_profile=profile)
        outputs[row["id"]] = payload
        candidate = notes_inside_ranges(normalize_notes(payload), row["ranges"])
        control = row["control"]
        control_metrics = row["controlMetrics"]
        candidate_metrics = evaluate(row["target"], candidate)
        control_left = row["controlPayload"].get("pianoArrangement", {}).get(
            "leftHandAccompaniment", {}
        )
        candidate_left = payload.get("pianoArrangement", {}).get(
            "leftHandAccompaniment", {}
        )
        deltas = {
            "exactF1_100ms": round(
                metric(candidate_metrics, "exactPitch", 100, "f1")
                - metric(control_metrics, "exactPitch", 100, "f1"), 6
            ),
            "exactRecall_100ms": round(
                metric(candidate_metrics, "exactPitch", 100, "recall")
                - metric(control_metrics, "exactPitch", 100, "recall"), 6
            ),
            "pitchClassF1_250ms": round(
                metric(candidate_metrics, "pitchClass", 250, "f1")
                - metric(control_metrics, "pitchClass", 250, "f1"), 6
            ),
            "pitchClassRecall_250ms": round(
                metric(candidate_metrics, "pitchClass", 250, "recall")
                - metric(control_metrics, "pitchClass", 250, "recall"), 6
            ),
            "durationMedianAbsoluteErrorSeconds": round(
                float(candidate_metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0)
                - float(control_metrics["duration"]["medianAbsoluteErrorSeconds"] or 0.0), 6
            ),
            "severeCutoffRate": round(
                float(candidate_metrics["duration"]["severeCutoffRate"])
                - float(control_metrics["duration"]["severeCutoffRate"]), 6
            ),
            "rapidRetriggersUnder100ms": int(
                candidate_metrics["rapidRetriggersUnder100ms"]
                - control_metrics["rapidRetriggersUnder100ms"]
            ),
            "outputNotes": len(candidate) - len(control),
            "outputOnsets": onset_count(candidate) - onset_count(control),
            "rightHandNotes": int(candidate_left.get("outputRightHandNotes", 0))
            - int(control_left.get("outputRightHandNotes", 0)),
            "leftHandNotes": int(candidate_left.get("outputLeftHandNotes", 0))
            - int(control_left.get("outputLeftHandNotes", 0)),
        }
        reports.append(
            {
                "id": row["id"],
                "weight": row["weight"],
                "referenceNotes": len(row["target"]),
                "controlNotes": len(control),
                "candidateNotes": len(candidate),
                "deltas": deltas,
                "supportingTone": {
                    key: candidate_left.get(key)
                    for key in (
                        "supportingToneSelectionModelApplied",
                        "supportingToneAlternativeShare",
                        "supportingToneChordCompletionShare",
                        "supportingToneSourceFamilies",
                        "supportingToneMinimumSourceMidi",
                        "supportingToneMaximumSourceMidi",
                        "chordCompletionChangedNotes",
                    )
                },
            }
        )
        for name, value in deltas.items():
            aggregate[name].append((float(value), row["weight"]))
    aggregate_deltas = {
        name: round(average(values), 6) for name, values in aggregate.items()
    }
    minima = {
        name: min(float(report["deltas"][name]) for report in reports)
        for name in (
            "exactF1_100ms",
            "exactRecall_100ms",
            "pitchClassF1_250ms",
            "pitchClassRecall_250ms",
        )
    }
    maximum_cutoff = max(
        float(report["deltas"]["severeCutoffRate"]) for report in reports
    )
    exact_structure = all(
        int(report["deltas"][name]) == 0
        for report in reports
        for name in ("outputNotes", "outputOnsets", "rightHandNotes", "leftHandNotes")
    )
    gates = {
        "exactHandAndDensityCountsPreserved": exact_structure,
        "aggregateExactF1DoesNotRegress": aggregate_deltas["exactF1_100ms"] >= -0.00025,
        "aggregateExactRecallDoesNotRegress": aggregate_deltas["exactRecall_100ms"] >= -0.00025,
        "aggregatePitchClassF1Improves": aggregate_deltas["pitchClassF1_250ms"] > 0.0,
        "aggregatePitchClassRecallDoesNotRegress": aggregate_deltas["pitchClassRecall_250ms"] >= -0.00025,
        "noSongExactF1RegressesOver002": minima["exactF1_100ms"] >= -0.002,
        "noSongPitchClassF1RegressesOver002": minima["pitchClassF1_250ms"] >= -0.002,
        "noSongPitchClassRecallRegressesOver002": minima["pitchClassRecall_250ms"] >= -0.002,
        "noSongSevereCutoffRateRegressesOver001": maximum_cutoff <= 0.001,
    }
    score = (
        5.0 * aggregate_deltas["exactF1_100ms"]
        + 3.0 * aggregate_deltas["exactRecall_100ms"]
        + 5.0 * aggregate_deltas["pitchClassF1_250ms"]
        + 3.0 * aggregate_deltas["pitchClassRecall_250ms"]
        + 0.5 * minima["pitchClassF1_250ms"]
        + 0.25 * minima["exactF1_100ms"]
        - 0.20 * max(0.0, aggregate_deltas["durationMedianAbsoluteErrorSeconds"])
        - 0.20 * max(0.0, aggregate_deltas["severeCutoffRate"])
        - (0.1 if not exact_structure else 0.0)
    )
    return (
        {
            "aggregateDeltas": aggregate_deltas,
            "minimumSongDeltas": {key: round(value, 6) for key, value in minima.items()},
            "maximumSongSevereCutoffRateRegression": round(maximum_cutoff, 6),
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
    parser.add_argument("--minimum-midi-grid", default="48,55,60")
    parser.add_argument("--maximum-midi-grid", default="76")
    parser.add_argument("--alternative-share-grid", default="0,0.5,1")
    parser.add_argument("--chord-completion-share-grid", default="0,0.5,1")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    base_path = Path(args.base_profile).resolve()
    alternative_path = Path(args.alternative_profile).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    base = load_json(base_path)
    alternative = load_json(alternative_path)
    songs = prepare_songs(list(manifest.get("songs") or []), base)
    trials = []
    outputs_by_id = {}
    for minimum in numeric_grid(args.minimum_midi_grid, integer=True):
        for maximum in numeric_grid(args.maximum_midi_grid, integer=True):
            if int(maximum) < int(minimum):
                continue
            for share in numeric_grid(args.alternative_share_grid):
                for chord_share in numeric_grid(args.chord_completion_share_grid):
                    trial_id = (
                        f"m{int(minimum):03d}-x{int(maximum):03d}"
                        f"-s{int(round(float(share)*100)):03d}"
                        f"-c{int(round(float(chord_share)*100)):03d}"
                    )
                    profile = apply_supporting_tone_ranker(
                        base,
                        alternative,
                        profile_id=f"pianella-supporting-tone-{trial_id}",
                        alternative_path=str(alternative_path),
                        alternative_share=float(share),
                        chord_completion_share=float(chord_share),
                        source_families=["guitar"],
                        minimum_source_midi=int(minimum),
                        maximum_source_midi=int(maximum),
                    )
                    result, outputs = evaluate_trial(songs, profile)
                    trials.append(
                        {
                            "id": trial_id,
                            "minimumSourceMidi": int(minimum),
                            "maximumSourceMidi": int(maximum),
                            "alternativeShare": float(share),
                            "chordCompletionShare": float(chord_share),
                            **result,
                        }
                    )
                    outputs_by_id[trial_id] = outputs
    trials.sort(
        key=lambda row: (
            not bool(row["passedAllGates"]),
            -float(row["selectionScore"]),
            -float(row["aggregateDeltas"]["pitchClassF1_250ms"]),
            float(row["alternativeShare"]) + float(row["chordCompletionShare"]),
        )
    )
    best = trials[0]
    report_path = output_dir / "SEARCH-REPORT.json"
    best_profile = apply_supporting_tone_ranker(
        base,
        alternative,
        profile_id=f"pianella-supporting-tone-{best['id']}",
        alternative_path=str(alternative_path),
        alternative_share=float(best["alternativeShare"]),
        chord_completion_share=float(best["chordCompletionShare"]),
        source_families=["guitar"],
        minimum_source_midi=int(best["minimumSourceMidi"]),
        maximum_source_midi=int(best["maximumSourceMidi"]),
        validation_report=str(report_path),
    )
    (output_dir / "best-profile.json").write_text(
        json.dumps(best_profile, indent=2) + "\n", encoding="utf-8"
    )
    best_output_dir = output_dir / "best-outputs"
    best_output_dir.mkdir(parents=True, exist_ok=True)
    for song_id, payload in outputs_by_id[best["id"]].items():
        (best_output_dir / f"{song_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    control_dir = output_dir / "control-outputs"
    control_dir.mkdir(parents=True, exist_ok=True)
    for row in songs:
        (control_dir / f"{row['id']}.json").write_text(
            json.dumps(row["controlPayload"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema": "polymath-supporting-tone-search-v1",
        "evidenceBoundary": (
            "Private aligned references only. Control and candidates use the same "
            "current code. Right-hand, hand counts, onset counts, and note counts "
            "must remain exact. Listening review is still required."
        ),
        "manifest": str(manifest_path),
        "baseProfile": str(base_path),
        "alternativeProfile": str(alternative_path),
        "trials": len(trials),
        "passingTrials": sum(bool(row["passedAllGates"]) for row in trials),
        "best": best,
        "topTrials": trials[:30],
        "allTrials": trials,
        "decision": "LISTENING_REQUIRED" if best["passedAllGates"] else "REJECTED",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "bestProfile": str(output_dir / "best-profile.json"),
                "bestOutputs": str(best_output_dir),
                "controlOutputs": str(control_dir),
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
