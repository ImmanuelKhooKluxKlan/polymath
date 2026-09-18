"""Evaluate raw-support recovery with one frozen whole-song holdout profile per song."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Package and direct-script execution.
    from .apply_raw_support_recovery import load_json, supplement_score
    from .evaluate_pianist_candidate_pair import evaluate_pair
    from .evaluate_piano_arranger import prepare_reference_notes
    from .search_default_piano_pipeline import clip_candidate, measure
except ImportError:  # pragma: no cover
    from apply_raw_support_recovery import load_json, supplement_score  # type: ignore
    from evaluate_pianist_candidate_pair import evaluate_pair  # type: ignore
    from evaluate_piano_arranger import prepare_reference_notes  # type: ignore
    from search_default_piano_pipeline import clip_candidate, measure  # type: ignore


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def weighted_delta(rows: list[dict[str, Any]], key: str) -> float:
    numerator = sum(
        float(row["evaluation"]["deltas"][key])
        * int(row["evaluation"]["referenceNotes"])
        for row in rows
    )
    denominator = sum(int(row["evaluation"]["referenceNotes"]) for row in rows)
    return round(numerator / max(1, denominator), 6)


def performance_summary(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    notes = metrics["notes"]
    duration = notes["duration"]
    visual = notes["visualDuration"]
    physical = notes["physicalDuration"]
    return {
        "durationMedianAbsoluteErrorSeconds": duration["medianAbsoluteErrorSeconds"],
        "durationSevereCutoffRate": duration["severeCutoffRate"],
        "durationOnsetOffsetF1_250ms": duration["onsetAndOffset250ms"]["f1"],
        "visualDurationMedianAbsoluteErrorSeconds": visual["medianAbsoluteErrorSeconds"],
        "visualSevereCutoffRate": visual["severeCutoffRate"],
        "visualOnsetOffsetF1_250ms": visual["onsetAndOffset250ms"]["f1"],
        "physicalDurationMedianAbsoluteErrorSeconds": physical["medianAbsoluteErrorSeconds"],
        "physicalSevereCutoffRate": physical["severeCutoffRate"],
        "physicalOnsetOffsetF1_250ms": physical["onsetAndOffset250ms"]["f1"],
        "rapidRetriggersUnder100ms": notes["rapidRetriggersUnder100ms"],
    }


def duration_gate(
    baseline: dict[str, float | int | None],
    candidate: dict[str, float | int | None],
) -> dict[str, bool]:
    def value(payload: dict[str, float | int | None], key: str) -> float:
        item = payload.get(key)
        return float(item) if item is not None else float("inf")

    return {
        "durationErrorNotOver10PercentWorse": (
            value(candidate, "durationMedianAbsoluteErrorSeconds")
            <= value(baseline, "durationMedianAbsoluteErrorSeconds") * 1.10 + 1e-9
        ),
        "visualCutoffRateNotWorse": (
            value(candidate, "visualSevereCutoffRate")
            <= value(baseline, "visualSevereCutoffRate") * 1.05 + 0.005 + 1e-9
        ),
        "physicalDurationErrorNotOver10PercentWorse": (
            value(candidate, "physicalDurationMedianAbsoluteErrorSeconds")
            <= value(baseline, "physicalDurationMedianAbsoluteErrorSeconds") * 1.10
            + 1e-9
        ),
        "physicalCutoffRateNotWorse": (
            value(candidate, "physicalSevereCutoffRate")
            <= value(baseline, "physicalSevereCutoffRate") * 1.05 + 0.005 + 1e-9
        ),
        "rapidRetriggersNotOver5PercentWorse": (
            value(candidate, "rapidRetriggersUnder100ms")
            <= value(baseline, "rapidRetriggersUnder100ms") * 1.05 + 1.0
        ),
    }


def evaluate_manifest(
    manifest: dict[str, Any],
    profiles_root: Path,
    output_root: Path,
    *,
    threshold: float,
    coverage_radius: float,
    cluster_seconds: float,
    onset_window: float,
    maximum_additions_per_onset: int,
    register_shift: int = 0,
    gesture_anchor_radius: float = 0.0,
    maximum_piano_source_midi: int | None = 52,
    style_velocity_blend: float = 0.75,
    style_duration_blend: float = 0.65,
) -> dict[str, Any]:
    songs: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row["id"])
        profile_path = profiles_root / song_id / "profile.json"
        starting_path = Path(str(row["candidate"])).resolve()
        source_path = Path(str(row["source"])).resolve()
        reference_path = Path(str(row["reference"])).resolve()
        alignment_path = Path(str(row["alignment"])).resolve()
        starting_payload = load_json(starting_path)
        source_payload = load_json(source_path)
        profile_payload = load_json(profile_path)
        reference_payload = load_json(reference_path)
        alignment_payload = load_json(alignment_path)
        candidate, recovery = supplement_score(
            starting_payload,
            source_payload,
            profile_payload,
            threshold=threshold,
            coverage_radius=coverage_radius,
            cluster_seconds=cluster_seconds,
            onset_window=onset_window,
            maximum_additions_per_onset=maximum_additions_per_onset,
            maximum_source_midi=59,
            register_shift=register_shift,
            minimum_output_midi=33,
            maximum_output_midi=71,
            gesture_anchor_radius=gesture_anchor_radius,
            maximum_piano_source_midi=maximum_piano_source_midi,
            style_velocity_blend=style_velocity_blend,
            style_duration_blend=style_duration_blend,
        )
        candidate_path = output_root / "candidates" / f"{song_id}.json"
        atomic_json(candidate_path, candidate)
        evaluation = evaluate_pair(
            reference_payload,
            alignment_payload,
            starting_payload,
            candidate,
            reference_already_aligned=bool(row.get("referenceAlreadyAligned")),
            reference_transpose_semitones=int(
                row.get("referenceTransposeSemitones") or 0
            ),
            reference_end_seconds=(
                float(row["referenceEndSeconds"])
                if row.get("referenceEndSeconds") is not None
                else None
            ),
            candidate_end_seconds=(
                float(row["candidateEndSeconds"])
                if row.get("candidateEndSeconds") is not None
                else None
            ),
        )
        evaluation_row = {
            "referenceAlreadyAligned": bool(row.get("referenceAlreadyAligned")),
            "referenceTransposeSemitones": int(
                row.get("referenceTransposeSemitones") or 0
            ),
            "referenceEndSeconds": row.get("referenceEndSeconds"),
            "candidateEndSeconds": row.get("candidateEndSeconds"),
        }
        reference_notes = prepare_reference_notes(
            evaluation_row, reference_payload, alignment_payload
        )
        if evaluation_row["candidateEndSeconds"] is not None:
            reference_notes = [
                note
                for note in reference_notes
                if float(note["time"])
                < float(evaluation_row["candidateEndSeconds"])
            ]
        baseline_performance = performance_summary(
            measure(
                reference_notes,
                clip_candidate(starting_payload, evaluation_row, alignment_payload),
            )
        )
        candidate_performance = performance_summary(
            measure(
                reference_notes,
                clip_candidate(candidate, evaluation_row, alignment_payload),
            )
        )
        performance_gates = duration_gate(
            baseline_performance, candidate_performance
        )
        atomic_json(output_root / "evaluations" / f"{song_id}.json", evaluation)
        songs.append(
            {
                "id": song_id,
                "startingCandidate": str(starting_path),
                "source": str(source_path),
                "profile": str(profile_path.resolve()),
                "candidate": str(candidate_path.resolve()),
                "recovery": recovery,
                "evaluation": evaluation,
                "performance": {
                    "baseline": baseline_performance,
                    "candidate": candidate_performance,
                    "gates": performance_gates,
                    "passes": all(performance_gates.values()),
                },
            }
        )

    keys = (
        "quality",
        "exactF1_100ms",
        "exactF1_250ms",
        "pitchClassF1_250ms",
        "pitchClassRecall_250ms",
        "velocityMeanAbsoluteError",
        "chordSizeDistance",
        "handOccupancyDistance",
    )
    aggregate = {
        "songs": len(songs),
        "referenceNotes": sum(
            int(row["evaluation"]["referenceNotes"]) for row in songs
        ),
        "acceptedAdditions": sum(
            int(row["recovery"]["acceptedAdditions"]) for row in songs
        ),
        "weightedDeltas": {key: weighted_delta(songs, key) for key in keys},
        "worstSongDeltas": {
            key: round(
                min(float(row["evaluation"]["deltas"][key]) for row in songs), 6
            )
            for key in keys
        },
        "rapidRetriggerIncrease": sum(
            max(
                0,
                int(row["evaluation"]["candidate"]["rapidRetriggersUnder100ms"])
                - int(row["evaluation"]["baseline"]["rapidRetriggersUnder100ms"]),
            )
            for row in songs
        ),
    }
    aggregate["passesStructuralGate"] = bool(
        aggregate["weightedDeltas"]["quality"] > 0
        and aggregate["weightedDeltas"]["pitchClassF1_250ms"] > 0
        and aggregate["worstSongDeltas"]["quality"] >= -0.0005
        and aggregate["worstSongDeltas"]["exactF1_100ms"] >= -0.002
        and aggregate["rapidRetriggerIncrease"] == 0
    )
    aggregate["durationAndCutoffGates"] = {
        key: all(bool(row["performance"]["gates"][key]) for row in songs)
        for key in next(iter(songs))["performance"]["gates"]
    }
    aggregate["passesDurationAndCutoffGate"] = all(
        aggregate["durationAndCutoffGates"].values()
    )
    return {
        "schema": "polymath-raw-support-recovery-loso-v1",
        "evidenceBoundary": (
            "Each song is decoded with a selector trained without that complete song. "
            "This is structural evidence only; duration/cutoff and blind-listening gates "
            "remain mandatory before promotion."
        ),
        "policy": {
            "threshold": threshold,
            "coverageRadiusSeconds": coverage_radius,
            "clusterSeconds": cluster_seconds,
            "onsetWindowSeconds": onset_window,
            "maximumAdditionsPerOnset": maximum_additions_per_onset,
            "registerShiftSemitones": register_shift,
            "gestureAnchorRadiusSeconds": round(gesture_anchor_radius, 4),
            "maximumPianoSourceMidi": maximum_piano_source_midi,
            "styleVelocityBlend": style_velocity_blend,
            "styleDurationBlend": style_duration_blend,
        },
        "aggregate": aggregate,
        "songs": songs,
        "decision": (
            "CANDIDATE_FOR_DURATION_AND_LISTENING_GATES"
            if aggregate["passesStructuralGate"]
            and aggregate["passesDurationAndCutoffGate"]
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profiles-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--coverage-radius-seconds", type=float, default=0.18)
    parser.add_argument("--cluster-seconds", type=float, default=0.08)
    parser.add_argument("--onset-window-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-additions-per-onset", type=int, default=2)
    parser.add_argument(
        "--register-shift-semitones",
        type=int,
        default=0,
        help=(
            "Octave-only shift for recovered notes. Zero preserves the detected "
            "register and folds only unavoidable range edges."
        ),
    )
    parser.add_argument(
        "--gesture-anchor-radius-seconds",
        type=float,
        default=0.0,
        help="Recover only notes near an existing arranged gesture when positive.",
    )
    parser.add_argument(
        "--maximum-piano-source-midi",
        type=int,
        default=52,
        help="Maximum raw MIDI accepted from piano-labelled recovery evidence.",
    )
    parser.add_argument("--style-velocity-blend", type=float, default=0.75)
    parser.add_argument("--style-duration-blend", type=float, default=0.65)
    args = parser.parse_args()
    if int(args.register_shift_semitones) % 12:
        parser.error("--register-shift-semitones must be an octave multiple")
    output_root = Path(args.output_root).resolve()
    report = evaluate_manifest(
        load_json(Path(args.manifest).resolve()),
        Path(args.profiles_root).resolve(),
        output_root,
        threshold=max(0.0, min(1.0, args.threshold)),
        coverage_radius=max(0.01, args.coverage_radius_seconds),
        cluster_seconds=max(0.01, args.cluster_seconds),
        onset_window=max(0.01, args.onset_window_seconds),
        maximum_additions_per_onset=max(1, args.maximum_additions_per_onset),
        register_shift=int(args.register_shift_semitones),
        gesture_anchor_radius=max(0.0, float(args.gesture_anchor_radius_seconds)),
        maximum_piano_source_midi=int(args.maximum_piano_source_midi),
        style_velocity_blend=max(0.0, min(1.0, args.style_velocity_blend)),
        style_duration_blend=max(0.0, min(1.0, args.style_duration_blend)),
    )
    report_path = output_root / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "aggregate": report["aggregate"],
                "songs": [
                    {
                        "id": row["id"],
                        "acceptedAdditions": row["recovery"]["acceptedAdditions"],
                        "deltas": row["evaluation"]["deltas"],
                        "performanceGates": row["performance"]["gates"],
                    }
                    for row in report["songs"]
                ],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
