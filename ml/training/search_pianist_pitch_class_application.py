"""Search a residual pitch-class decoder on complete development songs.

The fitted fold model for each song excludes that entire song.  This utility
only searches the small, shared application policy (add/remove thresholds and
placement strategy), then measures the resulting playable JSON against the
current candidate.  It never changes the frozen foundation transcription.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

from .apply_pianist_pitch_class_decoder import apply_decoder, load_json
from .evaluate_piano_arranger import prepare_reference_notes
from .evaluate_raw_support_recovery_loso import duration_gate, performance_summary
from .search_default_piano_pipeline import (
    atomic_json,
    clip_candidate,
    compact_metrics,
    load_rows,
    measure,
    scalar_quality,
)


def parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return values


def metric_deltas(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        key: round(float(candidate[key]) - float(control[key]), 6)
        for key in (
            "quality",
            "exactF1_100ms",
            "exactF1_250ms",
            "pitchClassF1_250ms",
            "pitchClassRecall_250ms",
            "velocityMeanAbsoluteError",
            "chordSizeDistance",
            "handOccupancyDistance",
        )
    }


def is_song_safe(
    deltas: dict[str, float], performance_gates: dict[str, bool]
) -> bool:
    tolerance = 0.0005
    return bool(
        float(deltas["quality"]) >= -tolerance
        and float(deltas["exactF1_100ms"]) >= -tolerance
        and float(deltas["exactF1_250ms"]) >= -tolerance
        and float(deltas["pitchClassF1_250ms"]) >= -tolerance
        and float(deltas["pitchClassRecall_250ms"]) >= -tolerance
        and all(bool(value) for value in performance_gates.values())
    )


def weighted_average(
    songs: Iterable[dict[str, Any]], key: str, *, nested: str = "deltas"
) -> float:
    rows = list(songs)
    total = sum(float(row.get("weight", 1.0)) for row in rows)
    return sum(
        float(row.get("weight", 1.0)) * float(row[nested][key]) for row in rows
    ) / max(1e-12, total)


def trial_selection_score(songs: list[dict[str, Any]]) -> float:
    return (
        weighted_average(songs, "quality")
        + 0.35 * weighted_average(songs, "exactF1_100ms")
        + 0.15 * weighted_average(songs, "exactF1_250ms")
        + 0.15 * weighted_average(songs, "pitchClassF1_250ms")
        + 0.05 * weighted_average(songs, "pitchClassRecall_250ms")
        - 0.04 * max(0.0, weighted_average(songs, "chordSizeDistance"))
        - 0.04 * max(0.0, weighted_average(songs, "handOccupancyDistance"))
    )


def audit_songs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = {
        str(song.get("id") or ""): song for song in payload.get("songs") or []
    }
    if "" in values:
        raise ValueError("Every audit song requires a non-empty id")
    return values


def application_paths(row: dict[str, Any]) -> tuple[Path, Path]:
    """Return (control, decoder-start) paths for replacement experiments.

    Historical manifests contain only ``candidate`` and therefore retain the
    old residual-search behavior.  A manifest may additionally freeze a
    production ``baseline`` while applying a challenger decoder to the shared
    pre-decoder ``candidate``.  This prevents accidentally stacking two
    alternative chord decoders during a replacement comparison.
    """

    starting = Path(str(row["candidate"])).resolve()
    control = Path(str(row.get("baseline") or row["candidate"])).resolve()
    return control, starting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--folds-root", type=Path)
    parser.add_argument(
        "--scores-root",
        type=Path,
        help="Directory of frozen <song-id>.json probability grids.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="JSON threshold/source-support policy used with --scores-root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--song",
        action="append",
        default=[],
        help="Song id to evaluate. Repeat it; default is every manifest song with a fold model.",
    )
    parser.add_argument(
        "--add-thresholds",
        type=parse_float_grid,
        default=(0.50, 0.55, 0.60, 0.65, 0.70),
    )
    parser.add_argument(
        "--remove-thresholds",
        type=parse_float_grid,
        default=(0.05, 0.075, 0.10, 0.125, 0.15, 0.175),
    )
    parser.add_argument(
        "--register-strategies",
        default="nearest-candidate",
        help="Comma-separated source, nearest-candidate, or family values.",
    )
    parser.add_argument(
        "--addition-timings",
        default="candidate",
        help="Comma-separated candidate, source, or hybrid values.",
    )
    parser.add_argument(
        "--preserve-melody-removals",
        action="store_true",
        help="Never remove a pitch class carried by a melody-role note.",
    )
    parser.add_argument(
        "--remove-generated-only",
        action="store_true",
        help="Only remove pitch classes whose candidate notes were decoder-generated.",
    )
    parser.add_argument("--maximum-generated-probability-for-removal", type=float)
    parser.add_argument("--require-exact-source-midi-for-removal", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.manifest.resolve())
    selected = set(str(value) for value in args.song if str(value))
    if selected:
        rows = [row for row in rows if str(row["id"]) in selected]
    audit = load_json(args.audit.resolve())
    audits = audit_songs(audit)
    if bool(args.scores_root) != bool(args.policy):
        raise ValueError("--scores-root and --policy must be provided together")
    if not args.folds_root and not args.scores_root:
        raise ValueError("Provide --folds-root or --scores-root with --policy")
    folds_root = args.folds_root.resolve() if args.folds_root else None
    scores_root = args.scores_root.resolve() if args.scores_root else None
    external_policy = load_json(args.policy.resolve()) if args.policy else None
    register_strategies = tuple(
        value.strip() for value in args.register_strategies.split(",") if value.strip()
    )
    addition_timings = tuple(
        value.strip() for value in args.addition_timings.split(",") if value.strip()
    )
    if not set(register_strategies) <= {"source", "nearest-candidate", "family"}:
        raise ValueError("Unsupported register strategy")
    if not set(addition_timings) <= {"candidate", "source", "hybrid"}:
        raise ValueError("Unsupported addition timing")

    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        song_id = str(row["id"])
        if song_id not in audits:
            raise ValueError(f"Audit is missing {song_id}")
        if scores_root is not None:
            model_path = args.policy.resolve()
            scores_path = scores_root / f"{song_id}.json"
            if not scores_path.exists():
                if selected:
                    raise ValueError(f"Frozen scores are missing for {song_id}")
                continue
            model = copy.deepcopy(external_policy)
            probability_scores = load_json(scores_path)
        else:
            assert folds_root is not None
            model_path = folds_root / song_id / "model.json"
            if not model_path.exists():
                if selected:
                    raise ValueError(f"Fold model is missing for {song_id}")
                continue
            model = load_json(model_path)
            scores_path = None
            probability_scores = None
        alignment = load_json(Path(str(row["alignment"])).resolve())
        reference = prepare_reference_notes(
            row,
            load_json(Path(str(row["reference"])).resolve()),
            alignment,
        )
        control_path, starting_path = application_paths(row)
        control_payload = load_json(control_path)
        starting_payload = load_json(starting_path)
        control_measure = measure(
            reference, clip_candidate(control_payload, row, alignment)
        )
        cache[song_id] = {
            "row": row,
            "alignment": alignment,
            "reference": reference,
            "source": load_json(Path(str(row["source"])).resolve()),
            "controlPayload": control_payload,
            "startingPayload": starting_payload,
            "controlPath": str(control_path),
            "startingPath": str(starting_path),
            "controlMeasure": control_measure,
            "control": compact_metrics(control_measure),
            "audit": audits[song_id],
            "model": model,
            "modelPath": str(model_path),
            "probabilityScores": probability_scores,
            "probabilityScoresPath": (
                str(scores_path) if scores_path is not None else None
            ),
        }
    if not cache:
        raise ValueError("No manifest song has a corresponding fold model")

    trials: list[dict[str, Any]] = []
    combinations = itertools.product(
        args.add_thresholds,
        args.remove_thresholds,
        register_strategies,
        addition_timings,
    )
    for add_threshold, remove_threshold, register_strategy, addition_timing in combinations:
        if not 0.0 <= remove_threshold < add_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= remove < add <= 1")
        key = (
            round(float(add_threshold), 6),
            round(float(remove_threshold), 6),
            register_strategy,
            addition_timing,
        )
        per_song: list[dict[str, Any]] = []
        for song_id, item in cache.items():
            model = copy.deepcopy(item["model"])
            model["addThreshold"] = float(add_threshold)
            model["removeThreshold"] = float(remove_threshold)
            candidate_payload, changes = apply_decoder(
                item["startingPayload"],
                item["source"],
                item["audit"],
                model,
                register_strategy=register_strategy,
                addition_timing=addition_timing,
                preserve_melody_removals=bool(args.preserve_melody_removals),
                remove_generated_only=bool(args.remove_generated_only),
                maximum_generated_probability_for_removal=(
                    args.maximum_generated_probability_for_removal
                ),
                require_exact_source_midi_for_removal=bool(
                    args.require_exact_source_midi_for_removal
                ),
                probabilities_override=item["probabilityScores"],
            )
            candidate_measure = measure(
                item["reference"],
                clip_candidate(candidate_payload, item["row"], item["alignment"]),
            )
            candidate = compact_metrics(candidate_measure)
            deltas = metric_deltas(item["control"], candidate)
            gates = duration_gate(
                performance_summary(item["controlMeasure"]),
                performance_summary(candidate_measure),
            )
            per_song.append(
                {
                    "id": song_id,
                    "weight": float(item["row"].get("weight", 1.0)),
                    "control": item["control"],
                    "candidate": candidate,
                    "deltas": deltas,
                    "performanceGates": gates,
                    "safe": is_song_safe(deltas, gates),
                    "changes": changes,
                }
            )
        safe = all(song["safe"] for song in per_song)
        total_changes = sum(
            int(song["changes"].get("acceptedAdditions", 0))
            + int(song["changes"].get("removedNotes", 0))
            for song in per_song
        )
        selection_score = trial_selection_score(per_song)
        trial = {
            "safe": safe,
            "materialImprovement": bool(total_changes and selection_score > 1e-7),
            "totalChanges": total_changes,
            "selectionScore": round(selection_score, 9),
            "addThreshold": float(add_threshold),
            "removeThreshold": float(remove_threshold),
            "registerStrategy": register_strategy,
            "additionTiming": addition_timing,
            "songs": per_song,
        }
        trials.append(trial)

    trials.sort(
        key=lambda row: (
            not (bool(row["safe"]) and bool(row["materialImprovement"])),
            not bool(row["safe"]),
            -float(row["selectionScore"]),
            -min(float(song["deltas"]["quality"]) for song in row["songs"]),
            float(row["addThreshold"]),
            float(row["removeThreshold"]),
        )
    )
    best = trials[0]
    output_dir = args.output_dir.resolve()
    if bool(best["safe"]) and bool(best["materialImprovement"]):
        for song_id, item in cache.items():
            model = copy.deepcopy(item["model"])
            model["addThreshold"] = float(best["addThreshold"])
            model["removeThreshold"] = float(best["removeThreshold"])
            payload, _changes = apply_decoder(
                item["startingPayload"],
                item["source"],
                item["audit"],
                model,
                register_strategy=str(best["registerStrategy"]),
                addition_timing=str(best["additionTiming"]),
                preserve_melody_removals=bool(args.preserve_melody_removals),
                remove_generated_only=bool(args.remove_generated_only),
                maximum_generated_probability_for_removal=(
                    args.maximum_generated_probability_for_removal
                ),
                require_exact_source_midi_for_removal=bool(
                    args.require_exact_source_midi_for_removal
                ),
                probabilities_override=item["probabilityScores"],
            )
            atomic_json(output_dir / "best-candidates" / f"{song_id}.json", payload)
    report = {
        "schema": "polymath-pianist-pitch-class-application-search-v2",
        "evidenceBoundary": (
            "Every coefficient model excludes its complete evaluated song. The shared "
            "application policy is selected on opened development songs, so a new untouched "
            "hard-song holdout and blind listening are still mandatory."
        ),
        "manifest": str(args.manifest.resolve()),
        "audit": str(args.audit.resolve()),
        "foldsRoot": str(folds_root) if folds_root is not None else None,
        "scoresRoot": str(scores_root) if scores_root is not None else None,
        "policy": str(args.policy.resolve()) if args.policy else None,
        "preserveMelodyRemovals": bool(args.preserve_melody_removals),
        "removeGeneratedOnly": bool(args.remove_generated_only),
        "maximumGeneratedProbabilityForRemoval": (
            args.maximum_generated_probability_for_removal
        ),
        "requireExactSourceMidiForRemoval": bool(
            args.require_exact_source_midi_for_removal
        ),
        "best": best,
        "trials": trials,
        "decision": (
            "RESEARCH_CANDIDATE"
            if bool(best["safe"]) and bool(best["materialImprovement"])
            else "REJECT_NO_MATERIAL_SAFE_IMPROVEMENT"
        ),
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "decision": report["decision"],
                "best": {
                    key: best[key]
                    for key in (
                        "safe",
                        "materialImprovement",
                        "totalChanges",
                        "selectionScore",
                        "addThreshold",
                        "removeThreshold",
                        "registerStrategy",
                        "additionTiming",
                    )
                },
                "songs": [
                    {"id": song["id"], "safe": song["safe"], **song["deltas"]}
                    for song in best["songs"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
