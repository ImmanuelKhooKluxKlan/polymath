"""Search conservative end-to-end policies for the swap-benefit ranker.

Both stacked components are whole-song out-of-fold for every development
song: the raw-support selector and the post-arranger benefit model.  Proposal
features and model scores are cached once; policy trials only filter and apply
those frozen scores.  Candidate evaluations are memoized by their actual swap
signature so a broad threshold grid remains inexpensive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .apply_raw_support_recovery import load_json
from .evaluate_pianist_candidate_pair import evaluate_pair
from .fit_raw_support_swap_benefit_ranker import build_song_examples
from .raw_support_swap_benefit import (
    apply_scored_proposals,
    generate_swap_proposals,
    score_feature_rows,
)


def numeric_grid(value: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("grid requires finite numeric values")
    return values


def register_set_grid(value: str) -> tuple[tuple[int, ...], ...]:
    choices: list[tuple[int, ...]] = []
    for group in value.split(";"):
        shifts = tuple(dict.fromkeys(int(item.strip()) for item in group.split(",") if item.strip()))
        if shifts and shifts not in choices:
            choices.append(shifts)
    if not choices:
        raise argparse.ArgumentTypeError("register sets use semicolons between comma-separated sets")
    return tuple(choices)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def weighted_delta(rows: list[dict[str, Any]], key: str) -> float:
    denominator = sum(int(row["evaluation"]["referenceNotes"]) for row in rows)
    numerator = sum(
        float(row["evaluation"]["deltas"][key]) * int(row["evaluation"]["referenceNotes"])
        for row in rows
    )
    return round(numerator / max(1, denominator), 6)


def policy_key(row: dict[str, Any]) -> tuple[float, ...]:
    weighted = row["weightedDeltas"]
    worst = row["worstSongDeltas"]
    return (
        float(row["passesStrictStructuralGate"]),
        float(weighted["quality"] > 0),
        float(weighted["exactF1_100ms"]),
        float(weighted["exactF1_250ms"]),
        float(weighted["pitchClassF1_250ms"]),
        float(worst["exactF1_250ms"]),
        float(worst["quality"]),
        -float(row["replacements"]),
    )


def swap_signature(candidate: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    signature: list[tuple[int, int, int]] = []
    for index, note in enumerate(candidate.get("notes") or []):
        if "rawSupportBenefitOriginalMidi" not in note:
            continue
        signature.append(
            (
                index,
                int(note["rawSupportBenefitOriginalMidi"]),
                int(note.get("midi", note.get("pitch"))),
            )
        )
    return tuple(signature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--benefit-profiles-root", type=Path)
    parser.add_argument(
        "--oof-scores-report",
        type=Path,
        help="Optional whole-song out-of-fold score report used instead of serialized benefit folds.",
    )
    parser.add_argument("--oof-model-name", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=numeric_grid,
        default=(0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.96, 0.98),
    )
    parser.add_argument(
        "--minimum-selector-probabilities",
        type=numeric_grid,
        default=(0.40, 0.55, 0.70, 0.85),
    )
    parser.add_argument(
        "--minimum-probability-gains",
        type=numeric_grid,
        default=(0.0, 0.15, 0.30, 0.45),
    )
    parser.add_argument(
        "--register-sets",
        type=register_set_grid,
        default=((0,), (12,), (0, 12)),
        help="Semicolon-separated choices; each choice is comma-separated, e.g. 0;12;0,12",
    )
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--minimum-source-midi", type=int, default=0)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    selector_root = args.selector_profiles_root.resolve()
    if bool(args.oof_scores_report) != bool(args.oof_model_name):
        raise ValueError("--oof-scores-report and --oof-model-name must be supplied together")
    if not args.oof_scores_report and not args.benefit_profiles_root:
        raise ValueError("A benefit profile root or an out-of-fold score report is required")
    benefit_root = args.benefit_profiles_root.resolve() if args.benefit_profiles_root else None
    external_model: dict[str, Any] | None = None
    if args.oof_scores_report:
        ablation = load_json(args.oof_scores_report.resolve())
        matches = [
            row
            for row in ablation.get("models") or []
            if str(row.get("name")) == str(args.oof_model_name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one out-of-fold model named {args.oof_model_name!r}")
        external_model = matches[0]
    output_root = args.output_root.resolve()
    manifest = load_json(manifest_path)
    union_shifts = tuple(sorted({shift for choice in args.register_sets for shift in choice}))
    prepared: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row["id"])
        paths = {
            "reference": Path(str(row["reference"])).resolve(),
            "alignment": Path(str(row["alignment"])).resolve(),
            "source": Path(str(row["source"])).resolve(),
            "baseline": Path(str(row["candidate"])).resolve(),
            "selector": (selector_root / song_id / "profile.json").resolve(),
        }
        if benefit_root is not None:
            paths["benefit"] = (benefit_root / song_id / "profile.json").resolve()
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{song_id} {label} is missing: {path}")
        item = {
            "id": song_id,
            "row": row,
            "reference": load_json(paths["reference"]),
            "alignment": load_json(paths["alignment"]),
            "source": load_json(paths["source"]),
            "baseline": load_json(paths["baseline"]),
            "selector": load_json(paths["selector"]),
            "benefit": (
                load_json(paths["benefit"])
                if "benefit" in paths
                else {"id": str(args.oof_model_name)}
            ),
            "evaluationCache": {},
        }
        if external_model is not None:
            examples = build_song_examples(
                row,
                item["selector"],
                source_radius=max(0.01, float(args.source_radius_seconds)),
                minimum_source_midi=int(args.minimum_source_midi),
                maximum_source_midi=int(args.maximum_source_midi),
                minimum_output_midi=int(args.minimum_output_midi),
                maximum_output_midi=int(args.maximum_output_midi),
                register_shifts=union_shifts,
            )
            proposals = examples["proposals"]
            generation = examples["diagnostics"]
        else:
            proposals, generation = generate_swap_proposals(
                item["baseline"],
                item["source"],
                item["selector"],
                source_radius=max(0.01, float(args.source_radius_seconds)),
                minimum_source_midi=int(args.minimum_source_midi),
                maximum_source_midi=int(args.maximum_source_midi),
                minimum_output_midi=int(args.minimum_output_midi),
                maximum_output_midi=int(args.maximum_output_midi),
                register_shifts=union_shifts,
            )
        item["proposals"] = proposals
        if external_model is not None:
            raw_scores = (external_model.get("probabilities") or {}).get(song_id)
            if not isinstance(raw_scores, list) or len(raw_scores) != len(proposals):
                raise ValueError(
                    f"{song_id} out-of-fold score count does not match proposals: "
                    f"{len(raw_scores) if isinstance(raw_scores, list) else 'missing'} != {len(proposals)}"
                )
            item["scores"] = [float(value) for value in raw_scores]
        else:
            model = item["benefit"].get("model") or item["benefit"].get("swapBenefitModel")
            if not isinstance(model, dict):
                raise ValueError(f"{song_id} benefit fold has no model")
            item["scores"] = score_feature_rows([proposal["features"] for proposal in proposals], model)
        item["generation"] = generation
        prepared.append(item)
    if len(prepared) < 4:
        raise ValueError("At least four songs are required for policy search.")

    metric_keys = (
        "quality",
        "exactF1_100ms",
        "exactF1_250ms",
        "pitchClassF1_250ms",
        "pitchClassRecall_250ms",
        "velocityMeanAbsoluteError",
        "chordSizeDistance",
        "handOccupancyDistance",
    )
    leaderboard: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        for selector_minimum in args.minimum_selector_probabilities:
            for gain_minimum in args.minimum_probability_gains:
                for register_shifts in args.register_sets:
                    songs: list[dict[str, Any]] = []
                    for item in prepared:
                        allowed = set(register_shifts)
                        kept_indices = [
                            index
                            for index, proposal in enumerate(item["proposals"])
                            if int(proposal["registerShift"]) in allowed
                        ]
                        proposals = [item["proposals"][index] for index in kept_indices]
                        scores = [item["scores"][index] for index in kept_indices]
                        generation = {**item["generation"], "registerShifts": list(register_shifts)}
                        candidate, diagnostics = apply_scored_proposals(
                            item["baseline"],
                            proposals,
                            scores,
                            item["benefit"],
                            generation,
                            threshold=max(0.0, min(1.0, threshold)),
                            minimum_selector_probability=max(0.0, min(1.0, selector_minimum)),
                            minimum_probability_gain=float(gain_minimum),
                            maximum_replacements_per_gesture=1,
                        )
                        signature = swap_signature(candidate)
                        cache = item["evaluationCache"]
                        if signature not in cache:
                            row = item["row"]
                            cache[signature] = evaluate_pair(
                                item["reference"],
                                item["alignment"],
                                item["baseline"],
                                candidate,
                                reference_already_aligned=bool(row.get("referenceAlreadyAligned")),
                                reference_transpose_semitones=int(row.get("referenceTransposeSemitones") or 0),
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
                        evaluation = cache[signature]
                        songs.append(
                            {
                                "id": item["id"],
                                "replacements": diagnostics["replacements"],
                                "evaluation": evaluation,
                            }
                        )

                    weighted = {key: weighted_delta(songs, key) for key in metric_keys}
                    worst = {
                        key: round(min(float(song["evaluation"]["deltas"][key]) for song in songs), 6)
                        for key in metric_keys
                    }
                    retrigger_increase = sum(
                        max(
                            0,
                            int(song["evaluation"]["candidate"]["rapidRetriggersUnder100ms"])
                            - int(song["evaluation"]["baseline"]["rapidRetriggersUnder100ms"]),
                        )
                        for song in songs
                    )
                    trial = {
                        "policy": {
                            "benefitThreshold": threshold,
                            "minimumSelectorProbability": selector_minimum,
                            "minimumProbabilityGain": gain_minimum,
                            "sourceRadiusSeconds": float(args.source_radius_seconds),
                            "maximumReplacementsPerGesture": 1,
                            "registerShifts": list(register_shifts),
                            "sourceMidiRange": [int(args.minimum_source_midi), int(args.maximum_source_midi)],
                            "outputMidiRange": [int(args.minimum_output_midi), int(args.maximum_output_midi)],
                        },
                        "weightedDeltas": weighted,
                        "worstSongDeltas": worst,
                        "replacements": sum(int(song["replacements"]) for song in songs),
                        "rapidRetriggerIncrease": retrigger_increase,
                        "songs": [
                            {
                                "id": song["id"],
                                "replacements": song["replacements"],
                                "deltas": song["evaluation"]["deltas"],
                            }
                            for song in songs
                        ],
                    }
                    trial["passesStrictStructuralGate"] = bool(
                        weighted["quality"] > 0
                        and weighted["exactF1_100ms"] > 0
                        and weighted["exactF1_250ms"] > 0
                        and weighted["pitchClassF1_250ms"] >= 0
                        and weighted["pitchClassRecall_250ms"] >= 0
                        and worst["quality"] >= -0.00025
                        and worst["exactF1_100ms"] >= 0
                        and worst["exactF1_250ms"] >= 0
                        and worst["pitchClassF1_250ms"] >= 0
                        and retrigger_increase == 0
                    )
                    leaderboard.append(trial)

    leaderboard.sort(key=policy_key, reverse=True)
    top = leaderboard[: max(1, int(args.top_k))]
    winner = top[0]
    winning_policy = winner["policy"]
    winning_registers = {int(value) for value in winning_policy["registerShifts"]}
    for item in prepared:
        kept_indices = [
            index
            for index, proposal in enumerate(item["proposals"])
            if int(proposal["registerShift"]) in winning_registers
        ]
        candidate, _diagnostics = apply_scored_proposals(
            item["baseline"],
            [item["proposals"][index] for index in kept_indices],
            [item["scores"][index] for index in kept_indices],
            item["benefit"],
            {**item["generation"], "registerShifts": sorted(winning_registers)},
            threshold=float(winning_policy["benefitThreshold"]),
            minimum_selector_probability=float(winning_policy["minimumSelectorProbability"]),
            minimum_probability_gain=float(winning_policy["minimumProbabilityGain"]),
            maximum_replacements_per_gesture=1,
        )
        atomic_json(output_root / "best-candidates" / f"{item['id']}.json", candidate)
    report = {
        "schema": "polymath-raw-support-swap-benefit-policy-search-v1",
        "evidenceBoundary": (
            "Every development song uses a raw-support selector and benefit ranker trained "
            "without that complete song. Policy selection uses only manifest songs. A later "
            "transfer song must remain absent from both model fitting and this grid."
        ),
        "developmentSongs": [item["id"] for item in prepared],
        "manifest": str(manifest_path),
        "benefitEvidence": (
            {
                "type": "whole-song-out-of-fold-score-report",
                "report": str(args.oof_scores_report.resolve()),
                "model": str(args.oof_model_name),
            }
            if args.oof_scores_report
            else {
                "type": "serialized-whole-song-holdout-profiles",
                "profilesRoot": str(benefit_root),
            }
        ),
        "trials": len(leaderboard),
        "uniqueCandidateEvaluations": {
            item["id"]: len(item["evaluationCache"]) for item in prepared
        },
        "best": winner,
        "top": top,
        "decision": (
            "CANDIDATE_FOR_UNSEEN_TRANSFER"
            if winner["passesStrictStructuralGate"]
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }
    atomic_json(output_root / "report.json", report)
    print(
        json.dumps(
            {
                "report": str((output_root / "report.json").resolve()),
                "trials": len(leaderboard),
                "uniqueCandidateEvaluations": report["uniqueCandidateEvaluations"],
                "best": winner,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
