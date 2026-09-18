"""Run leave-one-song-out validation for the learned piano arranger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from blend_piano_arranger_profiles import compose_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "ml" / "training" / "train_piano_arranger_adapter.py"
ARRANGER = REPO_ROOT / "server" / "piano_arranger.py"
EVALUATOR = REPO_ROOT / "ml" / "training" / "evaluate_piano_arranger.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rehash_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(profile)
    profile.pop("profileSha256", None)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    return profile


def run(*arguments: str | Path) -> None:
    subprocess.run([str(argument) for argument in arguments], cwd=REPO_ROOT, check=True)


def weighted_average(values: list[tuple[float, float]]) -> float:
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / max(total, 1e-9)


def pooled_weighted_rate(
    folds: dict[str, Any],
    side: str,
    section: str,
    numerator_field: str,
    denominator_field: str,
) -> float:
    """Pool event rates so tiny-match songs cannot dominate full-song evidence."""
    numerator = sum(
        float(fold[side][section][numerator_field]) * float(fold["heldOutWeight"])
        for fold in folds.values()
    )
    denominator = sum(
        float(fold[side][section][denominator_field]) * float(fold["heldOutWeight"])
        for fold in folds.values()
    )
    return numerator / max(denominator, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cleaned-source-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-feature-version",
        choices=("v1", "v2-lite", "v2"),
        default="v1",
        help="Selection feature contract passed identically to every fold trainer.",
    )
    parser.add_argument(
        "--density-multiplier",
        type=float,
        help="Validation-only override for decoder.preCleanupDensityMultiplier.",
    )
    parser.add_argument(
        "--duration-prediction-weight",
        type=float,
        default=0.0,
        help="Duration-model blend passed to every fold's trainer (default: 0).",
    )
    parser.add_argument(
        "--quota-backfill-ratio",
        type=float,
        help=(
            "Optional fraction of each local density quota that may be filled by "
            "events below the learned selection threshold."
        ),
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        help="Optional global selector threshold used consistently in every held-out fold.",
    )
    parser.add_argument(
        "--preferred-global-shift-semitones",
        type=int,
        help="Optional octave-multiple register lift applied after learned arrangement.",
    )
    parser.add_argument(
        "--source-duration-weight",
        type=float,
        help="Optional blend weight for source duration versus the learned role median.",
    )
    parser.add_argument(
        "--minimum-rendered-duration-seconds",
        type=float,
        help="Optional floor for unusually short arranged key holds.",
    )
    parser.add_argument(
        "--duration-extension-only",
        action="store_true",
        help="Permit the duration learner to lengthen source holds but never shorten them.",
    )
    parser.add_argument("--minimum-melody-duration-seconds", type=float)
    parser.add_argument("--minimum-bass-duration-seconds", type=float)
    parser.add_argument("--minimum-harmony-duration-seconds", type=float)
    parser.add_argument("--minimum-bass-family-duration-seconds", type=float)
    parser.add_argument(
        "--learn-register-model",
        action="store_true",
        help="Train and evaluate a trusted-evidence octave selector in every fold.",
    )
    parser.add_argument(
        "--register-fallback-shift-semitones",
        type=int,
        default=12,
        help="Fallback octave shift for register groups absent from a fold (default: 12).",
    )
    parser.add_argument(
        "--register-uncertain-shift-semitones",
        type=int,
        default=0,
        help="Octave shift used when fold evidence exists but is inconclusive.",
    )
    parser.add_argument(
        "--register-minimum-confidence",
        type=float,
        default=0.75,
        help="Minimum winning-class share for a learned octave choice.",
    )
    expansion = parser.add_mutually_exclusive_group()
    expansion.add_argument(
        "--expand-sparse-harmony",
        action="store_true",
        help="Train every fold with the same sparse-harmony expansion used by v010.",
    )
    expansion.add_argument(
        "--disable-sparse-expansion",
        action="store_true",
        help="Validation-only override that disables generated octave doublings.",
    )
    residual_base = parser.add_mutually_exclusive_group()
    residual_base.add_argument(
        "--residual-base-profile",
        help=(
            "Optional frozen profile blended with every fold's newly trained selector. "
            "This evaluates a conservative residual candidate without fitting the held-out song."
        ),
    )
    residual_base.add_argument(
        "--residual-base-profile-dir",
        help=(
            "Optional directory containing <held-out-id>/profile.json. Use this "
            "to ensure the frozen side of each residual fold also excludes its held-out song."
        ),
    )
    parser.add_argument(
        "--residual-base-share",
        type=float,
        default=0.95,
        help="Frozen-profile share when --residual-base-profile is used (default: 0.95).",
    )
    parser.add_argument(
        "--adaptive-policy-profile",
        help="Optional profile whose factual source-density policy is copied into every fold.",
    )
    parser.add_argument("--adaptive-selection-low-source-nps", type=float)
    parser.add_argument("--adaptive-selection-high-source-nps", type=float)
    parser.add_argument("--adaptive-selection-low-base-share", type=float)
    parser.add_argument("--adaptive-selection-high-base-share", type=float)
    parser.add_argument("--adaptive-selection-minimum-voice-ratio", type=float)
    args = parser.parse_args()

    if not 0.0 <= args.residual_base_share <= 1.0:
        raise ValueError("residual-base-share must be between 0 and 1")
    if args.adaptive_policy_profile and not (
        args.residual_base_profile or args.residual_base_profile_dir
    ):
        raise ValueError(
            "adaptive-policy-profile requires residual-base-profile or residual-base-profile-dir"
        )
    adaptive_selection_values = [
        args.adaptive_selection_low_source_nps,
        args.adaptive_selection_high_source_nps,
        args.adaptive_selection_low_base_share,
        args.adaptive_selection_high_base_share,
    ]
    if any(value is not None for value in adaptive_selection_values) and not all(
        value is not None for value in adaptive_selection_values
    ):
        raise ValueError(
            "All four adaptive selection-blend controls must be supplied together"
        )
    adaptive_selection_blend = None
    if all(value is not None for value in adaptive_selection_values):
        if not (args.residual_base_profile or args.residual_base_profile_dir):
            raise ValueError(
                "Adaptive selection blending requires a residual base profile"
            )
        if (
            args.adaptive_selection_high_source_nps
            <= args.adaptive_selection_low_source_nps
        ):
            raise ValueError(
                "adaptive-selection-high-source-nps must exceed the low threshold"
            )
        adaptive_selection_blend = {
            "lowSourceNotesPerSecond": args.adaptive_selection_low_source_nps,
            "highSourceNotesPerSecond": args.adaptive_selection_high_source_nps,
            "lowSourceBaseShare": args.adaptive_selection_low_base_share,
            "highSourceBaseShare": args.adaptive_selection_high_base_share,
        }
        if args.adaptive_selection_minimum_voice_ratio is not None:
            adaptive_selection_blend["minimumVoiceRatioForAggressiveBlend"] = (
                args.adaptive_selection_minimum_voice_ratio
            )
    elif args.adaptive_selection_minimum_voice_ratio is not None:
        raise ValueError(
            "adaptive-selection-minimum-voice-ratio requires all four selection-blend controls"
        )

    manifest = load_json(Path(args.manifest).resolve())
    pairs = manifest.get("pairs") or []
    if len(pairs) < 3:
        raise ValueError("Leave-one-song-out validation needs at least three songs.")
    auxiliary_pairs = manifest.get("auxiliaryPairs") or []
    if not isinstance(auxiliary_pairs, list):
        raise ValueError("auxiliaryPairs must be a list when provided.")
    pair_ids = [str(pair.get("id")) for pair in pairs]
    auxiliary_ids = [str(pair.get("id")) for pair in auxiliary_pairs]
    all_ids = pair_ids + auxiliary_ids
    duplicate_ids = {value for value in all_ids if all_ids.count(value) > 1}
    if duplicate_ids:
        raise ValueError(
            "Every evaluation and auxiliary pair id must be unique: "
            + ", ".join(sorted(duplicate_ids))
        )
    cleaned_source_dir = Path(args.cleaned_source_dir).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    fixed_residual_base = (
        load_json(Path(args.residual_base_profile).resolve())
        if args.residual_base_profile
        else None
    )
    residual_base_dir = (
        Path(args.residual_base_profile_dir).resolve()
        if args.residual_base_profile_dir
        else None
    )
    adaptive_policy = None
    if args.adaptive_policy_profile:
        policy_profile = load_json(Path(args.adaptive_policy_profile).resolve())
        adaptive_policy = copy.deepcopy(
            (policy_profile.get("decoder") or {}).get("adaptiveSourceDensity")
        )
        if not isinstance(adaptive_policy, dict):
            raise ValueError("adaptive-policy-profile has no decoder.adaptiveSourceDensity")
        adaptive_policy.pop("enabled", None)
    folds: dict[str, Any] = {}

    for held_out in pairs:
        held_out_id = str(held_out["id"])
        fold_residual_base = fixed_residual_base
        if residual_base_dir is not None:
            fold_base_path = residual_base_dir / held_out_id / "profile.json"
            if not fold_base_path.is_file():
                raise ValueError(f"Missing leakage-safe fold base profile: {fold_base_path}")
            fold_residual_base = load_json(fold_base_path)
        fold_root = output_dir / held_out_id
        training_manifest = {
            **manifest,
            "profileId": f"pianella-loso-without-{held_out_id}",
            "pairs": [pair for pair in pairs if pair["id"] != held_out_id],
        }
        evaluation_manifest = {
            **{key: value for key, value in manifest.items() if key != "auxiliaryPairs"},
            "profileId": f"pianella-loso-held-out-{held_out_id}",
            "pairs": [held_out],
        }
        training_manifest_path = fold_root / "training-manifest.json"
        evaluation_manifest_path = fold_root / "evaluation-manifest.json"
        profile_path = fold_root / "profile.json"
        training_report_path = fold_root / "training-report.json"
        candidate_dir = fold_root / "candidate"
        evaluation_path = fold_root / "evaluation.json"
        write_json(training_manifest_path, training_manifest)
        write_json(evaluation_manifest_path, evaluation_manifest)
        trainer_arguments: list[str | Path] = [
            sys.executable,
            TRAINER,
            "--manifest",
            training_manifest_path,
            "--output",
            profile_path,
            "--report",
            training_report_path,
            "--duration-prediction-weight",
            str(max(0.0, min(1.0, float(args.duration_prediction_weight)))),
            "--selection-feature-version",
            args.selection_feature_version,
        ]
        if args.expand_sparse_harmony:
            trainer_arguments.append("--expand-sparse-harmony")
        if args.learn_register_model:
            trainer_arguments.extend(
                [
                    "--learn-register-model",
                    "--register-fallback-shift-semitones",
                    str(args.register_fallback_shift_semitones),
                    "--register-uncertain-shift-semitones",
                    str(args.register_uncertain_shift_semitones),
                    "--register-minimum-confidence",
                    str(max(0.0, min(1.0, args.register_minimum_confidence))),
                ]
            )
        run(*trainer_arguments)
        profile = load_json(profile_path)
        if fold_residual_base is not None:
            profile = compose_profile(
                fold_residual_base,
                profile,
                base_share=args.residual_base_share,
                profile_id=f"pianella-loso-residual-without-{held_out_id}",
                take_duration_from_other=True,
                take_register_from_other=args.learn_register_model,
                duration_extension_only=(
                    True if args.duration_extension_only else None
                ),
                minimum_rendered_duration_by_role={
                    role: value
                    for role, value in {
                        "melody": args.minimum_melody_duration_seconds,
                        "bass": args.minimum_bass_duration_seconds,
                        "harmony": args.minimum_harmony_duration_seconds,
                    }.items()
                    if value is not None
                },
                minimum_rendered_duration_by_source_family=(
                    {"bass": args.minimum_bass_family_duration_seconds}
                    if args.minimum_bass_family_duration_seconds is not None
                    else {}
                ),
                adaptive_source_density=adaptive_policy,
                adaptive_selection_blend=adaptive_selection_blend,
            )
        decoder = profile.setdefault("decoder", {})
        if args.selection_threshold is not None:
            profile.setdefault("selectionModel", {})["threshold"] = max(
                0.0, min(1.0, float(args.selection_threshold))
            )
        if args.density_multiplier is not None:
            decoder["preCleanupDensityMultiplier"] = max(
                0.5, min(2.0, float(args.density_multiplier))
            )
        if args.quota_backfill_ratio is not None:
            decoder["quotaBackfillRatio"] = max(
                0.0, min(1.0, float(args.quota_backfill_ratio))
            )
        if args.preferred_global_shift_semitones is not None:
            decoder["preferredGlobalRegisterShiftSemitones"] = int(
                max(
                    -84,
                    min(
                        84,
                        round(float(args.preferred_global_shift_semitones) / 12.0) * 12,
                    ),
                )
            )
        if args.source_duration_weight is not None:
            decoder["sourceDurationWeight"] = max(
                0.0, min(1.0, float(args.source_duration_weight))
            )
        if args.minimum_rendered_duration_seconds is not None:
            decoder["minimumRenderedDurationSeconds"] = max(
                0.05, min(1.0, float(args.minimum_rendered_duration_seconds))
            )
        if args.duration_extension_only:
            decoder["durationExtensionOnly"] = True
        role_minimums = {
            role: max(0.05, min(1.0, float(value)))
            for role, value in {
                "melody": args.minimum_melody_duration_seconds,
                "bass": args.minimum_bass_duration_seconds,
                "harmony": args.minimum_harmony_duration_seconds,
            }.items()
            if value is not None
        }
        if role_minimums:
            decoder["minimumRenderedDurationByRole"] = role_minimums
        if args.minimum_bass_family_duration_seconds is not None:
            decoder["minimumRenderedDurationBySourceFamily"] = {
                "bass": max(
                    0.05,
                    min(1.0, float(args.minimum_bass_family_duration_seconds)),
                )
            }
        if args.disable_sparse_expansion:
            decoder["expandSparseHarmony"] = False
        elif args.expand_sparse_harmony:
            decoder["expandSparseHarmony"] = True
        write_json(profile_path, rehash_profile(profile))
        candidate_dir.mkdir(parents=True, exist_ok=True)
        run(
            sys.executable,
            ARRANGER,
            "--input",
            cleaned_source_dir / f"{held_out_id}.json",
            "--output",
            candidate_dir / f"{held_out_id}.json",
            "--mode",
            "full",
            "--profile",
            profile_path,
        )
        run(
            sys.executable,
            EVALUATOR,
            "--manifest",
            evaluation_manifest_path,
            "--baseline-dir",
            baseline_dir,
            "--candidate-dir",
            candidate_dir,
            "--output",
            evaluation_path,
        )
        evaluation = load_json(evaluation_path)
        song = evaluation["songs"][held_out_id]
        folds[held_out_id] = {
            "trainingSongIds": [
                pair["id"] for pair in pairs if pair["id"] != held_out_id
            ],
            "auxiliaryTrainingIds": [pair["id"] for pair in auxiliary_pairs],
            "heldOutWeight": float(held_out.get("weight", 1.0)),
            "profile": str(profile_path),
            "residualBaseProfileId": (
                None if fold_residual_base is None else fold_residual_base.get("id")
            ),
            "baseline": song["baseline"],
            "candidate": song["candidate"],
            "candidateMinusBaseline": evaluation["candidateMinusBaseline"],
            "foldDecision": evaluation["decision"],
        }

    aggregate: dict[str, dict[str, float]] = {"baseline": {}, "candidate": {}}
    metric_paths = {
        "exactF1_50ms": ("exactPitchOnset50ms", "f1"),
        "exactF1_100ms": ("exactPitchOnset100ms", "f1"),
        "exactF1_250ms": ("exactPitchOnset250ms", "f1"),
        "pitchClassF1_250ms": ("pitchClassOnset250ms", "f1"),
    }
    for side in ("baseline", "candidate"):
        for name, path in metric_paths.items():
            aggregate[side][name] = round(
                weighted_average(
                    [
                        (float(fold[side][path[0]][path[1]]), float(fold["heldOutWeight"]))
                        for fold in folds.values()
                    ]
                ),
                6,
            )
        aggregate[side]["durationMedianAbsoluteErrorSeconds"] = round(
            weighted_average(
                [
                    (
                        float(fold[side]["duration"]["medianAbsoluteErrorSeconds"] or 0.0),
                        float(fold["heldOutWeight"]),
                    )
                    for fold in folds.values()
                ]
            ),
            6,
        )
        aggregate[side]["severeCutoffs"] = round(
            weighted_average(
                [
                    (float(fold[side]["duration"]["severeCutoffs"]), float(fold["heldOutWeight"]))
                    for fold in folds.values()
                ]
            ),
            6,
        )
        for name, section, field in (
            (
                "visualDurationMedianAbsoluteErrorSeconds",
                "visualDuration",
                "medianAbsoluteErrorSeconds",
            ),
            (
                "physicalDurationMedianAbsoluteErrorSeconds",
                "physicalDuration",
                "medianAbsoluteErrorSeconds",
            ),
        ):
            aggregate[side][name] = round(
                weighted_average(
                    [
                        (
                            float(fold[side][section][field] or 0.0),
                            float(fold["heldOutWeight"]),
                        )
                        for fold in folds.values()
                    ]
                ),
                6,
            )
        for name, section in (
            ("severeCutoffRate", "duration"),
            ("visualSevereCutoffRate", "visualDuration"),
            ("physicalSevereCutoffRate", "physicalDuration"),
        ):
            aggregate[side][name] = round(
                pooled_weighted_rate(
                    folds,
                    side,
                    section,
                    "severeCutoffs",
                    "matchedNotes",
                ),
                6,
            )
        aggregate[side]["rapidRetriggersUnder100ms"] = round(
            weighted_average(
                [
                    (
                        float(fold[side]["rapidRetriggersUnder100ms"]),
                        float(fold["heldOutWeight"]),
                    )
                    for fold in folds.values()
                ]
            ),
            6,
        )

    deltas = {
        name: round(aggregate["candidate"][name] - aggregate["baseline"][name], 6)
        for name in aggregate["baseline"]
    }
    baseline = aggregate["baseline"]
    candidate = aggregate["candidate"]
    promotion_gates = {
        "unseen_exact_f1_100ms_improves": (
            candidate["exactF1_100ms"] >= baseline["exactF1_100ms"] + 0.005
        ),
        "unseen_exact_f1_250ms_does_not_regress": (
            candidate["exactF1_250ms"] >= baseline["exactF1_250ms"] - 0.002
        ),
        "unseen_pitch_class_f1_250ms_does_not_regress": (
            candidate["pitchClassF1_250ms"] >= baseline["pitchClassF1_250ms"] - 0.002
        ),
        "no_held_out_song_exact_f1_100ms_regresses_over_1_point": all(
            float(fold["candidateMinusBaseline"]["exactF1_100ms"]) >= -0.01
            for fold in folds.values()
        ),
        "unseen_duration_error_not_over_10_percent_worse": (
            candidate["durationMedianAbsoluteErrorSeconds"]
            <= baseline["durationMedianAbsoluteErrorSeconds"] * 1.10
        ),
        "unseen_severe_cutoff_rate_not_worse": (
            candidate["severeCutoffRate"]
            <= baseline["severeCutoffRate"] * 1.05 + 0.005
        ),
        "unseen_visual_duration_error_not_over_10_percent_worse": (
            candidate["visualDurationMedianAbsoluteErrorSeconds"]
            <= baseline["visualDurationMedianAbsoluteErrorSeconds"] * 1.10
        ),
        "unseen_visual_cutoff_rate_not_worse": (
            candidate["visualSevereCutoffRate"]
            <= baseline["visualSevereCutoffRate"] * 1.05 + 0.005
        ),
        "unseen_physical_duration_error_not_over_10_percent_worse": (
            candidate["physicalDurationMedianAbsoluteErrorSeconds"]
            <= baseline["physicalDurationMedianAbsoluteErrorSeconds"] * 1.10
        ),
        "unseen_physical_cutoff_rate_not_worse": (
            candidate["physicalSevereCutoffRate"]
            <= baseline["physicalSevereCutoffRate"] * 1.05 + 0.005
        ),
        "no_held_out_song_physical_cutoff_rate_regresses_over_3_points": all(
            float(fold["candidateMinusBaseline"]["physicalSevereCutoffRate"])
            <= 0.03
            for fold in folds.values()
        ),
        "unseen_rapid_retriggers_not_worse": (
            candidate["rapidRetriggersUnder100ms"]
            <= baseline["rapidRetriggersUnder100ms"] * 1.05 + 1.0
        ),
    }
    decision = "PROMOTE" if all(promotion_gates.values()) else "REJECT"
    if residual_base_dir is not None:
        validation_policy = (
            "both the base and residual selector exclude the song graded in each fold"
        )
    elif fixed_residual_base is not None:
        validation_policy = (
            "the residual selector excludes each graded song; the fixed base profile may have "
            "prior exposure and its provenance must be checked before claiming unseen performance"
        )
    else:
        validation_policy = "each candidate is trained without the song used for that fold's evaluation"
    if auxiliary_pairs:
        validation_policy += (
            "; declared auxiliary selection-only pairs are shared across folds and are never graded"
        )
    summary = {
        "schema": "polymath-piano-arranger-loso-v1",
        "policy": validation_policy,
        "folds": folds,
        "weightedAggregate": aggregate,
        "candidateMinusBaseline": deltas,
        "promotionGates": promotion_gates,
        "decision": decision,
        "decoderValidationOverrides": {
            "preCleanupDensityMultiplier": args.density_multiplier,
            "durationPredictionWeight": max(
                0.0, min(1.0, float(args.duration_prediction_weight))
            ),
            "quotaBackfillRatio": (
                None
                if args.quota_backfill_ratio is None
                else max(0.0, min(1.0, float(args.quota_backfill_ratio)))
            ),
            "selectionThreshold": (
                None
                if args.selection_threshold is None
                else max(0.0, min(1.0, float(args.selection_threshold)))
            ),
            "preferredGlobalRegisterShiftSemitones": (
                None
                if args.preferred_global_shift_semitones is None
                else int(
                    max(
                        -84,
                        min(
                            84,
                            round(float(args.preferred_global_shift_semitones) / 12.0) * 12,
                        ),
                    )
                )
            ),
            "sourceDurationWeight": (
                None
                if args.source_duration_weight is None
                else max(0.0, min(1.0, float(args.source_duration_weight)))
            ),
            "minimumRenderedDurationSeconds": (
                None
                if args.minimum_rendered_duration_seconds is None
                else max(
                    0.05, min(1.0, float(args.minimum_rendered_duration_seconds))
                )
            ),
            "durationExtensionOnly": bool(args.duration_extension_only),
            "minimumRenderedDurationByRole": {
                role: max(0.05, min(1.0, float(value)))
                for role, value in {
                    "melody": args.minimum_melody_duration_seconds,
                    "bass": args.minimum_bass_duration_seconds,
                    "harmony": args.minimum_harmony_duration_seconds,
                }.items()
                if value is not None
            },
            "minimumRenderedDurationBySourceFamily": (
                {
                    "bass": max(
                        0.05,
                        min(
                            1.0,
                            float(args.minimum_bass_family_duration_seconds),
                        ),
                    )
                }
                if args.minimum_bass_family_duration_seconds is not None
                else {}
            ),
            "learnRegisterModel": bool(args.learn_register_model),
            "registerFallbackShiftSemitones": (
                int(
                    max(
                        -48,
                        min(
                            48,
                            round(float(args.register_fallback_shift_semitones) / 12.0)
                            * 12,
                        ),
                    )
                )
                if args.learn_register_model
                else None
            ),
            "registerUncertainShiftSemitones": (
                int(
                    max(
                        -48,
                        min(
                            48,
                            round(float(args.register_uncertain_shift_semitones) / 12.0)
                            * 12,
                        ),
                    )
                )
                if args.learn_register_model
                else None
            ),
            "registerMinimumConfidence": (
                max(0.0, min(1.0, float(args.register_minimum_confidence)))
                if args.learn_register_model
                else None
            ),
            "expandSparseHarmony": (
                False
                if args.disable_sparse_expansion
                else True if args.expand_sparse_harmony else None
            ),
            "residualBaseProfile": (
                None if fixed_residual_base is None else fixed_residual_base.get("id")
            ),
            "residualBaseProfileDirectory": (
                None if residual_base_dir is None else str(residual_base_dir)
            ),
            "residualBaseShare": (
                None
                if fixed_residual_base is None and residual_base_dir is None
                else args.residual_base_share
            ),
            "adaptivePolicyProfile": args.adaptive_policy_profile,
            "adaptiveSelectionBlend": adaptive_selection_blend,
            "selectionFeatureVersion": args.selection_feature_version,
            "auxiliaryTrainingPairIds": auxiliary_ids,
        },
        "warning": (
            f"Only {len(pairs)} independently graded songs are in this manifest. "
            "This is a small-sample stress test, not production-scale validation."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "weightedAggregate": aggregate,
        "candidateMinusBaseline": deltas,
        "promotionGates": promotion_gates,
        "decision": decision,
        "decoderValidationOverrides": summary["decoderValidationOverrides"],
    }, indent=2))


if __name__ == "__main__":
    main()
