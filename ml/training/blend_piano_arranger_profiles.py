"""Compose an auditable residual blend of two piano-arranger profiles.

Each selector is a standardized logistic model.  Converting both logits back
to raw-feature coefficients lets us interpolate their decisions exactly,
without averaging incompatible standardized weights or retraining on a held-
out song.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def raw_logit_coefficients(model: dict[str, Any]) -> tuple[list[str], list[float]]:
    names = [str(value) for value in model.get("featureNames") or []]
    weights = [float(value) for value in model.get("weights") or []]
    means = [float(value) for value in model.get("means") or []]
    scales = [float(value) for value in model.get("scales") or []]
    if not names or not (len(names) == len(weights) == len(means) == len(scales)):
        raise ValueError("Selection model fields are missing or have different lengths")
    if any(abs(scale) < 1e-12 for scale in scales):
        raise ValueError("Selection model contains a zero feature scale")
    coefficients = [weight / scale for weight, scale in zip(weights, scales)]
    intercept = -sum(
        weight * mean / scale
        for weight, mean, scale in zip(weights, means, scales)
    )
    # The first feature is the constant bias feature in every current profile.
    if names[0] != "bias":
        raise ValueError("The first selector feature must be bias")
    coefficients[0] += intercept
    return names, coefficients


def blend_selection_models(
    base: dict[str, Any],
    other: dict[str, Any],
    base_share: float,
) -> dict[str, Any]:
    share = min(1.0, max(0.0, float(base_share)))
    base_names, base_coefficients = raw_logit_coefficients(base)
    other_names, other_coefficients = raw_logit_coefficients(other)
    if base_names != other_names:
        # Context contracts are append-only.  An older model is exactly
        # representable in the newer space by assigning zero raw coefficients
        # to every appended feature.  Reject every other mismatch.
        if len(base_names) < len(other_names) and other_names[: len(base_names)] == base_names:
            base_coefficients += [0.0] * (len(other_names) - len(base_names))
            base_names = list(other_names)
        elif len(other_names) < len(base_names) and base_names[: len(other_names)] == other_names:
            other_coefficients += [0.0] * (len(base_names) - len(other_names))
            other_names = list(base_names)
        else:
            raise ValueError("Selection profiles use incompatible feature contracts")
    coefficients = [
        share * left + (1.0 - share) * right
        for left, right in zip(base_coefficients, other_coefficients)
    ]
    return {
        "type": "raw-feature-residual-logit-blend-v1",
        "featureNames": base_names,
        "weights": [round(value, 10) for value in coefficients],
        "means": [0.0] * len(coefficients),
        "scales": [1.0] * len(coefficients),
        "threshold": round(
            share * float(base.get("threshold", 0.5))
            + (1.0 - share) * float(other.get("threshold", 0.5)),
            6,
        ),
    }


def compose_profile(
    base: dict[str, Any],
    other: dict[str, Any],
    *,
    base_share: float,
    profile_id: str,
    take_duration_from_other: bool = False,
    take_register_from_other: bool = False,
    take_roles_from_other: bool = False,
    take_style_from_other: bool = False,
    density_multiplier: float | None = None,
    duration_prediction_weight: float | None = None,
    preferred_global_shift_semitones: int | None = None,
    source_duration_weight: float | None = None,
    minimum_rendered_duration_seconds: float | None = None,
    minimum_rendered_duration_by_role: dict[str, float] | None = None,
    minimum_rendered_duration_by_source_family: dict[str, float] | None = None,
    duration_extension_only: bool | None = None,
    adaptive_source_density: dict[str, float] | None = None,
    adaptive_selection_blend: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile = copy.deepcopy(base)
    profile.pop("profileSha256", None)
    profile["id"] = profile_id
    profile["createdAt"] = datetime.now(timezone.utc).isoformat()
    profile["selectionModel"] = blend_selection_models(
        base.get("selectionModel") or {},
        other.get("selectionModel") or {},
        base_share,
    )
    if take_duration_from_other:
        profile["durationModel"] = copy.deepcopy(other.get("durationModel") or {})
    if take_register_from_other:
        register_model = copy.deepcopy(other.get("registerModel") or {})
        if not register_model:
            raise ValueError("Cannot take a register model from a profile that has none")
        profile["registerModel"] = register_model
    if take_roles_from_other:
        profile["roles"] = copy.deepcopy(other.get("roles") or {})
    if take_style_from_other:
        profile["style"] = copy.deepcopy(other.get("style") or {})
    if density_multiplier is not None:
        profile.setdefault("decoder", {})["preCleanupDensityMultiplier"] = round(
            min(3.0, max(0.5, float(density_multiplier))), 4
        )
    if duration_prediction_weight is not None:
        if not profile.get("durationModel"):
            raise ValueError("Cannot set duration prediction weight without a duration model")
        profile["durationModel"]["predictionWeight"] = round(
            min(1.0, max(0.0, float(duration_prediction_weight))), 4
        )
    if preferred_global_shift_semitones is not None:
        profile.setdefault("decoder", {})["preferredGlobalRegisterShiftSemitones"] = int(
            max(-84, min(84, round(float(preferred_global_shift_semitones) / 12.0) * 12))
        )
    if source_duration_weight is not None:
        profile.setdefault("decoder", {})["sourceDurationWeight"] = round(
            min(1.0, max(0.0, float(source_duration_weight))), 4
        )
    if minimum_rendered_duration_seconds is not None:
        profile.setdefault("decoder", {})["minimumRenderedDurationSeconds"] = round(
            min(1.0, max(0.05, float(minimum_rendered_duration_seconds))), 4
        )
    if minimum_rendered_duration_by_role:
        profile.setdefault("decoder", {})["minimumRenderedDurationByRole"] = {
            role: round(min(1.0, max(0.05, float(value))), 4)
            for role, value in minimum_rendered_duration_by_role.items()
            if role in {"melody", "bass", "harmony"}
        }
    if minimum_rendered_duration_by_source_family:
        profile.setdefault("decoder", {})[
            "minimumRenderedDurationBySourceFamily"
        ] = {
            family: round(min(1.0, max(0.05, float(value))), 4)
            for family, value in minimum_rendered_duration_by_source_family.items()
        }
    if duration_extension_only is not None:
        profile.setdefault("decoder", {})["durationExtensionOnly"] = bool(
            duration_extension_only
        )
    if adaptive_source_density is not None:
        profile.setdefault("decoder", {})["adaptiveSourceDensity"] = {
            "enabled": True,
            **{key: round(float(value), 4) for key, value in adaptive_source_density.items()},
        }
    if adaptive_selection_blend is not None:
        low_nps = float(adaptive_selection_blend["lowSourceNotesPerSecond"])
        high_nps = float(adaptive_selection_blend["highSourceNotesPerSecond"])
        low_share = min(
            1.0,
            max(0.0, float(adaptive_selection_blend["lowSourceBaseShare"])),
        )
        high_share = min(
            1.0,
            max(0.0, float(adaptive_selection_blend["highSourceBaseShare"])),
        )
        if high_nps <= low_nps:
            raise ValueError(
                "Adaptive selection high source density must exceed low source density"
            )
        profile.setdefault("decoder", {})["adaptiveSelectionBlend"] = {
            "enabled": True,
            "signal": "sourceNotesPerSecond",
            "lowSourceNotesPerSecond": round(low_nps, 4),
            "highSourceNotesPerSecond": round(high_nps, 4),
            "lowSourceBaseShare": round(low_share, 6),
            "highSourceBaseShare": round(high_share, 6),
            "lowSourceSelectionModel": blend_selection_models(
                base.get("selectionModel") or {},
                other.get("selectionModel") or {},
                low_share,
            ),
            "highSourceSelectionModel": blend_selection_models(
                base.get("selectionModel") or {},
                other.get("selectionModel") or {},
                high_share,
            ),
        }
        minimum_voice_ratio = adaptive_selection_blend.get(
            "minimumVoiceRatioForAggressiveBlend"
        )
        if minimum_voice_ratio is not None:
            profile["decoder"]["adaptiveSelectionBlend"][
                "minimumVoiceRatioForAggressiveBlend"
            ] = round(min(1.0, max(0.0, float(minimum_voice_ratio))), 4)
    profile["training"] = {
        "schema": "polymath-residual-profile-provenance-v1",
        "method": "raw-feature-logit-residual-blend",
        "baseProfile": {
            "id": base.get("id"),
            "profileSha256": base.get("profileSha256"),
            "trainingManifest": (base.get("training") or {}).get("manifest"),
        },
        "otherProfile": {
            "id": other.get("id"),
            "profileSha256": other.get("profileSha256"),
            "trainingManifest": (other.get("training") or {}).get("manifest"),
        },
        "commercialUseAllowed": False,
        "purpose": "private research evaluation",
    }
    profile["residualBlend"] = {
        "schema": "polymath-piano-arranger-residual-blend-v1",
        "baseProfileId": base.get("id"),
        "otherProfileId": other.get("id"),
        "baseShare": round(min(1.0, max(0.0, float(base_share))), 6),
        "otherShare": round(1.0 - min(1.0, max(0.0, float(base_share))), 6),
        "durationFrom": other.get("id") if take_duration_from_other else base.get("id"),
        "registerFrom": other.get("id") if take_register_from_other else base.get("id"),
        "rolesFrom": other.get("id") if take_roles_from_other else base.get("id"),
        "styleFrom": other.get("id") if take_style_from_other else base.get("id"),
        "densityMultiplier": profile.get("decoder", {}).get("preCleanupDensityMultiplier"),
        "durationPredictionWeight": profile.get("durationModel", {}).get("predictionWeight"),
        "preferredGlobalRegisterShiftSemitones": profile.get("decoder", {}).get(
            "preferredGlobalRegisterShiftSemitones", 24
        ),
        "sourceDurationWeight": profile.get("decoder", {}).get("sourceDurationWeight"),
        "minimumRenderedDurationSeconds": profile.get("decoder", {}).get(
            "minimumRenderedDurationSeconds", 0.05
        ),
        "minimumRenderedDurationByRole": profile.get("decoder", {}).get(
            "minimumRenderedDurationByRole", {}
        ),
        "minimumRenderedDurationBySourceFamily": profile.get("decoder", {}).get(
            "minimumRenderedDurationBySourceFamily", {}
        ),
        "durationExtensionOnly": bool(
            profile.get("decoder", {}).get("durationExtensionOnly", False)
        ),
        "adaptiveSourceDensity": profile.get("decoder", {}).get("adaptiveSourceDensity"),
        "adaptiveSelectionBlend": profile.get("decoder", {}).get(
            "adaptiveSelectionBlend"
        ),
        "commercialUseAllowed": False,
        "purpose": "private research candidate; requires held-out promotion gates",
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile["profileSha256"] = hashlib.sha256(canonical).hexdigest()
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--base-share", type=float, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--take-duration-from-other", action="store_true")
    parser.add_argument("--take-register-from-other", action="store_true")
    parser.add_argument("--take-roles-from-other", action="store_true")
    parser.add_argument("--take-style-from-other", action="store_true")
    parser.add_argument("--density-multiplier", type=float)
    parser.add_argument("--duration-prediction-weight", type=float)
    parser.add_argument("--preferred-global-shift-semitones", type=int)
    parser.add_argument("--source-duration-weight", type=float)
    parser.add_argument("--minimum-rendered-duration-seconds", type=float)
    parser.add_argument("--minimum-melody-duration-seconds", type=float)
    parser.add_argument("--minimum-bass-duration-seconds", type=float)
    parser.add_argument("--minimum-harmony-duration-seconds", type=float)
    parser.add_argument("--minimum-bass-family-duration-seconds", type=float)
    parser.add_argument("--duration-extension-only", action="store_true")
    parser.add_argument("--adaptive-low-source-nps", type=float)
    parser.add_argument("--adaptive-high-source-nps", type=float)
    parser.add_argument("--adaptive-low-density", type=float)
    parser.add_argument("--adaptive-high-density", type=float)
    parser.add_argument("--adaptive-low-duration-weight", type=float)
    parser.add_argument("--adaptive-high-duration-weight", type=float)
    parser.add_argument("--adaptive-short-source-median-duration", type=float)
    parser.add_argument("--adaptive-long-source-median-duration", type=float)
    parser.add_argument("--adaptive-minimum-voice-ratio-density", type=float)
    parser.add_argument("--adaptive-non-vocal-density", type=float)
    parser.add_argument("--adaptive-maximum-piano-ratio-vocal", type=float)
    parser.add_argument("--adaptive-minimum-voice-ratio-duration", type=float)
    parser.add_argument(
        "--adaptive-non-vocal-duration-weight", type=float, default=0.0
    )
    parser.add_argument("--adaptive-short-source-duration-weight", type=float)
    parser.add_argument("--adaptive-long-source-duration-weight", type=float)
    parser.add_argument(
        "--adaptive-non-vocal-source-duration-weight", type=float, default=1.0
    )
    parser.add_argument("--adaptive-low-quota-backfill-ratio", type=float)
    parser.add_argument("--adaptive-high-quota-backfill-ratio", type=float)
    parser.add_argument(
        "--adaptive-non-vocal-quota-backfill-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--adaptive-minimum-voice-ratio-sparse-disable", type=float
    )
    parser.add_argument("--adaptive-selection-low-source-nps", type=float)
    parser.add_argument("--adaptive-selection-high-source-nps", type=float)
    parser.add_argument("--adaptive-selection-low-base-share", type=float)
    parser.add_argument("--adaptive-selection-high-base-share", type=float)
    parser.add_argument("--adaptive-selection-minimum-voice-ratio", type=float)
    args = parser.parse_args()

    if not 0.0 <= args.base_share <= 1.0:
        raise ValueError("base-share must be between 0 and 1")
    base = json.loads(args.base.read_text(encoding="utf-8-sig"))
    other = json.loads(args.other.read_text(encoding="utf-8-sig"))
    adaptive_values = [
        args.adaptive_low_source_nps,
        args.adaptive_high_source_nps,
        args.adaptive_low_density,
        args.adaptive_high_density,
        args.adaptive_low_duration_weight,
        args.adaptive_high_duration_weight,
    ]
    if any(value is not None for value in adaptive_values) and not all(
        value is not None for value in adaptive_values
    ):
        raise ValueError("All six adaptive source-density controls must be supplied together")
    adaptive_source_density = None
    if all(value is not None for value in adaptive_values):
        if args.adaptive_high_source_nps <= args.adaptive_low_source_nps:
            raise ValueError("adaptive-high-source-nps must exceed adaptive-low-source-nps")
        adaptive_source_density = {
            "lowSourceNotesPerSecond": args.adaptive_low_source_nps,
            "highSourceNotesPerSecond": args.adaptive_high_source_nps,
            "lowDensityMultiplier": args.adaptive_low_density,
            "highDensityMultiplier": args.adaptive_high_density,
            "lowDurationPredictionWeight": args.adaptive_low_duration_weight,
            "highDurationPredictionWeight": args.adaptive_high_duration_weight,
        }
        density_voice_controls = [
            args.adaptive_minimum_voice_ratio_density,
            args.adaptive_non_vocal_density,
        ]
        if any(value is not None for value in density_voice_controls) and not all(
            value is not None for value in density_voice_controls
        ):
            raise ValueError(
                "Both adaptive density voice-gate controls must be supplied together"
            )
        if all(value is not None for value in density_voice_controls):
            adaptive_source_density.update(
                {
                    "minimumVoiceRatioForDensityAdaptation": min(
                        1.0,
                        max(0.0, args.adaptive_minimum_voice_ratio_density),
                    ),
                    "nonVocalDensityMultiplier": min(
                        3.0, max(0.5, args.adaptive_non_vocal_density)
                    ),
                }
            )
        if args.adaptive_minimum_voice_ratio_sparse_disable is not None:
            adaptive_source_density[
                "minimumVoiceRatioForDisablingSparseExpansion"
            ] = min(
                1.0,
                max(0.0, args.adaptive_minimum_voice_ratio_sparse_disable),
            )
        if args.adaptive_maximum_piano_ratio_vocal is not None:
            adaptive_source_density[
                "maximumPianoRatioForVocalAdaptation"
            ] = min(1.0, max(0.0, args.adaptive_maximum_piano_ratio_vocal))
        duration_thresholds = [
            args.adaptive_short_source_median_duration,
            args.adaptive_long_source_median_duration,
        ]
        if any(value is not None for value in duration_thresholds) and not all(
            value is not None for value in duration_thresholds
        ):
            raise ValueError(
                "Both adaptive source median-duration thresholds must be supplied together"
            )
        if all(value is not None for value in duration_thresholds):
            if (
                args.adaptive_long_source_median_duration
                <= args.adaptive_short_source_median_duration
            ):
                raise ValueError(
                    "adaptive-long-source-median-duration must exceed the short threshold"
                )
            adaptive_source_density.update(
                {
                    "shortSourceMedianDurationSeconds": args.adaptive_short_source_median_duration,
                    "longSourceMedianDurationSeconds": args.adaptive_long_source_median_duration,
                }
            )
        if args.adaptive_minimum_voice_ratio_duration is not None:
            adaptive_source_density.update(
                {
                    "minimumVoiceRatioForDurationPrediction": min(
                        1.0,
                        max(0.0, args.adaptive_minimum_voice_ratio_duration),
                    ),
                    "nonVocalDurationPredictionWeight": min(
                        1.0,
                        max(0.0, args.adaptive_non_vocal_duration_weight),
                    ),
                }
            )
        source_duration_weights = [
            args.adaptive_short_source_duration_weight,
            args.adaptive_long_source_duration_weight,
        ]
        if any(value is not None for value in source_duration_weights) and not all(
            value is not None for value in source_duration_weights
        ):
            raise ValueError(
                "Both adaptive source-duration blend weights must be supplied together"
            )
        if all(value is not None for value in source_duration_weights):
            adaptive_source_density.update(
                {
                    "shortSourceDurationWeight": min(
                        1.0, max(0.0, args.adaptive_short_source_duration_weight)
                    ),
                    "longSourceDurationWeight": min(
                        1.0, max(0.0, args.adaptive_long_source_duration_weight)
                    ),
                    "nonVocalSourceDurationWeight": min(
                        1.0,
                        max(0.0, args.adaptive_non_vocal_source_duration_weight),
                    ),
                }
            )
        quota_backfill_values = [
            args.adaptive_low_quota_backfill_ratio,
            args.adaptive_high_quota_backfill_ratio,
        ]
        if any(value is not None for value in quota_backfill_values) and not all(
            value is not None for value in quota_backfill_values
        ):
            raise ValueError(
                "Both adaptive quota-backfill ratios must be supplied together"
            )
        if all(value is not None for value in quota_backfill_values):
            adaptive_source_density.update(
                {
                    "lowQuotaBackfillRatio": min(
                        1.0, max(0.0, args.adaptive_low_quota_backfill_ratio)
                    ),
                    "highQuotaBackfillRatio": min(
                        1.0, max(0.0, args.adaptive_high_quota_backfill_ratio)
                    ),
                    "nonVocalQuotaBackfillRatio": min(
                        1.0,
                        max(
                            0.0,
                            args.adaptive_non_vocal_quota_backfill_ratio,
                        ),
                    ),
                }
            )
    elif any(
        value is not None
        for value in (
            args.adaptive_minimum_voice_ratio_density,
            args.adaptive_non_vocal_density,
            args.adaptive_maximum_piano_ratio_vocal,
            args.adaptive_minimum_voice_ratio_sparse_disable,
            args.adaptive_low_quota_backfill_ratio,
            args.adaptive_high_quota_backfill_ratio,
        )
    ):
        raise ValueError(
            "Adaptive voice-aware density/expansion controls require all six source-density controls"
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
    profile = compose_profile(
        base,
        other,
        base_share=args.base_share,
        profile_id=args.profile_id,
        take_duration_from_other=args.take_duration_from_other,
        take_register_from_other=args.take_register_from_other,
        take_roles_from_other=args.take_roles_from_other,
        take_style_from_other=args.take_style_from_other,
        density_multiplier=args.density_multiplier,
        duration_prediction_weight=args.duration_prediction_weight,
        preferred_global_shift_semitones=args.preferred_global_shift_semitones,
        source_duration_weight=args.source_duration_weight,
        minimum_rendered_duration_seconds=args.minimum_rendered_duration_seconds,
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
        duration_extension_only=True if args.duration_extension_only else None,
        adaptive_source_density=adaptive_source_density,
        adaptive_selection_blend=adaptive_selection_blend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "profileId": profile["id"],
        "baseShare": profile["residualBlend"]["baseShare"],
        "profileSha256": profile["profileSha256"],
    }))


if __name__ == "__main__":
    main()
