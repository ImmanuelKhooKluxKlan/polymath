"""Install a validated onset-gesture model into an arranger profile.

The onset trainer intentionally emits a small, standalone research artifact.
This utility performs the separate promotion step: it copies a complete piano
arranger profile, replaces only its left-hand onset selector, records the
provenance, and computes a new integrity hash.  No source profile is modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODEL_TYPE = "standardized-logistic-onset-gesture-ranker-v1"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def profile_sha256(profile: dict[str, Any]) -> str:
    payload = copy.deepcopy(profile)
    payload.pop("profileSha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def raw_feature_coefficients(model: dict[str, Any]) -> list[float]:
    """Convert a standardized linear logit into raw-feature coefficients."""

    names = list(model.get("featureNames") or [])
    weights = [float(value) for value in model.get("weights") or []]
    means = [float(value) for value in model.get("means") or []]
    scales = [float(value) for value in model.get("scales") or []]
    if not names or not (
        len(names) == len(weights) == len(means) == len(scales)
    ):
        raise ValueError("The onset model has inconsistent standardized feature arrays.")
    if "bias" not in names:
        raise ValueError("The onset model must contain a constant bias feature.")
    if any(abs(scale) < 1e-12 for scale in scales):
        raise ValueError("The onset model contains a zero feature scale.")

    coefficients = [weight / scale for weight, scale in zip(weights, scales)]
    constant = -sum(
        weight * mean / scale
        for weight, mean, scale in zip(weights, means, scales)
    )
    coefficients[names.index("bias")] += constant
    return coefficients


def blend_standardized_logistic_models(
    base_model: dict[str, Any],
    new_model: dict[str, Any],
    base_share: float,
) -> dict[str, Any]:
    """Blend logits while allowing each source model its own normalization."""

    if not 0.0 <= base_share <= 1.0:
        raise ValueError("base model share must be between 0 and 1")
    if base_model.get("type") != EXPECTED_MODEL_TYPE:
        raise ValueError("The base arranger onset model has an unsupported type.")
    base_names = list(base_model.get("featureNames") or [])
    new_names = list(new_model.get("featureNames") or [])
    if base_names != new_names:
        raise ValueError("The onset models use different feature orders and cannot be blended.")

    base_coefficients = raw_feature_coefficients(base_model)
    new_coefficients = raw_feature_coefficients(new_model)
    new_share = 1.0 - base_share
    return {
        "type": EXPECTED_MODEL_TYPE,
        "featureNames": base_names,
        "weights": [
            round(base_share * old + new_share * new, 10)
            for old, new in zip(base_coefficients, new_coefficients)
        ],
        "means": [0.0] * len(base_names),
        "scales": [1.0] * len(base_names),
        "threshold": round(
            base_share * float(base_model.get("threshold", 0.5))
            + new_share * float(new_model.get("threshold", 0.5)),
            10,
        ),
        "blend": {
            "policy": "raw-feature-logit-blend-v1",
            "baseShare": round(base_share, 6),
            "newShare": round(new_share, 6),
        },
    }


def apply_onset_profile(
    base: dict[str, Any],
    onset_profile: dict[str, Any],
    *,
    profile_id: str,
    onset_profile_path: str,
    validation_report_path: str | None = None,
    target_onset_keep_ratio: float | None = None,
    base_model_share: float = 0.0,
    created_at: str | None = None,
) -> dict[str, Any]:
    model = onset_profile.get("selectionModel")
    if not isinstance(model, dict) or model.get("type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            "The supplied onset profile does not contain a supported "
            f"{EXPECTED_MODEL_TYPE} selection model."
        )

    output = copy.deepcopy(base)
    decoder = output.setdefault("decoder", {})
    left_hand = decoder.setdefault("leftHandAccompaniment", {})
    if not left_hand.get("enabled"):
        raise ValueError("The base arranger has no enabled left-hand accompaniment decoder.")

    output["id"] = profile_id
    output["createdAt"] = created_at or datetime.now(timezone.utc).isoformat()
    old_model = left_hand.get("onsetSelectionModel")
    if base_model_share > 0.0:
        if not isinstance(old_model, dict):
            raise ValueError("The base arranger does not contain an onset model to blend.")
        left_hand["onsetSelectionModel"] = blend_standardized_logistic_models(
            old_model,
            model,
            base_model_share,
        )
    else:
        left_hand["onsetSelectionModel"] = copy.deepcopy(model)
    if target_onset_keep_ratio is not None:
        if not 0.1 <= target_onset_keep_ratio <= 1.0:
            raise ValueError("target onset keep ratio must be between 0.1 and 1.0")
        left_hand["targetOnsetKeepRatio"] = round(target_onset_keep_ratio, 6)

    training = output.setdefault("training", {})
    training["onsetGestureUpgrade"] = {
        "schema": "polymath-onset-gesture-promotion-v1",
        "profile": onset_profile_path,
        "profileId": onset_profile.get("id"),
        "profileSha256": onset_profile.get("profileSha256"),
        "validationReport": validation_report_path,
        "recommendedRawOnsetKeepRatio": onset_profile.get(
            "recommendedTargetOnsetKeepRatio"
        ),
        "deployedTargetOnsetKeepRatio": left_hand.get("targetOnsetKeepRatio"),
        "baseModelShare": round(base_model_share, 6),
        "newModelShare": round(1.0 - base_model_share, 6),
        "commercialUseAllowed": False,
        "decision": "EXPERIMENTAL_ONLY",
    }
    output.pop("profileSha256", None)
    output["profileSha256"] = profile_sha256(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--onset-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--validation-report")
    parser.add_argument("--target-onset-keep-ratio", type=float)
    parser.add_argument(
        "--base-model-share",
        type=float,
        default=0.0,
        help="Blend this share of the base arranger onset logit with the new model.",
    )
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    onset_path = Path(args.onset_profile).resolve()
    output_path = Path(args.output).resolve()
    validation_path = (
        str(Path(args.validation_report).resolve()) if args.validation_report else None
    )
    profile = apply_onset_profile(
        load_json(base_path),
        load_json(onset_path),
        profile_id=args.profile_id,
        onset_profile_path=str(onset_path),
        validation_report_path=validation_path,
        target_onset_keep_ratio=args.target_onset_keep_ratio,
        base_model_share=args.base_model_share,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "profileId": profile["id"],
                "profileSha256": profile["profileSha256"],
                "targetOnsetKeepRatio": profile["decoder"][
                    "leftHandAccompaniment"
                ].get("targetOnsetKeepRatio"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
