"""Install an alternate selector for one source family/register only.

This is a guarded research promotion step.  It leaves the base selector in
place and stores a richer contextual selector under an explicit runtime gate,
so an experiment aimed at upper guitar-labelled melody evidence cannot alter
the already reliable voice and bass paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def apply_conditional_selector(
    base: dict[str, Any],
    alternative: dict[str, Any],
    *,
    profile_id: str,
    alternative_profile_path: str,
    alternative_share: float,
    source_families: list[str],
    minimum_source_midi: int,
    maximum_source_midi: int,
    preserve_base_window_counts: bool = False,
    preserve_base_onset_counts: bool = False,
    onset_slot_window_seconds: float = 0.035,
    validation_report_path: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not 0.0 <= alternative_share <= 1.0:
        raise ValueError("alternative share must be between 0 and 1")
    if not 0 <= minimum_source_midi <= maximum_source_midi <= 127:
        raise ValueError("source MIDI range must be ordered and inside 0..127")
    families = sorted({str(value).strip().lower() for value in source_families if str(value).strip()})
    if not families:
        raise ValueError("At least one source family is required")
    if preserve_base_window_counts and preserve_base_onset_counts:
        raise ValueError("Choose either window-slot or onset-slot preservation, not both")
    if not 0.005 <= onset_slot_window_seconds <= 0.12:
        raise ValueError("onset slot window must be between 0.005 and 0.12 seconds")
    model = alternative.get("selectionModel")
    if not isinstance(model, dict):
        raise ValueError("The alternate profile has no selectionModel")
    names = model.get("featureNames") or []
    weights = model.get("weights") or []
    means = model.get("means") or []
    scales = model.get("scales") or []
    if not names or not (len(names) == len(weights) == len(means) == len(scales)):
        raise ValueError("The alternate selection model has inconsistent arrays")

    output = copy.deepcopy(base)
    output["id"] = profile_id
    output["createdAt"] = created_at or datetime.now(timezone.utc).isoformat()
    output.setdefault("decoder", {})["conditionalSelectionBlend"] = {
        "enabled": True,
        "policy": "threshold-centered-conditional-logit-blend-v1",
        "alternativeShare": round(float(alternative_share), 6),
        "minimumSourceMidi": int(minimum_source_midi),
        "maximumSourceMidi": int(maximum_source_midi),
        "sourceFamilies": families,
        "preserveBaseWindowCounts": bool(preserve_base_window_counts),
        "preserveBaseOnsetCounts": bool(preserve_base_onset_counts),
        "onsetSlotWindowSeconds": round(float(onset_slot_window_seconds), 6),
        "selectionModel": copy.deepcopy(model),
    }
    output.setdefault("training", {})["conditionalSelectionUpgrade"] = {
        "schema": "polymath-conditional-selection-promotion-v1",
        "alternativeProfile": alternative_profile_path,
        "alternativeProfileId": alternative.get("id"),
        "alternativeProfileSha256": alternative.get("profileSha256"),
        "validationReport": validation_report_path,
        "alternativeShare": round(float(alternative_share), 6),
        "sourceFamilies": families,
        "minimumSourceMidi": int(minimum_source_midi),
        "maximumSourceMidi": int(maximum_source_midi),
        "preserveBaseWindowCounts": bool(preserve_base_window_counts),
        "preserveBaseOnsetCounts": bool(preserve_base_onset_counts),
        "onsetSlotWindowSeconds": round(float(onset_slot_window_seconds), 6),
        "commercialUseAllowed": False,
        "decision": "EXPERIMENTAL_ONLY",
    }
    output.pop("profileSha256", None)
    output["profileSha256"] = profile_sha256(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--alternative-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--alternative-share", type=float, required=True)
    parser.add_argument("--source-families", default="guitar")
    parser.add_argument("--minimum-source-midi", type=int, default=60)
    parser.add_argument("--maximum-source-midi", type=int, default=84)
    parser.add_argument(
        "--preserve-base-window-counts",
        action="store_true",
        help="Rerank only existing family/register slots instead of adding density.",
    )
    parser.add_argument(
        "--preserve-base-onset-counts",
        action="store_true",
        help="Rerank pitches only inside each existing onset gesture.",
    )
    parser.add_argument("--onset-slot-window-seconds", type=float, default=0.035)
    parser.add_argument("--validation-report")
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    alternative_path = Path(args.alternative_profile).resolve()
    output_path = Path(args.output).resolve()
    profile = apply_conditional_selector(
        load_json(base_path),
        load_json(alternative_path),
        profile_id=args.profile_id,
        alternative_profile_path=str(alternative_path),
        alternative_share=args.alternative_share,
        source_families=args.source_families.split(","),
        minimum_source_midi=args.minimum_source_midi,
        maximum_source_midi=args.maximum_source_midi,
        preserve_base_window_counts=args.preserve_base_window_counts,
        preserve_base_onset_counts=args.preserve_base_onset_counts,
        onset_slot_window_seconds=args.onset_slot_window_seconds,
        validation_report_path=(
            str(Path(args.validation_report).resolve())
            if args.validation_report
            else None
        ),
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
                "conditionalSelectionBlend": profile["decoder"][
                    "conditionalSelectionBlend"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
