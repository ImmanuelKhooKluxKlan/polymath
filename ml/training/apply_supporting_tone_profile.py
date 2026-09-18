"""Install a learned supporting-tone ranker after hand/rhythm assignment."""

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


def apply_supporting_tone_ranker(
    base: dict[str, Any],
    alternative: dict[str, Any],
    *,
    profile_id: str,
    alternative_path: str,
    alternative_share: float,
    chord_completion_share: float,
    source_families: list[str],
    minimum_source_midi: int,
    maximum_source_midi: int,
    validation_report: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not 0.0 <= alternative_share <= 1.0:
        raise ValueError("alternative share must be between zero and one")
    if not 0.0 <= chord_completion_share <= 1.0:
        raise ValueError("chord completion share must be between zero and one")
    if not 0 <= minimum_source_midi <= maximum_source_midi <= 127:
        raise ValueError("source MIDI range must be ordered and inside 0..127")
    families = sorted(
        {str(value).strip().lower() for value in source_families if str(value).strip()}
    )
    if not families:
        raise ValueError("at least one source family is required")
    model = alternative.get("selectionModel")
    if not isinstance(model, dict):
        raise ValueError("alternative profile has no selectionModel")
    names = model.get("featureNames") or []
    if not names or not (
        len(names)
        == len(model.get("weights") or [])
        == len(model.get("means") or [])
        == len(model.get("scales") or [])
    ):
        raise ValueError("alternative selection model has inconsistent arrays")

    output = copy.deepcopy(base)
    output["id"] = profile_id
    output["createdAt"] = created_at or datetime.now(timezone.utc).isoformat()
    config = output.setdefault("decoder", {}).setdefault(
        "leftHandAccompaniment", {}
    )
    if not config.get("enabled"):
        raise ValueError("base profile has no enabled left-hand accompaniment stage")
    config["supportingToneSelectionModel"] = copy.deepcopy(model)
    config["supportingToneAlternativeShare"] = round(float(alternative_share), 6)
    config["supportingToneChordCompletionShare"] = round(
        float(chord_completion_share), 6
    )
    config["supportingToneSourceFamilies"] = families
    config["supportingToneMinimumSourceMidi"] = int(minimum_source_midi)
    config["supportingToneMaximumSourceMidi"] = int(maximum_source_midi)
    output.setdefault("training", {})["supportingToneUpgrade"] = {
        "schema": "polymath-supporting-tone-promotion-v1",
        "alternativeProfile": alternative_path,
        "alternativeProfileId": alternative.get("id"),
        "alternativeProfileSha256": alternative.get("profileSha256"),
        "validationReport": validation_report,
        "alternativeShare": round(float(alternative_share), 6),
        "chordCompletionShare": round(float(chord_completion_share), 6),
        "sourceFamilies": families,
        "minimumSourceMidi": int(minimum_source_midi),
        "maximumSourceMidi": int(maximum_source_midi),
        "rightHandFrozen": True,
        "onsetQuotasFrozen": True,
        "bassAnchorFrozenDuringSupportingToneRanking": True,
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
    parser.add_argument("--chord-completion-share", type=float, default=0.0)
    parser.add_argument("--source-families", default="guitar")
    parser.add_argument("--minimum-source-midi", type=int, default=48)
    parser.add_argument("--maximum-source-midi", type=int, default=76)
    parser.add_argument("--validation-report")
    args = parser.parse_args()
    base_path = Path(args.base).resolve()
    alternative_path = Path(args.alternative_profile).resolve()
    output_path = Path(args.output).resolve()
    profile = apply_supporting_tone_ranker(
        load_json(base_path),
        load_json(alternative_path),
        profile_id=args.profile_id,
        alternative_path=str(alternative_path),
        alternative_share=args.alternative_share,
        chord_completion_share=args.chord_completion_share,
        source_families=args.source_families.split(","),
        minimum_source_midi=args.minimum_source_midi,
        maximum_source_midi=args.maximum_source_midi,
        validation_report=(
            str(Path(args.validation_report).resolve())
            if args.validation_report
            else None
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "profileId": profile["id"],
                "profileSha256": profile["profileSha256"],
                "supportingTone": profile["decoder"]["leftHandAccompaniment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
