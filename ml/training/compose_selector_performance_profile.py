"""Compose a frozen note selector with a separately validated performance layer.

The selector continues to decide which notes exist.  Only the gesture-dynamics
block (velocity, written duration, and performance gain calibration) is copied
from the performance profile.  This keeps the experiment auditable and avoids
silently importing source-specific routing rules from the latter profile.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "polymath-piano-arranger-profile-v1"


def canonical_profile_hash(profile: dict[str, Any]) -> str:
    payload = copy.deepcopy(profile)
    payload.pop("profileSha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compose_selector_performance_profile(
    selector: dict[str, Any],
    performance: dict[str, Any],
    *,
    profile_id: str,
    created_at: str | None = None,
    conditional_route: dict[str, float | int] | None = None,
    monophonic_vocal_route: dict[str, float | int | str] | None = None,
) -> dict[str, Any]:
    if selector.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Selector profile has an unsupported schema")
    if performance.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Performance profile has an unsupported schema")
    gesture = (performance.get("decoder") or {}).get("gestureDynamics")
    if not isinstance(gesture, dict) or not gesture.get("enabled"):
        raise ValueError("Performance profile must contain enabled decoder.gestureDynamics")
    if (selector.get("decoder") or {}).get("defaultPipeline"):
        raise ValueError("Selector profile must use the learned-selector route")
    if not selector.get("selectionModel"):
        raise ValueError("Selector profile has no selectionModel")

    output = copy.deepcopy(selector)
    output.pop("profileSha256", None)
    output["id"] = profile_id
    output["createdAt"] = created_at or datetime.now(timezone.utc).isoformat()
    output.setdefault("decoder", {})["gestureDynamics"] = copy.deepcopy(gesture)
    routing_changed = bool(conditional_route)
    if conditional_route:
        fallback_decoder = copy.deepcopy(performance.get("decoder") or {})
        fallback_decoder.pop("conditionalLearnedRoute", None)
        if not fallback_decoder.get("defaultPipeline"):
            raise ValueError("Conditional fallback performance profile must use defaultPipeline")
        output["decoder"]["conditionalLearnedRoute"] = {
            "enabled": True,
            "maximumVoiceRatio": float(conditional_route["maximumVoiceRatio"]),
            "minimumBassRatio": float(conditional_route["minimumBassRatio"]),
            "maximumPianoRatio": float(conditional_route["maximumPianoRatio"]),
            "minimumSourceNotes": int(conditional_route["minimumSourceNotes"]),
            "fallbackProfileId": str(performance.get("id") or "default-fallback"),
            "fallbackDecoder": fallback_decoder,
        }
        if monophonic_vocal_route:
            output["decoder"]["conditionalLearnedRoute"][
                "monophonicVocalRoute"
            ] = {
                "enabled": True,
                "minimumVoiceRatio": float(
                    monophonic_vocal_route["minimumVoiceRatio"]
                ),
                "maximumBassRatio": float(
                    monophonic_vocal_route["maximumBassRatio"]
                ),
                "maximumPianoRatio": float(
                    monophonic_vocal_route["maximumPianoRatio"]
                ),
                "minimumSourceNotes": int(
                    monophonic_vocal_route["minimumSourceNotes"]
                ),
                "fallbackProfileId": str(
                    monophonic_vocal_route.get(
                        "fallbackProfileId", "monophonic-vocal-default"
                    )
                ),
                # Keep baseline note durations and physical holds on this
                # route.  The separately learned performance layer is for
                # full mixes and regressed the opened monophonic diagnostic.
                "fallbackDecoder": {
                    "defaultPipeline": True,
                    "maximumSourcePianoRatioForProfile": float(
                        monophonic_vocal_route.get("maximumPianoRatio", 0.02)
                    ),
                    "monophonicVocalCleanup": {
                        "enabled": True,
                        "maximumFloorVelocity": float(
                            monophonic_vocal_route["maximumFloorVelocity"]
                        ),
                        "minimumRunNotes": int(
                            monophonic_vocal_route["minimumRunNotes"]
                        ),
                        "maximumRunGapSeconds": float(
                            monophonic_vocal_route["maximumRunGapSeconds"]
                        ),
                        "onsetDelaySeconds": float(
                            monophonic_vocal_route["onsetDelaySeconds"]
                        ),
                    },
                },
            }
    elif monophonic_vocal_route:
        raise ValueError(
            "monophonic_vocal_route requires conditional_route to be enabled"
        )
    output.setdefault("training", {})["selectorPerformanceComposition"] = {
        "schema": "polymath-selector-performance-composition-v1",
        "selectorProfileId": str(selector.get("id") or ""),
        "selectorProfileSha256": str(selector.get("profileSha256") or ""),
        "performanceProfileId": str(performance.get("id") or ""),
        "performanceProfileSha256": str(performance.get("profileSha256") or ""),
        "copiedField": "decoder.gestureDynamics",
        "selectionModelChanged": False,
        "durationModelChanged": False,
        "routingChanged": routing_changed,
    }
    output["profileSha256"] = canonical_profile_hash(output)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--conditional-learned-route", action="store_true")
    parser.add_argument("--maximum-voice-ratio", type=float, default=0.02)
    parser.add_argument("--minimum-bass-ratio", type=float, default=0.50)
    parser.add_argument("--maximum-piano-ratio", type=float, default=0.08)
    parser.add_argument("--minimum-source-notes", type=int, default=64)
    parser.add_argument("--monophonic-vocal-route", action="store_true")
    parser.add_argument("--minimum-monophonic-voice-ratio", type=float, default=0.95)
    parser.add_argument("--maximum-monophonic-bass-ratio", type=float, default=0.02)
    parser.add_argument("--maximum-monophonic-piano-ratio", type=float, default=0.02)
    parser.add_argument("--minimum-monophonic-source-notes", type=int, default=32)
    parser.add_argument("--maximum-floor-velocity", type=float, default=0.46)
    parser.add_argument("--minimum-floor-run-notes", type=int, default=6)
    parser.add_argument("--maximum-floor-run-gap-seconds", type=float, default=0.35)
    parser.add_argument("--monophonic-onset-delay-seconds", type=float, default=0.02)
    args = parser.parse_args()

    selector = _read_json(args.selector.resolve())
    performance = _read_json(args.performance.resolve())
    profile = compose_selector_performance_profile(
        selector,
        performance,
        profile_id=args.profile_id,
        conditional_route=(
            {
                "maximumVoiceRatio": args.maximum_voice_ratio,
                "minimumBassRatio": args.minimum_bass_ratio,
                "maximumPianoRatio": args.maximum_piano_ratio,
                "minimumSourceNotes": args.minimum_source_notes,
            }
            if args.conditional_learned_route
            else None
        ),
        monophonic_vocal_route=(
            {
                "minimumVoiceRatio": args.minimum_monophonic_voice_ratio,
                "maximumBassRatio": args.maximum_monophonic_bass_ratio,
                "maximumPianoRatio": args.maximum_monophonic_piano_ratio,
                "minimumSourceNotes": args.minimum_monophonic_source_notes,
                "fallbackProfileId": "monophonic-vocal-default-v043",
                "maximumFloorVelocity": args.maximum_floor_velocity,
                "minimumRunNotes": args.minimum_floor_run_notes,
                "maximumRunGapSeconds": args.maximum_floor_run_gap_seconds,
                "onsetDelaySeconds": args.monophonic_onset_delay_seconds,
            }
            if args.monophonic_vocal_route
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "polymath-selector-performance-composition-report-v1",
        "output": str(args.output.resolve()),
        "profileId": profile["id"],
        "profileSha256": profile["profileSha256"],
        "selector": str(args.selector.resolve()),
        "performance": str(args.performance.resolve()),
        "composition": profile["training"]["selectorPerformanceComposition"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
