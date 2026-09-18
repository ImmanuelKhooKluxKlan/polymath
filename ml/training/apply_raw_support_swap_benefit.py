"""Apply one frozen post-arranger swap-benefit policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .apply_raw_support_recovery import load_json
from .raw_support_swap_benefit import apply_benefit_swaps


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selector-profile", type=Path, required=True)
    parser.add_argument("--benefit-profile", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    policy_payload = load_json(args.policy.resolve())
    policy = policy_payload.get("policy") or policy_payload
    source_range = policy.get("sourceMidiRange") or [0, 59]
    output_range = policy.get("outputMidiRange") or [33, 71]
    output, diagnostics = apply_benefit_swaps(
        load_json(args.candidate.resolve()),
        load_json(args.source.resolve()),
        load_json(args.selector_profile.resolve()),
        load_json(args.benefit_profile.resolve()),
        threshold=float(policy["benefitThreshold"]),
        source_radius=float(policy["sourceRadiusSeconds"]),
        minimum_selector_probability=float(policy["minimumSelectorProbability"]),
        minimum_probability_gain=float(policy["minimumProbabilityGain"]),
        maximum_replacements_per_gesture=int(policy.get("maximumReplacementsPerGesture", 1)),
        minimum_source_midi=int(source_range[0]),
        maximum_source_midi=int(source_range[1]),
        minimum_output_midi=int(output_range[0]),
        maximum_output_midi=int(output_range[1]),
        register_shifts=tuple(int(value) for value in policy.get("registerShifts") or [0]),
        source_families={str(value).lower() for value in policy.get("sourceFamilies") or []} or None,
        replace_hands={str(value).lower() for value in policy.get("replaceHands") or ["left"]},
        replace_roles={str(value).lower() for value in policy.get("replaceRoles") or ["bass", "harmony"]},
    )
    atomic_json(args.output.resolve(), output)
    atomic_json(
        args.report.resolve(),
        {
            "schema": "polymath-raw-support-swap-benefit-application-v1",
            "frozenPolicy": str(args.policy.resolve()),
            "candidate": str(args.candidate.resolve()),
            "source": str(args.source.resolve()),
            "selectorProfile": str(args.selector_profile.resolve()),
            "benefitProfile": str(args.benefit_profile.resolve()),
            "output": str(args.output.resolve()),
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
