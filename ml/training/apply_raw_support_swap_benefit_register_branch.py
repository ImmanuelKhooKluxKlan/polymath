"""Apply the frozen research-only register-branch swap policy.

Native-register proposals use the tiny MLP.  Octave-lift proposals use a
calibrated blend of that MLP and a song-relative context ranker.  Only pitch
is replaced; onset, duration, velocity, role, and gesture size remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .ablate_raw_support_swap_benefit_context_models import context_rows, proposal_key
from .apply_raw_support_recovery import load_json
from .compose_raw_support_swap_benefit_branch_scores import branch_score
from .raw_support_swap_benefit import (
    apply_scored_proposals,
    generate_swap_proposals,
    score_feature_rows,
)


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
    parser.add_argument("--context-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate.resolve())
    source = load_json(args.source.resolve())
    selector = load_json(args.selector_profile.resolve())
    benefit = load_json(args.benefit_profile.resolve())
    proposals, generation = generate_swap_proposals(
        candidate,
        source,
        selector,
        source_radius=0.18,
        minimum_source_midi=0,
        maximum_source_midi=59,
        minimum_output_midi=33,
        maximum_output_midi=71,
        register_shifts=(0, 12),
    )
    mlp_model = benefit.get("model") or benefit.get("swapBenefitModel")
    if not isinstance(mlp_model, dict):
        raise ValueError("Benefit profile has no tiny-MLP model")
    mlp_scores = np.asarray(
        score_feature_rows([proposal["features"] for proposal in proposals], mlp_model),
        dtype=np.float64,
    )
    lookup = context_rows(proposals)
    augmented = np.asarray(
        [lookup[proposal_key(proposal)] for proposal in proposals], dtype=np.float64
    )
    context_model = joblib.load(args.context_model.resolve())
    context_scores = np.asarray(context_model.predict_proba(augmented)[:, 1], dtype=np.float64)
    right_scores = 0.35 * mlp_scores + 0.65 * context_scores
    combined: list[float] = []
    branch_eligible = {"native": 0, "octaveLift": 0}
    for index, proposal in enumerate(proposals):
        if int(proposal["registerShift"]) == 0:
            score = branch_score(
                float(mlp_scores[index]),
                proposal,
                threshold=0.8,
                selector_minimum=0.4,
                gain_minimum=0.45,
                scale=1.0,
            )
            if score >= 0.8:
                branch_eligible["native"] += 1
        else:
            score = branch_score(
                float(right_scores[index]),
                proposal,
                threshold=0.35,
                selector_minimum=0.4,
                gain_minimum=0.3,
                scale=0.5,
            )
            if score >= 0.8:
                branch_eligible["octaveLift"] += 1
        combined.append(score)

    profile = {
        "id": "phase48-register-branch-v002-research",
        "researchOnly": True,
        "nativeModel": benefit.get("id"),
        "contextModel": str(args.context_model.resolve()),
    }
    output, application = apply_scored_proposals(
        candidate,
        proposals,
        combined,
        profile,
        {**generation, "branchEligible": branch_eligible},
        threshold=0.8,
        minimum_selector_probability=0.0,
        minimum_probability_gain=-99.0,
        maximum_replacements_per_gesture=1,
    )
    report = {
        "schema": "polymath-register-branch-application-v1",
        "researchOnly": True,
        "proposalCount": len(proposals),
        "branchEligible": branch_eligible,
        "application": application,
        "policy": {
            "native": {"register": 0, "threshold": 0.8, "selectorMinimum": 0.4, "gainMinimum": 0.45},
            "octaveLift": {
                "register": 12,
                "mlpWeight": 0.35,
                "contextWeight": 0.65,
                "threshold": 0.35,
                "selectorMinimum": 0.4,
                "gainMinimum": 0.3,
                "commonMarginScale": 0.5,
            },
            "commonThreshold": 0.8,
            "maximumReplacementsPerGesture": 1,
        },
    }
    atomic_json(args.output.resolve(), output)
    atomic_json(args.diagnostics.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
