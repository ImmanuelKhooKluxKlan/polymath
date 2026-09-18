"""Apply the frozen native-register benefit policy with a harm-only veto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .apply_raw_support_recovery import load_json
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


def harm_veto_scores(
    base_scores: np.ndarray, harm_scores: np.ndarray, threshold: float
) -> np.ndarray:
    if base_scores.shape != harm_scores.shape:
        raise ValueError(f"Score shape mismatch: {base_scores.shape} != {harm_scores.shape}")
    return np.where(harm_scores >= threshold, 0.0, base_scores)


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selector-profile", type=Path, required=True)
    parser.add_argument("--benefit-profile", type=Path, required=True)
    parser.add_argument("--harm-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--harm-threshold", type=float, default=0.45)
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
        register_shifts=(0,),
    )
    mlp_model = benefit.get("model") or benefit.get("swapBenefitModel")
    if not isinstance(mlp_model, dict):
        raise ValueError("Benefit profile has no tiny-MLP model")
    if proposals:
        feature_rows = np.asarray(
            [proposal["features"] for proposal in proposals], dtype=np.float64
        )
        base_scores = np.asarray(
            score_feature_rows(feature_rows.tolist(), mlp_model), dtype=np.float64
        )
        harm_features = np.column_stack((feature_rows, base_scores))
        harm_model = joblib.load(args.harm_model.resolve())
        harm_scores = np.asarray(
            harm_model.predict_proba(harm_features)[:, 1], dtype=np.float64
        )
    else:
        base_scores = np.empty(0, dtype=np.float64)
        harm_scores = np.empty(0, dtype=np.float64)
    threshold = float(args.harm_threshold)
    combined = harm_veto_scores(base_scores, harm_scores, threshold)
    veto_mask = harm_scores >= threshold
    eligible_before = (
        (base_scores >= 0.8)
        & np.asarray(
            [float(item["sourceProbability"]) >= 0.4 for item in proposals],
            dtype=bool,
        )
        & np.asarray(
            [float(item["probabilityGain"]) >= 0.45 for item in proposals],
            dtype=bool,
        )
    )
    eligible_audit = [
        {
            "groupTime": round(float(proposal["groupTime"]), 6),
            "incumbentIndex": int(proposal["incumbentIndex"]),
            "incumbentMidi": int(proposal["incumbentMidi"]),
            "targetMidi": int(proposal["targetMidi"]),
            "sourceFamily": str(proposal.get("sourceFamily") or ""),
            "role": str(proposal.get("role") or ""),
            "baseBenefitProbability": round(float(base_scores[index]), 8),
            "harmProbability": round(float(harm_scores[index]), 8),
            "vetoed": bool(veto_mask[index]),
        }
        for index, proposal in enumerate(proposals)
        if bool(eligible_before[index])
    ]
    profile = {
        "id": "phase51-native-register-harm-veto-v001-research",
        "researchOnly": True,
        "baseBenefitModel": benefit.get("id"),
        "harmModel": str(args.harm_model.resolve()),
    }
    output, application = apply_scored_proposals(
        candidate,
        proposals,
        combined.tolist(),
        profile,
        {
            **generation,
            "vetoedProposals": int(veto_mask.sum()),
            "vetoedBaseEligibleProposals": int((veto_mask & eligible_before).sum()),
        },
        threshold=0.8,
        minimum_selector_probability=0.4,
        minimum_probability_gain=0.45,
        maximum_replacements_per_gesture=1,
    )
    report = {
        "schema": "polymath-native-register-harm-veto-application-v1",
        "researchOnly": True,
        "proposalCount": len(proposals),
        "vetoedProposals": int(veto_mask.sum()),
        "vetoedBaseEligibleProposals": int((veto_mask & eligible_before).sum()),
        "baseEligibleProposalAudit": eligible_audit,
        "application": application,
        "policy": {
            "registerShifts": [0],
            "baseBenefitThreshold": 0.8,
            "selectorMinimum": 0.4,
            "probabilityGainMinimum": 0.45,
            "harmVetoThreshold": threshold,
            "maximumReplacementsPerGesture": 1,
        },
    }
    atomic_json(args.output.resolve(), output)
    atomic_json(args.diagnostics.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
