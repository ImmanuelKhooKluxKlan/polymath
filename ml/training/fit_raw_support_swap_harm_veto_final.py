"""Fit the frozen research-only harm veto after policy selection.

The meta-model is trained from out-of-fold selector and benefit scores.  Its
only runtime authority is to set an established benefit score to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .ablate_raw_support_swap_harm_veto_models import factories, prepare_song
from .apply_raw_support_recovery import load_json
from .raw_support_swap_benefit import SWAP_BENEFIT_FEATURE_NAMES


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--benefit-profiles-root", type=Path, required=True)
    parser.add_argument("--model-name", default="harm-extra-trees-d8-l10")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0x4841524D)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    rows = manifest.get("songs") or []
    if len(rows) < 4:
        raise ValueError("At least four complete songs are required.")
    selector_root = args.selector_profiles_root.resolve()
    benefit_root = args.benefit_profiles_root.resolve()
    songs = [
        prepare_song(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            load_json(benefit_root / str(row["id"]) / "profile.json"),
        )
        for row in rows
    ]
    available = factories(int(args.seed))
    if args.model_name not in available:
        raise ValueError(f"Unknown harm model: {args.model_name}")
    model = available[args.model_name]()
    features = np.vstack([song["features"] for song in songs])
    labels = np.concatenate([song["harmful"] for song in songs])
    weights = np.concatenate([song["harmWeights"] for song in songs])
    fit_arguments = (
        {f"{model.steps[-1][0]}__sample_weight": weights}
        if hasattr(model, "steps")
        else {"sample_weight": weights}
    )
    model.fit(features, labels, **fit_arguments)
    fitted = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)

    model_output = args.model_output.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_output.with_name(model_output.name + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(model_output)
    metadata = {
        "schema": "polymath-swap-harm-veto-final-v1",
        "id": "phase51-native-register-harm-veto-v001-research",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "researchOnly": True,
        "notProductionApproved": True,
        "modelName": args.model_name,
        "modelPath": str(model_output),
        "modelSha256": sha256(model_output),
        "featureNames": [*SWAP_BENEFIT_FEATURE_NAMES, "baseBenefitProbability"],
        "featureCount": int(features.shape[1]),
        "target": "utility < 0",
        "runtimeAuthority": "veto only; may replace a base score with zero but may not raise it",
        "training": {
            "manifest": str(manifest_path),
            "songs": [song["id"] for song in songs],
            "examples": int(len(labels)),
            "harmfulExamples": int(labels.sum()),
            "selectorEvidence": str(selector_root),
            "benefitEvidence": str(benefit_root),
            "stackingBoundary": "selector and benefit inputs are whole-song out-of-fold for each training song",
            "registerShiftsSeen": [0, 12],
        },
        "fitDiagnosticsOnlyNotGeneralizationEvidence": {
            "averagePrecision": round(
                float(average_precision_score(labels, fitted, sample_weight=weights)), 6
            ),
            "rocAuc": round(
                float(roc_auc_score(labels, fitted, sample_weight=weights)), 6
            ),
        },
    }
    atomic_json(args.metadata_output.resolve(), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
