"""Fit one frozen research context ranker on all development songs.

The joblib artifact is intentionally research-only.  Promotion would require
exporting an audited, runtime-safe representation rather than loading a pickle
inside the public worker.
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

from .ablate_raw_support_swap_benefit_context_models import augment_song, factories
from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--inference-selector-profile", type=Path, required=True)
    parser.add_argument("--model-name", default="context-hist-gb-n31-l20")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0x434F4E54)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    selector_root = args.selector_profiles_root.resolve()
    inference_selector_path = args.inference_selector_profile.resolve()
    manifest = load_json(manifest_path)
    songs: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        selector = load_json(selector_root / str(row["id"]) / "profile.json")
        song = build_song_examples(
            row,
            selector,
            source_radius=max(0.01, float(args.source_radius_seconds)),
            minimum_source_midi=0,
            maximum_source_midi=59,
            minimum_output_midi=33,
            maximum_output_midi=71,
            register_shifts=(0, 12),
        )
        songs.append(
            augment_song(song, row, selector, max(0.01, float(args.source_radius_seconds)))
        )
    available = factories(int(args.seed))
    if args.model_name not in available:
        raise ValueError(f"Unknown model {args.model_name!r}; choices: {sorted(available)}")
    features = np.vstack([song["features"] for song in songs])
    labels = np.concatenate([song["labels"] for song in songs]).astype(np.int64)
    weights = np.concatenate([song["weights"] for song in songs])
    model = available[args.model_name]()
    fit_arguments = (
        {f"{model.steps[-1][0]}__sample_weight": weights}
        if hasattr(model, "steps")
        else {"sample_weight": weights}
    )
    model.fit(features, labels, **fit_arguments)

    output_model = args.output_model.resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_model.with_name(output_model.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(output_model)
    metadata = {
        "schema": "polymath-swap-benefit-context-research-model-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modelName": args.model_name,
        "researchOnly": True,
        "runtimeSafeForPublicWorker": False,
        "manifest": str(manifest_path),
        "manifestSha256": sha256(manifest_path),
        "stackingEvidence": "Each development song used selector scores from a selector trained without that complete song.",
        "selectorProfilesRoot": str(selector_root),
        "inferenceSelectorProfile": str(inference_selector_path),
        "inferenceSelectorProfileSha256": sha256(inference_selector_path),
        "songs": [song["id"] for song in songs],
        "examples": int(len(labels)),
        "positives": int(labels.sum()),
        "featureCount": int(features.shape[1]),
        "model": str(output_model),
        "modelSha256": sha256(output_model),
    }
    atomic_json(args.output_metadata.resolve(), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
