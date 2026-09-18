"""Train whole-song holdout veto models for harmful pitch swaps.

The benefit ranker asks whether a replacement can help.  This second model
asks the asymmetric safety question: is there evidence the replacement will
destroy an exact or pitch-class match?  It may only veto an existing proposal;
it can never promote a new one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples
from .raw_support_swap_benefit import model_probabilities_numpy


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def factories(seed: int) -> dict[str, Callable[[], Any]]:
    models: dict[str, Callable[[], Any]] = {
        "harm-logistic-c0.1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2500, solver="lbfgs", random_state=seed),
        ),
        "harm-logistic-c1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2500, solver="lbfgs", random_state=seed),
        ),
    }
    for depth, leaf in ((4, 10), (6, 10), (8, 10), (8, 20)):
        models[f"harm-extra-trees-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    models["harm-random-forest-d8-l10"] = lambda: RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    for leaves, minimum_leaf in ((7, 10), (15, 10), (15, 20), (31, 20)):
        models[f"harm-hist-gb-n{leaves}-l{minimum_leaf}"] = lambda n=leaves, l=minimum_leaf: HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=240,
            max_leaf_nodes=n,
            min_samples_leaf=l,
            l2_regularization=3.0,
            random_state=seed,
        )
    return models


def prepare_song(
    row: dict[str, Any], selector: dict[str, Any], benefit: dict[str, Any]
) -> dict[str, Any]:
    song = build_song_examples(
        row,
        selector,
        source_radius=0.18,
        minimum_source_midi=0,
        maximum_source_midi=59,
        minimum_output_midi=33,
        maximum_output_midi=71,
        register_shifts=(0, 12),
    )
    base_scores = model_probabilities_numpy(song["features"], benefit["model"])
    harmful = (song["utilities"] < 0).astype(np.int64)
    harmful_count = int(harmful.sum())
    safe_count = len(harmful) - harmful_count
    if harmful_count < 10 or safe_count < 10:
        raise ValueError(f"{song['id']} lacks harm-veto classes")
    weight = max(0.01, float(row.get("weight", 1.0)))
    weights = np.asarray(
        [
            weight
            * len(harmful)
            * (0.5 / harmful_count if label else 0.5 / safe_count)
            * (1.0 + 0.12 * min(8.0, abs(float(utility))))
            for label, utility in zip(harmful, song["utilities"])
        ],
        dtype=np.float64,
    )
    return {
        **song,
        "features": np.column_stack((song["features"], base_scores)),
        "harmful": harmful,
        "harmWeights": weights,
        "baseScores": base_scores,
    }


def summarize(
    name: str, songs: list[dict[str, Any]], probabilities: dict[str, np.ndarray]
) -> dict[str, Any]:
    labels = np.concatenate([song["harmful"] for song in songs])
    scores = np.concatenate([probabilities[song["id"]] for song in songs])
    weights = np.concatenate([song["harmWeights"] for song in songs])
    per_song: dict[str, Any] = {}
    for song in songs:
        y = song["harmful"]
        p = probabilities[song["id"]]
        per_song[song["id"]] = {
            "examples": len(y),
            "harmful": int(y.sum()),
            "averagePrecision": round(float(average_precision_score(y, p)), 6),
            "rocAuc": round(float(roc_auc_score(y, p)), 6),
            "harmfulProbabilityMedian": round(float(np.median(p[y == 1])), 6),
            "safeProbabilityMedian": round(float(np.median(p[y == 0])), 6),
        }
    return {
        "name": name,
        "wholeSongOutOfFold": True,
        "aggregate": {
            "averagePrecision": round(
                float(average_precision_score(labels, scores, sample_weight=weights)), 6
            ),
            "rocAuc": round(float(roc_auc_score(labels, scores, sample_weight=weights)), 6),
            "minimumSongAveragePrecision": round(
                min(float(row["averagePrecision"]) for row in per_song.values()), 6
            ),
            "meanSongAveragePrecision": round(
                float(np.mean([row["averagePrecision"] for row in per_song.values()])), 6
            ),
        },
        "songs": per_song,
        "probabilities": {
            song["id"]: [round(float(value), 8) for value in probabilities[song["id"]]]
            for song in songs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--benefit-profiles-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0x4841524D)
    args = parser.parse_args()

    manifest = load_json(args.manifest.resolve())
    selector_root = args.selector_profiles_root.resolve()
    benefit_root = args.benefit_profiles_root.resolve()
    songs = [
        prepare_song(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            load_json(benefit_root / str(row["id"]) / "profile.json"),
        )
        for row in manifest.get("songs") or []
    ]
    results: list[dict[str, Any]] = []
    for model_index, (name, factory) in enumerate(factories(int(args.seed)).items()):
        probabilities: dict[str, np.ndarray] = {}
        for held_out in songs:
            training = [song for song in songs if song is not held_out]
            features = np.vstack([song["features"] for song in training])
            labels = np.concatenate([song["harmful"] for song in training])
            weights = np.concatenate([song["harmWeights"] for song in training])
            model = factory()
            fit_arguments = (
                {f"{model.steps[-1][0]}__sample_weight": weights}
                if hasattr(model, "steps")
                else {"sample_weight": weights}
            )
            model.fit(features, labels, **fit_arguments)
            probabilities[held_out["id"]] = model.predict_proba(held_out["features"])[:, 1]
        result = summarize(name, songs, probabilities)
        results.append(result)
        print(json.dumps({"completed": model_index + 1, "model": name, **result["aggregate"]}), flush=True)

    results.sort(
        key=lambda row: (
            float(row["aggregate"]["minimumSongAveragePrecision"]),
            float(row["aggregate"]["meanSongAveragePrecision"]),
            float(row["aggregate"]["averagePrecision"]),
        ),
        reverse=True,
    )
    report = {
        "schema": "polymath-swap-harm-veto-ablation-v1",
        "evidenceBoundary": "Every harmfulness score is whole-song out-of-fold; the benefit score appended as a feature is also out-of-fold or from a model that never trained on that auxiliary song.",
        "target": "utility < 0",
        "manifest": str(args.manifest.resolve()),
        "models": results,
        "baseScores": {
            song["id"]: [round(float(value), 8) for value in song["baseScores"]]
            for song in songs
        },
        "ranking": [
            {"rank": index + 1, "name": row["name"], **row["aggregate"]}
            for index, row in enumerate(results)
        ],
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "winner": report["ranking"][0]}, indent=2))


if __name__ == "__main__":
    main()
