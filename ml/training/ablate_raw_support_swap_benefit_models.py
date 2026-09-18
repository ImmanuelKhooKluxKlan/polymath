"""Compare proposal-ranker model families with whole-song holdouts.

This script is deliberately research-only and may use scikit-learn.  It never
ships a pickle or touches production.  Each model is trained on complete songs
other than the song it scores, and its main diagnostics are precision among
the first few proposals—the regime used by the conservative decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples, combine_songs
from .raw_support_swap_benefit import model_probabilities_numpy


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_factories(seed: int) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    for c_value in (0.1, 1.0, 10.0):
        factories[f"logistic-c{c_value:g}"] = lambda c=c_value: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed),
        )
    for depth, leaf in ((4, 10), (6, 10), (8, 10), (8, 20), (10, 20)):
        factories[f"random-forest-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: RandomForestClassifier(
            n_estimators=500,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt",
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )
        factories[f"extra-trees-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: ExtraTreesClassifier(
            n_estimators=500,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt",
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )
    for leaves, minimum_leaf, l2 in ((7, 10, 1.0), (15, 10, 1.0), (15, 20, 1.0), (15, 30, 3.0), (31, 20, 3.0)):
        factories[f"hist-gb-n{leaves}-l{minimum_leaf}-r{l2:g}"] = lambda n=leaves, l=minimum_leaf, r=l2: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=n,
            min_samples_leaf=l,
            l2_regularization=r,
            random_state=seed,
        )
    for depth, leaf in ((1, 10), (2, 10), (2, 20), (3, 20)):
        factories[f"gradient-boost-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=d,
            min_samples_leaf=l,
            subsample=0.85,
            random_state=seed,
        )
    return factories


def top_metrics(labels: np.ndarray, probabilities: np.ndarray, count: int) -> dict[str, Any]:
    kept = min(count, len(labels))
    indices = np.argsort(-probabilities, kind="stable")[:kept]
    positives = int(labels[indices].sum())
    return {
        "selected": kept,
        "truePositives": positives,
        "precision": round(positives / max(1, kept), 6),
        "positiveRecall": round(positives / max(1, int(labels.sum())), 6),
    }


def summarize_model(
    name: str,
    songs: list[dict[str, Any]],
    probabilities_by_song: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for song in songs:
        labels = song["labels"]
        probabilities = np.clip(probabilities_by_song[song["id"]], 1e-7, 1.0 - 1e-7)
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        all_weights.append(song["weights"])
        per_song[song["id"]] = {
            "examples": len(labels),
            "positives": int(labels.sum()),
            "averagePrecision": round(float(average_precision_score(labels, probabilities)), 6),
            "rocAuc": round(float(roc_auc_score(labels, probabilities)), 6),
            "top5": top_metrics(labels, probabilities, 5),
            "top10": top_metrics(labels, probabilities, 10),
            "top20": top_metrics(labels, probabilities, 20),
            "top40": top_metrics(labels, probabilities, 40),
        }
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    weights = np.concatenate(all_weights)
    average_precisions = [float(row["averagePrecision"]) for row in per_song.values()]
    top10 = [float(row["top10"]["precision"]) for row in per_song.values()]
    top20 = [float(row["top20"]["precision"]) for row in per_song.values()]
    return {
        "name": name,
        "wholeSongOutOfFold": True,
        "aggregate": {
            "averagePrecision": round(float(average_precision_score(labels, probabilities, sample_weight=weights)), 6),
            "rocAuc": round(float(roc_auc_score(labels, probabilities, sample_weight=weights)), 6),
            "weightedLogLoss": round(float(log_loss(labels, probabilities, sample_weight=weights)), 6),
            "meanSongAveragePrecision": round(sum(average_precisions) / len(average_precisions), 6),
            "minimumSongAveragePrecision": round(min(average_precisions), 6),
            "meanSongTop10Precision": round(sum(top10) / len(top10), 6),
            "minimumSongTop10Precision": round(min(top10), 6),
            "meanSongTop20Precision": round(sum(top20) / len(top20), 6),
            "minimumSongTop20Precision": round(min(top20), 6),
        },
        "songs": per_song,
        "probabilities": {
            song["id"]: [round(float(value), 8) for value in probabilities_by_song[song["id"]]]
            for song in songs
        },
    }


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    aggregate = row["aggregate"]
    return (
        float(aggregate["minimumSongTop10Precision"]),
        float(aggregate["meanSongTop10Precision"]),
        float(aggregate["minimumSongAveragePrecision"]),
        float(aggregate["meanSongAveragePrecision"]),
        -float(aggregate["weightedLogLoss"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--current-benefit-folds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0x41424C41)
    args = parser.parse_args()

    manifest = load_json(args.manifest.resolve())
    selector_root = args.selector_profiles_root.resolve()
    songs = [
        build_song_examples(
            row,
            load_json(selector_root / str(row["id"]) / "profile.json"),
            source_radius=max(0.01, float(args.source_radius_seconds)),
            minimum_source_midi=0,
            maximum_source_midi=59,
            minimum_output_midi=33,
            maximum_output_midi=71,
            register_shifts=(0, 12),
        )
        for row in manifest.get("songs") or []
    ]
    results: list[dict[str, Any]] = []
    current: dict[str, np.ndarray] = {}
    current_root = args.current_benefit_folds_root.resolve()
    for song in songs:
        profile = load_json(current_root / song["id"] / "profile.json")
        current[song["id"]] = model_probabilities_numpy(song["features"], profile["model"])
    results.append(summarize_model("current-tiny-mlp-v001", songs, current))

    for model_index, (name, factory) in enumerate(model_factories(int(args.seed)).items()):
        probabilities_by_song: dict[str, np.ndarray] = {}
        for fold_index, held_out in enumerate(songs):
            training_songs = [song for song in songs if song is not held_out]
            features, labels, weights, _training_mask = combine_songs(training_songs)
            model = factory()
            # Every song contributes balanced class weights in build_song_examples.
            model.fit(features, labels.astype(np.int64), **(
                {"sample_weight": weights}
                if not hasattr(model, "steps")
                else {f"{model.steps[-1][0]}__sample_weight": weights}
            ))
            probabilities_by_song[held_out["id"]] = model.predict_proba(held_out["features"])[:, 1]
        results.append(summarize_model(name, songs, probabilities_by_song))
        print(json.dumps({"completed": model_index + 1, "model": name, "aggregate": results[-1]["aggregate"]}), flush=True)

    results.sort(key=ranking_key, reverse=True)
    report = {
        "schema": "polymath-swap-benefit-model-ablation-v1",
        "evidenceBoundary": "Every probability is from a model trained without that complete song.",
        "manifest": str(args.manifest.resolve()),
        "ranking": [
            {"rank": index + 1, "name": row["name"], **row["aggregate"]}
            for index, row in enumerate(results)
        ],
        "models": results,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "winner": report["ranking"][0]}, indent=2))


if __name__ == "__main__":
    main()
