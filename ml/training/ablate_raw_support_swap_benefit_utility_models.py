"""Whole-song holdout ablation for signed swap-utility prediction.

The binary benefit target deliberately throws away useful information: a
proposal that is harmless-but-neutral and one that destroys three matches are
both labelled ``0``.  This research-only script predicts the signed reference
utility instead.  It also tests a decomposed variant which separately predicts
the exact-100 ms, exact-250 ms, and pitch-class deltas before recombining them.

References are used only to construct offline targets.  Every score written to
the report is produced by a model trained without that complete song.  The
report format intentionally matches ``search_raw_support_swap_benefit_policy``
so the real end-to-end musical gate remains the final authority.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error

from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sigmoid(values: np.ndarray, scale: float) -> np.ndarray:
    logits = np.clip(values / max(1e-6, scale), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def regression_factories(seed: int) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    for depth, leaf in ((4, 10), (6, 10), (8, 10), (8, 20), (10, 20)):
        factories[f"utility-extra-trees-d{depth}-l{leaf}"] = (
            lambda d=depth, l=leaf: ExtraTreesRegressor(
                n_estimators=300,
                max_depth=d,
                min_samples_leaf=l,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            )
        )
    for depth, leaf in ((6, 10), (8, 10), (10, 20)):
        factories[f"utility-random-forest-d{depth}-l{leaf}"] = (
            lambda d=depth, l=leaf: RandomForestRegressor(
                n_estimators=300,
                max_depth=d,
                min_samples_leaf=l,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            )
        )
    for leaves, minimum_leaf, l2 in ((7, 10, 1.0), (15, 10, 1.0), (15, 20, 3.0), (31, 20, 3.0)):
        factories[f"utility-hist-gb-n{leaves}-l{minimum_leaf}-r{l2:g}"] = (
            lambda n=leaves, l=minimum_leaf, r=l2: HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=220,
                max_leaf_nodes=n,
                min_samples_leaf=l,
                l2_regularization=r,
                loss="squared_error",
                random_state=seed,
            )
        )
    for depth, leaf, loss in ((1, 10, "huber"), (2, 10, "huber"), (2, 20, "huber"), (2, 20, "squared_error")):
        factories[f"utility-gradient-boost-d{depth}-l{leaf}-{loss}"] = (
            lambda d=depth, l=leaf, objective=loss: GradientBoostingRegressor(
                n_estimators=260,
                learning_rate=0.035,
                max_depth=d,
                min_samples_leaf=l,
                subsample=0.85,
                loss=objective,
                random_state=seed,
            )
        )
    return factories


def decomposed_factories(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "delta-extra-trees-d6-l10": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "delta-extra-trees-d8-l10": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "delta-hist-gb-n15-l20": lambda: HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=3.0,
            random_state=seed,
        ),
    }


def training_arrays(songs: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    features = np.vstack([song["features"] for song in songs])
    utilities = np.concatenate([song["utilities"] for song in songs])
    deltas = np.vstack(
        [
            np.asarray(
                [
                    [
                        detail["deltaExact100"],
                        detail["deltaExact250"],
                        detail["deltaPitchClass250"],
                    ]
                    for detail in song["labelDetails"]
                ],
                dtype=np.int64,
            )
            for song in songs
        ]
    )
    weights = np.concatenate(
        [
            np.full(len(song["utilities"]), 1.0 / len(song["utilities"]), dtype=np.float64)
            * (1.0 + 0.12 * np.abs(song["utilities"]))
            for song in songs
        ]
    )
    return features, utilities, deltas, weights


def expected_class_value(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities @ np.asarray(model.classes_, dtype=np.float64)


def summarize(
    name: str,
    songs: list[dict[str, Any]],
    raw_by_song: dict[str, np.ndarray],
    *,
    score_scale: float,
) -> dict[str, Any]:
    probabilities: dict[str, list[float]] = {}
    per_song: dict[str, Any] = {}
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []
    for song in songs:
        actual = song["utilities"]
        predicted = raw_by_song[song["id"]]
        ordering = np.argsort(-predicted, kind="stable")
        row: dict[str, Any] = {
            "examples": len(actual),
            "meanAbsoluteError": round(float(mean_absolute_error(actual, predicted)), 6),
        }
        for count in (5, 10, 20, 40):
            indices = ordering[: min(count, len(ordering))]
            chosen = actual[indices]
            row[f"top{count}"] = {
                "selected": len(indices),
                "positiveShare": round(float(np.mean(chosen > 0)), 6),
                "harmfulShare": round(float(np.mean(chosen < 0)), 6),
                "utilitySum": round(float(chosen.sum()), 6),
            }
        per_song[song["id"]] = row
        probabilities[song["id"]] = [
            round(float(value), 8) for value in sigmoid(predicted, score_scale)
        ]
        all_actual.append(actual)
        all_predicted.append(predicted)
    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_predicted)
    song_top10 = [row["top10"] for row in per_song.values()]
    return {
        "name": name,
        "wholeSongOutOfFold": True,
        "scoreTransform": {"type": "sigmoid", "scale": score_scale},
        "aggregate": {
            "meanAbsoluteError": round(float(mean_absolute_error(actual, predicted)), 6),
            "meanSongTop10PositiveShare": round(
                float(np.mean([row["positiveShare"] for row in song_top10])), 6
            ),
            "maximumSongTop10HarmfulShare": round(
                float(np.max([row["harmfulShare"] for row in song_top10])), 6
            ),
            "totalSongTop10Utility": round(
                float(np.sum([row["utilitySum"] for row in song_top10])), 6
            ),
        },
        "songs": per_song,
        "probabilities": probabilities,
    }


def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    aggregate = row["aggregate"]
    return (
        -float(aggregate["maximumSongTop10HarmfulShare"]),
        float(aggregate["meanSongTop10PositiveShare"]),
        float(aggregate["totalSongTop10Utility"]),
        -float(aggregate["meanAbsoluteError"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--score-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0x5554494C)
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
    for model_index, (name, factory) in enumerate(regression_factories(int(args.seed)).items()):
        predictions: dict[str, np.ndarray] = {}
        for held_out in songs:
            training = [song for song in songs if song is not held_out]
            features, utilities, _deltas, weights = training_arrays(training)
            model = factory()
            model.fit(features, utilities, sample_weight=weights)
            predictions[held_out["id"]] = np.asarray(
                model.predict(held_out["features"]), dtype=np.float64
            )
        result = summarize(name, songs, predictions, score_scale=float(args.score_scale))
        results.append(result)
        print(json.dumps({"completed": model_index + 1, "model": name, **result["aggregate"]}), flush=True)

    offset = len(results)
    coefficients = np.asarray([4.0, 3.0, 1.0], dtype=np.float64)
    for model_index, (name, factory) in enumerate(decomposed_factories(int(args.seed)).items()):
        predictions = {}
        for held_out in songs:
            training = [song for song in songs if song is not held_out]
            features, _utilities, deltas, weights = training_arrays(training)
            components: list[np.ndarray] = []
            for target_index in range(3):
                model = factory()
                model.fit(features, deltas[:, target_index], sample_weight=weights)
                components.append(expected_class_value(model, held_out["features"]))
            predictions[held_out["id"]] = np.column_stack(components) @ coefficients
        result = summarize(name, songs, predictions, score_scale=float(args.score_scale))
        results.append(result)
        print(json.dumps({"completed": offset + model_index + 1, "model": name, **result["aggregate"]}), flush=True)

    results.sort(key=ranking_key, reverse=True)
    report = {
        "schema": "polymath-swap-benefit-utility-model-ablation-v1",
        "evidenceBoundary": "Every score is from a model trained without that complete song; references create targets only.",
        "manifest": str(args.manifest.resolve()),
        "models": results,
        "ranking": [
            {"rank": index + 1, "name": row["name"], **row["aggregate"]}
            for index, row in enumerate(results)
        ],
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "winner": report["ranking"][0]}, indent=2))


if __name__ == "__main__":
    main()
