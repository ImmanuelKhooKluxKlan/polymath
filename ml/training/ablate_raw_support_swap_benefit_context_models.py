"""Test inference-safe song-relative features with whole-song holdouts.

Absolute selector probabilities, velocities, and register positions drift
between recordings.  This research ablation appends percentile/context fields
computed from the candidate proposal set itself.  No reference-derived value
is present in those fields, and every reported prediction comes from a model
trained without that complete song.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ablate_raw_support_swap_benefit_models import atomic_json, summarize_model
from .apply_raw_support_recovery import load_json
from .fit_raw_support_swap_benefit_ranker import build_song_examples
from .raw_support_swap_benefit import generate_swap_proposals


RANK_FEATURE_INDICES = (
    1,   # proposal selector probability
    2,   # incumbent selector probability
    3,   # selector probability gain
    5,   # source time distance
    6,   # source velocity
    7,   # source duration
    8,   # source register
    10,  # target register
    12,  # replacement interval size
    14,  # incumbent velocity
    15,  # incumbent duration
    16,  # gesture size
    17,  # gesture span
    35,  # source local density
    40,  # same-instrument pitch-class recurrence
    47,  # wide pitch-class support
    49,  # wide pitch-class dominance
    54,  # estimated-chord confidence
)


def proposal_key(proposal: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(proposal["incumbentIndex"]),
        int(proposal["sourceUsableIndex"]),
        int(proposal["targetMidi"]),
        int(proposal["registerShift"]),
    )


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return tie-aware ranks in (0, 1), invariant to song scale."""

    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    before = np.cumsum(np.r_[0, counts[:-1]])
    middle = before + (counts - 1.0) / 2.0 + 0.5
    return middle[inverse] / max(1, len(values))


def full_proposals(
    row: dict[str, Any], selector_profile: dict[str, Any], source_radius: float
) -> list[dict[str, Any]]:
    candidate = load_json(Path(str(row["candidate"])).resolve())
    source = load_json(Path(str(row["source"])).resolve())
    proposals, _diagnostics = generate_swap_proposals(
        candidate,
        source,
        selector_profile,
        source_radius=source_radius,
        minimum_source_midi=0,
        maximum_source_midi=59,
        minimum_output_midi=33,
        maximum_output_midi=71,
        register_shifts=(0, 12),
    )
    return proposals


def context_rows(proposals: list[dict[str, Any]]) -> dict[tuple[int, ...], np.ndarray]:
    raw = np.asarray([proposal["features"] for proposal in proposals], dtype=np.float64)
    ranks = np.column_stack(
        [percentile_ranks(raw[:, index]) for index in RANK_FEATURE_INDICES]
    )
    group_counts = Counter(int(proposal["groupNumber"]) for proposal in proposals)
    incumbent_counts = Counter(int(proposal["incumbentIndex"]) for proposal in proposals)
    target_counts = Counter(int(proposal["targetMidi"]) for proposal in proposals)
    family_prevalence = raw[:, 23:29].mean(axis=0)
    maximum_time = max(1e-6, max(float(proposal["groupTime"]) for proposal in proposals))
    rows: dict[tuple[int, ...], np.ndarray] = {}
    for index, proposal in enumerate(proposals):
        family_index = int(np.argmax(raw[index, 23:29]))
        extra = np.asarray(
            [
                *ranks[index],
                min(1.0, float(proposal["groupTime"]) / maximum_time),
                np.log1p(group_counts[int(proposal["groupNumber"])]),
                np.log1p(incumbent_counts[int(proposal["incumbentIndex"])]),
                np.log1p(target_counts[int(proposal["targetMidi"])]),
                family_prevalence[family_index],
            ],
            dtype=np.float64,
        )
        rows[proposal_key(proposal)] = np.concatenate((raw[index], extra))
    return rows


def augment_song(
    song: dict[str, Any], row: dict[str, Any], selector: dict[str, Any], source_radius: float
) -> dict[str, Any]:
    lookup = context_rows(full_proposals(row, selector, source_radius))
    augmented = np.asarray(
        [lookup[proposal_key(proposal)] for proposal in song["proposals"]],
        dtype=np.float64,
    )
    return {**song, "features": augmented}


def factories(seed: int) -> dict[str, Callable[[], Any]]:
    models: dict[str, Callable[[], Any]] = {
        "context-logistic-c0.1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2500, solver="lbfgs", random_state=seed),
        ),
        "context-logistic-c1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2500, solver="lbfgs", random_state=seed),
        ),
    }
    for depth, leaf in ((4, 10), (6, 10), (8, 10), (8, 20)):
        models[f"context-extra-trees-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    models["context-random-forest-d8-l10"] = lambda: RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    for leaves, minimum_leaf in ((7, 10), (15, 10), (15, 20), (31, 20)):
        models[f"context-hist-gb-n{leaves}-l{minimum_leaf}"] = lambda n=leaves, l=minimum_leaf: HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=240,
            max_leaf_nodes=n,
            min_samples_leaf=l,
            l2_regularization=3.0,
            random_state=seed,
        )
    for depth, leaf in ((1, 10), (2, 10), (2, 20)):
        models[f"context-gradient-boost-d{depth}-l{leaf}"] = lambda d=depth, l=leaf: GradientBoostingClassifier(
            n_estimators=240,
            learning_rate=0.04,
            max_depth=d,
            min_samples_leaf=l,
            subsample=0.85,
            random_state=seed,
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-profiles-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-radius-seconds", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0x434F4E54)
    args = parser.parse_args()

    manifest = load_json(args.manifest.resolve())
    selector_root = args.selector_profiles_root.resolve()
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

    results: list[dict[str, Any]] = []
    for model_index, (name, factory) in enumerate(factories(int(args.seed)).items()):
        probabilities: dict[str, np.ndarray] = {}
        for held_out in songs:
            training = [song for song in songs if song is not held_out]
            features = np.vstack([song["features"] for song in training])
            labels = np.concatenate([song["labels"] for song in training]).astype(np.int64)
            weights = np.concatenate([song["weights"] for song in training])
            model = factory()
            fit_arguments = (
                {f"{model.steps[-1][0]}__sample_weight": weights}
                if hasattr(model, "steps")
                else {"sample_weight": weights}
            )
            model.fit(features, labels, **fit_arguments)
            probabilities[held_out["id"]] = model.predict_proba(held_out["features"])[:, 1]
        result = summarize_model(name, songs, probabilities)
        results.append(result)
        print(json.dumps({"completed": model_index + 1, "model": name, **result["aggregate"]}), flush=True)

    results.sort(
        key=lambda row: (
            float(row["aggregate"]["minimumSongTop10Precision"]),
            float(row["aggregate"]["meanSongTop10Precision"]),
            float(row["aggregate"]["minimumSongAveragePrecision"]),
            float(row["aggregate"]["meanSongAveragePrecision"]),
        ),
        reverse=True,
    )
    report = {
        "schema": "polymath-swap-benefit-context-model-ablation-v1",
        "evidenceBoundary": "Context uses only the inference-time proposal set; every score is whole-song out-of-fold.",
        "manifest": str(args.manifest.resolve()),
        "rankFeatureIndices": list(RANK_FEATURE_INDICES),
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
