import numpy as np

from ml.training.search_checkpoint_event_selector_loso import (
    LogisticModel,
    cluster_variants,
    fit_logistic,
    select_events,
)


def note(midi: int, time: float, duration: float = 0.2) -> dict:
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "visualDuration": duration,
        "audioDuration": duration,
        "velocity": 0.7,
    }


def test_cluster_variants_keeps_fast_incumbent_repeats_separate() -> None:
    variants = {
        "incumbent": [note(60, 1.0), note(60, 1.1)],
        "checkpointA": [note(60, 1.02), note(60, 1.12)],
        "checkpointB": [note(60, 0.99), note(60, 1.09)],
    }
    clusters = cluster_variants(variants, 0.08)
    assert len(clusters) == 2
    assert all(len(cluster["members"]) == 3 for cluster in clusters)


def test_logistic_model_learns_separable_signal() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    model = fit_logistic(features, labels, 0.1)
    probabilities = model.probabilities(features)
    assert probabilities[0] < probabilities[1] < probabilities[2] < probabilities[3]
    assert probabilities[0] < 0.5 < probabilities[3]


def test_selector_preserves_incumbent_when_margin_is_not_cleared() -> None:
    variants = {
        "incumbent": [note(60, 1.0)],
        "checkpointA": [note(60, 1.04)],
        "checkpointB": [note(60, 1.05)],
    }
    # Zero feature weights give every proposal probability 0.5.
    model = LogisticModel(np.zeros(32), np.ones(32), np.zeros(33))
    output, diagnostics = select_events(
        variants,
        radius_seconds=0.1,
        model=model,
        switch_margin=0.01,
        addition_threshold=0.9,
    )
    assert output[0]["time"] == 1.0
    assert diagnostics["switchedIncumbentOnsets"] == 0


def test_selector_can_add_high_confidence_non_incumbent_cluster() -> None:
    variants = {
        "incumbent": [],
        "checkpointA": [note(64, 2.0)],
        "checkpointB": [note(64, 2.01)],
    }
    # Positive intercept makes every proposal high confidence.
    weights = np.zeros(33)
    weights[0] = 5.0
    model = LogisticModel(np.zeros(32), np.ones(32), weights)
    output, diagnostics = select_events(
        variants,
        radius_seconds=0.1,
        model=model,
        switch_margin=0.0,
        addition_threshold=0.8,
    )
    assert len(output) == 1
    assert diagnostics["addedNonIncumbentEvents"] == 1
