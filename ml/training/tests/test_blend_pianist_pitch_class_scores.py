from __future__ import annotations

import pytest

from ml.training.blend_pianist_pitch_class_scores import blend_payloads


def payload(model: str, values: list[float]) -> dict:
    return {
        "songId": "held-out",
        "modelId": model,
        "modelSha256": model + "-hash",
        "trainingSongIds": ["train-a", "train-b"],
        "referenceFieldsRead": False,
        "decisionScope": "additions",
        "cells": [{"cellIndex": 0, "probabilities": values}],
    }


def test_minimum_blend_requires_both_models_to_agree() -> None:
    left = payload("left", [1.0] + [0.8] * 11)
    right = payload("right", [1.0] + [0.5] * 11)
    result = blend_payloads([left, right], mode="minimum")
    assert result["cells"][0]["probabilities"] == [1.0] + [0.5] * 11
    assert result["referenceFieldsRead"] is False
    assert result["trainingSongIds"] == ["train-a", "train-b"]


def test_blend_rejects_any_payload_without_blind_proof() -> None:
    left = payload("left", [1.0] * 12)
    right = payload("right", [1.0] * 12)
    right["referenceFieldsRead"] = True
    with pytest.raises(ValueError, match="blind-inference"):
        blend_payloads([left, right], mode="minimum")
