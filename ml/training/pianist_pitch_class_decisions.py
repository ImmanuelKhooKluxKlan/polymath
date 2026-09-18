"""Lightweight decision masks shared by pitch-class research estimators."""

from __future__ import annotations

from typing import Any

import numpy as np


def membership_decision_mask(arrays: dict[str, Any]) -> np.ndarray:
    candidate = arrays["candidate"].reshape(-1)
    source = arrays["source"].reshape(-1)
    return (candidate > 0) | (source > 0)
