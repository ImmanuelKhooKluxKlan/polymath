"""Inference-safe long-context features for pianist pitch-class decisions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fit_pianist_pitch_class_decoder import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    examples_from_song,
    pitch_bits,
)


SUMMARY_CONTEXTS = (
    "global_candidate",
    "global_source",
    "local_candidate_radius4",
    "local_source_radius4",
)
OFFSET_CONTEXTS = tuple(
    f"offset_{offset:+d}_{kind}"
    for offset in (-4, -2, 2, 4)
    for kind in ("candidate", "source")
)
CONTEXT_FEATURE_NAMES = tuple(
    f"{context}_interval_{interval:+d}"
    for context in SUMMARY_CONTEXTS + OFFSET_CONTEXTS
    for interval in range(12)
)
FEATURE_NAMES = BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


def context_examples_from_song(song: dict[str, Any]) -> list[dict[str, Any]]:
    """Extend the v1 examples without reading any reference-derived field.

    Labels still come from ``examples_from_song`` for training, but every new
    feature uses only candidate/source pitch classes available at inference.
    """

    rows = list(song.get("cells") or [])
    base = examples_from_song(song)
    if len(base) != len(rows) * 12:
        raise ValueError("Base pitch-class example grid is malformed")
    candidate = np.asarray(
        [pitch_bits(row.get("candidatePitchClasses") or []) for row in rows]
    )
    source = np.asarray(
        [pitch_bits(row.get("sourcePitchClasses") or []) for row in rows]
    )
    global_candidate = candidate.mean(axis=0) if len(rows) else np.zeros(12)
    global_source = source.mean(axis=0) if len(rows) else np.zeros(12)
    zero = np.zeros(12, dtype=float)
    output: list[dict[str, Any]] = []
    for cell_index, _row in enumerate(rows):
        start = max(0, cell_index - 4)
        end = min(len(rows), cell_index + 5)
        local_candidate = candidate[start:end].mean(axis=0)
        local_source = source[start:end].mean(axis=0)
        offset_contexts: list[np.ndarray] = []
        for offset in (-4, -2, 2, 4):
            index = cell_index + offset
            offset_contexts.extend(
                (
                    candidate[index] if 0 <= index < len(rows) else zero,
                    source[index] if 0 <= index < len(rows) else zero,
                )
            )
        for pitch_class in range(12):
            row = base[cell_index * 12 + pitch_class]
            extra = [
                float(value)
                for context in (
                    global_candidate,
                    global_source,
                    local_candidate,
                    local_source,
                    *offset_contexts,
                )
                for value in np.roll(context, -pitch_class)
            ]
            if len(extra) != len(CONTEXT_FEATURE_NAMES):
                raise AssertionError("Long-context feature contract drifted")
            output.append({**row, "features": list(row["features"]) + extra})
    return output
