"""Normalize MuScriptor beat-grid events without importing the GPU runtime."""

from __future__ import annotations

import math
from typing import Any


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_beat_grid(value: Any) -> dict[str, float | int | None] | None:
    if value is None:
        return None
    bpm = _number(_field(value, "bpm"))
    beats_per_bar = _number(_field(value, "beats_per_bar", "beatsPerBar"))
    first_downbeat = _number(_field(value, "first_downbeat", "firstDownbeat"))
    onset_delay = _number(_field(value, "onset_delay", "onsetDelay"))
    if bpm is None or not 20 <= bpm <= 400:
        return None
    if beats_per_bar is not None and not 1 <= beats_per_bar <= 16:
        return None
    return {
        "bpm": round(bpm, 3),
        "beatsPerBar": None if beats_per_bar is None else int(round(beats_per_bar)),
        "firstDownbeatSeconds": None if first_downbeat is None else round(first_downbeat, 4),
        "onsetDelaySeconds": 0.0 if onset_delay is None else round(onset_delay, 4),
    }


def apply_onset_delay(notes: list[dict[str, Any]], beat_grid: dict[str, Any] | None) -> float:
    delay = _number((beat_grid or {}).get("onsetDelaySeconds")) or 0.0
    delay = max(-0.10, min(0.10, delay))
    if not delay:
        return 0.0
    for note in notes:
        note["time"] = round(max(0.0, float(note["time"]) - delay), 4)
    return delay


__all__ = ["apply_onset_delay", "normalize_beat_grid"]
