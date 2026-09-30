"""Attach synchronized full-mix/stem evidence to pitch-class texture cells.

The output remains inference-safe: every feature comes from model detections at
the frozen candidate onset.  References already present in a texture audit are
copied unchanged and are never consulted while building view evidence.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
from pathlib import Path
from typing import Any

try:
    from .analyze_arranger_error_budget import is_percussion
except ImportError:  # pragma: no cover
    from analyze_arranger_error_budget import is_percussion  # type: ignore


VIEW_NAMES = ("fullMix", "vocals", "backing")


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def load_notes(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    notes: list[dict[str, Any]] = []
    for item in payload.get("notes") or []:
        if not isinstance(item, dict) or is_percussion(item):
            continue
        midi = item.get("midi", item.get("pitch"))
        time = item.get("time", item.get("startTime", item.get("start")))
        try:
            midi_value = int(round(float(midi)))
            time_value = float(time)
        except (TypeError, ValueError):
            continue
        if not (0 <= midi_value <= 127 and math.isfinite(time_value) and time_value >= 0):
            continue
        velocity = finite(item.get("velocity"), 0.72)
        if velocity > 1.0:
            velocity /= 127.0
        notes.append(
            {
                "midi": midi_value,
                "time": time_value,
                "velocity": max(0.0, min(1.0, velocity)),
            }
        )
    notes.sort(key=lambda note: (note["time"], note["midi"]))
    return notes, [float(note["time"]) for note in notes]


def local_pitch_class_rows(
    views: dict[str, tuple[list[dict[str, Any]], list[float]]],
    source_time: float,
    *,
    attack_window: float,
    context_window: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pitch_class in range(12):
        view_rows: dict[str, Any] = {}
        agreement = 0
        attack_agreement = 0
        for view_name in VIEW_NAMES:
            packed = views.get(view_name)
            if packed is None:
                view_rows[view_name] = {
                    "available": False,
                    "attackCount": 0,
                    "nearCount": 0,
                    "nearestOnsetDistanceSeconds": context_window,
                    "maximumVelocity": 0.0,
                }
                continue
            notes, times = packed
            left = bisect.bisect_left(times, source_time - context_window)
            right = bisect.bisect_right(times, source_time + context_window)
            matching = [
                note for note in notes[left:right] if int(note["midi"]) % 12 == pitch_class
            ]
            attacks = [
                note
                for note in matching
                if abs(float(note["time"]) - source_time) <= attack_window
            ]
            near_count = len(matching)
            attack_count = len(attacks)
            agreement += int(near_count > 0)
            attack_agreement += int(attack_count > 0)
            view_rows[view_name] = {
                "available": True,
                "attackCount": attack_count,
                "nearCount": near_count,
                "nearestOnsetDistanceSeconds": round(
                    min(
                        (abs(float(note["time"]) - source_time) for note in matching),
                        default=context_window,
                    ),
                    6,
                ),
                "maximumVelocity": round(
                    max((float(note["velocity"]) for note in matching), default=0.0),
                    6,
                ),
            }
        rows.append(
            {
                "pitchClass": pitch_class,
                "viewAgreementCount": agreement,
                "attackAgreementCount": attack_agreement,
                "views": view_rows,
            }
        )
    return rows


def augment_audits(
    audits: list[dict[str, Any]],
    view_manifest: dict[str, Any],
    *,
    attack_window: float,
    context_window: float,
) -> dict[str, Any]:
    view_config = {
        str(song["id"]): song.get("views") or {}
        for song in view_manifest.get("songs") or []
    }
    loaded: dict[str, dict[str, tuple[list[dict[str, Any]], list[float]]]] = {}
    for song_id, configured in view_config.items():
        loaded[song_id] = {}
        for view_name, filename in configured.items():
            if view_name not in VIEW_NAMES:
                raise ValueError(f"Unsupported view {view_name!r} for {song_id}")
            path = Path(str(filename)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded[song_id][view_name] = load_notes(path)

    output_songs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for audit in audits:
        for original_song in audit.get("songs") or []:
            song_id = str(original_song.get("id") or "")
            if song_id in seen_ids:
                continue
            seen_ids.add(song_id)
            song = copy.deepcopy(original_song)
            views = loaded.get(song_id, {})
            song["multiViewAvailable"] = bool(views)
            song["multiViewNames"] = sorted(views)
            for cell in song.get("cells") or []:
                cell["multiViewPitchClassEvidence"] = local_pitch_class_rows(
                    views,
                    finite(cell.get("sourceTime")),
                    attack_window=attack_window,
                    context_window=context_window,
                )
                cell["multiViewAvailable"] = bool(views)
                cell["multiViewCount"] = len(views)
            output_songs.append(song)

    return {
        "schema": "polymath-multiview-texture-audit-v1",
        "evidenceBoundary": (
            "View features use synchronized transcription detections only; "
            "references are copied for later scoring and never build features."
        ),
        "settings": {
            "attackWindowSeconds": attack_window,
            "contextWindowSeconds": context_window,
            "supportedViews": list(VIEW_NAMES),
        },
        "songs": output_songs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-audit", action="append", required=True)
    parser.add_argument("--view-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attack-window-seconds", type=float, default=0.10)
    parser.add_argument("--context-window-seconds", type=float, default=0.25)
    args = parser.parse_args()
    attack_window = max(0.01, float(args.attack_window_seconds))
    context_window = max(attack_window, float(args.context_window_seconds))
    audits = [
        json.loads(Path(filename).read_text(encoding="utf-8-sig"))
        for filename in args.texture_audit
    ]
    manifest = json.loads(
        Path(args.view_manifest).read_text(encoding="utf-8-sig")
    )
    output = augment_audits(
        audits,
        manifest,
        attack_window=attack_window,
        context_window=context_window,
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "songs": len(output["songs"]),
                "songsWithMultipleViews": sum(
                    bool(song.get("multiViewAvailable")) for song in output["songs"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
