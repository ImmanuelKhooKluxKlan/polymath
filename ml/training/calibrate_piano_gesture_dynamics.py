"""Fit a monotonic pianist-velocity distribution to an arranger profile.

The gesture renderer first predicts *relative* intensity from source audio and
chord structure. A small corpus can then teach its desired dynamic range via
per-song quantiles. Averaging quantiles per song prevents a long track from
silently dominating a short one and preserves the predicted ordering rather
than copying timestamps or notes from a reference song.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
DEFAULT_VELOCITY_BY_CHORD_SIZE = {
    1: 0.68,
    2: 0.67,
    3: 0.71,
    4: 0.88,
    5: 0.90,
    6: 0.86,
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, fraction))
    left = int(math.floor(position))
    right = min(len(ordered) - 1, left + 1)
    if left == right:
        return ordered[left]
    ratio = position - left
    return ordered[left] * (1.0 - ratio) + ordered[right] * ratio


def gesture_rows(payload: dict[str, Any], onset_window: float) -> list[dict[str, Any]]:
    """Return eligible human gestures without allowing rejected labels through.

    Aligned training targets mark unsafe/review windows with
    ``trainingEligible: false``.  Keeping that check here is important because
    a time-trimmed target can still contain low-confidence material inside its
    approved time boundary.
    """

    notes: list[dict[str, float | int]] = []
    for item in payload.get("notes") or []:
        if not isinstance(item, dict) or item.get("trainingEligible") is False:
            continue
        time = finite(item.get("time", item.get("startTime")), -1.0)
        velocity = finite(item.get("velocity"), -1.0)
        midi = int(round(finite(item.get("midi", item.get("pitch")), -1.0)))
        if time >= 0 and 0 < velocity <= 1:
            notes.append({"time": time, "velocity": velocity, "midi": midi})
    notes.sort(key=lambda note: float(note["time"]))
    groups: list[list[dict[str, float | int]]] = []
    group_starts: list[float] = []
    for note in notes:
        note_time = float(note["time"])
        if not groups or note_time - group_starts[-1] > onset_window:
            groups.append([note])
            group_starts.append(note_time)
        else:
            groups[-1].append(note)
    return [
        {
            "velocity": median(float(note["velocity"]) for note in group),
            "size": min(6, len({int(note["midi"]) for note in group})),
        }
        for group in groups
    ]


def gesture_velocities(payload: dict[str, Any], onset_window: float) -> list[float]:
    return [float(row["velocity"]) for row in gesture_rows(payload, onset_window)]


def fit_quantile_map(
    songs: list[tuple[list[float], float]], fractions: tuple[float, ...]
) -> dict[str, float]:
    if not songs:
        raise ValueError("At least one usable target is required")
    total_weight = sum(weight for _values, weight in songs)
    if total_weight <= 0:
        raise ValueError("Target weights must have a positive sum")
    result: dict[str, float] = {}
    previous = 0.0
    for fraction in fractions:
        value = sum(quantile(values, fraction) * weight for values, weight in songs)
        value /= total_weight
        # Sampling noise can make independently averaged knots microscopically
        # non-monotonic. Isotonic clipping is sufficient at seven fixed knots.
        value = max(previous, min(1.0, max(0.01, value)))
        result[str(round(fraction, 6))] = round(value, 6)
        previous = value
    return result


def fit_chord_size_map(
    songs: list[tuple[list[dict[str, Any]], float]],
) -> dict[str, float]:
    """Fit chord-size strength without letting long songs dominate.

    Each song contributes one median per size.  Missing sizes fall back to the
    decoder defaults rather than silently retaining values from a potentially
    superseded profile.
    """

    result: dict[str, float] = {}
    for size, fallback in DEFAULT_VELOCITY_BY_CHORD_SIZE.items():
        values: list[tuple[float, float]] = []
        for rows, weight in songs:
            matching = [float(row["velocity"]) for row in rows if row["size"] == size]
            if matching:
                values.append((median(matching), weight))
        total_weight = sum(weight for _value, weight in values)
        value = (
            sum(item * weight for item, weight in values) / total_weight
            if total_weight > 0
            else fallback
        )
        result[str(size)] = round(min(1.0, max(0.01, value)), 6)
    return result


def blend_chord_size_map(
    existing: dict[str, Any] | None,
    fitted: dict[str, float],
    fitted_share: float,
    minimum_fitted_size: int = 1,
) -> dict[str, float]:
    """Blend learned chord strength separately from quantile calibration.

    Chord-size medians and the global velocity distribution answer different
    questions.  Keeping separate blend controls lets a listening winner retain
    its proven 1--3 note touch while a clean target distribution is evaluated,
    instead of silently replacing both behaviours in one experiment.
    """

    share = min(1.0, max(0.0, float(fitted_share)))
    minimum_size = max(1, min(6, int(minimum_fitted_size)))
    current = existing or {}
    result: dict[str, float] = {}
    for size, fallback in DEFAULT_VELOCITY_BY_CHORD_SIZE.items():
        key = str(size)
        old_value = finite(current.get(key), fallback)
        new_value = finite(fitted.get(key), fallback)
        effective_share = share if size >= minimum_size else 0.0
        result[key] = round(
            min(
                1.0,
                max(
                    0.01,
                    old_value * (1.0 - effective_share)
                    + new_value * effective_share,
                ),
            ),
            6,
        )
    return result


def profile_hash(profile: dict[str, Any]) -> str:
    value = copy.deepcopy(profile)
    value.pop("profileSha256", None)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--gesture-template",
        help=(
            "Optional profile whose static gesture renderer settings are copied "
            "onto a base that does not already enable gesture dynamics. Learned "
            "velocity maps and chord-size strengths are always replaced."
        ),
    )
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--target-weight", action="append", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--onset-window-seconds", type=float, default=0.035)
    parser.add_argument("--calibration-blend", type=float, default=1.0)
    parser.add_argument(
        "--chord-size-blend",
        type=float,
        default=1.0,
        help=(
            "Share of fitted chord-size medians to install (0 preserves the "
            "base/template chord touch; 1 uses the fitted map)."
        ),
    )
    parser.add_argument(
        "--fitted-chord-minimum-size",
        type=int,
        default=1,
        help=(
            "Apply the fitted chord-size blend only at this size and above. "
            "For example, 4 preserves the listening winner's 1-3 note touch."
        ),
    )
    args = parser.parse_args()

    target_paths = [Path(value).resolve() for value in args.target]
    weights = args.target_weight or [1.0] * len(target_paths)
    if len(weights) != len(target_paths):
        raise ValueError("Supply one --target-weight for every --target, or none")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Every target weight must be positive")
    onset_window = max(0.005, min(0.080, args.onset_window_seconds))
    blend = max(0.0, min(1.0, args.calibration_blend))
    chord_size_blend = max(0.0, min(1.0, args.chord_size_blend))
    fitted_chord_minimum_size = max(
        1, min(6, int(args.fitted_chord_minimum_size))
    )

    songs: list[tuple[list[float], float]] = []
    gesture_songs: list[tuple[list[dict[str, Any]], float]] = []
    target_rows: list[dict[str, Any]] = []
    for path, weight in zip(target_paths, weights):
        rows = gesture_rows(load_json(path), onset_window)
        values = [float(row["velocity"]) for row in rows]
        if len(values) < 20:
            raise ValueError(f"{path} has too few usable gestures ({len(values)})")
        songs.append((values, weight))
        gesture_songs.append((rows, weight))
        size_counts = {
            str(size): sum(row["size"] == size for row in rows)
            for size in DEFAULT_VELOCITY_BY_CHORD_SIZE
        }
        target_rows.append(
            {
                "path": str(path),
                "weight": weight,
                "gestures": len(values),
                "minimum": round(min(values), 6),
                "median": round(quantile(values, 0.5), 6),
                "maximum": round(max(values), 6),
                "gestureCountByChordSize": size_counts,
            }
        )
    velocity_map = fit_quantile_map(songs, DEFAULT_QUANTILES)
    fitted_velocity_by_chord_size = fit_chord_size_map(gesture_songs)

    base_path = Path(args.base).resolve()
    profile = copy.deepcopy(load_json(base_path))
    profile["id"] = args.profile_id
    profile["createdAt"] = datetime.now(timezone.utc).isoformat()
    template_row = None
    if args.gesture_template:
        template_path = Path(args.gesture_template).resolve()
        template_profile = load_json(template_path)
        template_gesture = (
            (template_profile.get("decoder") or {}).get("gestureDynamics") or {}
        )
        if not template_gesture.get("enabled"):
            raise ValueError("The gesture template does not enable gesture dynamics")
        profile.setdefault("decoder", {})["gestureDynamics"] = copy.deepcopy(
            template_gesture
        )
        template_row = {
            "path": str(template_path),
            "profileId": template_profile.get("id"),
            "profileSha256": template_profile.get("profileSha256"),
            "copiedFieldsAreStaticHyperparametersOnly": True,
        }
    gesture = profile.setdefault("decoder", {}).setdefault("gestureDynamics", {})
    if not gesture.get("enabled"):
        raise ValueError(
            "The base profile does not enable gesture dynamics; supply "
            "--gesture-template to install its static renderer settings"
        )
    velocity_by_chord_size = blend_chord_size_map(
        gesture.get("velocityByChordSize"),
        fitted_velocity_by_chord_size,
        chord_size_blend,
        fitted_chord_minimum_size,
    )
    gesture["velocityQuantileMap"] = velocity_map
    gesture["velocityByChordSize"] = velocity_by_chord_size
    gesture["minimumVelocity"] = velocity_map["0.0"]
    gesture["maximumVelocity"] = velocity_map["1.0"]
    gesture["quantileCalibrationBlend"] = round(blend, 6)
    training = profile.setdefault("training", {})
    training["gestureVelocityCalibration"] = {
        "schema": "polymath-gesture-velocity-calibration-v1",
        "method": "weighted-per-song-quantiles-plus-chord-size-medians",
        "targets": target_rows,
        "gestureTemplate": template_row,
        "onsetWindowSeconds": round(onset_window, 6),
        "fittedVelocityByChordSize": fitted_velocity_by_chord_size,
        "velocityByChordSize": velocity_by_chord_size,
        "chordSizeBlend": round(chord_size_blend, 6),
        "fittedChordMinimumSize": fitted_chord_minimum_size,
        "commercialUseAllowed": False,
        "decision": "EXPERIMENTAL_ONLY",
    }
    profile.pop("profileSha256", None)
    profile["profileSha256"] = profile_hash(profile)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    report = {
        "schema": "polymath-gesture-velocity-calibration-report-v1",
        "profile": str(output_path),
        "profileSha256": profile["profileSha256"],
        "baseProfile": str(base_path),
        "targets": target_rows,
        "velocityQuantileMap": velocity_map,
        "fittedVelocityByChordSize": fitted_velocity_by_chord_size,
        "velocityByChordSize": velocity_by_chord_size,
        "gestureTemplate": template_row,
        "calibrationBlend": blend,
        "chordSizeBlend": chord_size_blend,
        "fittedChordMinimumSize": fitted_chord_minimum_size,
        "decision": "RESEARCH_ONLY",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
