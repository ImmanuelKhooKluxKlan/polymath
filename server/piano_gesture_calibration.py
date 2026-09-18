"""Shared feature contract for supervised pianist-gesture calibration.

The full-mix transcriber supplies a useful loudness contour, but that contour
is not the same thing as a pianist's hammer velocity.  This small deterministic
layer lets a training-only ridge model correct the *gesture* after the normal
source/chord dynamics pass.  Every simultaneous note still receives one hammer
velocity; melody balance remains a separate performance-gain decision.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from statistics import median
from typing import Any


GESTURE_CALIBRATION_FEATURE_NAMES = (
    "bias",
    "base_velocity",
    "base_velocity_squared",
    "source_velocity",
    "source_velocity_squared",
    "duration_expression",
    "chord_size_2",
    "chord_size_3",
    "chord_size_4",
    "chord_size_5",
    "chord_size_6_plus",
    "extra_note_count",
    "melody_share",
    "bass_share",
    "left_hand_share",
    "mean_midi_centered",
    "pitch_span",
    "source_voice_share",
    "source_guitar_share",
    "source_bass_share",
    "song_progress",
    "intro_strength",
    "outro_strength",
    "previous_gap_expression",
    "next_gap_expression",
    "local_4s_velocity_mean",
    "local_4s_velocity_std",
    "local_8s_velocity_mean",
    "velocity_minus_local_4s",
    "velocity_z_local_4s",
    "local_4s_source_velocity_mean",
    "source_velocity_minus_local_4s",
    "local_4s_onsets_per_second",
    "local_4s_notes_per_onset",
    "local_4s_large_chord_share",
    "local_4s_melody_share",
    "local_4s_bass_share",
    "local_4s_duration_expression",
    "local_density_change",
    "large_chord_x_source_velocity",
    "large_chord_x_local_velocity",
    "large_chord_x_local_share",
    "large_chord_x_density",
    "melody_x_source_velocity",
    "bass_x_source_velocity",
)


GESTURE_DURATION_FEATURE_NAMES = (
    "bias",
    "base_duration_log",
    "base_duration_log_squared",
    "base_velocity",
    "source_velocity",
    "chord_size_2",
    "chord_size_3",
    "chord_size_4_plus",
    "melody_share",
    "bass_share",
    "left_hand_share",
    "mean_midi_centered",
    "pitch_span",
    "source_voice_share",
    "source_guitar_share",
    "source_bass_share",
    "song_progress",
    "intro_strength",
    "outro_strength",
    "previous_gap_log",
    "next_gap_log",
    "previous_same_pitch_gap_log",
    "next_same_pitch_gap_log",
    "has_previous_same_pitch",
    "has_next_same_pitch",
    "duration_over_next_gap",
    "local_4s_onsets_per_second",
    "local_4s_notes_per_onset",
    "local_4s_duration_log",
    "local_density_change",
)


def _finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _role(note: dict[str, Any]) -> str:
    return str(note.get("arrangementRole") or note.get("role") or "").lower()


def _instrument_family(note: dict[str, Any]) -> str:
    value = str(
        note.get("sourceInstrument") or note.get("instrument") or ""
    ).lower()
    if "voice" in value or "vocal" in value or "choir" in value:
        return "voice"
    if "guitar" in value:
        return "guitar"
    if "bass" in value:
        return "bass"
    return "other"


def gesture_calibration_features(
    group: list[dict[str, Any]],
    base_velocity: float,
    context: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return inference-safe features for one near-simultaneous key strike."""

    if not group:
        raise ValueError("A gesture calibration group cannot be empty")
    count = len(group)
    midis = [int(round(_finite(note.get("midi"), 60.0))) for note in group]
    durations = [
        max(0.03, _finite(note.get("duration"), 0.2)) for note in group
    ]
    source_velocities = [
        _clamp(
            _finite(
                note.get("sourceVelocityBeforeArrangement"),
                _finite(note.get("velocity"), base_velocity),
            ),
            0.01,
            1.0,
        )
        for note in group
    ]
    structural_size = min(6, max(1, len(set(midis))))
    median_duration = median(durations)
    duration_expression = _clamp(
        (math.sqrt(median_duration) - math.sqrt(0.12))
        / (math.sqrt(0.85) - math.sqrt(0.12)),
        0.0,
        1.0,
    )
    families = [_instrument_family(note) for note in group]
    roles = [_role(note) for note in group]
    hands = [
        str(note.get("hand") or "").lower()
        or ("left" if midi < 72 else "right")
        for note, midi in zip(group, midis)
    ]
    base = _clamp(float(base_velocity), 0.01, 1.0)
    source = median(source_velocities)
    sequence = context or {}
    large_chord = float(structural_size >= 4)
    local_velocity = _clamp(
        _finite(sequence.get("local_4s_velocity_mean"), base), 0.01, 1.0
    )
    local_velocity_std = _clamp(
        _finite(sequence.get("local_4s_velocity_std"), 0.0), 0.0, 0.5
    )
    velocity_delta = _clamp(
        _finite(sequence.get("velocity_minus_local_4s"), 0.0), -1.0, 1.0
    )
    local_source_velocity = _clamp(
        _finite(sequence.get("local_4s_source_velocity_mean"), source),
        0.01,
        1.0,
    )
    local_large_share = _clamp(
        _finite(sequence.get("local_4s_large_chord_share"), 0.0), 0.0, 1.0
    )
    local_density = _clamp(
        _finite(sequence.get("local_4s_onsets_per_second"), 0.0) / 10.0,
        0.0,
        2.0,
    )
    melody_share = roles.count("melody") / count
    bass_share = roles.count("bass") / count
    return {
        "bias": 1.0,
        "base_velocity": base,
        "base_velocity_squared": base * base,
        "source_velocity": source,
        "source_velocity_squared": source * source,
        "duration_expression": duration_expression,
        "chord_size_2": float(structural_size == 2),
        "chord_size_3": float(structural_size == 3),
        "chord_size_4": float(structural_size == 4),
        "chord_size_5": float(structural_size == 5),
        "chord_size_6_plus": float(structural_size >= 6),
        "extra_note_count": _clamp((count - 1) / 5.0, 0.0, 1.5),
        "melody_share": melody_share,
        "bass_share": bass_share,
        "left_hand_share": hands.count("left") / count,
        "mean_midi_centered": _clamp(
            (sum(midis) / count - 66.0) / 24.0, -1.5, 1.5
        ),
        "pitch_span": _clamp((max(midis) - min(midis)) / 24.0, 0.0, 2.0),
        "source_voice_share": families.count("voice") / count,
        "source_guitar_share": families.count("guitar") / count,
        "source_bass_share": families.count("bass") / count,
        "song_progress": _clamp(_finite(sequence.get("song_progress"), 0.5), 0.0, 1.0),
        "intro_strength": _clamp(_finite(sequence.get("intro_strength"), 0.0), 0.0, 1.0),
        "outro_strength": _clamp(_finite(sequence.get("outro_strength"), 0.0), 0.0, 1.0),
        "previous_gap_expression": _clamp(
            _finite(sequence.get("previous_gap_expression"), 0.0), 0.0, 1.0
        ),
        "next_gap_expression": _clamp(
            _finite(sequence.get("next_gap_expression"), 0.0), 0.0, 1.0
        ),
        "local_4s_velocity_mean": local_velocity,
        "local_4s_velocity_std": local_velocity_std,
        "local_8s_velocity_mean": _clamp(
            _finite(sequence.get("local_8s_velocity_mean"), base), 0.01, 1.0
        ),
        "velocity_minus_local_4s": velocity_delta,
        "velocity_z_local_4s": _clamp(
            velocity_delta / max(0.025, local_velocity_std), -4.0, 4.0
        ),
        "local_4s_source_velocity_mean": local_source_velocity,
        "source_velocity_minus_local_4s": _clamp(
            source - local_source_velocity, -1.0, 1.0
        ),
        "local_4s_onsets_per_second": local_density,
        "local_4s_notes_per_onset": _clamp(
            _finite(sequence.get("local_4s_notes_per_onset"), 1.0) / 6.0,
            0.0,
            2.0,
        ),
        "local_4s_large_chord_share": local_large_share,
        "local_4s_melody_share": _clamp(
            _finite(sequence.get("local_4s_melody_share"), 0.0), 0.0, 1.0
        ),
        "local_4s_bass_share": _clamp(
            _finite(sequence.get("local_4s_bass_share"), 0.0), 0.0, 1.0
        ),
        "local_4s_duration_expression": _clamp(
            _finite(sequence.get("local_4s_duration_expression"), 0.0), 0.0, 1.0
        ),
        "local_density_change": _clamp(
            _finite(sequence.get("local_density_change"), 0.0) / 10.0,
            -2.0,
            2.0,
        ),
        "large_chord_x_source_velocity": large_chord * source,
        "large_chord_x_local_velocity": large_chord * local_velocity,
        "large_chord_x_local_share": large_chord * local_large_share,
        "large_chord_x_density": large_chord * local_density,
        "melody_x_source_velocity": melody_share * source,
        "bass_x_source_velocity": bass_share * source,
    }


def gesture_duration_features(
    group: list[dict[str, Any]],
    base_duration: float,
    context: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return inference-safe predictors for one pianist key-hold gesture."""

    if not group:
        raise ValueError("A gesture duration group cannot be empty")
    sequence = context or {}
    count = len(group)
    midis = [int(round(_finite(note.get("midi"), 60.0))) for note in group]
    roles = [_role(note) for note in group]
    families = [_instrument_family(note) for note in group]
    hands = [
        str(note.get("hand") or "").lower()
        or ("left" if midi < 72 else "right")
        for note, midi in zip(group, midis)
    ]
    source_velocities = [
        _clamp(
            _finite(
                note.get("sourceVelocityBeforeArrangement"),
                _finite(note.get("velocity"), 0.72),
            ),
            0.01,
            1.0,
        )
        for note in group
    ]
    base = _clamp(float(base_duration), 0.03, 8.0)
    base_log = math.log1p(base)
    previous_gap = _clamp(
        _finite(sequence.get("previous_gap_seconds"), base), 0.0, 8.0
    )
    next_gap = _clamp(
        _finite(sequence.get("next_gap_seconds"), base), 0.0, 8.0
    )
    previous_same_pitch = _finite(
        sequence.get("previous_same_pitch_gap_seconds"), -1.0
    )
    next_same_pitch = _finite(
        sequence.get("next_same_pitch_gap_seconds"), -1.0
    )
    structural_size = len(set(midis))
    return {
        "bias": 1.0,
        "base_duration_log": base_log,
        "base_duration_log_squared": base_log * base_log,
        "base_velocity": _clamp(
            median(_finite(note.get("velocity"), 0.72) for note in group),
            0.01,
            1.0,
        ),
        "source_velocity": median(source_velocities),
        "chord_size_2": float(structural_size == 2),
        "chord_size_3": float(structural_size == 3),
        "chord_size_4_plus": float(structural_size >= 4),
        "melody_share": roles.count("melody") / count,
        "bass_share": roles.count("bass") / count,
        "left_hand_share": hands.count("left") / count,
        "mean_midi_centered": _clamp(
            (sum(midis) / count - 66.0) / 24.0, -1.5, 1.5
        ),
        "pitch_span": _clamp((max(midis) - min(midis)) / 24.0, 0.0, 2.0),
        "source_voice_share": families.count("voice") / count,
        "source_guitar_share": families.count("guitar") / count,
        "source_bass_share": families.count("bass") / count,
        "song_progress": _clamp(
            _finite(sequence.get("song_progress"), 0.5), 0.0, 1.0
        ),
        "intro_strength": _clamp(
            _finite(sequence.get("intro_strength"), 0.0), 0.0, 1.0
        ),
        "outro_strength": _clamp(
            _finite(sequence.get("outro_strength"), 0.0), 0.0, 1.0
        ),
        "previous_gap_log": math.log1p(previous_gap),
        "next_gap_log": math.log1p(next_gap),
        "previous_same_pitch_gap_log": math.log1p(
            previous_same_pitch if previous_same_pitch >= 0.0 else 8.0
        ),
        "next_same_pitch_gap_log": math.log1p(
            next_same_pitch if next_same_pitch >= 0.0 else 8.0
        ),
        "has_previous_same_pitch": float(previous_same_pitch >= 0.0),
        "has_next_same_pitch": float(next_same_pitch >= 0.0),
        "duration_over_next_gap": _clamp(
            base / max(0.03, next_gap), 0.0, 4.0
        ),
        "local_4s_onsets_per_second": _clamp(
            _finite(sequence.get("local_4s_onsets_per_second"), 0.0) / 10.0,
            0.0,
            2.0,
        ),
        "local_4s_notes_per_onset": _clamp(
            _finite(sequence.get("local_4s_notes_per_onset"), 1.0) / 6.0,
            0.0,
            2.0,
        ),
        "local_4s_duration_log": math.log1p(
            _clamp(
                _finite(sequence.get("local_4s_duration_seconds"), base),
                0.03,
                8.0,
            )
        ),
        "local_density_change": _clamp(
            _finite(sequence.get("local_density_change"), 0.0) / 10.0,
            -2.0,
            2.0,
        ),
    }


def gesture_sequence_contexts(
    groups: list[list[dict[str, Any]]],
    base_velocities: list[float],
) -> list[dict[str, float]]:
    """Describe the local phrase around every gesture without song identity.

    Pianist dynamics are phrase-shaped: an intro, verse, chorus and final lift
    cannot be inferred from one isolated hammer strike.  These deterministic
    rolling features give a small supervised model that context while keeping
    inference linear in the number of gestures.
    """

    if len(groups) != len(base_velocities):
        raise ValueError("Gesture groups and base velocities must have equal length")
    if not groups:
        return []
    times = [float(group[0].get("time", 0.0)) for group in groups]
    velocities = [_clamp(float(value), 0.01, 1.0) for value in base_velocities]
    source_velocities = [
        median(
            _clamp(
                _finite(
                    note.get("sourceVelocityBeforeArrangement"),
                    _finite(note.get("velocity"), velocities[index]),
                ),
                0.01,
                1.0,
            )
            for note in group
        )
        for index, group in enumerate(groups)
    ]
    note_counts = [max(1, len({int(round(_finite(note.get("midi"), 60))) for note in group})) for group in groups]
    large = [float(count >= 4) for count in note_counts]
    melody = [float(any(_role(note) == "melody" for note in group)) for group in groups]
    bass = [float(any(_role(note) == "bass" for note in group)) for group in groups]
    durations = [
        _clamp(
            (math.sqrt(max(0.03, median(_finite(note.get("duration"), 0.2) for note in group))) - math.sqrt(0.12))
            / (math.sqrt(0.85) - math.sqrt(0.12)),
            0.0,
            1.0,
        )
        for group in groups
    ]
    duration_seconds = [
        max(
            0.03,
            median(
                _finite(
                    note.get("scoreDuration", note.get("duration")), 0.2
                )
                for note in group
            ),
        )
        for group in groups
    ]
    group_pitches = [
        {int(round(_finite(note.get("midi"), 60))) for note in group}
        for group in groups
    ]
    previous_same_pitch_gaps: list[float | None] = [None] * len(groups)
    next_same_pitch_gaps: list[float | None] = [None] * len(groups)
    previous_by_pitch: dict[int, float] = {}
    for index, (time, pitches) in enumerate(zip(times, group_pitches)):
        gaps = [time - previous_by_pitch[pitch] for pitch in pitches if pitch in previous_by_pitch]
        previous_same_pitch_gaps[index] = min(gaps) if gaps else None
        for pitch in pitches:
            previous_by_pitch[pitch] = time
    next_by_pitch: dict[int, float] = {}
    for index in range(len(groups) - 1, -1, -1):
        time = times[index]
        pitches = group_pitches[index]
        gaps = [next_by_pitch[pitch] - time for pitch in pitches if pitch in next_by_pitch]
        next_same_pitch_gaps[index] = min(gaps) if gaps else None
        for pitch in pitches:
            next_by_pitch[pitch] = time

    def prefix(values: list[float]) -> list[float]:
        output = [0.0]
        for value in values:
            output.append(output[-1] + float(value))
        return output

    velocity_prefix = prefix(velocities)
    velocity_square_prefix = prefix([value * value for value in velocities])
    source_velocity_prefix = prefix(source_velocities)
    notes_prefix = prefix([float(value) for value in note_counts])
    large_prefix = prefix(large)
    melody_prefix = prefix(melody)
    bass_prefix = prefix(bass)
    duration_prefix = prefix(durations)
    duration_seconds_prefix = prefix(duration_seconds)
    total_duration = max(1.0, times[-1] - times[0])

    def bounds(center: float, radius: float) -> tuple[int, int]:
        return (
            bisect_left(times, center - radius),
            bisect_right(times, center + radius),
        )

    def mean_between(values_prefix: list[float], left: int, right: int) -> float:
        return (values_prefix[right] - values_prefix[left]) / max(1, right - left)

    contexts: list[dict[str, float]] = []
    for index, time in enumerate(times):
        left4, right4 = bounds(time, 2.0)
        left8, right8 = bounds(time, 4.0)
        local4_velocity = mean_between(velocity_prefix, left4, right4)
        local4_velocity_square = mean_between(
            velocity_square_prefix, left4, right4
        )
        local4_velocity_std = math.sqrt(
            max(0.0, local4_velocity_square - local4_velocity * local4_velocity)
        )
        past_left = bisect_left(times, time - 4.0)
        past_right = bisect_left(times, time)
        future_left = bisect_right(times, time)
        future_right = bisect_right(times, time + 4.0)
        past_density = (past_right - past_left) / 4.0
        future_density = (future_right - future_left) / 4.0
        contexts.append(
            {
                "song_progress": _clamp((time - times[0]) / total_duration, 0.0, 1.0),
                "intro_strength": math.exp(-max(0.0, time - times[0]) / 20.0),
                "outro_strength": math.exp(-max(0.0, times[-1] - time) / 20.0),
                "previous_gap_expression": _clamp(
                    (time - times[index - 1]) / 1.0 if index else 1.0, 0.0, 1.0
                ),
                "next_gap_expression": _clamp(
                    (times[index + 1] - time) / 1.0 if index + 1 < len(times) else 1.0,
                    0.0,
                    1.0,
                ),
                "previous_gap_seconds": (
                    time - times[index - 1] if index else 1.0
                ),
                "next_gap_seconds": (
                    times[index + 1] - time
                    if index + 1 < len(times)
                    else 1.0
                ),
                "previous_same_pitch_gap_seconds": (
                    previous_same_pitch_gaps[index]
                    if previous_same_pitch_gaps[index] is not None
                    else -1.0
                ),
                "next_same_pitch_gap_seconds": (
                    next_same_pitch_gaps[index]
                    if next_same_pitch_gaps[index] is not None
                    else -1.0
                ),
                "local_4s_velocity_mean": local4_velocity,
                "local_4s_velocity_std": local4_velocity_std,
                "local_8s_velocity_mean": mean_between(velocity_prefix, left8, right8),
                "velocity_minus_local_4s": velocities[index] - local4_velocity,
                "local_4s_source_velocity_mean": mean_between(
                    source_velocity_prefix, left4, right4
                ),
                "local_4s_onsets_per_second": (right4 - left4) / 4.0,
                "local_4s_notes_per_onset": mean_between(notes_prefix, left4, right4),
                "local_4s_large_chord_share": mean_between(large_prefix, left4, right4),
                "local_4s_melody_share": mean_between(melody_prefix, left4, right4),
                "local_4s_bass_share": mean_between(bass_prefix, left4, right4),
                "local_4s_duration_expression": mean_between(duration_prefix, left4, right4),
                "local_4s_duration_seconds": mean_between(
                    duration_seconds_prefix, left4, right4
                ),
                "local_density_change": future_density - past_density,
            }
        )
    return contexts


def calibrated_gesture_velocity(
    group: list[dict[str, Any]],
    base_velocity: float,
    config: dict[str, Any] | None,
    *,
    minimum_velocity: float,
    maximum_velocity: float,
    context: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Apply a standardized linear correction and return value plus delta."""

    if not config or not config.get("enabled"):
        return base_velocity, 0.0
    names = [str(value) for value in config.get("featureNames") or []]
    weights = [_finite(value, 0.0) for value in config.get("weights") or []]
    means = [_finite(value, 0.0) for value in config.get("means") or []]
    scales = [_finite(value, 1.0) for value in config.get("scales") or []]
    if not names or not (
        len(names) == len(weights) == len(means) == len(scales)
    ):
        return base_velocity, 0.0
    supported = set(GESTURE_CALIBRATION_FEATURE_NAMES)
    if any(name not in supported for name in names):
        return base_velocity, 0.0
    features = gesture_calibration_features(group, base_velocity, context)
    raw_correction = sum(
        weight * (features[name] - mean) / max(1e-9, abs(scale))
        for name, weight, mean, scale in zip(names, weights, means, scales)
    )
    maximum_correction = _clamp(
        _finite(config.get("maximumCorrection"), 0.18), 0.0, 0.35
    )
    blend = _clamp(_finite(config.get("blend"), 1.0), 0.0, 1.0)
    correction = blend * _clamp(
        raw_correction, -maximum_correction, maximum_correction
    )
    value = _clamp(
        float(base_velocity) + correction, minimum_velocity, maximum_velocity
    )
    return value, value - float(base_velocity)


def calibrated_gesture_duration(
    group: list[dict[str, Any]],
    base_duration: float,
    config: dict[str, Any] | None,
    *,
    minimum_duration: float = 0.05,
    maximum_duration: float = 4.0,
    context: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Apply a bounded log-duration correction to a complete onset gesture."""

    if not config or not config.get("enabled"):
        return base_duration, 0.0
    names = [str(value) for value in config.get("featureNames") or []]
    weights = [_finite(value, 0.0) for value in config.get("weights") or []]
    means = [_finite(value, 0.0) for value in config.get("means") or []]
    scales = [_finite(value, 1.0) for value in config.get("scales") or []]
    if not names or not (
        len(names) == len(weights) == len(means) == len(scales)
    ):
        return base_duration, 0.0
    supported = set(GESTURE_DURATION_FEATURE_NAMES)
    if any(name not in supported for name in names):
        return base_duration, 0.0
    features = gesture_duration_features(group, base_duration, context)
    minimum_melody_share = _clamp(
        _finite(config.get("minimumMelodyShare"), 0.0), 0.0, 1.0
    )
    maximum_non_melody_duration = _clamp(
        _finite(config.get("maximumNonMelodyBaseDurationSeconds"), 8.0),
        0.0,
        8.0,
    )
    if (
        minimum_melody_share > 0.0
        and features["melody_share"] < minimum_melody_share
        and base_duration > maximum_non_melody_duration
    ):
        return base_duration, 0.0
    raw_prediction = sum(
        weight * (features[name] - mean) / max(1e-9, abs(scale))
        for name, weight, mean, scale in zip(names, weights, means, scales)
    )
    if config.get("predictionMode") == "articulation-ratio":
        predicted_ratio = math.exp(
            _clamp(raw_prediction, math.log(0.15), math.log(5.0))
        )
        next_gap = _clamp(
            _finite((context or {}).get("next_gap_seconds"), base_duration),
            0.03,
            8.0,
        )
        predicted_duration = predicted_ratio * next_gap
        raw_log_correction = math.log1p(predicted_duration) - math.log1p(
            max(0.01, float(base_duration))
        )
    else:
        raw_log_correction = raw_prediction
    maximum_log_correction = _clamp(
        _finite(config.get("maximumLogCorrection"), 0.70), 0.0, 2.0
    )
    blend = _clamp(_finite(config.get("blend"), 1.0), 0.0, 1.0)
    corrected_log = math.log1p(max(0.01, float(base_duration))) + blend * _clamp(
        raw_log_correction, -maximum_log_correction, maximum_log_correction
    )
    value = _clamp(
        math.expm1(corrected_log), minimum_duration, maximum_duration
    )
    return value, value - float(base_duration)


__all__ = [
    "GESTURE_CALIBRATION_FEATURE_NAMES",
    "GESTURE_DURATION_FEATURE_NAMES",
    "calibrated_gesture_duration",
    "calibrated_gesture_velocity",
    "gesture_calibration_features",
    "gesture_duration_features",
    "gesture_sequence_contexts",
]
