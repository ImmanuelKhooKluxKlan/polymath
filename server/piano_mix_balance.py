"""Inference-safe melody/accompaniment balance for the Iowa piano pack.

MIDI velocity is not a loudness percentage.  The 88 sampled keys have very
different recorded levels, and several accompaniment notes can mask one lead
note.  This module estimates the two stem energies from facts available at
inference time, then changes only ``performanceGain``.  Pitch, onset, duration,
hammer velocity, and note count remain frozen.
"""

from __future__ import annotations

import math
from typing import Any


# PCM RMS of the first 0.5 seconds of public/samples/iowa-mf, indexed A0--C8.
# The short attack window predicts the fixed research renderer substantially
# better than whole-file RMS because piano samples have long, uneven tails.
IOWA_MF_ATTACK_RMS = (
    0.01372762, 0.01359176, 0.01350215, 0.02267524, 0.02032261, 0.02753776,
    0.02893144, 0.02028063, 0.02611511, 0.03921529, 0.02848677, 0.02543501,
    0.02608511, 0.02770901, 0.03117676, 0.04823331, 0.04145482, 0.04150181,
    0.04837124, 0.04453677, 0.06543013, 0.02977242, 0.03266819, 0.03629959,
    0.04174923, 0.05142995, 0.05559367, 0.05498811, 0.05001672, 0.03705449,
    0.04609645, 0.05730396, 0.03904226, 0.03908094, 0.03749539, 0.04047024,
    0.02416416, 0.04189392, 0.03700518, 0.03562314, 0.04082543, 0.04456226,
    0.03374890, 0.02709751, 0.03981648, 0.04091264, 0.05648932, 0.08018938,
    0.04219824, 0.06290360, 0.12274073, 0.06932649, 0.03065719, 0.02561658,
    0.03237542, 0.07222936, 0.04070198, 0.06159839, 0.03122423, 0.02061678,
    0.02462964, 0.03493649, 0.02103645, 0.01801691, 0.01114210, 0.02086877,
    0.02294324, 0.02835539, 0.02535980, 0.01616449, 0.01333878, 0.01893867,
    0.01408464, 0.01948822, 0.01004622, 0.00653643, 0.01031963, 0.00666000,
    0.00689811, 0.00891749, 0.00700015, 0.02220107, 0.00570341, 0.00634342,
    0.00699170, 0.00555820, 0.00685032, 0.00566372,
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        elif end > start:
            merged.append((start, end))
    return merged


def _overlap_seconds(
    start: float, end: float, spans: list[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(end, span_end) - max(start, span_start))
        for span_start, span_end in spans
        if span_start < end and span_end > start
    )


def _attack_rms(midi: int) -> float:
    if 21 <= midi <= 108:
        return IOWA_MF_ATTACK_RMS[midi - 21]
    return 0.03


def adaptive_melody_mix_balance(
    notes: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve role gains from estimated sample energy without target leakage."""

    policy = config if isinstance(config, dict) else {}
    if not policy.get("enabled") or not notes:
        return notes, {
            "applied": False,
            "profile": "adaptive-iowa-melody-mix-v1",
            "reason": "disabled" if notes else "no-notes",
        }

    release_seconds = _clamp(
        _finite(policy.get("estimatedReleaseSeconds"), 0.72), 0.10, 1.50
    )
    target_share = _clamp(
        _finite(policy.get("targetEstimatedMelodyShare"), 0.59), 0.45, 0.75
    )
    minimum_melody_gain = _clamp(
        _finite(policy.get("minimumMelodyGain"), 0.80), 0.25, 1.50
    )
    maximum_melody_gain = _clamp(
        _finite(policy.get("maximumMelodyGain"), 1.30), minimum_melody_gain, 1.50
    )
    minimum_accompaniment_gain = _clamp(
        _finite(policy.get("minimumAccompanimentGain"), 0.25), 0.25, 1.25
    )
    maximum_accompaniment_gain = _clamp(
        _finite(policy.get("maximumAccompanimentGain"), 1.25),
        minimum_accompaniment_gain,
        1.25,
    )
    neutral_gain = _clamp(
        _finite(policy.get("neutralPerformanceGain"), 1.0), 0.25, 1.25
    )

    shaped = [dict(note) for note in notes]
    melody = [
        note for note in shaped if str(note.get("arrangementRole")) == "melody"
    ]
    accompaniment = [
        note for note in shaped if str(note.get("arrangementRole")) != "melody"
    ]
    if not melody or not accompaniment:
        return notes, {
            "applied": False,
            "profile": "adaptive-iowa-melody-mix-v1",
            "reason": "missing-melody-or-accompaniment",
            "melodyNotes": len(melody),
            "accompanimentNotes": len(accompaniment),
        }

    def audible_span(note: dict[str, Any]) -> tuple[float, float]:
        start = max(0.0, _finite(note.get("time"), 0.0))
        hold = max(
            0.01,
            _finite(note.get("audioDuration", note.get("duration")), 0.20),
        )
        return start, start + hold + release_seconds

    melody_spans = _merge_spans([audible_span(note) for note in melody])
    melody_energy = 0.0
    accompaniment_energy = 0.0
    overlapping_accompaniment_ids: set[int] = set()
    for note in shaped:
        start, end = audible_span(note)
        overlap = _overlap_seconds(start, end, melody_spans)
        if overlap <= 0:
            continue
        midi = int(round(_finite(note.get("midi", note.get("pitch")), 60.0)))
        velocity = _clamp(_finite(note.get("velocity"), 0.72), 0.01, 1.0)
        renderer_velocity_gain = 0.12 + 0.88 * (velocity**1.7)
        amplitude = renderer_velocity_gain * _attack_rms(midi)
        energy = amplitude * amplitude * overlap
        if str(note.get("arrangementRole")) == "melody":
            melody_energy += energy
        else:
            accompaniment_energy += energy
            overlapping_accompaniment_ids.add(id(note))

    if melody_energy <= 0 or accompaniment_energy <= 0:
        return notes, {
            "applied": False,
            "profile": "adaptive-iowa-melody-mix-v1",
            "reason": "insufficient-overlapping-energy",
            "melodyEnergy": round(melody_energy, 10),
            "accompanimentEnergy": round(accompaniment_energy, 10),
        }

    melody_amplitude = math.sqrt(melody_energy)
    accompaniment_amplitude = math.sqrt(accompaniment_energy)
    unity_ratio = melody_amplitude / accompaniment_amplitude
    target_ratio = target_share / (1.0 - target_share)
    required_melody_gain = target_ratio / max(1e-12, unity_ratio)
    if required_melody_gain <= maximum_melody_gain:
        melody_gain = _clamp(
            required_melody_gain,
            minimum_melody_gain,
            maximum_melody_gain,
        )
        accompaniment_gain = _clamp(
            melody_gain * unity_ratio / target_ratio,
            minimum_accompaniment_gain,
            maximum_accompaniment_gain,
        )
    else:
        melody_gain = maximum_melody_gain
        accompaniment_gain = _clamp(
            melody_gain * unity_ratio / target_ratio,
            minimum_accompaniment_gain,
            maximum_accompaniment_gain,
        )

    changed = 0
    for note in shaped:
        if str(note.get("arrangementRole")) == "melody":
            new_gain = melody_gain
        elif id(note) in overlapping_accompaniment_ids:
            new_gain = accompaniment_gain
        else:
            new_gain = neutral_gain
        previous = _finite(note.get("performanceGain"), 1.0)
        note["performanceGainBeforeAdaptiveMix"] = round(previous, 6)
        note["performanceGain"] = round(new_gain, 6)
        changed += int(abs(previous - new_gain) > 1e-9)

    predicted_ratio = (melody_gain * melody_amplitude) / max(
        1e-12, accompaniment_gain * accompaniment_amplitude
    )
    predicted_share = predicted_ratio / (1.0 + predicted_ratio)
    unity_share = unity_ratio / (1.0 + unity_ratio)
    return shaped, {
        "applied": True,
        "profile": "adaptive-iowa-melody-mix-v1",
        "energyModel": "iowa-mf-first-500ms-rms-x-renderer-velocity-v1",
        "referenceAnswersUsed": False,
        "targetEstimatedMelodyShare": round(target_share, 6),
        "estimatedUnityMelodyShare": round(unity_share, 6),
        "estimatedBalancedMelodyShare": round(predicted_share, 6),
        "melodyPerformanceGain": round(melody_gain, 6),
        "accompanimentPerformanceGain": round(accompaniment_gain, 6),
        "neutralPerformanceGain": round(neutral_gain, 6),
        "melodyNotes": len(melody),
        "accompanimentNotes": len(accompaniment),
        "overlappingAccompanimentNotes": len(overlapping_accompaniment_ids),
        "changedPerformanceGainNotes": changed,
        "estimatedReleaseSeconds": round(release_seconds, 6),
        "pitchesFrozen": True,
        "onsetsFrozen": True,
        "durationsFrozen": True,
        "hammerVelocitiesFrozen": True,
    }
