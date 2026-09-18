"""Fit tempo-independent two-hand piano style from an authored score.

This does not claim that a shortened MIDI is time-aligned ground truth for a
full music video. It learns only reusable arrangement evidence: staff balance,
register bands, and the distribution of rhythmic hold units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * max(0.0, min(1.0, fraction)))]


def _reference_notes(
    payload: dict[str, Any], transpose_semitones: int = 0
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for index, source in enumerate(payload.get("notes") or []):
        if not isinstance(source, dict):
            continue
        midi = round(_finite(source.get("midi"), -1)) + int(transpose_semitones)
        time = _finite(source.get("time"), -1)
        duration = _finite(source.get("duration"), 0)
        if 21 <= midi <= 108 and time >= 0 and duration > 0:
            notes.append({**source, "midi": int(midi), "time": time, "duration": duration, "_index": index})
    if not notes:
        raise ValueError("Reference score contains no playable piano notes.")
    return notes


def _split_staves(notes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    explicit_left = [
        note for note in notes if str(note.get("hand") or "").strip().lower() == "left"
    ]
    explicit_right = [
        note for note in notes if str(note.get("hand") or "").strip().lower() == "right"
    ]
    if explicit_left and explicit_right and len(explicit_left) + len(explicit_right) == len(notes):
        return explicit_left, explicit_right, {
            "method": "explicit-reference-hand-labels",
            "lowerTrack": "left",
            "upperTrack": "right",
            "ignoredMiddleTrackNotes": 0,
        }

    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        track = note.get("sourceTrack")
        if track is None:
            track = note.get("track")
        if track is not None:
            tracks[str(track)].append(note)
    if len(tracks) >= 2:
        ranked = sorted(
            tracks.items(),
            key=lambda item: median(note["midi"] for note in item[1]),
        )
        lower_name, lower = ranked[0]
        upper_name, upper = ranked[-1]
        ignored = sum(len(group) for _, group in ranked[1:-1])
        return lower, upper, {
            "method": "lowest-and-highest-source-track-medians",
            "lowerTrack": lower_name,
            "upperTrack": upper_name,
            "ignoredMiddleTrackNotes": ignored,
        }
    lower = [note for note in notes if note["midi"] < 60]
    upper = [note for note in notes if note["midi"] >= 60]
    if not lower or not upper:
        raise ValueError("Reference score cannot be separated into two hands.")
    return lower, upper, {"method": "midi-60-fallback"}


def _subdivision_seconds(notes: list[dict[str, Any]]) -> float:
    onsets: list[float] = []
    for note in sorted(notes, key=lambda item: item["time"]):
        if not onsets or note["time"] - onsets[-1] > 0.035:
            onsets.append(note["time"])
    gaps = [
        following - previous
        for previous, following in zip(onsets, onsets[1:])
        if 0.065 <= following - previous <= 0.55
    ]
    if not gaps:
        return 0.20
    cutoff = _quantile(gaps, 0.50)
    return max(0.07, min(0.35, _quantile([gap for gap in gaps if gap <= cutoff + 0.012], 0.50)))


def _register_summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    midis = sorted(note["midi"] for note in notes)
    return {
        "minimumMidi": int(midis[0]),
        "p10Midi": round(_quantile(midis, 0.10), 3),
        "medianMidi": round(_quantile(midis, 0.50), 3),
        "p90Midi": round(_quantile(midis, 0.90), 3),
        "maximumMidi": int(midis[-1]),
    }


def _duration_histogram(notes: list[dict[str, Any]], subdivision: float) -> tuple[list[dict[str, Any]], float]:
    counts: Counter[float] = Counter()
    release_ratios: list[float] = []
    for note in notes:
        units = float(max(1, min(32, round(note["duration"] / subdivision))))
        counts[units] += 1
        release_ratios.append(note["duration"] / (units * subdivision))
    total = sum(counts.values())
    histogram = [
        {"units": units, "notes": count, "share": round(count / total, 6)}
        for units, count in sorted(counts.items())
    ]
    return histogram, max(0.70, min(1.10, median(release_ratios)))


def _profile_sha256(profile: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(profile))
    clone.pop("profileSha256", None)
    encoded = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit(
    base: dict[str, Any],
    reference: dict[str, Any],
    *,
    profile_id: str,
    density: float,
    melody_upper_shift: int = 0,
    bass_lower_shift: int = 0,
    harmony_upper_shift: int = 0,
    harmony_lower_shift: int = 0,
    reference_transpose_semitones: int = 0,
    preserve_base_density: bool = False,
) -> dict[str, Any]:
    notes = _reference_notes(reference, reference_transpose_semitones)
    lower, upper, split = _split_staves(notes)
    subdivision = _subdivision_seconds(notes)
    lower_histogram, lower_release = _duration_histogram(lower, subdivision)
    upper_histogram, upper_release = _duration_histogram(upper, subdivision)
    written_release_ratio = median([lower_release, upper_release])

    result = json.loads(json.dumps(base))
    result["id"] = profile_id
    decoder = result.setdefault("decoder", {})
    decoder["preferredGlobalRegisterShiftSemitones"] = 0
    if not preserve_base_density:
        adaptive = decoder.setdefault("adaptiveSourceDensity", {})
        adaptive["lowDensityMultiplier"] = density
        adaptive["highDensityMultiplier"] = density
        adaptive["longSourceDurationWeight"] = 0.65
    decoder["authoredTwoHandStyle"] = {
        "enabled": True,
        "schema": "polymath-authored-two-hand-style-v1",
        "harmonyHandAssignment": "preserve-native-register",
        "targetUpperNoteShare": round(len(upper) / len(notes), 6),
        "lower": _register_summary(lower),
        "upper": _register_summary(upper),
        "preferredOctaveShiftsSemitones": {
            "melodyUpper": melody_upper_shift,
            "bassLower": bass_lower_shift,
            "harmonyUpper": harmony_upper_shift,
            "harmonyLower": harmony_lower_shift,
        },
        "durationGrid": {
            "enabled": True,
            "referenceSubdivisionSeconds": round(subdivision, 6),
            "writtenReleaseRatio": round(written_release_ratio, 6),
            "maximumUnits": 12,
            "lowerHistogram": lower_histogram,
            "upperHistogram": upper_histogram,
        },
        "referenceEvidence": {
            "notes": len(notes),
            "lowerNotes": len(lower),
            "upperNotes": len(upper),
            "staffSplit": split,
            "registerTransposeSemitones": int(reference_transpose_semitones),
            "scope": "style-only-not-time-aligned-note-accuracy",
        },
    }
    result.setdefault("trainingProvenance", {})["authoredStyleReference"] = {
        "scope": "register-staff-balance-and-tempo-relative-duration-only",
        "warning": "The reference may be shortened or differently timed and is not treated as frame-aligned truth.",
    }
    result["profileSha256"] = _profile_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-id", default="pianella-authored-two-hand-v047")
    parser.add_argument("--adaptive-density", type=float, default=1.20)
    parser.add_argument(
        "--preserve-base-density",
        action="store_true",
        help="Change only two-hand register/duration style and keep the base decoder density policy intact.",
    )
    parser.add_argument("--melody-upper-shift", type=int, choices=(-12, 0, 12), default=0)
    parser.add_argument("--bass-lower-shift", type=int, choices=(-12, 0, 12), default=0)
    parser.add_argument("--harmony-upper-shift", type=int, choices=(-12, 0, 12), default=0)
    parser.add_argument("--harmony-lower-shift", type=int, choices=(-12, 0, 12), default=0)
    parser.add_argument(
        "--reference-transpose-semitones",
        type=int,
        choices=(-24, -12, 0, 12, 24),
        default=0,
        help="Transpose only the learned register bands; timing and pitch classes are unchanged.",
    )
    args = parser.parse_args()
    base_path = Path(args.base_profile).resolve()
    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()
    result = fit(
        json.loads(base_path.read_text(encoding="utf-8-sig")),
        json.loads(reference_path.read_text(encoding="utf-8-sig")),
        profile_id=args.profile_id,
        density=max(0.5, min(3.0, args.adaptive_density)),
        melody_upper_shift=args.melody_upper_shift,
        bass_lower_shift=args.bass_lower_shift,
        harmony_upper_shift=args.harmony_upper_shift,
        harmony_lower_shift=args.harmony_lower_shift,
        reference_transpose_semitones=args.reference_transpose_semitones,
        preserve_base_density=args.preserve_base_density,
    )
    result["trainingProvenance"]["authoredStyleReference"].update({
        "referencePath": str(reference_path),
        "referenceSha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "baseProfilePath": str(base_path),
        "baseProfileSha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    })
    result["profileSha256"] = _profile_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "profileId": result["id"],
        "profileSha256": result["profileSha256"],
        "authoredTwoHandStyle": result["decoder"]["authoredTwoHandStyle"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
