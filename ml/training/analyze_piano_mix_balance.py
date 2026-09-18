"""Measure melody-versus-accompaniment balance through the fixed piano renderer.

MIDI velocity is not a loudness percentage: three accompaniment notes can sum
to more energy than one melody note, and the sampled piano has different levels
across its register.  This diagnostic therefore renders the two arrangement
roles as isolated stems with the same non-normalizing chain used for blind
listening tests.  It measures RMS amplitude while melody is active and reports
an intuitive lead share, ``lead / (lead + accompaniment)``.

The result is a reproducible mix diagnostic, not a mastering claim or a LUFS
measurement.  It never changes a checkpoint or production configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support package execution and direct script execution.
    from .render_piano_json_audio import (
        RELEASE_SECONDS,
        SAMPLE_RATE,
        normalize_render_notes,
        read_pcm16_stereo,
        render_payload,
        sha256_file,
    )
except ImportError:  # pragma: no cover - used by the direct CLI path.
    from render_piano_json_audio import (
        RELEASE_SECONDS,
        SAMPLE_RATE,
        normalize_render_notes,
        read_pcm16_stereo,
        render_payload,
        sha256_file,
    )


def payload_for_roles(
    payload: dict[str, Any],
    roles: set[str],
    *,
    invert: bool = False,
) -> dict[str, Any]:
    """Return a metadata-preserving payload containing selected role names."""

    selected = []
    for note in payload.get("notes", []):
        if not isinstance(note, dict):
            continue
        role = str(note.get("arrangementRole") or "").strip().lower()
        matches = role in roles
        if matches != invert:
            selected.append(note)
    result = dict(payload)
    result["notes"] = selected
    return result


def active_mask(
    payload: dict[str, Any],
    frames: int,
    *,
    release_seconds: float = RELEASE_SECONDS,
) -> np.ndarray:
    """Return the union of audible note spans for one rendered stem."""

    mask = np.zeros(max(0, frames), dtype=bool)
    for note in normalize_render_notes(payload):
        start = max(0, round(float(note["time"]) * SAMPLE_RATE))
        end = min(
            frames,
            math.ceil(
                (float(note["time"]) + float(note["duration"]) + release_seconds)
                * SAMPLE_RATE
            ),
        )
        if start < end:
            mask[start:end] = True
    return mask


def rms(audio: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        audio = audio[mask]
    if not audio.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def amplitude_share(lead_rms: float, accompaniment_rms: float) -> float:
    total = lead_rms + accompaniment_rms
    return lead_rms / total if total > 0 else 0.0


def db_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return 20.0 * math.log10(numerator / denominator)


def analyze_mix_balance(
    payload: dict[str, Any],
    samples_directory: Path,
    output_directory: Path,
    *,
    duration_seconds: float | None = None,
    target_lead_share: float = 0.6,
    tolerance: float = 0.04,
) -> dict[str, Any]:
    if not 0 < target_lead_share < 1:
        raise ValueError("target lead share must be between zero and one")
    if not 0 <= tolerance < 0.5:
        raise ValueError("tolerance must be at least zero and below 0.5")

    lead_payload = payload_for_roles(payload, {"melody"})
    accompaniment_payload = payload_for_roles(payload, {"melody"}, invert=True)
    lead_count = len(normalize_render_notes(lead_payload))
    accompaniment_count = len(normalize_render_notes(accompaniment_payload))
    if not lead_count:
        raise ValueError("Input JSON has no playable arrangementRole=melody notes")
    if not accompaniment_count:
        raise ValueError("Input JSON has no playable accompaniment notes")

    # Keep every stem on one identical timeline. Letting each isolated role
    # stop at its own final note would bias whole-song RMS and make the WAVs
    # awkward to compare in a DAW.
    if duration_seconds is None:
        all_notes = normalize_render_notes(payload)
        duration_seconds = max(
            float(note["time"]) + float(note["duration"]) + RELEASE_SECONDS
            for note in all_notes
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    lead_path = output_directory / "01-MELODY-STEM.wav"
    accompaniment_path = output_directory / "02-ACCOMPANIMENT-STEM.wav"
    combined_path = output_directory / "03-COMBINED.wav"
    lead_render = render_payload(
        lead_payload,
        samples_directory,
        lead_path,
        duration_seconds=duration_seconds,
    )
    accompaniment_render = render_payload(
        accompaniment_payload,
        samples_directory,
        accompaniment_path,
        duration_seconds=duration_seconds,
    )
    combined_render = render_payload(
        payload,
        samples_directory,
        combined_path,
        duration_seconds=duration_seconds,
    )

    lead_audio = read_pcm16_stereo(lead_path)
    accompaniment_audio = read_pcm16_stereo(accompaniment_path)
    frames = min(len(lead_audio), len(accompaniment_audio))
    lead_audio = lead_audio[:frames]
    accompaniment_audio = accompaniment_audio[:frames]
    mask = active_mask(lead_payload, frames)
    lead_active_rms = rms(lead_audio, mask)
    accompaniment_during_lead_rms = rms(accompaniment_audio, mask)
    active_share = amplitude_share(lead_active_rms, accompaniment_during_lead_rms)
    whole_lead_rms = rms(lead_audio)
    whole_accompaniment_rms = rms(accompaniment_audio)
    whole_share = amplitude_share(whole_lead_rms, whole_accompaniment_rms)

    result = {
        "schema": "polymath-piano-mix-balance-v1",
        "definition": (
            "RMS amplitude share = melody RMS / (melody RMS + accompaniment RMS); "
            "primary measurement uses only spans where melody is audibly active."
        ),
        "target": {
            "melodyShare": round(target_lead_share, 6),
            "accompanimentShare": round(1.0 - target_lead_share, 6),
            "tolerance": round(tolerance, 6),
        },
        "melodyActive": {
            "seconds": round(float(np.count_nonzero(mask)) / SAMPLE_RATE, 6),
            "melodyRms": round(lead_active_rms, 8),
            "accompanimentRms": round(accompaniment_during_lead_rms, 8),
            "melodyShare": round(active_share, 6),
            "accompanimentShare": round(1.0 - active_share, 6),
            "melodyDbOverAccompaniment": (
                round(value, 4)
                if (value := db_ratio(lead_active_rms, accompaniment_during_lead_rms))
                is not None
                else None
            ),
            "targetMet": abs(active_share - target_lead_share) <= tolerance,
        },
        "wholeSong": {
            "melodyRms": round(whole_lead_rms, 8),
            "accompanimentRms": round(whole_accompaniment_rms, 8),
            "melodyShare": round(whole_share, 6),
            "accompanimentShare": round(1.0 - whole_share, 6),
        },
        "notes": {
            "melody": lead_count,
            "accompaniment": accompaniment_count,
            "total": lead_count + accompaniment_count,
        },
        "renders": {
            "melody": lead_render,
            "accompaniment": accompaniment_render,
            "combined": combined_render,
        },
    }
    report_path = output_directory / "MIX-BALANCE.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    result["report"] = str(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--target-lead-share", type=float, default=0.6)
    parser.add_argument("--tolerance", type=float, default=0.04)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    result = analyze_mix_balance(
        json.loads(input_path.read_text(encoding="utf-8-sig")),
        Path(args.samples).resolve(),
        Path(args.output_dir).resolve(),
        duration_seconds=args.duration_seconds,
        target_lead_share=args.target_lead_share,
        tolerance=args.tolerance,
    )
    result["sourceJson"] = str(input_path)
    result["sourceJsonSha256"] = sha256_file(input_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
