"""Convert a dense human melody-f0 curve into an auditable note reference.

MedleyDB melody annotations describe frequency at roughly 5.8 ms intervals.
They do not contain note boundaries.  This converter freezes a conservative,
model-independent segmentation policy before a transcription is scored:

* a short median filter removes vibrato-scale frequency jitter;
* semitone hysteresis prevents boundary chatter;
* very short pitch excursions are discarded;
* only brief unvoiced gaps between the same pitch are bridged; and
* notes shorter than the declared musical floor are excluded.

The result is evaluation evidence, not a newly inferred training label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DEFAULT_MEDIAN_FILTER_FRAMES = 9
DEFAULT_HYSTERESIS_SEMITONES = 0.55
DEFAULT_BRIDGE_GAP_SECONDS = 0.08
DEFAULT_MINIMUM_STABLE_RUN_SECONDS = 0.035
DEFAULT_MINIMUM_NOTE_SECONDS = 0.08


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frequency_to_midi(frequency: float) -> float:
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be a positive finite number")
    return 69.0 + 12.0 * math.log2(frequency / 440.0)


def load_f0_csv(path: Path) -> list[tuple[float, float]]:
    frames: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_number}: expected time,frequency")
            try:
                timestamp = float(row[0])
                frequency = float(row[1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: values must be numeric") from error
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"{path}:{line_number}: invalid timestamp")
            if not math.isfinite(frequency) or frequency < 0:
                raise ValueError(f"{path}:{line_number}: invalid frequency")
            if frames and timestamp <= frames[-1][0]:
                raise ValueError(f"{path}:{line_number}: timestamps must strictly increase")
            frames.append((timestamp, frequency))
    if len(frames) < 2:
        raise ValueError(f"{path}: expected at least two f0 frames")
    return frames


def infer_hop_seconds(frames: list[tuple[float, float]]) -> float:
    hops = [right[0] - left[0] for left, right in zip(frames, frames[1:])]
    hop = float(median(hops))
    if not math.isfinite(hop) or hop <= 0:
        raise ValueError("could not infer a positive annotation hop")
    return hop


def _odd_window(value: int) -> int:
    integer = max(1, int(value))
    return integer if integer % 2 else integer + 1


def smooth_voiced_midi(
    frames: list[tuple[float, float]],
    window_frames: int,
) -> list[float | None]:
    radius = _odd_window(window_frames) // 2
    raw = [frequency_to_midi(value) if value > 0 else None for _time, value in frames]
    smoothed: list[float | None] = []
    for index, current in enumerate(raw):
        if current is None:
            smoothed.append(None)
            continue
        local = [
            value
            for value in raw[max(0, index - radius) : min(len(raw), index + radius + 1)]
            if value is not None
        ]
        smoothed.append(float(median(local)) if local else current)
    return smoothed


def quantize_with_hysteresis(
    values: Iterable[float | None],
    hysteresis_semitones: float,
) -> list[int | None]:
    if not 0.5 <= hysteresis_semitones <= 1.5:
        raise ValueError("hysteresis_semitones must be between 0.5 and 1.5")
    labels: list[int | None] = []
    previous: int | None = None
    for value in values:
        if value is None:
            labels.append(None)
            previous = None
            continue
        if previous is not None and abs(value - previous) <= hysteresis_semitones:
            label = previous
        else:
            label = int(round(value))
        labels.append(label)
        previous = label
    return labels


def _runs(labels: list[int | None]) -> list[tuple[int, int, int | None]]:
    if not labels:
        return []
    runs: list[tuple[int, int, int | None]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append((start, index, labels[start]))
            start = index
    return runs


def remove_unstable_pitch_runs(
    labels: list[int | None],
    hop_seconds: float,
    minimum_stable_seconds: float,
) -> list[int | None]:
    output = list(labels)
    for start, end, pitch in _runs(labels):
        if pitch is None or (end - start) * hop_seconds >= minimum_stable_seconds:
            continue
        left = labels[start - 1] if start > 0 else None
        right = labels[end] if end < len(labels) else None
        replacement = left if left is not None and left == right else None
        output[start:end] = [replacement] * (end - start)
    return output


def bridge_same_pitch_gaps(
    labels: list[int | None],
    hop_seconds: float,
    maximum_gap_seconds: float,
) -> list[int | None]:
    output = list(labels)
    for start, end, pitch in _runs(labels):
        if pitch is not None or start == 0 or end >= len(labels):
            continue
        left = labels[start - 1]
        right = labels[end]
        if (
            left is not None
            and left == right
            and (end - start) * hop_seconds <= maximum_gap_seconds
        ):
            output[start:end] = [left] * (end - start)
    return output


def note_events(
    frames: list[tuple[float, float]],
    labels: list[int | None],
    hop_seconds: float,
    minimum_note_seconds: float,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for start, end, pitch in _runs(labels):
        if pitch is None:
            continue
        onset = frames[start][0]
        ending = frames[end - 1][0] + hop_seconds
        duration = ending - onset
        if duration + 1e-12 < minimum_note_seconds:
            continue
        notes.append(
            {
                "midi": int(pitch),
                "time": round(onset, 6),
                "duration": round(duration, 6),
                "visualDuration": round(duration, 6),
                "audioDuration": round(duration, 6),
                "velocity": 0.76,
                "referenceKind": "human-f0-segment",
            }
        )
    return notes


def convert_f0_reference(
    frames: list[tuple[float, float]],
    *,
    median_filter_frames: int = DEFAULT_MEDIAN_FILTER_FRAMES,
    hysteresis_semitones: float = DEFAULT_HYSTERESIS_SEMITONES,
    bridge_gap_seconds: float = DEFAULT_BRIDGE_GAP_SECONDS,
    minimum_stable_run_seconds: float = DEFAULT_MINIMUM_STABLE_RUN_SECONDS,
    minimum_note_seconds: float = DEFAULT_MINIMUM_NOTE_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hop = infer_hop_seconds(frames)
    smoothed = smooth_voiced_midi(frames, median_filter_frames)
    labels = quantize_with_hysteresis(smoothed, hysteresis_semitones)
    labels = remove_unstable_pitch_runs(
        labels, hop, minimum_stable_run_seconds
    )
    labels = bridge_same_pitch_gaps(labels, hop, bridge_gap_seconds)
    notes = note_events(frames, labels, hop, minimum_note_seconds)
    if not notes:
        raise ValueError("the f0 curve produced no stable note events")
    voiced_frames = sum(1 for _time, frequency in frames if frequency > 0)
    return notes, {
        "frameCount": len(frames),
        "voicedFrameCount": voiced_frames,
        "voicedFrameRatio": round(voiced_frames / len(frames), 6),
        "hopSeconds": round(hop, 12),
        "medianFilterFrames": _odd_window(median_filter_frames),
        "pitchHysteresisSemitones": hysteresis_semitones,
        "bridgeSamePitchUnvoicedGapSeconds": bridge_gap_seconds,
        "minimumStableRunSeconds": minimum_stable_run_seconds,
        "minimumOutputNoteSeconds": minimum_note_seconds,
        "outputNotes": len(notes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Human melody reference")
    parser.add_argument("--median-filter-frames", type=int, default=DEFAULT_MEDIAN_FILTER_FRAMES)
    parser.add_argument("--hysteresis-semitones", type=float, default=DEFAULT_HYSTERESIS_SEMITONES)
    parser.add_argument("--bridge-gap-seconds", type=float, default=DEFAULT_BRIDGE_GAP_SECONDS)
    parser.add_argument(
        "--minimum-stable-run-seconds",
        type=float,
        default=DEFAULT_MINIMUM_STABLE_RUN_SECONDS,
    )
    parser.add_argument("--minimum-note-seconds", type=float, default=DEFAULT_MINIMUM_NOTE_SECONDS)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    frames = load_f0_csv(source)
    notes, policy = convert_f0_reference(
        frames,
        median_filter_frames=args.median_filter_frames,
        hysteresis_semitones=args.hysteresis_semitones,
        bridge_gap_seconds=args.bridge_gap_seconds,
        minimum_stable_run_seconds=args.minimum_stable_run_seconds,
        minimum_note_seconds=args.minimum_note_seconds,
    )
    payload = {
        "schema": "polymath-human-f0-note-reference-v1",
        "title": str(args.title)[:160],
        "sourceType": "human-melody-f0-annotation",
        "readyToPlayFormat": "polymath-musician-json-v1",
        "notes": notes,
        "referenceConversion": {
            **policy,
            "sourcePath": str(source),
            "sourceSha256": sha256_file(source),
            "modelOutputWasRead": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **policy}, indent=2))


if __name__ == "__main__":
    main()
