"""Extract note labels from a fixed-camera falling-note piano video.

This is deliberately a label-generation tool, not a general video
transcriber.  It targets recordings where:

* an 88-key piano is stationary and spans a known horizontal interval;
* rectangular note bars fall vertically at a constant speed; and
* the bars reach a known strike row when the corresponding key sounds.

Two observation rows are decoded.  The median travel time between them is
used to extrapolate each note to the strike row.  This keeps label timing
independent of any audio transcription that will later be trained against
the labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import numpy as np


WHITE_PITCH_CLASSES = frozenset({0, 2, 4, 5, 7, 9, 11})
LOWEST_MIDI = 21
HIGHEST_MIDI = 108
WHITE_KEY_COUNT = 52


class PianoRollExtractionError(RuntimeError):
    """Raised when visual evidence is insufficient or inconsistent."""


@dataclass(frozen=True)
class VisualEvent:
    midi: int
    start: float
    duration: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midi_note_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def piano_key_centers(left: float, right: float) -> dict[int, float]:
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        raise PianoRollExtractionError("keyboard bounds must be finite and increasing")
    white_width = (right - left) / WHITE_KEY_COUNT
    centers: dict[int, float] = {}
    white_before = 0
    for midi in range(LOWEST_MIDI, HIGHEST_MIDI + 1):
        if midi % 12 in WHITE_PITCH_CLASSES:
            centers[midi] = left + (white_before + 0.5) * white_width
            white_before += 1
        else:
            centers[midi] = left + white_before * white_width
    if white_before != WHITE_KEY_COUNT:
        raise PianoRollExtractionError("internal 88-key geometry error")
    return centers


def boolean_runs(values: Iterable[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    materialized = list(values)
    for index, value in enumerate([*materialized, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def active_midis_for_row(
    frame: np.ndarray,
    *,
    row: int,
    row_radius: int,
    keyboard_left: int,
    keyboard_right: int,
    centers: dict[int, float],
    brightness_threshold: int,
    minimum_column_fill: float,
    closing_width: int,
    minimum_run_width: int,
    maximum_run_width: int,
    center_tolerance: float,
) -> set[int]:
    height, width = frame.shape[:2]
    y0 = max(0, row - row_radius)
    y1 = min(height, row + row_radius)
    x0 = max(0, keyboard_left)
    x1 = min(width, keyboard_right)
    if y1 <= y0 or x1 <= x0:
        raise PianoRollExtractionError("observation strip falls outside the video frame")

    # max(B, G, R) is the HSV value channel without a full-frame conversion.
    # Falling bars are luminous against a nearly black background.
    value = frame[y0:y1, x0:x1].max(axis=2)
    active_columns = (value > brightness_threshold).mean(axis=0) >= minimum_column_fill
    if closing_width > 1:
        kernel = np.ones((1, closing_width), dtype=np.uint8)
        active_columns = cv2.morphologyEx(
            active_columns.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, kernel
        )[0].astype(bool)

    midi_values = np.array(sorted(centers), dtype=np.int16)
    center_values = np.array([centers[int(midi)] for midi in midi_values], dtype=np.float64)
    detected: set[int] = set()
    for local_start, local_end in boolean_runs(active_columns.tolist()):
        run_width = local_end - local_start
        if not minimum_run_width <= run_width <= maximum_run_width:
            continue
        center = x0 + (local_start + local_end - 1) / 2.0
        nearest = int(np.argmin(np.abs(center_values - center)))
        if abs(float(center_values[nearest]) - center) <= center_tolerance:
            detected.add(int(midi_values[nearest]))
    return detected


def events_from_timestamps(
    timestamps_by_midi: dict[int, list[float]],
    *,
    frame_interval: float,
    maximum_gap: float,
    minimum_duration: float,
) -> list[VisualEvent]:
    events: list[VisualEvent] = []
    for midi, timestamps in sorted(timestamps_by_midi.items()):
        if not timestamps:
            continue
        start = previous = float(timestamps[0])
        for timestamp in [*timestamps[1:], math.inf]:
            timestamp = float(timestamp)
            if timestamp - previous > maximum_gap:
                duration = previous - start + frame_interval
                if duration >= minimum_duration:
                    events.append(VisualEvent(midi=midi, start=start, duration=duration))
                start = timestamp
            previous = timestamp
    return sorted(events, key=lambda event: (event.start, event.midi))


def estimate_row_travel_seconds(
    primary: list[VisualEvent],
    calibration: list[VisualEvent],
    *,
    minimum_delta: float,
    maximum_delta: float,
) -> tuple[float, list[float]]:
    calibration_by_midi: dict[int, list[VisualEvent]] = defaultdict(list)
    for event in calibration:
        calibration_by_midi[event.midi].append(event)

    deltas: list[float] = []
    used: dict[int, set[int]] = defaultdict(set)
    for event in primary:
        choices: list[tuple[float, int]] = []
        for index, candidate in enumerate(calibration_by_midi.get(event.midi, [])):
            if index in used[event.midi]:
                continue
            delta = candidate.start - event.start
            if minimum_delta <= delta <= maximum_delta:
                duration_error = abs(candidate.duration - event.duration)
                choices.append((duration_error + 0.05 * delta, index))
        if not choices:
            continue
        _score, index = min(choices)
        used[event.midi].add(index)
        deltas.append(calibration_by_midi[event.midi][index].start - event.start)

    if len(deltas) < 20:
        raise PianoRollExtractionError(
            f"only {len(deltas)} cross-row note tracks were found; at least 20 are required"
        )
    center = median(deltas)
    deviations = [abs(delta - center) for delta in deltas]
    mad = median(deviations)
    tolerance = max(0.05, 4.0 * mad)
    retained = [delta for delta in deltas if abs(delta - center) <= tolerance]
    if len(retained) < 20:
        raise PianoRollExtractionError(
            f"only {len(retained)} stable cross-row tracks remained after outlier rejection"
        )
    return float(median(retained)), retained


def extract_labels(
    video_path: Path,
    *,
    title: str,
    keyboard_left: int,
    keyboard_right: int,
    observation_row: int,
    calibration_row: int,
    strike_row: int,
    video_to_audio_offset: float,
    start_seconds: float,
    end_seconds: float | None,
    brightness_threshold: int,
    minimum_column_fill: float,
    row_radius: int,
    closing_width: int,
    minimum_run_width: int,
    maximum_run_width: int,
    center_tolerance: float,
    maximum_gap: float,
    minimum_duration: float,
    minimum_row_delta: float,
    maximum_row_delta: float,
    velocity: float,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise PianoRollExtractionError(f"video does not exist: {video_path}")
    if not observation_row < calibration_row < strike_row:
        raise PianoRollExtractionError(
            "expected observation_row < calibration_row < strike_row"
        )
    if not 0.0 < velocity <= 1.0:
        raise PianoRollExtractionError("velocity must be in (0, 1]")

    centers = piano_key_centers(float(keyboard_left), float(keyboard_right))
    primary_times: dict[int, list[float]] = defaultdict(list)
    calibration_times: dict[int, list[float]] = defaultdict(list)
    frame_times: list[float] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PianoRollExtractionError(f"OpenCV could not open: {video_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            frame_times.append(timestamp)
            if timestamp < start_seconds:
                continue
            if end_seconds is not None and timestamp > end_seconds:
                break
            decoded_frames += 1
            common = {
                "frame": frame,
                "row_radius": row_radius,
                "keyboard_left": keyboard_left,
                "keyboard_right": keyboard_right,
                "centers": centers,
                "brightness_threshold": brightness_threshold,
                "minimum_column_fill": minimum_column_fill,
                "closing_width": closing_width,
                "minimum_run_width": minimum_run_width,
                "maximum_run_width": maximum_run_width,
                "center_tolerance": center_tolerance,
            }
            for midi in active_midis_for_row(row=observation_row, **common):
                primary_times[midi].append(timestamp)
            for midi in active_midis_for_row(row=calibration_row, **common):
                calibration_times[midi].append(timestamp)
    finally:
        capture.release()

    positive_intervals = [
        right - left
        for left, right in zip(frame_times, frame_times[1:])
        if right > left
    ]
    if not positive_intervals:
        raise PianoRollExtractionError("video yielded no increasing frame timestamps")
    frame_interval = float(median(positive_intervals))
    primary_events = events_from_timestamps(
        primary_times,
        frame_interval=frame_interval,
        maximum_gap=maximum_gap,
        minimum_duration=minimum_duration,
    )
    calibration_events = events_from_timestamps(
        calibration_times,
        frame_interval=frame_interval,
        maximum_gap=maximum_gap,
        minimum_duration=minimum_duration,
    )
    row_travel_seconds, stable_deltas = estimate_row_travel_seconds(
        primary_events,
        calibration_events,
        minimum_delta=minimum_row_delta,
        maximum_delta=maximum_row_delta,
    )
    pixels_between_rows = calibration_row - observation_row
    travel_to_strike = row_travel_seconds * (
        (strike_row - observation_row) / pixels_between_rows
    )
    total_time_shift = video_to_audio_offset + travel_to_strike

    notes = [
        {
            "midi": event.midi,
            "note": midi_note_name(event.midi),
            "time": round(max(0.0, event.start + total_time_shift), 6),
            "duration": round(max(0.01, event.duration), 6),
            "visualDuration": round(max(0.01, event.duration), 6),
            "audioDuration": round(max(0.01, event.duration), 6),
            "velocity": round(velocity, 4),
            "instrument": "acoustic_piano",
            "hand": "left" if event.midi < 60 else "right",
            "source": "piano-roll-video-geometry-v1",
        }
        for event in primary_events
    ]
    return {
        "schema": "polymath-ready-to-play-v1",
        "title": title,
        "instrument": "acoustic_piano",
        "sourceType": "piano-roll-video-geometry",
        "notes": notes,
        "durationSeconds": round(max((note["time"] + note["duration"] for note in notes), default=0.0), 6),
        "visualExtraction": {
            "schema": "polymath-piano-roll-video-extraction-v1",
            "sourceVideo": str(video_path.resolve()),
            "sourceVideoSha256": sha256_file(video_path),
            "frameSize": {"width": frame_width, "height": frame_height},
            "decodedFrames": decoded_frames,
            "medianFrameIntervalSeconds": round(frame_interval, 9),
            "keyboard": {
                "lowestMidi": LOWEST_MIDI,
                "highestMidi": HIGHEST_MIDI,
                "leftPixel": keyboard_left,
                "rightPixel": keyboard_right,
            },
            "rows": {
                "observation": observation_row,
                "calibration": calibration_row,
                "strike": strike_row,
            },
            "timing": {
                "videoToAudioOffsetSeconds": round(video_to_audio_offset, 9),
                "medianObservationToCalibrationSeconds": round(row_travel_seconds, 9),
                "travelToStrikeSeconds": round(travel_to_strike, 9),
                "totalVisualToAudioShiftSeconds": round(total_time_shift, 9),
                "crossRowTracks": len(stable_deltas),
                "crossRowDeltaMinimumSeconds": round(min(stable_deltas), 9),
                "crossRowDeltaMaximumSeconds": round(max(stable_deltas), 9),
            },
            "detection": {
                "brightnessThreshold": brightness_threshold,
                "minimumColumnFill": minimum_column_fill,
                "rowRadius": row_radius,
                "closingWidth": closing_width,
                "minimumRunWidth": minimum_run_width,
                "maximumRunWidth": maximum_run_width,
                "centerTolerancePixels": center_tolerance,
                "maximumGapSeconds": maximum_gap,
                "minimumDurationSeconds": minimum_duration,
                "primaryEvents": len(primary_events),
                "calibrationEvents": len(calibration_events),
            },
            "limitations": [
                "Velocity is not encoded reliably by this visualization and is assigned a constant value.",
                "Adjacent note bars that visually overlap may require manual review.",
                "This geometry is valid only for the fixed keyboard crop recorded in the source video.",
            ],
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--keyboard-left", type=int, default=0)
    parser.add_argument("--keyboard-right", type=int, default=944)
    parser.add_argument("--observation-row", type=int, default=300)
    parser.add_argument("--calibration-row", type=int, default=380)
    parser.add_argument("--strike-row", type=int, default=430)
    parser.add_argument("--video-to-audio-offset", type=float, default=0.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--brightness-threshold", type=int, default=80)
    parser.add_argument("--minimum-column-fill", type=float, default=0.35)
    parser.add_argument("--row-radius", type=int, default=5)
    parser.add_argument("--closing-width", type=int, default=5)
    parser.add_argument("--minimum-run-width", type=int, default=12)
    parser.add_argument("--maximum-run-width", type=int, default=36)
    parser.add_argument("--center-tolerance", type=float, default=5.0)
    parser.add_argument("--maximum-gap", type=float, default=0.075)
    parser.add_argument("--minimum-duration", type=float, default=0.06)
    parser.add_argument("--minimum-row-delta", type=float, default=0.45)
    parser.add_argument("--maximum-row-delta", type=float, default=1.05)
    parser.add_argument("--velocity", type=float, default=0.78)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.out.exists():
        raise PianoRollExtractionError(f"refusing to overwrite existing output: {args.out}")
    result = extract_labels(
        args.video.resolve(),
        title=args.title,
        keyboard_left=args.keyboard_left,
        keyboard_right=args.keyboard_right,
        observation_row=args.observation_row,
        calibration_row=args.calibration_row,
        strike_row=args.strike_row,
        video_to_audio_offset=args.video_to_audio_offset,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        brightness_threshold=args.brightness_threshold,
        minimum_column_fill=args.minimum_column_fill,
        row_radius=args.row_radius,
        closing_width=args.closing_width,
        minimum_run_width=args.minimum_run_width,
        maximum_run_width=args.maximum_run_width,
        center_tolerance=args.center_tolerance,
        maximum_gap=args.maximum_gap,
        minimum_duration=args.minimum_duration,
        minimum_row_delta=args.minimum_row_delta,
        maximum_row_delta=args.maximum_row_delta,
        velocity=args.velocity,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out.resolve()),
        "sha256": sha256_file(args.out.resolve()),
        "notes": len(result["notes"]),
        "timing": result["visualExtraction"]["timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
