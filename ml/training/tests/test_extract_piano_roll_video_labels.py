from __future__ import annotations

import numpy as np

from ml.training.extract_piano_roll_video_labels import (
    VisualEvent,
    active_midis_for_row,
    estimate_row_travel_seconds,
    events_from_timestamps,
    piano_key_centers,
)


def test_full_keyboard_geometry_has_expected_landmarks() -> None:
    centers = piano_key_centers(0.0, 944.0)
    assert len(centers) == 88
    assert round(centers[21], 3) == 9.077
    assert round(centers[36], 3) == 172.462
    assert round(centers[60], 3) == 426.615
    assert round(centers[108], 3) == 934.923


def test_luminous_bar_maps_to_nearest_key_center() -> None:
    centers = piano_key_centers(0.0, 944.0)
    frame = np.zeros((500, 960, 3), dtype=np.uint8)
    center = round(centers[64])
    frame[295:305, center - 12 : center + 13] = 180
    detected = active_midis_for_row(
        frame,
        row=300,
        row_radius=5,
        keyboard_left=0,
        keyboard_right=944,
        centers=centers,
        brightness_threshold=80,
        minimum_column_fill=0.35,
        closing_width=5,
        minimum_run_width=12,
        maximum_run_width=36,
        center_tolerance=5.0,
    )
    assert detected == {64}


def test_timestamps_preserve_retriggers_but_bridge_one_frame_flicker() -> None:
    timestamps = {
        60: [1.0, 1.033, 1.099, 1.132, 1.30, 1.333],
    }
    events = events_from_timestamps(
        timestamps,
        frame_interval=0.033,
        maximum_gap=0.075,
        minimum_duration=0.06,
    )
    assert [(event.midi, round(event.start, 3)) for event in events] == [
        (60, 1.0),
        (60, 1.3),
    ]
    assert round(events[0].duration, 3) == 0.165


def test_cross_row_estimate_rejects_one_large_outlier() -> None:
    primary = [VisualEvent(60 + index % 4, index * 0.4, 0.2) for index in range(30)]
    calibration = [
        VisualEvent(event.midi, event.start + 0.75, event.duration)
        for event in primary
    ]
    calibration[4] = VisualEvent(calibration[4].midi, primary[4].start + 0.98, 0.2)
    delta, retained = estimate_row_travel_seconds(
        primary,
        calibration,
        minimum_delta=0.45,
        maximum_delta=1.05,
    )
    assert round(delta, 3) == 0.75
    assert len(retained) >= 29
