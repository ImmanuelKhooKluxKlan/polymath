from __future__ import annotations

import unittest

from ml.training.align_piano_sections import add_source_ranges_to_windows
from ml.training.train_piano_arranger_adapter import (
    alignment_training_ranges,
    note_is_in_training_range,
)


class PianoAlignmentSafetyTests(unittest.TestCase):
    def test_target_windows_are_mapped_onto_source_time(self) -> None:
        windows = [
            {
                "id": "window-001",
                "targetStartSeconds": 10.0,
                "targetEndSeconds": 20.0,
                "status": "trusted",
            }
        ]
        anchors = [
            {"referenceTime": 0.0, "observedTime": 5.0},
            {"referenceTime": 20.0, "observedTime": 25.0},
        ]

        enriched = add_source_ranges_to_windows(windows, anchors)

        self.assertEqual(enriched[0]["sourceStartSeconds"], 15.0)
        self.assertEqual(enriched[0]["sourceEndSeconds"], 25.0)

    def test_legacy_report_keeps_whole_song_compatibility(self) -> None:
        self.assertIsNone(alignment_training_ranges({"matches": []}))
        self.assertTrue(note_is_in_training_range({"time": 123.0}, None))

    def test_only_trusted_source_windows_become_training_ranges(self) -> None:
        report = {
            "qualityWindows": [
                {
                    "status": "trusted",
                    "sourceStartSeconds": 12.0,
                    "sourceEndSeconds": 22.0,
                },
                {
                    "status": "review",
                    "sourceStartSeconds": 22.0,
                    "sourceEndSeconds": 32.0,
                },
                {
                    "status": "unsafe",
                    "sourceStartSeconds": 32.0,
                    "sourceEndSeconds": 42.0,
                },
            ]
        }

        ranges = alignment_training_ranges(report)

        self.assertEqual(ranges, [(12.0, 22.0)])
        self.assertTrue(note_is_in_training_range({"time": 12.0}, ranges))
        self.assertTrue(note_is_in_training_range({"time": 21.999}, ranges))
        self.assertFalse(note_is_in_training_range({"time": 22.0}, ranges))
        self.assertFalse(note_is_in_training_range({"time": 35.0}, ranges))

    def test_explicit_windows_without_trusted_ranges_train_nothing(self) -> None:
        report = {
            "qualityWindows": [
                {
                    "status": "unsafe",
                    "sourceStartSeconds": 0.0,
                    "sourceEndSeconds": 10.0,
                }
            ]
        }

        ranges = alignment_training_ranges(report)

        self.assertEqual(ranges, [])
        self.assertFalse(note_is_in_training_range({"time": 5.0}, ranges))


if __name__ == "__main__":
    unittest.main()
