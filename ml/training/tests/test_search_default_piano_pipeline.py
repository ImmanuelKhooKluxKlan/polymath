from __future__ import annotations

import unittest

from ml.training.search_default_piano_pipeline import notes_in_time_window


class NotesInTimeWindowTests(unittest.TestCase):
    def test_uses_half_open_source_clock_window(self) -> None:
        notes = [
            {"time": 0.0, "midi": 60},
            {"time": 4.999, "midi": 61},
            {"time": 5.0, "midi": 62},
            {"time": 10.0, "midi": 63},
        ]

        clipped = notes_in_time_window(
            notes,
            start_seconds=5.0,
            end_seconds=10.0,
        )

        self.assertEqual([62], [note["midi"] for note in clipped])

    def test_rejects_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            notes_in_time_window([], start_seconds=4.0, end_seconds=4.0)

    def test_allows_a_validation_window_before_training(self) -> None:
        notes = [
            {"time": 2.0, "midi": 60},
            {"time": 8.0, "midi": 62},
        ]

        validation = notes_in_time_window(notes, start_seconds=0.0, end_seconds=5.0)
        training = notes_in_time_window(notes, start_seconds=5.0, end_seconds=10.0)

        self.assertEqual([60], [note["midi"] for note in validation])
        self.assertEqual([62], [note["midi"] for note in training])


if __name__ == "__main__":
    unittest.main()
