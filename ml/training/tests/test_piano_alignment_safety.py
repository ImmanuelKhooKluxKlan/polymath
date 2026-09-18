from __future__ import annotations

import unittest

from ml.training.align_piano_sections import add_source_ranges_to_windows
from ml.training.train_piano_arranger_adapter import (
    alignment_training_ranges,
    manifest_training_pairs,
    note_is_in_training_range,
    pair_supervision_tasks,
    source_index_match_map,
    trusted_alignment_matches,
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

    def test_style_and_register_matches_obey_the_same_trusted_ranges(self) -> None:
        matches = {
            0: {"exactPitch": True},
            1: {"exactPitch": False},
            2: {"exactPitch": True},
        }
        source_notes = [
            {"sourceIndex": 0, "time": 4.0},
            {"sourceIndex": 1, "time": 8.0},
            {"sourceIndex": 2, "time": 12.0},
        ]

        filtered = trusted_alignment_matches(
            matches, source_notes, [(5.0, 10.0)]
        )

        self.assertEqual(set(filtered), {1})

    def test_source_index_pseudo_labels_choose_the_most_literal_target(self) -> None:
        source = [
            {
                "sourceIndex": 0,
                "midi": 60,
                "time": 1.0,
                "duration": 0.5,
                "velocity": 0.7,
                "instrument": "voice",
            }
        ]
        target = {
            "notes": [
                {
                    "sourceIndex": 0,
                    "midi": 72,
                    "time": 1.0,
                    "duration": 0.6,
                    "velocity": 0.8,
                    "generatedBy": "octave-doubling",
                },
                {
                    "sourceIndex": 0,
                    "midi": 60,
                    "time": 1.0,
                    "duration": 0.5,
                    "velocity": 0.7,
                },
                {"sourceIndex": 999, "midi": 64, "time": 2.0, "duration": 0.2},
            ]
        }

        matches = source_index_match_map(source, target)

        self.assertEqual(set(matches), {0})
        self.assertTrue(matches[0]["exactPitch"])
        self.assertEqual(matches[0]["reference"]["midi"], 60)

    def test_auxiliary_pair_can_be_selection_only(self) -> None:
        self.assertEqual(
            pair_supervision_tasks(
                {"id": "teacher", "supervisionTasks": ["selection"]}
            ),
            frozenset({"selection"}),
        )
        with self.assertRaisesRegex(ValueError, "unsupported supervision tasks"):
            pair_supervision_tasks({"id": "bad", "supervisionTasks": ["magic"]})

    def test_manifest_combines_unique_primary_and_auxiliary_pairs(self) -> None:
        combined = manifest_training_pairs(
            {
                "pairs": [{"id": "real"}],
                "auxiliaryDefaults": {
                    "pseudoLabel": True,
                    "supervisionTasks": ["selection"],
                },
                "auxiliaryPairs": [{"id": "teacher"}],
            }
        )
        self.assertEqual([pair["id"] for pair in combined], ["real", "teacher"])
        self.assertTrue(combined[1]["pseudoLabel"])
        self.assertEqual(combined[1]["supervisionTasks"], ["selection"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            manifest_training_pairs(
                {"pairs": [{"id": "same"}], "auxiliaryPairs": [{"id": "same"}]}
            )


if __name__ == "__main__":
    unittest.main()
