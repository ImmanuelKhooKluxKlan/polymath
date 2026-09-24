import unittest

from ml.training.search_overlap_boundary_policy import (
    coverage_edge_distance,
    merge_song_predictions,
    primary_boundary_distance,
)


def note(midi, time, duration=0.4):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "instrument": "acoustic_piano",
    }


class OverlapBoundaryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.primary = [
            {"sourceStart": 0.0, "durationSeconds": 5.0},
            {"sourceStart": 5.0, "durationSeconds": 5.0},
        ]
        self.overlap = [{"sourceStart": 2.5, "durationSeconds": 5.0}]

    def test_coverage_distance_is_largest_at_window_center(self):
        self.assertEqual(coverage_edge_distance(5.0, self.overlap), 2.5)
        self.assertEqual(coverage_edge_distance(2.5, self.overlap), 0.0)
        self.assertIsNone(coverage_edge_distance(1.0, self.overlap))

    def test_primary_boundary_distance(self):
        self.assertAlmostEqual(primary_boundary_distance(4.9), 0.1)
        self.assertAlmostEqual(primary_boundary_distance(7.5), 2.5)

    def test_overlap_replaces_only_the_boundary_zone(self):
        merged = merge_song_predictions(
            [note(60, 3.0), note(61, 5.0), note(62, 7.0)],
            [note(70, 3.0), note(71, 5.0), note(72, 7.0)],
            self.primary,
            self.overlap,
            0.25,
        )
        self.assertEqual([(row["midi"], row["time"]) for row in merged], [
            (60, 3.0),
            (71, 5.0),
            (62, 7.0),
        ])

    def test_primary_is_kept_when_overlap_has_no_coverage(self):
        merged = merge_song_predictions(
            [note(60, 1.0)],
            [note(70, 3.0)],
            self.primary,
            self.overlap,
            1.5,
        )
        self.assertEqual([(row["midi"], row["time"]) for row in merged], [(60, 1.0)])


if __name__ == "__main__":
    unittest.main()
