import unittest

from ml.training.evaluate_piano_arranger import notes_inside_ranges, trusted_source_ranges


class PianoArrangerEvaluationSafetyTests(unittest.TestCase):
    def test_explicit_windows_only_score_approved_source_ranges(self):
        report = {
            "qualityWindows": [
                {"sourceStartSeconds": 0.0, "sourceEndSeconds": 5.0, "status": "unsafe"},
                {"sourceStartSeconds": 5.0, "sourceEndSeconds": 10.0, "status": "trusted"},
                {"sourceStartSeconds": 10.0, "sourceEndSeconds": 15.0, "status": "review"},
                {"sourceStartSeconds": 15.0, "sourceEndSeconds": 20.0, "status": "accepted-manually"},
            ]
        }
        ranges = trusted_source_ranges(report)
        notes = [
            {"midi": 60, "time": 2.0, "duration": 0.2},
            {"midi": 61, "time": 7.0, "duration": 0.2},
            {"midi": 62, "time": 12.0, "duration": 0.2},
            {"midi": 63, "time": 17.0, "duration": 0.2},
        ]
        self.assertEqual(ranges, [(5.0, 10.0), (15.0, 20.0)])
        self.assertEqual(
            [note["midi"] for note in notes_inside_ranges(notes, ranges)],
            [61, 63],
        )

    def test_legacy_report_retains_whole_song_compatibility(self):
        self.assertIsNone(trusted_source_ranges({"anchors": []}))

    def test_explicit_unapproved_report_fails_closed(self):
        ranges = trusted_source_ranges(
            {"qualityWindows": [{"sourceStart": 0.0, "sourceEnd": 5.0, "status": "unsafe"}]}
        )
        self.assertEqual(ranges, [])
        self.assertEqual(notes_inside_ranges([{"midi": 60, "time": 2.0}], ranges), [])


if __name__ == "__main__":
    unittest.main()
