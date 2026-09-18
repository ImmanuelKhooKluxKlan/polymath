import unittest

from ml.training.evaluate_piano_arranger import (
    evaluate,
    normalize_notes,
    notes_inside_ranges,
    prepare_reference_notes,
    reference_transpose_semitones,
    trusted_source_ranges,
)


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

    def test_duration_scoring_does_not_reward_a_wrong_octave(self):
        reference = [
            {"midi": 60, "time": 1.0, "duration": 1.0, "velocity": 0.8}
        ]
        observed = [
            {
                "midi": 72,
                "time": 1.0,
                "duration": 0.1,
                "visualDuration": 0.1,
                "audioDuration": 0.1,
                "velocity": 0.5,
            }
        ]

        metrics = evaluate(reference, observed)

        self.assertEqual(metrics["exactPitchOnset250ms"]["matches"], 0)
        self.assertEqual(metrics["duration"]["matchedNotes"], 1)
        self.assertEqual(metrics["duration"]["severeCutoffs"], 1)
        self.assertEqual(
            metrics["durationMatchingPolicy"], "pitch-class-onset-250ms"
        )
        self.assertAlmostEqual(metrics["velocity"]["meanAbsoluteError"], 0.3)

    def test_predeclared_reference_octave_is_applied_before_scoring(self):
        notes = normalize_notes(
            {"notes": [{"midi": 60, "time": 1.0, "duration": 0.5}]},
            transpose_semitones=12,
        )

        self.assertEqual(notes[0]["midi"], 72)
        self.assertEqual(reference_transpose_semitones({}), 0)
        self.assertEqual(
            reference_transpose_semitones({"referenceTransposeSemitones": 12}),
            12,
        )

    def test_reference_transpose_rejects_non_octave_candidate_corrections(self):
        with self.assertRaisesRegex(ValueError, "octave multiple"):
            reference_transpose_semitones({"referenceTransposeSemitones": 7})

    def test_already_aligned_reference_is_not_warped_twice(self):
        payload = {"notes": [{"midi": 60, "time": 4.0, "duration": 1.0}]}
        report = {
            "anchors": [
                {"referenceTime": 0.0, "observedTime": 10.0},
                {"referenceTime": 10.0, "observedTime": 20.0},
            ]
        }

        kept = prepare_reference_notes(
            {"referenceAlreadyAligned": True}, payload, report
        )
        warped = prepare_reference_notes({}, payload, report)

        self.assertEqual(kept[0]["time"], 4.0)
        self.assertEqual(warped[0]["time"], 14.0)

    def test_partial_reference_cutoffs_use_the_correct_clock(self):
        payload = {
            "notes": [
                {"midi": 60, "time": 2.0, "duration": 0.5},
                {"midi": 62, "time": 7.0, "duration": 0.5},
            ]
        }
        report = {
            "anchors": [
                {"referenceTime": 0.0, "observedTime": 5.0},
                {"referenceTime": 10.0, "observedTime": 15.0},
            ]
        }

        notes = prepare_reference_notes(
            {"referenceEndSeconds": 6.0, "candidateEndSeconds": 8.0},
            payload,
            report,
        )

        self.assertEqual([(note["midi"], note["time"]) for note in notes], [(60, 7.0)])


if __name__ == "__main__":
    unittest.main()
