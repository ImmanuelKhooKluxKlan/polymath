import unittest

from ml.training.analyze_full_song_checkpoint_regimes import analyze_predictions


def note(midi: int, time: float) -> dict[str, float | int]:
    return {
        "midi": midi,
        "time": time,
        "duration": 0.4,
        "visualDuration": 0.4,
        "audioDuration": 0.4,
        "velocity": 0.7,
    }


class FullSongCheckpointRegimeAnalysisTests(unittest.TestCase):
    def test_accounts_for_recovered_lost_added_and_removed_events(self) -> None:
        reference = [note(60, 1.0), note(72, 2.0), note(74, 3.0)]
        baseline = [note(60, 1.0), note(72, 2.0), note(40, 2.5)]
        candidate = [note(60, 1.0), note(74, 3.0), note(41, 2.6)]

        report = analyze_predictions(
            reference,
            {"base": baseline, "candidate": candidate},
            [{"start": 0.0, "end": 5.0}],
            baseline_label="base",
        )

        candidate_report = report["predictions"]["candidate"]
        changes = candidate_report["eventChangesAgainstBaseline"]
        self.assertEqual(changes["addedEvents"], 2)
        self.assertEqual(changes["removedEvents"], 2)
        self.assertEqual(changes["100ms"]["recoveredReferenceNotes"], 1)
        self.assertEqual(changes["100ms"]["lostReferenceNotes"], 1)
        self.assertEqual(changes["100ms"]["netReferenceMatches"], 0)
        self.assertEqual(
            candidate_report["perPitchBand"]["C5-B5"]["deltaAgainstBaseline"]["100ms"]["matches"],
            0.0,
        )

    def test_reports_window_specific_false_positive_increase(self) -> None:
        reference = [note(60, 1.0), note(64, 6.0)]
        baseline = [note(60, 1.0), note(64, 6.0)]
        candidate = [note(60, 1.0), note(61, 2.0), note(64, 6.0)]

        report = analyze_predictions(
            reference,
            {"base": baseline, "candidate": candidate},
            [{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 10.0}],
            baseline_label="base",
        )

        windows = report["predictions"]["candidate"]["perWindow"]
        self.assertEqual(windows[0]["deltaAgainstBaseline"]["predictedNotes"], 1)
        self.assertLess(windows[0]["deltaAgainstBaseline"]["100ms"]["f1"], 0)
        self.assertEqual(windows[1]["deltaAgainstBaseline"]["predictedNotes"], 0)
        self.assertEqual(windows[1]["deltaAgainstBaseline"]["100ms"]["f1"], 0.0)

    def test_rejects_missing_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown baseline"):
            analyze_predictions(
                [note(60, 1.0)],
                {"candidate": [note(60, 1.0)]},
                [{"start": 0.0, "end": 2.0}],
                baseline_label="base",
            )


if __name__ == "__main__":
    unittest.main()
