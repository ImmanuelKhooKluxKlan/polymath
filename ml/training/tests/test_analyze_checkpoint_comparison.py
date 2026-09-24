import unittest

from ml.training.analyze_checkpoint_comparison import (
    analyze_song_change,
    merge_profiles,
    profile_rates,
)


def note(midi, time, velocity=0.75):
    return {
        "midi": midi,
        "time": time,
        "duration": 0.4,
        "velocity": velocity,
        "instrument": "acoustic_piano",
    }


class CheckpointChangeAttributionTests(unittest.TestCase):
    def test_recovered_and_regressed_notes_are_separated(self):
        references = [
            note(60, 0.1),
            note(62, 1.0, 0.15),
            note(64, 2.0),
            note(65, 3.0),
        ]
        baseline = [note(60, 0.1), note(64, 2.0), note(67, 4.0)]
        candidate = [note(60, 0.1), note(62, 1.0), note(69, 4.0)]
        result = analyze_song_change(references, baseline, candidate)
        self.assertEqual(result["commonMatches"], 1)
        self.assertEqual(result["recoveredReferenceNotes"], 1)
        self.assertEqual(result["regressedReferenceNotes"], 1)
        self.assertEqual(result["persistentMisses"], 1)
        self.assertEqual(result["netMatchChange"], 0)
        self.assertEqual(result["baselineFalsePositives"], 1)
        self.assertEqual(result["candidateFalsePositives"], 1)
        self.assertEqual(result["profiles"]["recovered"]["velocity"]["under_0.20"], 1)

    def test_nearby_exact_pitch_is_a_match(self):
        result = analyze_song_change(
            [note(60, 1.0)],
            [],
            [note(60, 1.08)],
            tolerance=0.10,
        )
        self.assertEqual(result["recoveredReferenceNotes"], 1)
        self.assertEqual(result["candidateMatchedNotes"], 1)

    def test_profile_uses_the_full_reference_chord_context(self):
        references = [note(60, 1.0), note(64, 1.01), note(67, 1.02)]
        result = analyze_song_change(
            references,
            [note(60, 1.0), note(67, 1.02)],
            [note(60, 1.0), note(64, 1.01), note(67, 1.02)],
        )
        self.assertEqual(
            result["profiles"]["recovered"]["chordSize"]["two_or_three"],
            1,
        )

    def test_profile_merging_sums_each_dimension(self):
        first = analyze_song_change([note(60, 1.0, 0.1)], [], [note(60, 1.0)])
        second = analyze_song_change([note(48, 2.0, 0.1)], [], [note(48, 2.0)])
        merged = merge_profiles([
            first["profiles"]["recovered"],
            second["profiles"]["recovered"],
        ])
        self.assertEqual(merged["notes"], 2)
        self.assertEqual(merged["velocity"]["under_0.20"], 2)

    def test_profile_rate_uses_reference_bucket_denominators(self):
        reference = analyze_song_change(
            [note(60, 1.0, 0.1), note(62, 2.0, 0.1)], [], [],
        )["profiles"]["reference"]
        subset = analyze_song_change(
            [note(60, 1.0, 0.1)], [], [],
        )["profiles"]["reference"]
        rates = profile_rates(subset, reference)
        self.assertEqual(rates["notes"], 0.5)
        self.assertEqual(rates["velocity"]["under_0.20"], 0.5)


if __name__ == "__main__":
    unittest.main()
